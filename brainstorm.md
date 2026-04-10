# Brainstorm: Linear Attention for Flow Estimation

**Date:** 2026-04-10  
**Participants:** rdarder, AI assistant  
**Context:** Re-focusing on flow estimation after simplifying embeddings model

---

## The Wish

Build an efficient flow estimation system that:
- Works with our hierarchical embeddings (16-dim vectors per pixel)
- Uses self/cross attention within non-overlapping windows (16×16)
- Produces one flow vector per pixel (dense estimation)
- Fits within ~128M FLOPs budget for 64 windows (8×8 grid of 16×16 windows)
- Can be trained self-supervised (no ground-truth flow required)

**Core insight:** Softmax is a nonlinearity that prevents factorization and forces O(N²) computation. Can we break away from softmax and work in the low-rank (D-dimensional) space instead?

---

## Core Tension

**The problem with softmax attention:**

For a 16×16 window (256 positions, D=16):

| Operation | FLOPs per window | FLOPs for 64 windows |
|-----------|------------------|----------------------|
| Self-attention logits (Q@K.T) | 1.05M | 67M |
| Self-attention softmax | 0.52M | 33M |
| Cross-attention logits | 1.05M | 67M |
| Cross-attention softmax | 0.52M | 33M |
| **Total (attention only)** | **~3.1M** | **~200M** |

This exceeds our 128M budget *before* extracting any features or estimating flow.

**The key observation:** The logit matrix L = Q @ K.T is (256, 256) but has **rank at most D=16**. We're materializing 65K values that live in a 16-dimensional subspace.

---

## Explored Options

### Option 1: Sparse/Localized Attention
**Idea:** Only compute attention in a local neighborhood (e.g., 5×5 around each position).  
**Status:** ❌ Discarded for now  
**Why:** Large motions would be missed. Hierarchical model should handle motion range, but we want a solution that works at a single level first.

### Option 2: Argmax + Logit Statistics
**Idea:** Skip softmax, use argmax of logits + simple statistics (peak value, logit variance).  
**Status:** ⚠️ Explored but concerns raised  
**Why:** Argmax is brittle. Self-attention peak is always at self position (logit=1 for normalized embeddings), so it doesn't give useful information.

### Option 3: Linear Attention (Adopted)
**Idea:** Use the low-rank structure directly. Pre-compute K statistics, then decode per-query in O(D) instead of O(N).  
**Status:** ✅ Adopted for prototyping  
**Why:** 100× speedup, natural confidence metrics, aligns with how embeddings are trained.

---

## Linear Attention: The Mechanism

### What It Is

Instead of asking "which K vectors match me?" (softmax attention), linear attention asks "where do K vectors like me tend to be?"

**Pre-compute once per window:**
```python
K_coords = K.T @ coords  # (D, 2) - for each embedding dimension, where in space does it fire?
```

**Per query (O(D) not O(N)):**
```python
com = Q @ K_coords  # (2,) - center of mass
```

### Geometric Intuition

- **Softmax attention:** Instance-based matching. "Find my twins in this crowd, average their positions."
- **Linear attention:** Dimension-based decoding. "Which embedding dimensions define me? Where do those dimensions tend to live?"

**The bet:** The model can learn a linear map (K_coords) from embedding space to spatial space. Dimensions don't have fixed spatial meanings (no positional encoding). Instead, K_coords is a learned "dictionary": each dimension d has an associated spatial signature (where in space that dimension tends to fire). The query Q activates a combination of dimensions, and K_coords decodes that combination back to position. There's no requirement that dimension 0 = left side — just that the dictionary is consistent within each window.

### Detailed FLOP Analysis

**Setup:** 16×16 window (N=256 positions), D=16 embedding dimensions, coords=(256, 2)

#### Softmax Attention (Per Window)

| Operation | Calculation | FLOPs |
|-----------|-------------|-------|
| **Self-attention** | | |
| Logits (Q @ K.T) | 256 × 256 × 16 (matmul) | 1.31M |
| Softmax per row | 256 rows × 256 positions × ~8 ops (exp, sum, div) | 0.52M |
| COM (weights @ coords) | 256 × 256 × 2 (matmul) | 0.13M |
| Self-attention subtotal | | **1.96M** |
| **Cross-attention** | | |
| Logits (Q @ K.T) | 256 × 256 × 16 | 1.31M |
| Softmax per row | 256 × 256 × ~8 | 0.52M |
| COM (weights @ coords) | 256 × 256 × 2 | 0.13M |
| Cross-attention subtotal | | **1.96M** |
| **Per window total** | | **~3.9M** |
| **64 windows** | | **~250M** |

*Note: FLOP count for matmul is 2×M×N×K for M×N @ N×K, but we count multiply-add as 1 FLOP for simplicity.*

#### Linear Attention (Per Window)

| Operation | Calculation | FLOPs |
|-----------|-------------|-------|
| **Pre-compute (once per window)** | | |
| K_coords_self = Q.T @ coords | 16 × 256 × 2 | 8K |
| K_coords_cross = K.T @ coords | 16 × 256 × 2 | 8K |
| Pre-compute subtotal | | **16K** |
| **Per-query (×256 queries)** | | |
| self_com = Q @ K_coords_self | 256 × (16 × 2) | 8K |
| cross_com = Q @ K_coords_cross | 256 × (16 × 2) | 8K |
| flow = cross - self | 256 × 2 | 1K |
| Per-query subtotal | | **17K** |
| **Flow concordance (confidence)** | | |
| flow_per_dim = cross_pos_per_dim - self_pos_per_dim | 256 × 16 × 2 (elementwise) | 8K |
| flow_variance = var(flow_per_dim, axis=0) | 256 × 16 × ~4 ops | 16K |
| Concordance subtotal | | **24K** |
| **Per window total** | | **~57K** |
| **64 windows** | | **~3.6M** |

#### Speedup Summary

| Metric | Softmax | Linear | Ratio |
|--------|---------|--------|-------|
| Per window | 3.9M | 57K | 68× |
| 64 windows | 250M | 3.6M | 69× |
| + MLP refinement (optional) | - | +9M | Still 25× |

**With MLP refinement:** Add a small MLP (15→32→2) per pixel for flow refinement: ~550 FLOPs/pixel × 256 pixels × 64 windows ≈ 9M FLOPs. Total: ~12.6M, still 20× faster than softmax baseline.

### Flow Estimation

```python
# Self-attention COM (where do I point in frame 1?)
self_com = Q @ (Q.T @ coords)

# Cross-attention COM (where do I point in frame 2?)
cross_com = Q @ (K.T @ coords)

# Raw flow estimate
flow = cross_com - self_com  # (2,) per pixel
```

**Confidence metric:** Flow concordance (variance across dimensions)
```python
# Per-dimension flow contribution
flow_per_dim = cross_pos_per_dim - self_pos_per_dim  # (D, 2)

# Variance = disagreement between dimensions
confidence = -flow_per_dim.var(axis=0).mean()
```

Low variance = all dimensions agree on flow = confident match.  
High variance = dimensions disagree = ambiguous match.

---

## Loss Function Discussion

### What We Want

Self-supervised signal that:
1. Encourages embeddings to be temporally stable (same position in frame 1 and 2 should have similar embeddings)
2. Prevents collapse (constant embeddings, zero flow)
3. Doesn't impose unnecessary structure (self-attention doesn't need to be centered)

### Explored Losses

| Loss | What It Does | Status |
|------|--------------|--------|
| **Self-attention position reconstruction** | Encourage self-COM to match actual position | ❌ Discarded - imposes structure we don't need |
| **Warped reconstruction** | Warp cross-embeddings by predicted flow, compare to self-embeddings | ✅ Adopted - core self-supervised signal |
| **Embedding diversity** | Encourage embeddings to vary across positions | ✅ Adopted - prevents constant collapse |
| **Flow concordance** | Encourage dimensions to agree on flow direction/magnitude | ✅ Adopted - confidence metric + auxiliary loss |

### Final Loss Components

```python
total_loss = (
    λ₁ * warped_reconstruction_loss +    # Core signal: frame1 ≈ warp(frame2, flow)
    λ₂ * embedding_diversity_loss +       # Prevent collapse: embeddings should vary
    λ₃ * flow_concordance_loss            # Encourage coherent dimensions
)
```

**Warped reconstruction:** Uses GatherND (available at training time) to warp cross-frame embeddings by predicted flow. Compares warped embeddings to original self-embeddings.

**Embedding diversity:** Maximizes variance of embeddings across spatial positions. Prevents "all positions output same embedding" collapse.

**Flow concordance:** Minimizes variance of per-dimension flow contributions. Encourages all dimensions to "flow together."

---

## Collapse Modes and Prevention

| Collapse Mode | Description | Prevention |
|---------------|-------------|------------|
| **Constant embedding** | All positions output same embedding | Embedding diversity loss |
| **Zero flow** | Model always predicts no motion | Dataset diversity (varied motions in training data) |
| **Frame-specific** | Frame 1 = c₁, Frame 2 = c₂ (different constants) | Warped reconstruction loss |
| **Random noise** | Each position has random embedding | Flow concordance + warped reconstruction |

**Key insight:** Warped reconstruction alone doesn't prevent constant embedding collapse (if all embeddings are constant, warping does nothing). Need explicit diversity pressure.

---

## Unknowns / Open Questions

| Question | Status |
|----------|--------|
| **Feature map for linear attention** | Using identity (φ(Q) = Q) for now. Alternatives: ReLU, exp() for sharper responses. |
| **MLP for flow refinement** | First implementation: raw flow (cross - self). Long-term: small MLP taking per-dimension flow contributions as input. |
| **Loss weights (λ₁, λ₂, λ₃)** | TBD through experimentation. Expect λ₁ (reconstruction) to dominate. |
| **Gradient stopping on target** | JEPA literature suggests stop_grad on warped cross-embeddings. Not yet decided. |
| **Momentum encoder** | JEPA technique for stability. Not adopted yet — keep as reference. |
| **Embedding dimension (D=16)** | Linear attention dimensions may be less expressive than softmax attention (each dimension must encode spatial information, not just support matching). Current D=16 might be a bottleneck. Inverted bottleneck architecture compresses to D=16 — consider keeping expanded dimensions (D=32 or higher)? Trade-off: higher D increases linear attention cost proportionally (O(D) per query). **Topic for future exploration.** |

---

## False Starts

### Softmax-Free Logit Statistics
**Initial idea:** Compute statistics (mean, variance, argmax) from raw logits without softmax.  
**Why it didn't work:** Argmax is brittle. Self-attention diagonal is always 1 (normalized embeddings), so peak location is trivial. Logit variance requires full matrix anyway.

### Pre-computed Softmax Moments
**Initial idea:** Pre-compute K statistics to approximate softmax-weighted COM without full matrix.  
**Why it failed:** Partition function Z = Σⱼ exp(Qᵢ·Kⱼ/T) is nonlinear — can't factorize. Requires all N dot products per query anyway.

**Pivot:** Abandon softmax entirely. Linear attention gives us O(D) per query with natural confidence metrics.

---

## Detailed Spec (First Prototype)

### Architecture
- **Input:** Two frames, each with embeddings (B, H, W, D=16)
- **Windowing:** Non-overlapping 16×16 windows
- **Per window:**
  - Pre-compute: K_coords_self = Q.T @ coords, K_coords_cross = K.T @ coords
  - Per query: self_com = Q @ K_coords_self, cross_com = Q @ K_coords_cross
  - Flow: cross_com - self_com
  - Confidence: -var(flow_per_dim)

### Training
- **Warped reconstruction:** Use GatherND to sample cross-embeddings at (pos + flow), compare to self-embeddings
- **Embedding diversity:** Maximize spatial variance of embeddings
- **Flow concordance:** Minimize variance of per-dimension flow

### Inference
- Same as training, but no warped reconstruction (no GatherND at inference)
- Output: flow (H, W, 2) + confidence (H, W)

---

## JEPA Connection (Reference, Not Adopted)

Our approach is structurally similar to V-JEPA (video JEPA):
- Two views (frame 1, frame 2)
- Encoder produces latent representations
- Predictor estimates transformation (flow)
- Self-supervised via reconstruction

**JEPA techniques discussed but not adopted:**
- Gradient stopping on target
- Momentum encoder (target network)
- Masked prediction (subset of positions)
- Prediction head MLP

Keep these in mind if we hit training instability or collapse. Not implementing yet — stay minimal.

---

## Next Steps (To Be Decided)

1. **Embeddings-first:** Change embeddings loss to linear-attention-based, keep rest of infra. Train embeddings in isolation.
2. **Flow estimation:** Once embeddings are stable, add flow estimation on top.
3. **Compare approaches:** Linear attention vs. softmax attention vs. other baselines.

---

## Notes for Implementation

- **GatherND is available at training time** — the inference constraint (no GatherND) only applies to deployment.
- **Hierarchical model unchanged** — this brainstorm focused on single-level flow estimation.
- **Config system already in place** — can add linear attention options alongside existing softmax attention config.
