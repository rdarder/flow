# Changes: Linear Attention for Embeddings Training

## Goal
Enable efficient self-supervised flow training within ~2M FLOPs per lookup window by replacing softmax attention with linear attention in the embeddings generator.

## Core Tension
**Efficiency vs. Expressiveness** — Softmax attention provides strong matching signals but requires O(N²) computation that exceeds FLOPs budget. Linear attention reduces to O(D) per query by exploiting the low-rank structure of the logit matrix, but each dimension must now encode both matching and spatial information.

## The Bet
The model can learn a linear map from embedding space to spatial space without explicit positional encoding. Dimensions don't have fixed spatial meanings — instead, a learned "dictionary" (K_coords) maps dimension activations to positions. This enables 70× speedup while maintaining self-supervised training signals.

## Constraints
- ~2M FLOPs per lookup window (simpler heuristic than total budget)
- No GatherND at inference time (training-only operation)
- Window size remains 16×16 for now (not a hard constraint, just current scope)

## Decisions
| Idea | Status | Why |
|------|--------|-----|
| Linear attention mechanism | ✅ Adopted | 70× speedup, natural confidence metrics, aligns with low-rank embedding structure |
| Warped reconstruction loss | ✅ Adopted | Core self-supervised signal: frame1 ≈ warp(frame2, flow) |
| Embedding diversity loss | ✅ Adopted | Prevents constant embedding collapse |
| Flow concordance loss | ✅ Adopted | Training regularizer: encourages dimensions to encode spatial information coherently. Not a standalone confidence metric (D=16 is too low for consistent agreement), but helps the flow estimator infer channel reliability. |
| Self-attention position reconstruction | ❌ Rejected | Imposes structure we don't need; self-COM doesn't need to match actual position |
| Sparse/localized attention | ❌ Rejected | Would miss large motions; want single-level solution first |
| Argmax + logit statistics | ❌ Rejected | Argmax is brittle; self-attention peak is always at self position |

## Unknowns / Ablation Questions
- **Feature map φ**: Start with identity (φ(Q) = Q). Alternatives (ReLU, exp()) for sharper responses — experiment later.
- **MLP for flow refinement**: Start with raw flow (cross_com - self_com). Later: experiment with MLP taking per-dimension flow contributions as input for full estimation.
- **Loss weights**: λ_reconstruction=1.0, λ_diversity=0.1, λ_concordance=0.1. Concordance is a mild regularizer, not a dominant signal.
- **Embedding dimension D=16**: To be revisited in a separate brainstorm — increasing D affects the inverted bottleneck architecture and has broader implications beyond linear attention.
- **Diversity scope**: Current implementation uses per-window diversity. Alternative: global diversity across entire feature map. Per-window matches attention structure better; global may provide stronger collapse prevention. Topic for future ablation.
- **Concordance max variance**: Set to 0.5 as initial guess. May need tuning based on observed variance during training.

## Key Insights
- **Low-rank structure**: The logit matrix L = Q @ K.T is (256, 256) but has rank at most D=16. We're materializing 65K values that live in a 16-dimensional subspace.
- **Geometric intuition**: Softmax asks "which K vectors match me?" (instance-based). Linear attention asks "where do K vectors like me tend to be?" (dimension-based decoding).
- **Collapse prevention**: Warped reconstruction alone doesn't prevent constant embedding collapse. Need explicit diversity pressure.
- **Concordance as regularizer**: Flow concordance encourages coherent spatial encoding across dimensions. Not a standalone confidence metric (D=16 too low), but the flow estimator can use per-channel statistics (e.g., activation magnitude) to infer whether disagreement is expected or problematic.

## Numbers to Preserve
- Target: ~2M FLOPs per lookup window
- Window: 16×16 (256 positions)
- Embedding dimension: D=16 (current, may ablate)
- Expected speedup: ~70× vs softmax attention
- Linear attention per window: ~57K FLOPs (vs ~3.9M for softmax)

## Detailed Spec: Linear Attention Mechanism

### Pre-compute (once per window)
```python
# Self-attention: where do my dimensions point in frame 1?
K_coords_self = Q.T @ coords  # (D, 2)

# Cross-attention: where do K dimensions point in frame 2?
K_coords_cross = K.T @ coords  # (D, 2)
```

### Per-query decoding (O(D) not O(N))
```python
# Center of mass in frame 1
self_com = Q @ K_coords_self  # (2,)

# Center of mass in frame 2
cross_com = Q @ K_coords_cross  # (2,)

# Raw flow estimate
flow = cross_com - self_com  # (2,) per pixel
```

### Confidence metric
```python
# Per-dimension flow contribution
flow_per_dim = cross_pos_per_dim - self_pos_per_dim  # (D, 2)

# Variance = disagreement between dimensions
confidence = -flow_per_dim.var(axis=0).mean()
```

### Loss components
```python
total_loss = (
    λ₁ * warped_reconstruction_loss +    # frame1 ≈ warp(frame2, flow)
    λ₂ * embedding_diversity_loss +       # prevent collapse
    λ₃ * flow_concordance_loss            # encourage coherent dimensions
)
```

## Scope Clarification

**This intent covers the embeddings module only.** The loss function estimates flow as part of training embeddings, but there is no dedicated flow estimator module yet. Once embeddings are convincing, a separate flow estimator module will be implemented (may be as simple as moving loss function logic to a model, or more complex).

## References
- `brainstorm.md` — Full mental model, FLOP analysis, and JEPA connection
- **JEPA reference** (deferred): Gradient stopping on target / momentum encoder techniques from V-JEPA literature — revisit if training shows instability or collapse.
