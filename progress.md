# Progress: Linear Attention Embeddings

## Current State

The embeddings training system now uses **linear attention** for self-supervised optical flow learning, replacing the previous softmax-based spatial variance loss.

### Session: 2026-04-10 — Normalized Linear Attention and Bounded Losses

Transitioned from raw linear attention to **normalized weight averaging** with interpretable, bounded loss values. All loss components are now positive numbers where 0 represents the theoretical optimum.

#### Behavior Changes

- **COM is now a true weighted average** — Attention weights are normalized by their sum, ensuring center-of-mass stays within coordinate bounds [0, 1] instead of potentially exceeding them
- **Flow has interpretable scale** — `flow = 1.0` now means "full window span" (16 pixels for 16×16 window), making flow magnitudes interpretable during training
- **Diversity loss is bounded** — Changed from unbounded negative variance to normalized loss: `1 - (variance / 0.25)`, where 0 = maximum diversity, 1 = complete collapse
- **All losses positive, 0 = perfect** — Total loss is now a meaningful sum where lower is always better (previously could go negative)
- **Embeddings use ReLU activation** — Embedding generator outputs are now ReLU'd before L2 normalization, restricting to positive orthant (required for weight normalization to be meaningful)

#### Diagnostic Additions

- **Flow range monitoring** — Logs self_com min/max, cross_com min/max, flow min/max to verify COM stays in expected [0, 1] range
- **Weight sum monitoring** — Logs average activation sum per position to detect embedding collapse (near-zero weights)
- **Diversity score** — Reports variance as percentage of theoretical maximum (0.25) for quick assessment

#### Implementation Details

- Weight normalization: `COM = (Q @ K_coords) / Q.sum(axis=-1)`
- Flow rescaling: `flow_pixels = flow_normalized * window_size` before warping
- Max theoretical variance: 0.25 for L2-normalized embeddings in positive orthant
- New aux field `diversity_variance` preserves raw variance for reference

### Linear Attention Mechanism

Each 16×16 window computes flow via center-of-mass decoding:
- **Pre-compute per window:** `K_coords = K.T @ coords` maps each embedding dimension to a spatial position
- **Per-query decoding:** `COM = Q @ K_coords` finds where the query embedding points (O(D) not O(N²))
- **Flow estimate:** `flow = cross_frame_COM - self_frame_COM`

This reduces attention computation from ~3.9M FLOPs per window (softmax) to ~57K FLOPs (linear) — a 70× speedup.

### Training Signal

**Warped reconstruction loss** is the core self-supervised signal:
1. Predict flow from linear attention
2. Warp frame2 embeddings by predicted flow using bilinear interpolation
3. Compare warped frame2 to frame1 (MSE)

The model learns embeddings that are temporally stable — the same spatial position in frame1 and frame2 should have similar embeddings after warping.

**Embedding diversity loss** prevents collapse by encouraging embeddings to vary across spatial positions (maximizes spatial variance).

### Architecture

- **Input:** 81×81 RGB frames (configured for 1×1 coarse grid with 16×16 windows)
- **Pyramid:** 3 levels (73×73, 35×35, 16×16) with MobileNet V4-inspired UIB blocks
- **Output:** 16-dim L2-normalized embeddings per pixel
- **Flow:** 2D vector per pixel, computed independently per window

### Training Configuration

- **Loss weights:** λ_reconstruction=1.0, λ_diversity=0.1
- **Level weighting:** Uniform (decay=1.0) across pyramid levels
- **Diversity scope:** Per-window (matches attention structure)

### Known Behaviors

- Flow is clipped implicitly by the linear attention mechanism (dimensions encode spatial information, not arbitrary offsets)
- Coarsest level (16×16) produces exactly one window — minimal but functional
- Early training produces high-variance flow; reconstruction loss drives convergence

### Deviations from Original Plan

- **Flow concordance loss deferred:** The original intent included a third loss term encouraging dimensions to agree on flow direction. Removed to keep scope minimal — can be added once base training is stable.
- **No MLP refinement:** Raw flow (cross_com - self_com) is used directly. MLP refinement deferred.

---

## Next Session Considerations

The gap between this progress and `intent.md` determines what's next. Current intent includes flow estimation module (separate from embeddings loss) and potential ablations (feature maps, diversity scope, loss weights).
