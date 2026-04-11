# Progress: Linear Attention Embeddings

## Current State

The embeddings training system uses **linear attention** for self-supervised optical flow learning. Flow is estimated via center-of-mass decoding, with three loss components: warped reconstruction, embedding diversity, and flow concordance.

### Linear Attention Mechanism

Each 16×16 window computes flow via center-of-mass decoding:
- **Pre-compute per window:** `K_coords = K.T @ coords` maps each embedding dimension to a spatial position
- **Per-query decoding:** `COM = Q @ K_coords` finds where the query embedding points (O(D) not O(N²))
- **Flow estimate:** `flow = cross_frame_COM - self_frame_COM`

This reduces attention computation from ~3.9M FLOPs per window (softmax) to ~57K FLOPs (linear) — a 70× speedup.

### Training Signal

**Warped reconstruction loss** (λ=1.0) is the core self-supervised signal:
1. Predict flow from linear attention
2. Warp frame2 embeddings by predicted flow using bilinear interpolation
3. Compare warped frame2 to frame1 (MSE)

The model learns embeddings that are temporally stable — the same spatial position in frame1 and frame2 should have similar embeddings after warping.

**Embedding diversity loss** (λ=0.1) prevents collapse by encouraging embeddings to vary across spatial positions. Normalized: 0 = maximum diversity, 1 = complete collapse.

**Flow concordance loss** (λ=0.1) acts as a regularizer, penalizing variance in per-dimension flow predictions. Encourages dimensions to encode spatial information coherently. Normalized: 0 = perfect agreement, 1 = max disagreement (max_variance=1.0 derived empirically for D=16).

### Architecture

- **Input:** 81×81 RGB frames (configured for 1×1 coarse grid with 16×16 windows)
- **Pyramid:** 3 levels (73×73, 35×35, 16×16) with MobileNet V4-inspired UIB blocks
- **Output:** 16-dim L2-normalized embeddings per pixel
- **Flow:** 2D vector per pixel, computed independently per window

### Training Configuration

- **Loss weights:** λ_reconstruction=1.0, λ_diversity=0.1, λ_concordance=0.1
- **Level weighting:** Uniform (decay=1.0) across pyramid levels
- **Diversity scope:** Per-window (matches attention structure)

### Implementation Details

- **COM normalization:** Attention weights normalized by sum, ensuring center-of-mass stays within [0, 1]
- **Flow scale:** `flow = 1.0` means "full window span" (16 pixels for 16×16 window)
- **Efficient computation:** Per-dimension positions computed first, then aggregated to COM (avoids redundant einsum)
- **All losses positive, 0 = perfect:** Total loss is a meaningful sum where lower is always better

### Diagnostic Additions

- **Flow range monitoring:** Logs self_com min/max, cross_com min/max, flow min/max to verify COM stays in expected [0, 1] range
- **Weight sum monitoring:** Logs average activation sum per position to detect embedding collapse (near-zero weights)
- **Loss breakdown:** TensorBoard logs reconstruction, diversity, and concordance separately, plus per-level breakdown

### Known Behaviors

- Flow is clipped implicitly by the linear attention mechanism (dimensions encode spatial information, not arbitrary offsets)
- Coarsest level (16×16) produces exactly one window — minimal but functional
- Early training produces high-variance flow; reconstruction loss drives convergence
- Concordance loss starts low (~0.01) even with untrained embeddings due to L2 normalization structure

### Deviations from Original Plan

- **No MLP refinement:** Raw flow (cross_com - self_com) is used directly. MLP refinement deferred.
- **Flow concordance as regularizer:** Original intent suggested concordance as confidence metric, but D=16 is too low for consistent agreement. Now used purely as training regularizer.

---

## Next Session Considerations

The gap between this progress and `intent.md` determines what's next. Current intent includes flow estimation module (separate from embeddings loss) and potential ablations (feature maps, diversity scope, loss weights).
