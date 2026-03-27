# Barevision: Hierarchical Optical Flow Architecture

This document outlines the design and architecture of the Barevision optical flow model. The system is divided into two primary components:

1. **The Embedding Engine** (Currently Implemented): A self-supervised hierarchical feature extractor trained with spatial variance loss.
2. **The Flow Estimation Pipeline** (Currently Implemented): A hierarchical tracker utilizing attention-based feature matching with reconstruction loss.

---

## 1. The Embedding Engine (Implemented)

The embedding engine processes raw RGB frames into a multi-scale pyramid of dense, visually unique feature embeddings. It is designed to be highly efficient on NPUs by strictly utilizing dense and grouped convolutions without complex routing or spatial padding.

### Architecture: The "Decoupled Cascade"

To prevent the destruction of high-frequency structural details during downsampling, feature extraction is strictly decoupled from spatial reduction.

* **Strictly `VALID` Padding:** The network uses no artificial padding (`padding='VALID'`). This drops edge pixels at every convolution, structurally preventing misleading boundary data (zero-padding) from corrupting the embeddings.
* **L2 Normalized Hypersphere:** The final embeddings at each level are L2 normalized to unit length. Crucially, no ReLU activation is applied before normalization, allowing the network to utilize the full [-1.0, 1.0] cosine similarity space.

**Pyramid Blocks:**

1. **StemBlock (Level 0):** Operates on raw 3-channel RGB. Expands the receptive field to 25 pixels using two stacked 3x3 `stride=1` convolutions.
   * `Conv(Dense, 3→32)` → `GroupNorm` → `GELU`
   * `Conv(Groups=8, 32→32)` → `GroupNorm` → `GELU`
   * *Branch A (Embed):* 1x1 Conv (32→16) → L2 Norm
   * *Branch B (Downsample):* 3x3 Conv, `stride=2`, `VALID`

2. **StandardBlock (Levels 1 to N):** Refines features for coarser levels.
   * `Conv(Groups=8, 32→32)` → `GroupNorm` → `GELU`
   * *Branch A (Embed):* 1x1 Conv (32→16) → L2 Norm
   * *Branch B (Downsample):* 3x3 Conv, `stride=2`, `VALID` (Omitted on the final level).

### Resolution and the Spatial Buffer

Because the network drops edge pixels via `VALID` padding, the physical resolution of the feature maps shrinks slightly more than a standard 2x downsample at every level. For example, to get a clean 16x16 grid at Level 2, the network must ingest an 83x83 crop at Level 0.

This means the finer levels (Level 0 and Level 1) physically cover a wider field of view than the coarsest level. This "extra" resolution acts as a crucial spatial buffer that becomes necessary during the flow estimation phase.

---

## 2. Flow Estimation Pipeline (Implemented)

The flow estimation pipeline estimates optical flow using attention-based feature matching across pyramid levels. The system processes each level independently and uses reconstruction loss for training.

### Hierarchical Flow Estimation

**Architecture:** The flow estimator operates at all pyramid levels:

1. **Per-Level Processing:** Each level's embeddings are split into 16×16 windows
2. **Independent Estimation:** A dedicated MLP predicts flow for each window
3. **Multi-Level Loss:** Reconstruction loss computed at each level

**Spatial Buffer Utilization:** The centered crop strategy (from VALID padding) provides symmetric buffer space on all sides:
- Level 0: 79×79 → 64×64 (7-8 pixels buffer per side)
- Level 1: 37×37 → 32×32 (2-3 pixels buffer per side)
- Level 2: 16×16 → 16×16 (no crop needed)

This buffer accommodates motion near window boundaries and enables future window-shifting strategies.

### The Constellation & Centroids

Because the embeddings form stable "signatures" or "constellations" of attention peaks rather than single solid blobs, tracking relies on comparing the center of mass (Centroid) of the Self-Attention map to the Cross-Attention map.

### Temperature Scaling

Two separate temperature parameters control attention sharpness during embeddings training:

* **`self_temperature` (default: 0.3):** Controls self-attention softmax sharpness. Lower values produce sharper peaks concentrated near the source pixel.
* **`cross_temperature` (default: 0.3):** Controls cross-attention softmax sharpness. Lower values produce sharper peaks for confident cross-frame matching.

This separation allows independent tuning of self-attention (embedding uniqueness) and cross-attention (matching confidence) behavior.

### The Boundary Problem (Centroid Drag)

When a feature moves near the edge of a 16x16 window, the boundary physically truncates part of the attention signature. Taking the geometric centroid of a truncated signature artificially skews the flow estimate backward (Centroid Drag).

### Solution: The Flow Estimator

Instead of utilizing explicit mathematical geometry to correct boundary clipping, the pipeline employs a small, per-embedding MLP to statistically predict the local residual flow based on cheap spatial features.

For every embedding, the following 8-float feature vector is computed:

1. **Self-Relative Centroid (2 floats):** Offset of self-attention centroid from source position. Should be ~0 for well-formed self-attention.
2. **Cross-Relative Centroid (2 floats):** Offset of cross-attention centroid from source position. This is the primary flow signal.
3. **Cross-Absolute Centroid (2 floats):** Absolute position of cross-attention centroid in the window [0, 1]. Provides boundary context for detecting edge clipping.
4. **Self Max Peak (1 float):** Maximum attention weight in self-attention map. Indicates self-attention sharpness (confidence).
5. **Cross Max Peak (1 float):** Maximum attention weight in cross-attention map. Indicates matching confidence.

These 8 floats are passed through a 2-layer MLP:
```
Linear(8 → 16) → ReLU → Linear(16 → 16) → ReLU → Linear(16 → 2, no bias) → tanh → scale
```

The network outputs two values:
* **Residual U:** The local X displacement.
* **Residual V:** The local Y displacement.

Output is bounded to [-0.5, 0.5] in normalized coordinates (half-window maximum flow).

**Design rationale:**

* **Translation invariance:** By using relative centroids (offset from source), the same flow pattern produces identical features regardless of position in the window. This removes the need for absolute source position and reduces the learning burden.
* **Boundary detection:** Cross-absolute centroid position provides context for detecting when flow approaches window edges, helping the model distinguish real flow from boundary clipping artifacts.
* **Confidence signals:** Max peak values for both self and cross attention serve as confidence indicators, useful for downstream aggregation and detecting ambiguous matches.
* **Conservative initialization:** Output layer uses small kernel weights (normal(0.02)) with no bias, starting with near-zero predictions and no direction preference.

### Flow Aggregation

At each level, the predicted residual flows are aggregated across the entire image. The confidence features (max peaks) are available for future confidence-weighted aggregation. This robust aggregation is then used for training with reconstruction loss.

---

## 3. Training Strategy (Implemented)

The model uses a two-phase training approach with separate objectives for embeddings and flow estimation.

### Phase 1: Standalone Embeddings Training (Primary)

Embeddings are trained independently using **Spatial Variance Loss** to encourage spatially concentrated attention patterns:

**Spatial Variance Loss Formulation:**
- For each query position, compute the weighted mean position from attention weights
- Measure the variance of attention-weighted coordinates around that mean
- Minimize variance → attention peaks become spatially localized

**Properties:**
- Lower variance = attention concentrates near specific locations
- Self-attention: peaks cluster around the source pixel (encourages unique embeddings)
- Cross-attention: finds specific matches in the target frame (encourages confident matching)
- Space-aware: Unlike entropy, spatial variance penalizes scattered attention even if the distribution is peaky

**Loss combination:**
```
total_loss = lambda_self * self_variance + (1 - lambda_self) * cross_variance
```

Where:
- `self_variance`: Spatial variance of self-attention (default weight: 0.5)
- `cross_variance`: Spatial variance of cross-attention (default weight: 0.5)
- Both computed hierarchically across pyramid levels with configurable weighting

This formulation ensures embeddings are both distinctive (sharp self-attention) and matchable (sharp cross-attention), with the spatial constraint preventing scattered attention patterns.

### Phase 2: Flow Estimation Training

Flow estimation is trained using **Reconstruction Loss** in latent space:

**Reconstruction Loss:**
- Uses estimated flow to warp Frame 1 embeddings
- Minimizes L2 distance between warped Frame 1 and Frame 2 embeddings
- Computed independently at each pyramid level
- Ensures embeddings are trackable across frames

**Future: Joint Fine-Tuning**
The architecture supports loading pretrained embeddings and fine-tuning jointly with flow estimation, combining spatial variance and reconstruction losses. This is deferred pending validation of the standalone embeddings baseline.

---

## 4. Training Infrastructure

### Checkpointing

**Standalone Embeddings Checkpoints:**
- Saved during embeddings training (`embeddings/training.py`)
- Contains model state, step, and `EmbeddingsSettings` configuration
- Can be loaded into flow estimation pipeline later

**Joint Training Checkpoints (Outdated):**
- Legacy checkpoints from combined training approach
- Contains full `Settings` object with joint configuration

### Validation

Training includes automatic validation on a held-out set (15% of videos):
- Runs at the end of each epoch (configurable frequency)
- Tracks best validation loss for model selection
- Logs metrics to TensorBoard for monitoring

### Inference

Trained models can be used for optical flow estimation via the inference script, which loads checkpoints and estimates flow between arbitrary image pairs.

See `src/barevision/flow/ARCHITECTURE.md` for detailed implementation documentation.
