# Barevision: Hierarchical Optical Flow Architecture

This document outlines the design and architecture of the Barevision optical flow model. The system is divided into two primary components:

1. **The Embedding Engine** (Currently Implemented): A self-supervised hierarchical feature extractor.
2. **The Flow Estimation Pipeline** (Planned Design): A coarse-to-fine tracker utilizing a learned residual corrector and latent reconstruction loss.

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

## 2. Flow Estimation Pipeline (Partially Implemented)

The flow estimation components are implemented but the full hierarchical cascading inference pipeline is still under development. The reconstruction loss is already integrated into training.

### Hierarchical Cascading and Window Shifting

**Design:** At the coarsest level (Level 2), the network estimates the macro-flow of the scene. This median flow vector is upscaled and passed as a prior to the next finer level (Level 1).

To cancel out global/ego-motion, Frame 2's attention window at Level 1 is physically shifted by this prior to re-center the target object. **This is where the spatial buffer from Part 1 is utilized:** Because Level 1 ingested a wider field of view without artificial padding, we have the valid, uncorrupted feature context required to safely slide this shifted window without falling off the edge of the image tensor.

**Current status:** The FlowEstimator MLP and reconstruction loss are implemented. The full hierarchical cascading inference (passing priors down the pyramid with window shifting) is planned for future implementation.

### The Constellation & Centroids

Because the embeddings form stable "signatures" or "constellations" of attention peaks rather than single solid blobs, tracking relies on comparing the center of mass (Centroid) of the Self-Attention map to the Cross-Attention map.

### Temperature Scaling

Two separate temperature parameters control attention sharpness:

* **`entropy_temperature` (default: 1.0):** Used for entropy loss computation. Fixed at 1.0 for temperature-independent loss values, making entropy comparable across training runs.
* **`flow_temperature` (default: 0.3):** Used during flow estimation forward pass. Controls output sharpness - lower values produce sharper attention peaks for precise flow, higher values produce smoother centroids for robust tracking.

This decoupling prevents the loss landscape from changing when tuning flow estimation behavior.

### The Boundary Problem (Centroid Drag)

When a feature moves near the edge of a 16x16 window, the boundary physically truncates part of the attention signature. Taking the geometric centroid of a truncated signature artificially skews the flow estimate backward (Centroid Drag).

### Solution: The Flow Estimator

### Solution: The Flow Estimator

Instead of utilizing explicit mathematical geometry to correct boundary clipping, the pipeline employs a small, per-embedding MLP to statistically predict the local residual flow based on cheap spatial features.

For every embedding, the following 8-float feature vector is computed:

1. **Self-Relative Centroid (2 floats):** Offset of self-attention centroid from source position. Should be ~0 for well-formed self-attention.
2. **Cross-Relative Centroid (2 floats):** Offset of cross-attention centroid from source position. This is the primary flow signal.
3. **Cross-Absolute Centroid (2 floats):** Absolute position of cross-attention centroid in the window [0, 1]. Provides boundary context for detecting edge clipping.
4. **Self Max Peak (1 float):** Maximum attention weight in self-attention map. Indicates self-attention sharpness (confidence).
5. **Cross Max Peak (1 float):** Maximum attention weight in cross-attention map. Indicates matching confidence.

These 8 floats are passed through a 3-layer MLP (`Linear(8→32) → ReLU → Linear(32→32) → ReLU → Linear(32→32) → ReLU → Linear(32→2)`).
The network outputs two values:

* **Residual U:** The local X displacement.
* **Residual V:** The local Y displacement.

*Note: The residual U and V outputs are continuous values that represent the full local flow of that specific embedding, which can span multiple pixels.*

**Design rationale:**

* **Translation invariance:** By using relative centroids (offset from source), the same flow pattern produces identical features regardless of position in the window. This removes the need for absolute source position and reduces the learning burden.
* **Boundary detection:** Cross-absolute centroid position provides context for detecting when flow approaches window edges, helping the model distinguish real flow from boundary clipping artifacts.
* **Confidence signals:** Max peak values for both self and cross attention serve as confidence indicators, useful for downstream aggregation and detecting ambiguous matches.

*Note: Prior features such as Prior Flow (from coarser levels), Quadrant Masses, and explicit Confidence output are deferred for future implementation once this baseline is validated.*

### Flow Aggregation

At each level, the predicted residual flows are aggregated across the entire image using a **median**, which naturally discards outliers. The confidence features (max peaks) are available for future confidence-weighted aggregation once the baseline is validated. This robust median flow is then upscaled and passed to the next finer level as the new prior.

---

## 3. Dual Loss Formulation (Implemented)

The model is trained end-to-end using a dual objective that balances structural distinctness with accurate tracking:

1. **Entropy Minimization Loss:** Applied to attention maps at all pyramid levels (Self and Cross). This regularizer prevents embeddings from collapsing into trivial, perfectly smooth solutions. It forces embeddings to remain locally unique and visually distinct.
2. **Reconstruction Loss (Latent Space):** The network uses the estimated flow field to warp Frame 1 embeddings and minimizes the L2 distance between warped Frame 1 embeddings and Frame 2 embeddings. This ensures features are trackable across frames.

**Loss combination:**
```
total_loss = entropy_loss + recon_weight * reconstruction_loss
```

Where:
- `entropy_loss`: Primary objective ensuring distinctive embeddings
- `reconstruction_loss`: Secondary objective ensuring embeddings are trackable  
- `recon_weight`: Controls relative importance (default: 0.1)

This formulation treats entropy as the foundation (without distinctive embeddings, flow is ambiguous) and reconstruction as a grounding signal (ensuring distinctive features actually correspond across frames).

**Entropy loss details:**
- Computed per 16×16 window at each pyramid level
- Normalized by theoretical maximum `log(window_size²)` to [0, 1] range
- Aggregated across levels with configurable weighting (default: uniform per-pixel)
- Temperature scaling (default: 0.2) controls softmax sharpness

By optimizing both simultaneously, the embedding engine learns features that are specifically optimized to be trackable, while entropy loss guarantees those features remain sharply grounded in visual structure.

---

## 4. Training Infrastructure

### Checkpointing

The training pipeline automatically saves model checkpoints at three points:

1. **Periodic**: Every N steps during training (configurable)
2. **Best Model**: When validation loss improves (automatic)
3. **Final**: When training completes

Checkpoints include model state, training step, and full configuration for reconstruction during inference.

### Validation

Training includes automatic validation on a held-out set (15% of videos):
- Runs at the end of each epoch (configurable frequency)
- Tracks best validation loss for model selection
- Logs metrics to TensorBoard for monitoring

### Inference

Trained models can be used for optical flow estimation via the inference script, which loads checkpoints and estimates flow between arbitrary image pairs.

See `src/barevision/flow/ARCHITECTURE.md` for detailed implementation documentation.
