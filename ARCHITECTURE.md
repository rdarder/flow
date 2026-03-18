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

*Note: The Flow Estimator described below is partially implemented. The current implementation uses only 6 input features (Source Position + Centroids) and outputs 2 values (Residual U, V). The full feature set including Prior Flow, Quadrant Masses, Max Peak Value, and Confidence output is planned for future implementation.*

Instead of utilizing explicit mathematical geometry to correct boundary clipping, the pipeline employs a small, per-embedding MLP to statistically predict the local residual flow based on extremely cheap spatial features.

For every embedding, the following 18-float feature vector is **planned**:

1. **Source Position:** Local `X, Y` coordinates (2 floats).
2. **Prior Flow:** `U, V` vector passed down from the parent level (2 floats).
3. **Centroids:** `Cx, Cy` for both Self and Cross attention maps (4 floats).
4. **Quadrant Masses:** The unnormalized sum of attention weights split into 4 spatial quadrants for both Self and Cross maps. This acts as a cheap geometric proxy for boundary truncation (8 floats).
5. **Max Peak Value:** The absolute maximum attention weight for Self and Cross maps. Serves as a cheap proxy for variance/scatter (2 floats).

These 18 floats are passed through a small neural network (`Linear(18→24) → ReLU → Linear(24→3)`).
The network outputs three values:

* **Residual U:** The local X displacement relative to the shifted window.
* **Residual V:** The local Y displacement relative to the shifted window.
* **Confidence:** A score [0, 1] predicting the reliability of the patch. *Not yet implemented.*

*Note: The residual U and V outputs are continuous values that represent the full local flow of that specific embedding, which can span multiple pixels.*

### Flow Aggregation

*Note: Confidence-weighted aggregation is planned but not yet implemented. Current implementation uses simple median.*

At each level, the predicted residual flows are added to the prior flow. The flows are then aggregated across the entire image using a **Confidence-Weighted Median**, which naturally discards ambiguous or occluded patches. This robust median flow is then upscaled and passed to the next finer level as the new prior.

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
