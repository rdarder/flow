# Code Map: Optical Flow Training

This document maps concepts to source files. It tells you **what is where**, not how things work. Read the code for details.

---

## Quick Map

| Concept | Primary File | What's There |
|---------|-------------|--------------|
| Embedding Pyramid | `embeddings/model.py` | `StemBlock`, `StandardBlock`, `HierarchicalEmbeddingModel` |
| Spatial Variance Loss | `embeddings/spatial_losses.py` | Coordinate-based variance loss, self/cross attention, hierarchical aggregation |
| Standalone Training | `embeddings/training.py` | `EmbeddingsTrainer`, independent embeddings training loop |
| Level Flow Estimator | `matching/model.py` | `LevelFlowEstimator` (MLP), `AttentionFeatures` (8 features) |
| Hierarchical Flow | `matching/model.py` | `HierarchicalFlowEstimator` (orchestrates multi-level estimation) |
| Reconstruction Loss | `matching/losses.py` | `warp_embeddings`, `reconstruction_loss_core`, `hierarchical_reconstruction_loss` |
| Joint Training (Outdated) | `joint/training.py` | Combined embeddings + flow training (superseded by standalone) |
| Joint Loss (Outdated) | `joint/losses.py` | Combined spatial variance + reconstruction loss |
| Data Loading | `dataset/video.py` | Frame pair generation, batching |
| Hyperparameters | `settings.py` | tyro CLI config, dataclasses |

---

## Package Structure

```
barevision/flow/
├── embeddings/              # Standalone embeddings training (PRIMARY)
│   ├── model.py            # HierarchicalEmbeddingModel (StemBlock, StandardBlock)
│   ├── spatial_losses.py   # Spatial variance loss functions
│   ├── training.py         # Entry point: python -m barevision.flow.embeddings.training
│   ├── visualization_train.py # Training visualizations (variance maps, attention)
│   ├── visualization.py    # Attention map figures, frame grid overlays
│   ├── test_model.py       # Tests for embedding model
│   └── test_spatial_losses.py # Tests for spatial variance loss
│
├── matching/                # Attention-based feature matching
│   ├── model.py            # LevelFlowEstimator (MLP), HierarchicalFlowEstimator
│   ├── losses.py           # Warp + reconstruction loss
│   ├── visualization.py    # Flow colorwheel, arrow visualizations
│   ├── test_model.py       # Tests for LevelFlowEstimator
│   └── test_hierarchical_model.py  # Tests for HierarchicalFlowEstimator
│
├── joint/                   # Joint training (OUTDATED - kept for reference)
│   ├── model.py            # Combined embeddings + flow model
│   ├── losses.py           # Combined loss (spatial variance + reconstruction)
│   ├── training.py         # Joint training loop (superseded)
│   └── smoke_test.py       # Joint training smoke test
│
├── dataset/video.py         # Data loading (shared)
├── settings.py              # Hyperparameters (shared)
├── checkpoint_utils.py      # Model persistence utilities
├── inference.py             # Flow estimation from checkpoint
├── logging_utils.py         # Console/TensorBoard logging utilities
└── ARCHITECTURE.md          # This file
```

---

## Training Architecture

### Standalone Embeddings Training (Primary)

Embeddings are trained independently using spatial variance loss:

```python
# Entry point
python -m barevision.flow.embeddings.training \
    --model.num_levels=3 \
    --loss.spatial_variance.window_size=16 \
    --loss.spatial_variance.lambda_self=0.5 \
    --training.epochs=10
```

**Key Features**:
- No flow estimator or reconstruction loss
- Faster iteration (no flow computation overhead)
- Spatial variance loss encourages localized attention patterns
- Checkpoints save `EmbeddingsSettings` directly

### Joint Training (Outdated)

The `joint/` package contains the original combined training approach where embeddings and flow were trained together. This has been superseded by standalone embeddings training. The code remains for reference and may be updated in the future to support loading pretrained embeddings for fine-tuning.

---

## Spatial Variance Loss

Replaces entropy minimization with spatial variance to encourage spatially concentrated attention patterns.

**Core Idea**: Attention weights should concentrate near specific spatial locations, not just be peaky in distribution.

**How it Works**:
1. For each query position, compute weighted mean position from attention weights
2. Measure variance of attention-weighted coordinates around that mean
3. Minimize variance → attention peaks become spatially localized

**Self-Attention**: Encourages each pixel to attend primarily to itself and nearby neighbors within the window.

**Cross-Attention**: Encourages finding specific matching locations in the target frame.

**Temperature Control**: Separate temperature parameters for self and cross attention control softmax sharpness (lower = sharper peaks).

---

## File Roles

### Embeddings Package (`embeddings/`)

**`model.py`** — Hierarchical embedding pyramid.
- `StemBlock`: Level 0, RGB input, two stacked 3×3 convs
- `StandardBlock`: Levels 1-N, single 3×3 conv with group normalization
- `HierarchicalEmbeddingModel`: Full pyramid, L2-normalized outputs
- `count_parameters`: Utility for parameter counting
- Read this for embedding architecture changes.

**`spatial_losses.py`** — Spatial variance minimization.
- `_generate_normalized_coordinates`: Creates [0,1] coordinate grids for windows
- `_compute_spatial_variance`: Core variance computation (E[X²] - E[X]²)
- `self_attention_spatial_variance`: Variance loss for self-attention
- `cross_attention_spatial_variance`: Variance loss for cross-attention
- `windowed_spatial_variance_losses`: Single-level window-based loss computation
- `compute_hierarchical_spatial_variance_loss`: Multi-level aggregation with weighting
- `HierarchicalSpatialVarianceLoss`: Main loss class
- Read this for loss function changes.

**`training.py`** — Standalone training orchestrator.
- `EmbeddingsTrainer`: Main training class
- `train_step`: JIT-compiled gradient update
- `validation_step`: Validation without gradients
- `_save_checkpoint`, `_save_best_checkpoint`: Checkpoint helpers
- Entry point: `python -m barevision.flow.embeddings.training`
- Read this for training loop changes.

**`visualization_train.py`** — Training visualizations.
- `log_visualizations`: Orchestrates per-level visualization
- Variance map heatmaps (shows spatial concentration)
- Attention maps for selected query pixels
- Called every N steps during training
- Diagnostic only, safe to modify.

**`visualization.py`** — Attention map figures.
- `create_attention_maps_figure`: Self/cross attention heatmaps for selected pixels
- `create_frame_with_grid_figure`: Frame overlays with window grid
- Used by both standalone and joint training
- Diagnostic only, safe to modify.

### Matching Package (`matching/`)

**`model.py`** — Attention-based feature matching.
- `AttentionFeatures`: Computes 8 spatial and confidence features from attention maps
  - self_relative (2): centroid offset from source
  - cross_relative (2): flow vector
  - cross_absolute (2): boundary context
  - self_max_peak, cross_max_peak (2): confidence signals
- `LevelFlowEstimator`: MLP predicting flow from features (8→16→16→2)
  - 2 hidden layers with ReLU, 16-dim default
  - Output bounded to [-0.5, 0.5] via tanh
- `HierarchicalFlowEstimator`: Orchestrates multi-level flow estimation
  - Processes each pyramid level independently
  - Crops embeddings to grid-aligned dimensions
  - Returns list of flow fields, one per level
- Read this for matching architecture changes.

**`losses.py`** — Reconstruction objective.
- `warp_embeddings`: Backward warp embeddings using flow field
- `reconstruction_loss_core`: L2 distance between warped and target
- `hierarchical_reconstruction_loss`: Multi-level aggregation
- Read this for reconstruction loss changes.

**`visualization.py`** — Flow diagnostics.
- `flow_to_colorwheel`: Direction/magnitude encoding as RGB
- `flow_to_arrows`: Quiver plot overlay
- Diagnostic only, safe to modify.

### Joint Training Package (`joint/`) — Outdated

**`model.py`** — Combined model (outdated).
- `JointEmbeddingFlowModel`: Combines embeddings + flow estimator
- Superseded by standalone embeddings training

**`losses.py`** — Combined loss (outdated).
- `JointEmbeddingReconstructionLoss`: Spatial variance + reconstruction
- Now uses spatial variance loss (updated from entropy)

**`training.py`** — Joint training loop (outdated).
- Original combined training approach
- Superseded by standalone embeddings training

### Shared Modules

**`dataset/video.py`** — Data pipeline.
- `VideoFrameDataset`: Frame pair (t, t+k) generation
- `create_dataloader`: Batch iterator with train/val split
- 85/15 train/val split by video (not frame)
- JAX PRNG-based shuffling for reproducibility

**`settings.py`** — Hyperparameters and CLI.
- `EmbeddingsSettings`: Standalone training configuration
- `SpatialVarianceLossSettings`: Loss hyperparameters
- `EmbeddingsModelSettings`: Model architecture
- `DatasetSettings`, `TrainingSettings`, `LoggingSettings`, etc.
- `Settings`: Joint training configuration (outdated)
- Change values here, not in code.

**`logging_utils.py`** — Logging utilities.
- `TensorboardLogger` wrapper functions
- `log_attention_statistics`: Spatial variance distributions
- `log_embedding_statistics`: Embedding value distributions
- `log_gradient_statistics`: Parameter/gradient histograms
- `log_progress`: Console output formatting
- Pure utilities, safe to modify.

**`checkpoint_utils.py`** — Model persistence utilities.
- `save_checkpoint`, `save_best_checkpoint`: Joint training checkpoints
- `load_checkpoint`, `restore_model_from_checkpoint`: Loading utilities
- `generate_run_name`: Unique run identifier
- Standalone training has its own checkpoint helpers in `embeddings/training.py`

**`inference.py`** — Flow estimation from checkpoint.
- Loads model from checkpoint
- Estimates flow between two images
- Saves flow field as numpy array

---

## Checkpointing

### Standalone Embeddings Checkpoints

Saved during standalone training (`embeddings/training.py`):

**Structure**:
```
checkpoints/{run_name}/
├── step_000002/        # Periodic checkpoint
│   ├── model/          # Model state
│   ├── config/         # EmbeddingsSettings (as dict)
│   └── step            # Global step number
└── best/               # Best validation loss
    └── ...
```

**Settings**: Saved as `EmbeddingsSettings` directly (no conversion)

### Joint Training Checkpoints

Saved during joint training (`checkpoint_utils.py`):

**Structure**:
```
checkpoints/{run_name}/
├── step_000100/        # Periodic
├── best/               # Best validation
└── final/              # Training complete
```

**Settings**: Saved as full `Settings` object (joint configuration)

### Validation System

- **Frequency**: Configurable via `--validation.every_epochs`
- **Metrics**: Validation loss logged to TensorBoard
- **Best Model**: Automatically tracked and saved
- **Split**: 85% train / 15% val by video (JAX PRNG shuffle)

---

## Utility Modules (External)

**`../utils/grid.py`** — Spatial operations.
- `WindowGrid`: Split/stitch embeddings into 16×16 windows
- `crop_to_grid_aligned`: Centered crop to window-multiple dimensions
- Core utility for window-based processing

**`../utils/path.py`** — Path helpers.
- `get_datasets_dir`: Project datasets directory resolver

**`../utils/logging.py`** — TensorBoard logging.
- `TensorboardLogger`: Scalar, image, histogram logging

**`../utils/cache.py`** — JAX compilation cache.
- `setup_jax_compilation_cache()`: Persistent compilation cache
- Auto-configured on import
- Cache location: `~/.cache/barevision/jax_cache`

---

## Entry Points

```bash
# Standalone embeddings training (PRIMARY)
python -m barevision.flow.embeddings.training \
    --model.num_levels=3 \
    --loss.spatial_variance.window_size=16 \
    --training.epochs=10

# Embeddings smoke test
python -m barevision.flow.embeddings.smoke_test

# Joint training (OUTDATED)
python -m barevision.flow.joint.training \
    --dataset.batch_size=4 \
    --training.epochs=10

# Joint smoke test
python -m barevision.flow.joint.smoke_test

# Inference (estimate flow between two images)
python -m barevision.flow.inference \
    --checkpoint_path checkpoints/flow_20260317_143052/final \
    --image1 frame1.png \
    --image2 frame2.png \
    --output flow.npy

# Tests
pytest src/barevision/flow/embeddings/test_model.py
pytest src/barevision/flow/embeddings/test_spatial_losses.py
pytest src/barevision/flow/matching/test_model.py
pytest src/barevision/flow/dataset/test_video.py
```

---

## Typical Workflows

| Task | Files to Read |
|------|---------------|
| Change pyramid depth/channels | `embeddings/model.py`, `settings.py` |
| Modify spatial variance loss | `embeddings/spatial_losses.py` |
| Change standalone training loop | `embeddings/training.py` |
| Change matching architecture | `matching/model.py` |
| Modify reconstruction loss | `matching/losses.py` |
| Adjust hyperparameters | `settings.py` (CLI or code) |
| Modify visualization | `embeddings/visualization.py`, `embeddings/visualization_train.py`, `matching/visualization.py` |
| Change dataset format | `dataset/video.py` |

---

## Related Documentation

- **Conceptual Design**: [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) — Algorithm, loss formulation, pyramid design rationale
