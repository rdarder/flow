# Code Map: Optical Flow Training

This document maps concepts to source files. It tells you **what is where**, not how things work. Read the code for details.

---

## Quick Map

| Concept | Primary File | What's There |
|---------|-------------|--------------|
| Embedding Pyramid | `embeddings/model.py` | `StemBlock`, `StandardBlock`, `HierarchicalEmbeddingModel` |
| Entropy Loss | `embeddings/losses.py` | Self/cross attention loss, hierarchical aggregation, `crop_to_grid_aligned` |
| Level Flow Estimator | `matching/model.py` | `LevelFlowEstimator` (MLP), `AttentionFeatures` (8 features) |
| Hierarchical Flow | `matching/model.py` | `HierarchicalFlowEstimator` (orchestrates multi-level estimation) |
| Reconstruction Loss | `matching/losses.py` | `warp_embeddings`, `reconstruction_loss_core`, `hierarchical_reconstruction_loss` |
| Training Orchestrator | `training/model.py` | `Model` (combines embeddings + hierarchical flow) |
| Combined Loss | `training/losses.py` | `compute_loss` (entropy + hierarchical reconstruction) |
| Training Loop | `training/__main__.py` | `train_step`, `run_epoch`, `train` |
| Visualization (Embeddings) | `embeddings/visualization.py` | Grid overlays, attention map figures |
| Visualization (Matching) | `matching/visualization.py` | Flow colorwheel, arrow visualizations |
| Visualization (Orchestrator) | `training/visualization.py` | Combines both packages for training logs |
| Data Loading | `video_dataset.py` | Frame pair generation, batching |
| Hyperparameters | `settings.py` | tyro CLI config, dataclasses |

---

## Package Structure

```
barevision/flow/
├── embeddings/           # Feature pyramid extraction
│   ├── model.py         # HierarchicalEmbeddingModel (StemBlock, StandardBlock)
│   ├── losses.py        # Entropy loss functions, crop_to_grid_aligned
│   ├── visualization.py # Attention map figures, frame grid overlays
│   ├── test_model.py    # Tests for embedding model
│   └── test_losses.py   # Tests for entropy loss
│
├── matching/             # Attention-based feature matching
│   ├── model.py         # LevelFlowEstimator (MLP), HierarchicalFlowEstimator, AttentionFeatures
│   ├── losses.py        # Warp + reconstruction loss (single-level and hierarchical)
│   ├── visualization.py # Flow colorwheel, arrow visualizations
│   ├── test_model.py    # Tests for LevelFlowEstimator and attention features
│   └── test_hierarchical_model.py  # Tests for HierarchicalFlowEstimator
│
├── joint/               # Joint training orchestration
│   ├── model.py         # Model (embeddings + HierarchicalFlowEstimator)
│   ├── losses.py        # Combined loss function (entropy + hierarchical reconstruction)
│   ├── visualization.py # Orchestrates visualization + window extraction helpers
│   └── training.py      # Entry point: python -m barevision.flow.joint.training
│
├── video_dataset.py      # Data loading (shared)
├── settings.py           # Hyperparameters (shared)
├── checkpoint_utils.py   # Model persistence and restoration
├── inference.py          # Inference script for flow estimation
├── logging_utils.py      # Console output (shared)
└── ARCHITECTURE.md       # This file
```

---

## File Roles

### Embeddings Package (`embeddings/`)

**`model.py`** — Hierarchical embedding pyramid.
- `StemBlock`: Level 0, RGB input, two stacked 3×3 convs
- `StandardBlock`: Levels 1-N, single 3×3 conv
- `HierarchicalEmbeddingModel`: Full pyramid, 3 levels by default
- `count_parameters`: Utility for parameter counting
- `calculate_required_input_size`, `calculate_coarse_output_size`: Resolution math
- Read this for embedding architecture changes.

**`losses.py`** — Entropy minimization objectives.
- `self_attention_entropy_loss_core`: Sharp self-peaks (unique embeddings)
- `cross_attention_entropy_loss_core`: Sharp cross-peaks (confident matching)
- `compute_window_attention_losses`: Single-level window-based loss
- `compute_hierarchical_entropy_loss`: Multi-level aggregation with level weighting
- `crop_to_grid_aligned`: Utility for resolution alignment
- Read this for loss function changes.

**`visualization.py`** — Embedding diagnostics.
- `create_attention_maps_figure`: Self/cross attention heatmaps for selected pixels
- Diagnostic only. Safe to modify without affecting training.

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
  - No output bias for centered initialization
- `HierarchicalFlowEstimator`: Orchestrates multi-level flow estimation
  - Runs LevelFlowEstimator at each pyramid level (V1: independent)
  - Crops embeddings to grid-aligned dimensions (centered crop)
  - Splits into 16×16 windows for processing
  - Returns list of flow fields, one per level
  - V2: Will add window shifting and priors
- `create_source_position_grid`: Normalized coordinate grid
- `flow_to_dense`: Reshape flow from token to spatial format
- Read this for matching architecture changes.

**`losses.py`** — Reconstruction objective.
- `warp_embeddings`: Backward warp embeddings using flow field
- `reconstruction_loss_core`: L2 distance between warped and target embeddings
- `hierarchical_reconstruction_loss`: Multi-level reconstruction loss
  - Crops each level to grid-aligned dimensions
  - Warps embeddings at each level using level-specific flow
  - Averages loss across all levels
- Read this for reconstruction loss changes.

**`visualization.py`** — Flow diagnostics.
- `flow_to_colorwheel`: Direction/magnitude encoding as RGB
- `flow_to_arrows`: Quiver plot overlay on magnitude background
- Diagnostic only. Safe to modify without affecting training.

### Training Package (`training/`)

**`model.py`** — Combined model orchestrator.
- `Model`: Combines `HierarchicalEmbeddingModel` + `HierarchicalFlowEstimator`
- Single forward pass returns `(flows, pyramid1, pyramid2)`
  - flows: List of flow fields, one per pyramid level
  - pyramid1, pyramid2: Embedding lists from both frames
- `extract_embeddings`: Convenience method for embedding-only use
- Read this for model integration changes.

**`losses.py`** — Combined training objective.
- `compute_loss`: Combines entropy + hierarchical reconstruction loss
- `total = entropy_loss + recon_weight * reconstruction_loss`
- Entropy is primary (distinctive embeddings), reconstruction is secondary (trackable)
- Reconstruction loss computed across all pyramid levels
- Returns aux dict with per-level losses for logging
- Read this for loss weighting changes.

**`visualization.py`** — Training visualization orchestrator.
- `log_visualizations`: Calls both embeddings + matching visualization
- Logs per-level:
  - Flow colorwheel and arrows (`Level0/flow_colorwheel`, `Level0/flow_arrows`, etc.)
  - Frame grid overlays (`Level0/Frame_Grid`)
  - Attention maps (`Level0/Attention_Maps`)
- Grid overlay uses centered crop to match actual processing
- Bright white grid lines at pixel edges for visibility
- Called every N steps, not in training loss path
- Diagnostic only. Safe to modify without affecting training.

**`__main__.py`** — Training entry point.
- `train_step`: JIT-compiled gradient update (via `nnx.jit`)
- `run_epoch`: DataLoader iteration, logging calls
- `train`: Model/optimizer setup, main loop
- Entry point: `python -m barevision.flow.training`
- Rarely changes. Read for debugging training flow.

### Shared Modules

**`video_dataset.py`** — Data pipeline.
- `VideoFrameDataset`: Frame pair (t, t+k) generation
- `create_dataloader`: Batch iterator
- Only touch for dataset or pairing logic changes.

**`settings.py`** — Hyperparameters and CLI.
- `DatasetSettings`, `ModelSettings`, `TrainingSettings`, `LoggingSettings`
- `Settings`: Root dataclass, tyro CLI entry point
- Change values here, not in code.

**`logging_utils.py`** — Console output helpers.
- `print_header`, `print_footer`, `log_progress`
- `log_metrics`: TensorBoard scalar logging
- `log_diagnostics`: Embedding statistics
- Pure formatting. No logic.

**`checkpoint_utils.py`** — Model persistence utilities.
- `save_checkpoint`: Save periodic/final checkpoints
- `save_best_checkpoint`: Save best model by validation loss
- `load_checkpoint`: Load checkpoint data
- `restore_model_from_checkpoint`: Restore model state
- `generate_run_name`: Unique run identifier with timestamp
- `config_from_checkpoint`: Extract config without loading model

**`inference.py`** — Flow estimation from checkpoint.
- CLI entry point: `python -m barevision.flow.inference`
- Loads model from checkpoint
- Estimates flow between two images
- Saves flow field as numpy array and optional visualization

---

## Checkpointing and Validation

### Checkpoint System (`checkpoint_utils.py`)

The training pipeline supports three types of checkpoints:

1. **Periodic Checkpoints**: Saved every N steps during training
   - Configured via `--checkpoint.every_steps <N>`
   - Path: `checkpoints/{run_name}/step_{STEP:06d}/`
   - Set to 0 to disable

2. **Best Model Checkpoint**: Automatically saved when validation loss improves
   - Enabled by default (`--validation.save_best=true`)
   - Path: `checkpoints/{run_name}/best/`
   - Contains `best_val_loss` metadata

3. **Final Checkpoint**: Saved when training completes
   - Configured via `--checkpoint.save_final=true` (default)
   - Path: `checkpoints/{run_name}/final/`

Each checkpoint contains:
- Model state (NNX state dict)
- Global step number
- Full configuration (for model reconstruction)
- Validation loss (for best checkpoints only)

### Validation System

Validation runs at the end of each epoch on the held-out validation set (15% of videos):

- **Frequency**: Configurable via `--validation.every_epochs <N>` (default: 1)
- **Metrics**: Validation loss logged to TensorBoard as `Loss/validation`
- **Best Model Tracking**: Automatically tracks and saves best model by validation loss

Validation uses the same loss function as training (entropy + reconstruction) but without gradient computation.

### Train/Validation Split

The dataset splits videos (not frames) into train/val sets:
- **Training**: 85% of videos (rounded down)
- **Validation**: 15% of videos (rounded up)
- **Split Method**: JAX PRNG-based shuffling with configurable seed
- **Reproducibility**: Same seed produces identical splits

Split is configured via `--dataset.seed` (default: 42).

---

## Utility Modules (External)

**`../utils/grid.py`** — Spatial operations.
- `WindowGrid`: Split/stitch embeddings into 16×16 windows
  - Core utility for window-based processing
  - Used by embeddings and matching packages
- Dead code removed: `compute_valid_resolution`, `validate_resolution`, `crop_to_valid`, etc.
  - These were only used in tests and have been superseded

**`../utils/path.py`** — Path helpers.
- `get_datasets_dir`: Project datasets directory resolver

**`../utils/logging.py`** — `JaxLogger` wrapper for TensorBoard.

**`../utils/cache.py`** — JAX compilation cache configuration.
- `setup_jax_compilation_cache()`: Configures persistent compilation cache
- Auto-configured on `import barevision`
- Cache location: `~/.cache/barevision/jax_cache`

---

## Legacy Code (Ignore)

**`old_flow/`** — Previous implementation iteration.
- Contains hierarchical flow estimation, checkpointing, older training scripts
- **Do not modify.** Superseded by current `flow/` implementation.

---

## Entry Points

```bash
# Training
python -m barevision.flow.joint.training --dataset.batch_size=4 --training.epochs=10

# Smoke test (quick validation)
python -m barevision.flow.joint.smoke_test 

# Inference (estimate flow between two images)
python -m barevision.flow.inference --checkpoint_path checkpoints/flow_20260317_143052/final \
                                    --image1 frame1.png \
                                    --image2 frame2.png \
                                    --output flow.npy

# Tests
pytest src/barevision/flow/embeddings/test_model.py
pytest src/barevision/flow/embeddings/test_losses.py
pytest src/barevision/flow/matching/test_model.py
pytest src/barevision/flow/test_video_dataset.py
pytest src/barevision/flow/test_visualization.py
```

---

## Typical Workflows

| Task | Files to Read |
|------|---------------|
| Change pyramid depth/channels | `embeddings/model.py`, `settings.py` |
| Modify embedding loss | `embeddings/losses.py` |
| Change matching architecture | `matching/model.py` |
| Modify reconstruction loss | `matching/losses.py` |
| Change training loop | `training/__main__.py` |
| Add logging metric | `training/__main__.py`, `logging_utils.py` |
| Change dataset format | `video_dataset.py` |
| Adjust hyperparameters | `settings.py` (CLI or code) |
| Modify visualization | `embeddings/visualization.py`, `matching/visualization.py`, `training/visualization.py` |

---

## Related Documentation

- **Conceptual Design**: [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) — Algorithm, loss formulation, pyramid design rationale
