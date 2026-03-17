# Code Map: Optical Flow Training

This document maps concepts to source files. It tells you **what is where**, not how things work. Read the code for details.

---

## Quick Map

| Concept | Primary File | What's There |
|---------|-------------|--------------|
| Embedding Pyramid | `embeddings/model.py` | `StemBlock`, `StandardBlock`, `HierarchicalEmbeddingModel` |
| Entropy Loss | `embeddings/losses.py` | Self/cross attention loss, hierarchical aggregation |
| Feature Matching | `matching/model.py` | `FlowEstimator`, `AttentionCentroids`, centroid-based flow prediction |
| Reconstruction Loss | `matching/losses.py` | `warp_embeddings`, `reconstruction_loss_core` |
| Training Orchestrator | `training/model.py` | `Model` (combines embeddings + matching) |
| Combined Loss | `training/losses.py` | `compute_loss` (entropy + reconstruction) |
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
│   ├── model.py         # HierarchicalEmbeddingModel
│   ├── losses.py        # Entropy loss functions
│   ├── visualization.py # Attention map figures
│   ├── test_model.py
│   └── test_losses.py
│
├── matching/             # Attention-based feature matching
│   ├── model.py         # FlowEstimator, AttentionCentroids
│   ├── losses.py        # Warp + reconstruction loss
│   ├── visualization.py # Flow field visualizations
│   └── test_model.py
│
├── training/             # Combined training orchestration
│   ├── model.py         # Model (embeddings + matching)
│   ├── losses.py        # Combined loss function
│   ├── visualization.py # Orchestrates visualization + window extraction helpers
│   └── __main__.py      # Entry point: python -m barevision.flow.training
│
├── video_dataset.py      # Data loading (shared)
├── settings.py           # Hyperparameters (shared)
└── logging_utils.py      # Console output (shared)
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
- `AttentionCentroids`: Computes center-of-mass from attention maps
- `FlowEstimator`: MLP predicting flow from centroids + positions
- `create_source_position_grid`: Normalized coordinate grid
- `flow_to_dense`: Reshape flow from token to spatial format
- Read this for matching architecture changes.

**`losses.py`** — Reconstruction objective.
- `warp_embeddings`: Backward warp embeddings using flow field
- `reconstruction_loss_core`: L2 distance between warped and target embeddings
- Read this for reconstruction loss changes.

**`visualization.py`** — Flow diagnostics.
- `flow_to_colorwheel`: Direction/magnitude encoding as RGB
- `flow_to_arrows`: Quiver plot overlay on magnitude background
- Diagnostic only. Safe to modify without affecting training.

### Training Package (`training/`)

**`model.py`** — Combined model orchestrator.
- `Model`: Combines `HierarchicalEmbeddingModel` + `FlowEstimator`
- Single forward pass returns `(flow, pyramid1, pyramid2)`
- `extract_embeddings`: Convenience method for embedding-only use
- Read this for model integration changes.

**`losses.py`** — Combined training objective.
- `compute_loss`: Combines entropy + reconstruction loss
- `total = (1 - lambda_recon) * entropy + lambda_recon * reconstruction`
- Read this for loss weighting changes.

**`visualization.py`** — Training visualization orchestrator.
- `log_visualizations`: Calls both embeddings + matching visualization
- Logs flow colorwheel/arrows + attention maps per pyramid level
- `_extract_window_attention_data`: Helper for extracting window attention (internal)
- `_extract_pixel_attention_maps`: Helper for pixel extraction (internal)
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
- `create_smoke_test_settings`: Quick validation config
- Change values here, not in code.

**`logging_utils.py`** — Console output helpers.
- `print_header`, `print_footer`, `log_progress`
- `log_metrics`: TensorBoard scalar logging
- `log_diagnostics`: Embedding statistics
- Pure formatting. No logic.

---

## Utility Modules (External)

**`../utils/grid.py`** — Spatial operations.
- `WindowGrid`: Split/stitch embeddings into 16×16 windows
- `compute_valid_resolution`, `validate_resolution`: Shape helpers
- Used by `embeddings/losses.py` for window splitting.

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
python -m barevision.flow.training --dataset.batch_size=4 --training.epochs=10

# Smoke test (quick validation)
python -m barevision.flow.training --smoke-test

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
