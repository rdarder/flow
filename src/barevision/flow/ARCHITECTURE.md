# Code Map: Embedding Engine

This document maps concepts to source files. It tells you **what is where**, not how things work. Read the code for details.

---

## Quick Map

| Concept | Primary File | What's There |
|---------|-------------|--------------|
| Pyramid Model | `model.py` | `StemBlock`, `StandardBlock`, `HierarchicalEmbeddingModel` |
| Entropy Loss | `loss.py` | Self/cross attention loss, hierarchical aggregation |
| Training Loop | `train.py` | Epoch/step orchestration, logging hooks |
| Data Loading | `video_dataset.py` | Frame pair generation, batching |
| Hyperparameters | `settings.py` | tyro CLI config, dataclasses |
| Visualization | `visualization.py` | TensorBoard figures (diagnostic only) |
| Window Utilities | `../utils/grid.py` | `WindowGrid` split/stitch, resolution helpers |

---

## File Roles

### Core (Read These First)

**`model.py`** — The embedding pyramid architecture.
- `StemBlock`: Level 0, RGB input, two stacked 3×3 convs
- `StandardBlock`: Levels 1-N, single 3×3 conv
- `HierarchicalEmbeddingModel`: Full pyramid, 3 levels by default
- Read this for architecture changes.

**`loss.py`** — Training objectives.
- `self_attention_entropy_loss_core`: Sharp self-peaks
- `cross_attention_entropy_loss_core`: Sharp cross-peaks
- `compute_hierarchical_embedding_losses`: Multi-level aggregation with level weighting
- Read this for loss function changes.

### Orchestration (Read When Needed)

**`train.py`** — Training loop.
- `train_step`: JIT-compiled gradient update
- `run_epoch`: DataLoader iteration, logging calls
- `train`: Model/optimizer setup, main loop
- Rarely changes. Read for debugging training flow.

**`video_dataset.py`** — Data pipeline.
- `VideoFrameDataset`: Frame pair (t, t+k) generation
- `create_dataloader`: Batch iterator
- Only touch for dataset or pairing logic changes.

### Configuration

**`settings.py`** — Hyperparameters and CLI.
- `DatasetSettings`, `ModelSettings`, `TrainingSettings`, `LoggingSettings`
- `Settings`: Root dataclass, tyro CLI entry point
- `create_smoke_test_settings`: Quick validation config
- Change values here, not in code.

### Diagnostics

**`visualization.py`** — TensorBoard logging figures.
- `create_frame_with_grid_figure`: Input frames with 16×16 grid overlay
- `create_attention_maps_figure`: Self/cross attention heatmaps
- `log_visualizations`: Called every N steps, not in training loss path
- Diagnostic only. Safe to modify without affecting training.

**`logging_utils.py`** — Console output helpers.
- `print_header`, `print_footer`, `log_progress`
- Pure formatting. No logic.

---

## Utility Modules

**`../utils/grid.py`** — Spatial operations.
- `WindowGrid`: Split/stitch embeddings into 16×16 windows
- `compute_valid_resolution`, `validate_resolution`: Shape helpers
- Used by `loss.py` for window splitting.

**`../utils/path.py`** — Path helpers.
- `get_datasets_dir`: Project datasets directory resolver

**`../utils/logging.py`** — `JaxLogger` wrapper for TensorBoard.

---

## Legacy Code (Ignore)

**`old_flow/`** — Previous implementation iteration.
- Contains hierarchical flow estimation, checkpointing, older training scripts
- **Do not modify.** Superseded by current `flow/` implementation.

---

## Entry Points

```bash
# Training
python -m barevision.flow.train --dataset.batch_size=4 --training.epochs=10

# Smoke test (quick validation)
python -m barevision.flow.train --smoke-test

# Tests
pytest src/barevision/flow/test_model.py
pytest src/barevision/flow/test_loss.py
pytest src/barevision/flow/test_video_dataset.py
pytest src/barevision/flow/test_visualization.py
```

---

## Typical Workflows

| Task | Files to Read |
|------|---------------|
| Change pyramid depth/channels | `model.py`, `settings.py` |
| Modify loss function | `loss.py` |
| Debug training crash | `train.py`, `video_dataset.py` |
| Add logging metric | `train.py`, `logging_utils.py` |
| Change dataset format | `video_dataset.py` |
| Adjust hyperparameters | `settings.py` (CLI or code) |

---

## Related Documentation

- **Conceptual Design**: [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) — Algorithm, loss formulation, pyramid design rationale
