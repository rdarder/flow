# Code Navigation

Maps source files to their responsibilities. Updated on change.

---

## Project Root

**config.yaml:** YAML configuration for embeddings training. Defines model architecture (UIB pyramid), dataset settings, loss hyperparameters, training schedule, logging, checkpointing, and validation. Loaded via `barevision.config.load_config()` which validates against Pydantic models.

## src/barevision

Core package. Embeddings training pipeline for hierarchical optical flow.

**config.py:** Root configuration and YAML loading. Ties together all config subsections from business modules. Uses Pydantic with frozen=True for JAX/JIT compatibility. Exports: `RootConfig`, `load_config()`, `TrainingConfig`, `LoggingConfig`, `ValidationConfig`, `LossConfig`.

## src/barevision/dataset

Data loading for video frames.

**video.py:** Loads video frames and generates sparse frame pairs (t, t+k) for self-supervised training. Handles 85/15 train/val split by video with configurable seed. `create_dataloader()` accepts `image_size` parameter (calculated by caller). Config: `DatasetConfig` (Pydantic, frozen). Exports: `VideoFrameDataset`, `FramePair`, `create_dataloader`, `DatasetConfig`.

## src/barevision/embeddings

Embedding pyramid model, training loop, and spatial variance loss.

**checkpointer.py:** Orbax CheckpointManager wrapper for embeddings training. Epoch-based checkpointing with validation-loss-based preservation policy. Exports: `CheckpointManagerWrapper`, `CheckpointConfig`.

**logging_utils.py:** Training diagnostics and console logging utilities. Exports: `log_diagnostics`, `log_metrics`, `log_progress`.

**model.py:** Hierarchical embedding pyramid with MobileNet V4-inspired Universal Inverted Blocks (UIB). Configuration: `UIBConfig` → `LevelConfig` → `HierarchicalModelConfig` (tuple of levels). Config classes use Pydantic with frozen=True for JAX/JIT compatibility. Model config owns size methods and can build model via `build_model()`. Model classes (`UniversalInvertedBlock`, `Level`, `HierarchicalEmbeddingModel`) have no size methods. Each level has configurable UIBs. GroupNorm + ReLU after each conv, L2 norm on level outputs. Exports: `HierarchicalEmbeddingModel`, `HierarchicalModelConfig`, `Level`, `LevelConfig`, `UniversalInvertedBlock`, `UIBConfig`, `count_parameters`.

**spatial_losses.py:** Spatial variance loss for attention concentration. Computes variance of attention-weighted positions per query. Supports self and cross-attention with temperature scaling. Config: `SpatialVarianceLossConfig` (Pydantic, frozen). Exports: `HierarchicalSpatialVarianceLoss`, `SpatialVarianceLossConfig`, `compute_hierarchical_spatial_variance_loss`.

**training.py:** EmbeddingsTrainer - main training loop with TensorBoard logging, validation, checkpointing. Loads model from YAML config via `barevision.config.load_config()`. Exports: `EmbeddingsTrainer`, `TrainingConfig`, `LoggingConfig`, `ValidationConfig`.

**visualization.py:** Diagnostic visualizations for attention maps, embeddings, and training metrics. TensorBoard figure generation.

## src/barevision/flow

Flow estimation pipeline (placeholder structure).

## src/barevision/utils

Shared utilities.

**cache.py:** File-based caching for expensive computations.

**checks.py:** Assertion helpers for configuration validation. Exports: `check_value`.

**console.py:** Console output wrapper with consistent formatting. Exports: `ConsoleLogger`.

**grid.py:** Window splitting/stitching for spatial operations. Splits embeddings into non-overlapping windows for attention. Exports: `WindowGrid`.

**logging.py:** TensorBoard logger wrapper. Exports: `TensorboardLogger`.

**path.py:** Path resolution for project directories (datasets, checkpoints, etc.).

---
