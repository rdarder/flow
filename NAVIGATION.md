# Code Navigation

Maps source files to their responsibilities. Updated on change.

---

## src/barevision

Core package. Embeddings training pipeline for hierarchical optical flow.

## src/barevision/dataset

Data loading for video frames.

**video.py:** Loads video frames and generates sparse frame pairs (t, t+k) for self-supervised training. Handles 85/15 train/val split by video with configurable seed. Exports: `VideoFrameDataset`, `FramePair`, `create_dataloader`.

## src/barevision/embeddings

Embedding pyramid model, training loop, and spatial variance loss.

**checkpointer.py:** Orbax CheckpointManager wrapper for embeddings training. Epoch-based checkpointing with validation-loss-based preservation policy. Exports: `CheckpointManagerWrapper`, `CheckpointSettings`.

**logging_utils.py:** Training diagnostics and console logging utilities. Exports: `log_diagnostics`, `ConsoleLogger`.

**model.py:** Hierarchical embedding pyramid with MobileNet V4-inspired Universal Inverted Blocks (UIB). Each level has 2 UIBs with configurable DW convs, PW expand/compress, and downsampling. GroupNorm + ReLU after each conv, L2 norm on level outputs. Exports: `HierarchicalEmbeddingModel`, `UniversalInvertedBlock`, `UIBConfig`, `count_parameters`.

**settings.py:** Tyro-based CLI configuration. Simplified ModelSettings (embed_dim, num_levels) for UIB architecture. Exports: `Settings` and component dataclasses.

**spatial_losses.py:** Spatial variance loss for attention concentration. Computes variance of attention-weighted positions per query. Supports self and cross-attention with temperature scaling. Exports: `HierarchicalSpatialVarianceLoss`, `compute_per_channel_variance`.

**training.py:** EmbeddingsTrainer - main training loop with TensorBoard logging, validation, checkpointing. Exports: `EmbeddingsTrainer`.

**visualization.py:** Diagnostic visualizations for attention maps, embeddings, and training metrics. TensorBoard figure generation.

## src/barevision/flow

Flow estimation pipeline (placeholder structure).

## src/barevision/utils

Shared utilities.

**cache.py:** File-based caching for expensive computations.

**checks.py:** Assertion helpers for configuration validation. Exports: `check_value`.

**console.py:** Console output wrapper with consistent formatting. Exports: `ConsoleLogger`.

**grid.py:** Window splitting/stitching for spatial operations. Splits embeddings into non-overlapping windows for attention. Exports: `WindowGrid`.

**image.py:** Image resolution calculations for UIB-based pyramid. Computes required input/output sizes per UIB and per level with VALID padding convolutions. Exports: `calculate_uib_output_size`, `calculate_level_output_size`, `calculate_required_input_size`, `calculate_coarse_output_size`, `image_size`.

**logging.py:** TensorBoard logger wrapper. Exports: `TensorboardLogger`.

**path.py:** Path resolution for project directories (datasets, checkpoints, etc.).

---
