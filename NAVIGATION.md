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

**video.py:** Loads video frames and generates sparse frame pairs (t, t+k) for self-supervised training. Handles 85/15 train/val split by video with configurable seed. Pre-loads all unique frames into memory as JAX arrays to eliminate per-step JPEG decoding overhead. `create_dataloader()` accepts `image_size` parameter (calculated by caller). Config: `DatasetConfig` (Pydantic, frozen, includes `frame_cache_max_mb`). Exports: `VideoFrameDataset`, `FramePair`, `PreloadedFrameDataset`, `create_dataloader`, `DatasetConfig`.

**test_fixtures/:** Small synthetic dataset for testing (2 videos, 9 frames total).

**test_fixtures/generate_fixtures.py:** Script to generate synthetic test frames.

## src/barevision/embeddings

Embedding pyramid model, training loop, and linear attention flow loss.

**checkpointer.py:** Orbax CheckpointManager wrapper for embeddings training. Epoch-based checkpointing with validation-loss-based preservation policy. Exports: `CheckpointManagerWrapper`, `CheckpointConfig`.

**linear_attention_loss.py:** Linear attention flow loss for self-supervised training. Replaces softmax attention with O(D) linear attention mechanism. Computes flow via center-of-mass decoding, warped reconstruction loss, and embedding diversity loss. Config: `LinearAttentionFlowLossConfig` (Pydantic, frozen). Exports: `HierarchicalLinearAttentionFlowLoss`, `LinearAttentionFlowLossConfig`, `compute_hierarchical_linear_attention_loss`.

**logging_utils.py:** Training diagnostics and console logging utilities. Logs flow statistics, embedding statistics, and gradient statistics. Exports: `log_diagnostics`, `log_metrics`, `log_progress`, `log_flow_statistics`.

**model.py:** Hierarchical embedding pyramid with MobileNet V4-inspired Universal Inverted Blocks (UIB). Configuration: `UIBConfig` → `LevelConfig` → `HierarchicalModelConfig` (tuple of levels). Config classes use Pydantic with frozen=True for JAX/JIT compatibility. Model config owns size methods and can build model via `build_model()`. Model classes (`UniversalInvertedBlock`, `Level`, `HierarchicalEmbeddingModel`) have no size methods. Each level has configurable UIBs. GroupNorm + ReLU after each conv, L2 norm on level outputs. Exports: `HierarchicalEmbeddingModel`, `HierarchicalModelConfig`, `Level`, `LevelConfig`, `UniversalInvertedBlock`, `UIBConfig`, `count_parameters`.

**training.py:** EmbeddingsTrainer - main training loop with TensorBoard logging, validation, checkpointing. Loads model from YAML config via `barevision.config.load_config()`. Exports: `EmbeddingsTrainer`, `TrainingConfig`, `LoggingConfig`, `ValidationConfig`.

**visualization.py:** Diagnostic visualizations for per-dimension activation patterns and frame grid overlays. TensorBoard figure generation.

## src/barevision/flow

Flow estimation pipeline (placeholder structure).

## src/barevision/utils

Shared utilities.

**cache.py:** File-based caching for expensive computations.

**checks.py:** Assertion helpers for configuration validation. Exports: `check_value`.

**console.py:** Console output wrapper with consistent formatting. Exports: `ConsoleLogger`.

**grid.py:** Window splitting/stitching for spatial operations. Splits embeddings into non-overlapping windows for attention. Exports: `WindowGrid`.

**logging.py:** TensorBoard logger wrapper. Exports: `TensorboardLogger`.

**path.py:** Path resolution for project directories (datasets, checkpoints, etc.). Includes `set_datasets_dir_override()` for test isolation.

---
