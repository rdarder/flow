# Progress: UIB Architecture with YAML Configuration System

## Current State

The model uses **MobileNet V4-inspired Universal Inverted Blocks (UIB)** arranged in a 3-level pyramid with 2 UIBs per level.

**Architecture:**
- Each level: 2 UIBs in sequence, second UIB downsamples
- UIB structure: DW (optional) → PW Expand → DW (optional) → PW Compress → Downsample (optional)
- GroupNorm + ReLU after every convolution
- L2 normalization on level outputs (configurable per UIB)
- VALID padding throughout
- **Downsampling initialized with Gaussian kernels** (sigma=1.0) for proper spatial averaging from training start

**Configuration hierarchy:**
- `UIBConfig`: Single block config with `output_size()`, `required_input_size()` methods, and `downsample_gaussian_sigma` field
- `LevelConfig`: Holds tuple of `UIBConfig`s, iterates for size math
- `HierarchicalModelConfig`: Simple container with tuple of `LevelConfig`s; owns all size methods and `build_model()`
- Model classes (`UniversalInvertedBlock`, `Level`, `HierarchicalEmbeddingModel`) have no size methods — use config

**Configuration System:**
- **YAML-based**: All settings loaded from `config.yaml` at project root
- **Pydantic validation**: All config classes use `BaseModel` with `ConfigDict(frozen=True)` for JAX/JIT compatibility
- **Hierarchical structure**: `RootConfig` ties together model, dataset, loss, training, logging, checkpoint, and validation subsections
- **Config classes co-located with behavior**:
  - `model.py`: `UIBConfig`, `LevelConfig`, `HierarchicalModelConfig`
  - `dataset/video.py`: `DatasetConfig`
  - `spatial_losses.py`: `SpatialVarianceLossConfig`
  - `checkpointer.py`: `CheckpointConfig`
  - `config.py`: `RootConfig`, `LossConfig`, `TrainingConfig`, `LoggingConfig`, `ValidationConfig`, `load_config()`
- **CLI interface**: `python -m barevision.embeddings.training --config config.yaml`
- **Removed**: `settings.py` module and `make_default_model_config()` function

**Size calculation:**
- Config classes own all size math (forward and inverse)
- `HierarchicalModelConfig.target_to_input(coarsest_grid_size, window_size)` calculates required input image size
- No duplication: size methods live only in config, not in model classes
- `create_dataloader()` accepts `image_size` parameter (calculated by caller)

**Training pipeline:**
- Spatial variance loss (self + cross attention)
- GroupNorm enables stable training with ReLU activations
- Smoke test validates full pipeline
- All 66 existing unit tests pass
