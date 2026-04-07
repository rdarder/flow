# Progress: Embedding Model Simplification

## Current State

The embedding model has been simplified to a cleaner architecture with identical blocks at all pyramid levels.

### Architecture

**Simplified EmbeddingBlock** (all 3 levels identical):
- PW Compact: Dense 1×1 convolution (3→4 for first block, 16→4 for others)
- DW ×8: Depthwise 3×3 with 8 filters per channel, stride=2, VALID padding
- PW Project: Grouped 1×1 projection (32→16, 4 groups)
- L2 Norm: Required on output to prevent softmax collapse

**Removed**:
- Preprocessor layer (first block now handles RGB directly)
- Mean subtraction / local contrast normalization
- GroupNorm layers
- MeanConv for downsampling

**Resolution** (3 levels, 16×16 coarsest):
- Input: 135×135
- Level 0: 67×67 × 16 channels
- Level 1: 33×33 × 16 channels
- Level 2: 16×16 × 16 channels

### Performance Characteristics

- **Parameters**: ~1,496 (down from ~7,248)
- **FLOPs per pixel**: ~512 per level (down from ~2,048)
- **Total FLOPs** (3 levels, 135×135 input): ~12M (down from ~50M)

### Configuration

ModelSettings now uses:
- `compact_channels=4` (default)
- `depthwise_multiplier=8` (default)
- `project_groups=4` (default)
- `embed_dim=16` (default)
- `num_levels=3` (default)

Removed settings: `use_preprocessor`, `use_group_norm`, `use_mean_subtraction`, `use_l2_norm`, `use_mean_conv_for_downsampling`, `hidden_dim`, `num_groups` (kept as alias for project_groups).

### Training Pipeline

- Smoke test passes with 2-level, 67×67 input configuration
- All 48 unit tests pass
- Visualization module compatible (operates on pyramid levels generically)
- Loss function unchanged (spatial variance with temperature 0.25)

### Known Limitations

- Visualizations untested with new resolutions (67×67, 33×33, 16×16)
- No ablation results yet on optimal compact_channels, depthwise_multiplier, project_groups
- First block uses 3→4 compact; may need 3→8 if RGB has less redundancy
