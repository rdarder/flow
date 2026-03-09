# Flow: Hierarchical Embedding Training for Patch Matching

This package trains hierarchical embedding representations optimized for attention-based matching in optical flow estimation.

## Overview

We use a coarse-to-fine pyramid architecture to learn embeddings at multiple scales. **Phase 2 implements Deep Supervision** by applying entropy loss at ALL pyramid levels simultaneously.

**Key insight**: By training embeddings at multiple spatial scales with deep supervision, we ensure gradients flow equally into macro-structures (coarse levels) and micro-structures (fine levels), forcing all convolutional layers to learn trackable features immediately.

**Phase 2 approach**:
- Applies entropy loss at every pyramid level (not just coarsest)
- Restricts training to adjacent frames (max_frame_distance=2) so physical motion stays within 16×16 windows
- Crops each level to grid-aligned dimensions for clean 16×16 window splitting
- Averages loss per-level first, then sums across levels (prevents fine levels from dominating)

## Architecture

### Pyramid Structure (Phase 2 Default)

```
Input: (B, 135, 135, 3) RGB
  ↓
Level 0: Conv(3×3, stride=2) → 1×1 → 16 channels → (B, 67, 67, 16) → crop to 64×64 → 4×4 grid
  ↓
Level 1: Conv(3×3, stride=2) → 1×1 → 16 channels → (B, 33, 33, 16) → crop to 32×32 → 2×2 grid
  ↓
Level 2: Conv(3×3, stride=2) → 1×1 → 16 channels → (B, 16, 16, 16) → 1×1 grid
```

Each level:
- Uses VALID padding (no padding) to avoid border artifacts
- Stride=2 convolution for 2× spatial downsampling
- 1×1 convolution to maintain 16 channels
- L2 normalization to unit norm
- **Phase 2**: Crops to grid-aligned dimensions (divisible by 16) for clean window splitting

### Output

The model returns a list of feature maps: `[Level_0, Level_1, Level_2]`

**Phase 2 Training**: All levels are used for loss computation with deep supervision.
- Level 0: 64×64 spatial → 4×4 grid of 16×16 windows (16 windows)
- Level 1: 32×32 spatial → 2×2 grid of 16×16 windows (4 windows)
- Level 2: 16×16 spatial → 1×1 grid of 16×16 windows (1 window)

## Input Dimension Calculation

Because we use VALID padding with stride=2, spatial dimensions shrink at each level. The dataloader must provide exactly the right input size to yield the target coarse dimensions.

**Formula** (working backwards from target):
```
input_size = (output_size - 1) * stride + kernel_size
```

For 3 levels targeting 16×16 at coarsest (Phase 2 default):
- Level 2 output: 16×16
- Level 1 output → Level 2 input: (16-1)*2 + 3 = 33
- Level 0 output → Level 1 input: (33-1)*2 + 3 = 67
- Raw input → Level 0 input: (67-1)*2 + 3 = **135**

The `DatasetSettings.img_size` property automatically calculates this based on `num_levels`, `coarse_grid_size`, and `window_size`.

## Loss Functions

**Phase 2: Deep Supervision** - Loss is applied at ALL pyramid levels simultaneously.

### Grid Alignment (Phase 2)

Each level is cropped to dimensions divisible by window_size (16) before loss computation:
- Level 0: 67×67 → crop to 64×64 → 4×4 grid of windows
- Level 1: 33×33 → crop to 32×32 → 2×2 grid of windows
- Level 2: 16×16 → no crop → 1×1 grid of windows

### Per-Level Loss Computation

For each pyramid level:
1. Crop feature maps to grid-aligned dimensions
2. Split into 16×16 windows
3. Compute self-attention entropy within each window
4. Compute cross-attention entropy between corresponding windows

### Combined Loss (Phase 2)

```
loss_L[i] = α * self_entropy_L[i] + β * cross_entropy_L[i]  # per level
Total_Loss = loss_L0 + loss_L1 + loss_L2  # sum across levels
```

Default weights: α=1.0, β=0.1

**Why sum per-level losses?** If we flattened all windows from all levels into a single batch, Level 0 (16 windows) would statistically drown out Level 2 (1 window). By averaging within each level first, then summing, all levels contribute equally to the gradient.

## Training Stabilizers

### L2 Normalization

All embeddings are L2-normalized to unit norm before computing attention. This prevents high-norm embeddings from dominating attention regardless of content.

### Temperature Scaling

Attention logits are divided by temperature τ=0.05 before softmax. Low temperature sharpens the distribution, amplifying small differences to select clear winners.

## Training Data

Video frame pairs from single continuous takes (no cuts). **Phase 2 restricts to adjacent frames** (max_frame_distance=2) to ensure physical pixel displacement stays within 16×16 attention windows at the finest resolutions.

The dataset automatically resizes frames to the exact calculated input dimension (135×135 for default 3-level pyramid with 1×1 coarse grid).

## Configuration

```bash
# Phase 2 default (3 levels, 1×1 coarse grid, 16×16 windows, adjacent frames)
python -m barevision.flow.train

# Custom configuration
python -m barevision.flow.train \
  --model.num-levels 3 \
  --model.embed-dim 16 \
  --model.window-size 16 \
  --dataset.coarse-grid-size 1 \
  --dataset.max-frame-distance 2 \
  --dataset.batch-size 4

# Smoke test
python -m barevision.flow.train --smoke-test
```

Key settings in `settings.py`:
- `ModelSettings.num_levels`: Number of pyramid levels (default 3)
- `ModelSettings.embed_dim`: Output channels per level (default 16)
- `ModelSettings.window_size`: Attention window size (default 16)
- `DatasetSettings.coarse_grid_size`: Target coarse grid dimension (default 1 for 1×1 grid)
- `DatasetSettings.max_frame_distance`: Max temporal distance (default 2 for Phase 2)
- `DatasetSettings.img_size`: **Calculated automatically** based on above parameters

## Visualization

**Phase 2**: Visualizations are generated for ALL pyramid levels independently.

For each level:
1. Original RGB images are downscaled to match that level's embedding dimensions
2. A random 16×16 window is selected at that level's resolution
3. Attention maps are overlaid directly on downscaled images
4. Figures are logged with level-specific tags: `Level0/`, `Level1/`, `Level2/`

This allows visual inspection of how each level tracks image structure at its native resolution.

## Future Phases

### Phase 2: Multi-Level Deep Supervision ✓ COMPLETED
Train all pyramid levels with deep supervision using adjacent frames.

### Phase 3: Flow-Based Window Shifting
Add flow priors to shift attention windows at finer levels, canceling ego-motion to track patches across larger displacements.

### Phase 4: Full Integration
Integrate with flow estimation pipeline for end-to-end optical flow.

## Testing

```bash
# Unit tests
pytest src/barevision/flow/test_model.py
pytest src/barevision/flow/test_loss.py

# Smoke test
python -m barevision.flow.train --smoke-test
```

## Parameter Count

For default 3-level pyramid with 16 channels:
- Level 0 (3→16 ch): ~720 parameters
- Level 1 (16→16 ch): ~2,592 parameters
- Level 2 (16→16 ch): ~2,592 parameters
- **Total: ~5,904 parameters**
