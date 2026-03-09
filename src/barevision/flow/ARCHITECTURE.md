# Flow: Hierarchical Embedding Training for Patch Matching

This package trains hierarchical embedding representations optimized for attention-based matching in optical flow estimation.

## Overview

We use a coarse-to-fine pyramid architecture to learn embeddings at multiple scales. This Phase 1 implementation focuses on training the coarsest level only, establishing the foundation for future phases that will add flow-based window shifting at finer levels.

**Key insight**: By training embeddings at a coarse spatial scale (e.g., 3×3 grid of 16×16 windows), we learn representations that capture larger spatial context, which will later enable tracking patches across large displacements when combined with flow priors.

## Architecture

### Pyramid Structure

```
Input: (B, 391, 391, 3) RGB
  ↓
Level 0: Conv(3×3, stride=2) → 1×1 → 16 channels → (B, 195, 195, 16)
  ↓
Level 1: Conv(3×3, stride=2) → 1×1 → 16 channels → (B, 97, 97, 16)
  ↓
Level 2: Conv(3×3, stride=2) → 1×1 → 16 channels → (B, 48, 48, 16)
```

Each level:
- Uses VALID padding (no padding) to avoid border artifacts
- Stride=2 convolution for 2× spatial downsampling
- 1×1 convolution to maintain 16 channels
- L2 normalization to unit norm

### Output

The model returns a list of feature maps: `[Level_0, Level_1, Level_2]`

For Phase 1 training, we use only the coarsest level (Level 2: 48×48 spatial, arranged as 3×3 grid of 16×16 windows).

## Input Dimension Calculation

Because we use VALID padding with stride=2, spatial dimensions shrink at each level. The dataloader must provide exactly the right input size to yield the target coarse dimensions.

**Formula** (working backwards from target):
```
input_size = (output_size - 1) * stride + kernel_size
```

For 3 levels targeting 48×48 at coarsest:
- Level 2 output: 48×48
- Level 1 output → Level 2 input: (48-1)*2 + 3 = 97
- Level 0 output → Level 1 input: (97-1)*2 + 3 = 195
- Raw input → Level 0 input: (195-1)*2 + 3 = **391**

The `DatasetSettings.img_size` property automatically calculates this based on `num_levels`, `coarse_grid_size`, and `window_size`.

## Loss Functions

Training uses the coarsest pyramid level only (Phase 1). The loss functions remain unchanged from the original design:

### Self-Attention Entropy (Coarsest Level)

For the 3×3 grid of 16×16 windows at the coarsest level:
- Split each 48×48 feature map into nine 16×16 windows
- Compute self-attention within each window
- Minimize entropy to encourage unique embeddings

### Cross-Attention Entropy (Coarsest Level)

For corresponding windows between frame t and t+k:
- Compute cross-frame attention between matching windows
- Minimize entropy to encourage sharp matches

### Combined Loss

```
loss = α * self_entropy + β * cross_entropy
```

Default weights: α=1.0, β=0.1

## Training Stabilizers

### L2 Normalization

All embeddings are L2-normalized to unit norm before computing attention. This prevents high-norm embeddings from dominating attention regardless of content.

### Temperature Scaling

Attention logits are divided by temperature τ=0.05 before softmax. Low temperature sharpens the distribution, amplifying small differences to select clear winners.

## Training Data

Video frame pairs from single continuous takes (no cuts). The dataset automatically resizes frames to the exact calculated input dimension (391×391 for default 3-level pyramid).

## Configuration

```bash
# Default training (3 levels, 3×3 coarse grid, 16×16 windows)
python -m barevision.flow.train

# Custom configuration
python -m barevision.flow.train \
  --model.num-levels 3 \
  --model.embed-dim 16 \
  --model.window-size 16 \
  --dataset.coarse-grid-size 3 \
  --dataset.batch-size 4

# Smoke test
python -m barevision.flow.train --smoke-test
```

Key settings in `settings.py`:
- `ModelSettings.num_levels`: Number of pyramid levels (default 3)
- `ModelSettings.embed_dim`: Output channels per level (default 16)
- `ModelSettings.window_size`: Attention window size (default 16)
- `DatasetSettings.coarse_grid_size`: Target coarse grid dimension (default 3)
- `DatasetSettings.img_size`: **Calculated automatically** based on above parameters

## Visualization

Visualizations are adapted for the pyramid:
1. Original RGB images are downscaled to match coarse embedding dimensions (48×48)
2. Attention maps from coarsest level are overlaid directly on downscaled images
3. No complex coordinate mapping needed - 1:1 correspondence between coarse pixels and downscaled RGB

## Future Phases

### Phase 2: Multi-Level Training
Train intermediate levels with flow-based window shifting to cancel ego-motion.

### Phase 3: Fine-Level Refinement
Add finest pyramid level for pixel-accurate flow estimation.

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
