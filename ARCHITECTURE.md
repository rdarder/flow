# Hierarchical Optical Flow: Architecture

An attention-based optical flow estimator designed for cheap NPUs without gatherND (warping) support.

## Overview

This model estimates optical flow between two frames using a hierarchical pyramid with windowed attention. Instead of warping pixels (which requires gatherND), we use attention mechanisms to find correspondences between frames. The pyramid structure enables capturing large motions while keeping attention matrices tractable (16×16 windows).

**Constraint-driven design**: Traditional optical flow methods warp frame 2 toward frame 1 using estimated flow. Our target NPU cannot do this efficiently. We use pure attention-based matching instead.

## Architecture Layers

The system processes images through four abstraction layers:

```
Image (B, H, W, C)
    ↓ [Embedding Pyramid: 2×2 patchify → 1×1 conv]
Grid (B, H', W', embed_dim) - embeddings at each pyramid level
    ↓ [GridFlowEstimator: split into windows]
Window (B, num_windows, 16, 16, embed_dim) - batched attention windows
    ↓ [Token-level attention: grid_to_tokens]
Token (B*num_windows, 256, embed_dim) - flattened for attention
```

## Component Map

| Concept | File | Primary Class/Function | Purpose |
|---------|------|------------------------|---------|
| **End-to-end model** | `hierarchical_model.py` | `HierarchicalFlowModel` | Orchestrates pyramid + flow estimation across levels |
| **Embedding pyramid** | `embedding_pyramid.py` | `EmbeddingPyramid` | Generates multi-scale embeddings (2×2 → 1×1 conv) |
| **Grid flow estimation** | `grid_flow.py` | `GridFlowEstimator` | Splits grids into windows, runs attention, stitches results |
| **Token attention** | `token_attention.py` | `TokenCrossAttention`, `TokenSelfAttention` | Core attention mechanisms (cross-frame + self-frame) |
| **Flow blending** | `flow_blender.py` | `PriorBlender`, `upsample_flow_2x` | Confidence-weighted blend of lookup and prior flow |
| **Spatial utilities** | `window_grid.py` | `WindowGrid`, `create_coordinate_grid` | Split/stitch windows, coordinate generation, grid↔token transforms |
| **Training** | `train.py` | `create_train_state`, `train_step` | Main training loop with loss computation |

## Data Flow (Forward Pass)

**Input**: Two frames `img1`, `img2` (B, 64, 64, 3) → auto-cropped to valid size

### Phase 1: Embedding Pyramid
```python
pyramid1 = EmbeddingPyramid(img1)  # [level_0 (16×16), level_1 (32×32)]
pyramid2 = EmbeddingPyramid(img2)
```

Each pyramid level follows the same pattern:
- Group 2×2 spatial region → flatten → 1×1 conv → 16-dim embedding
- Level 1 (finest): 2×2 pixels → 16-dim
- Level 0 (coarser): 2×2 Level-1 embeddings → 16-dim

### Phase 2: Level-by-Level Flow Estimation

**Level 0 (Coarsest)**:
```python
# Initialize priors (hardcoded)
prior_flow = zeros(B, 16, 16, 2)
prior_confidence = full(B, 16, 16, 1, 0.5)

# Estimate flow
flow_0, conf_0 = GridFlowEstimator(emb1_0, emb2_0, prior_flow, prior_confidence)
```

**Level 1 (Finest)**:
```python
# Upsample coarse flow as prior
prior_flow = upsample_flow_2x(flow_0)      # (B, 32, 32, 2)
prior_confidence = upsample_confidence_2x(conf_0)  # (B, 32, 32, 1)

# Estimate with prior guidance
flow_1, conf_1 = GridFlowEstimator(emb1_1, emb2_1, prior_flow, prior_confidence)
```

### Phase 3: Output Conversion
```python
# Convert normalized flow to pixel coordinates
flow_pixels = flow_1 * scale  # scale = image_size (e.g., 64)
```

**Output**: `flow_pixels` (B, 32, 32, 2) - flow field at finest pyramid resolution

## Key Algorithms

### 1. Token Cross-Attention (Cross-Frame Matching)

Located in `token_attention.py:TokenCrossAttention`

**Purpose**: Match each token in frame 1 to tokens in frame 2, guided by prior flow.

**Mechanism**:
```
attention_score = visual_similarity + spatial_proximity

visual_similarity = dot(q_features, k_features) * learned_scale
spatial_proximity = gaussian_kernel(q_pos + prior_flow, k_pos) * prior_confidence
```

The prior flow shifts the query position ("search near where coarse flow points"), and prior confidence modulates the spatial penalty ("trust coarse flow more when it's confident").

**Outside Window Penalty**: When prior flow points outside the 16×16 window, we add a distance-based penalty since we can't verify those matches locally.

**Output**: Flow estimate (where did each pixel move?), confidence (attention max), attention weights.

### 2. Flow Blending (Hierarchical Integration)

Located in `flow_blender.py:PriorBlender`

**Purpose**: Combine token-level lookup results with coarse-level prior flow.

**Formula**:
```
weight_lookup = conf_lookup
weight_prior = conf_prior
flow_blended = (weight_lookup * flow_lookup + weight_prior * flow_prior) 
                / (weight_lookup + weight_prior)
conf_blended = (conf_lookup + conf_prior) / 2
```

**Intuition**: 
- High lookup confidence → trust what we found locally
- High prior confidence → trust the upsampled coarse estimate
- This enables flow from outside the current window to enter via the prior

### 3. Token Self-Attention (Peer Propagation)

Located in `token_attention.py:TokenSelfAttention`

**Purpose**: Refine flow estimates by borrowing from confident neighbors in the same frame.

**Mechanism**: Roughly: if the predicted flow via cross attention and blending is not confident enough, 
then look for similar embeddings in the same frame, those similar embeddings are more likely to be part of 
the same object and they would be flowing similarly, so copy them or get influenced by them.

```
attention_score = visual_similarity + spatial_proximity + confidence_bias

flow_refined = softmax(attention_score) @ neighbor_flows
```

The confidence bias term gives more weight to tokens with high consensus. This fills gaps in textureless regions and occlusions.

**Output**: Refined flow estimate and updated confidence (consensus from neighbors).

### 4. Window Grid Operations

Located in `window_grid.py`

**Key Functions**:
- `WindowGrid.split()`: (B, H, W, C) → (B, num_windows, 16, 16, C)
- `WindowGrid.stitch()`: (B, num_windows, 16, 16, C) → (B, H, W, C)
- `create_coordinate_grid()`: Generate normalized [0,1] coordinates for positions
- `grid_to_tokens()`: (B, H, W, C) → (B, H*W, C) for attention
- `tokens_to_grid()`: (B, H*W, C) → (B, H, W, C) after attention

## Abstraction Boundaries

### `hierarchical_model.py` (Orchestration)
**Owns**: Multi-level coordination, input validation, output conversion
**Doesn't own**: Attention mechanics, spatial operations

### `grid_flow.py` (Grid-Level Processing)
**Owns**: Window splitting/stitching, batching windows together, calling attention modules
**Doesn't own**: How attention works, how blending works

### `token_attention.py` (Token-Level Attention)
**Owns**: Attention score computation (visual + spatial + confidence), softmax, flow aggregation
**Doesn't know**: Whether it's processing a full image or a window

### `flow_blender.py` (Flow Combination)
**Owns**: Confidence-weighted blending formula, upsampling utilities
**Doesn't know**: Where flow estimates come from

### `window_grid.py` (Spatial Utilities)
**Owns**: Coordinate generation, grid↔token transforms, resolution validation
**No learned parameters**: Pure geometric operations

### `embedding_pyramid.py` (Feature Extraction)
**Owns**: 2×2 patchify, 1×1 conv projection, level-wise feature extraction
**Uniform operation**: Same code path for all pyramid levels

## Resolution Requirements

For `num_levels` pyramid levels and `window_size=16`:

```
valid_resolution = window_size * 2^num_levels

Examples:
- 1 level: 16 * 2^1 = 32×32 → 16×16 flow
- 2 levels: 16 * 2^2 = 64×64 → 32×32 flow
- 3 levels: 16 * 2^3 = 128×128 → 64×64 flow
```

The model auto-crops inputs if `auto_crop=True` (default).

## Configuration

Training configured via `settings.py` with tyro CLI:

```bash
python -m flow.train --model.num-levels 2 --dataset.img-size 64 --training.epochs 50
```

Key settings:
- `ModelSettings.num_levels`: Pyramid depth (default 2)
- `ModelSettings.embed_dim`: Embedding dimension (default 16)
- `ModelSettings.window_size`: Attention window size (default 16)
- `TrainingSettings.auto_crop`: Enable automatic input cropping (default True)

## Development Notes

- **Testing**: Run smoke test after architectural changes: `python -m flow.train --smoke-test`
- **Methodology**: "Integrate immediately, verify always"
- **Attention matrices**: Fixed at 256×256 (16×16 windows) regardless of image size
- **No warping**: The model never warps images; all correspondence via attention
- **Gradient isolation**: Each pyramid level trains independently (stop-gradient on priors)

## Future Directions (v2+)

### Shift/Crop Search Windows
Use average prior flow to crop frame 2 before matching, recentering search without full 16×16 search.

### Overlapping Windows
Process 18×18 regions, keep center 16×16. Gives PeerPropagation access to neighbors across window boundaries.

### Learned Upsampling
Replace 2× replication with small convolution for smoother flow upsampling.

### Confidence Calibration
Learn temperature per level so confidence scores are comparable across scales.

### More Levels
Add Level 2 (8×8 windows) for even coarser motion or finer resolution at Level 1.
