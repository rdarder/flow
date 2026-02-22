# Implementation Plan: Hierarchical Optical Flow

This plan implements the foundation for the hierarchical optical flow model described in ARCHITECTURE.md. We proceed in small, verifiable steps.

## Phase 1: Multi-Resolution Synthetic Dataset

### Goal
Extend the synthetic dataset to support arbitrary resolutions while keeping train.py working.

### Implementation
1. Modify `src/synthetic_dataset.py`:
   - Add `image_size` parameter to `SyntheticFlowDataset.__init__` (default: 18 for backward compatibility)
   - Ensure generated images and flow fields match the specified size
   - Motion generation should scale appropriately with image size

2. Create `tests/test_synthetic_dataset.py`:
   - Test that dataset generates correct shapes for 16×16, 32×32, 64×64
   - Test that flow field has same spatial dimensions as images
   - Test that motion magnitudes are reasonable relative to image size

3. Modify `src/train.py`:
   - Explicitly pass `image_size=18` to dataset initialization
   - No functional changes to training logic
   - Train.py continues to work exactly as before

### Verification
Run: `python -m pytest tests/test_synthetic_dataset.py -v`
Expected: All tests pass, showing correct shapes for different resolutions.

Run: `python -m src.train`
Expected: Training runs normally with 18×18 images (unchanged behavior).

---

## Phase 2: Embedding Pyramid Module

### Goal
Create a module that generates multi-scale embeddings from an image using the unified 2×2→flatten→1×1 conv approach.

### Implementation
1. Create `src/embedding_pyramid.py`:
   - Implement `EmbeddingPyramid` class
   - Input: image tensor (B, H, W, C) and config (num_levels, embedding_dim)
   - Output: list of embeddings [level_0, level_1, ...] from coarsest to finest
   - Each level:
     - Takes 2×2 spatial region from input (pixels for level 1, embeddings for level 0)
     - Flattens to vector (4*C channels)
     - Applies 1×1 convolution to embedding_dim
   - Each level learns its own 1×1 conv weights

2. Create `tests/test_embedding_pyramid.py`:
   - Test with 64×64 grayscale image, num_levels=2
     - Expect: level_0 shape (B, 16, 16, 16), level_1 shape (B, 32, 32, 16)
   - Test with 64×64 RGB image, num_levels=2
     - Expect: level_1 has 4*3=12 channels flattened, level_0 has 4*16=64 channels flattened
   - Test that different levels have different weights (not shared)
   - Test with num_levels=1 (single level, 32×32 output for 64×64 input)
   - Test shape consistency across batch dimension

### Verification
Run: `python -m pytest tests/test_embedding_pyramid.py -v`
Expected: All tests pass, showing correct pyramid structure and independent weights per level.

Run: `python -c "from src.embedding_pyramid import EmbeddingPyramid; import jax.numpy as jnp; pyramid = EmbeddingPyramid(num_levels=2, embedding_dim=16); x = jnp.zeros((1, 64, 64, 1)); out = pyramid(x); print([o.shape for o in out])"`
Expected: `[(1, 16, 16, 16), (1, 32, 32, 16)]`

---

## Success Criteria

- [ ] Phase 1: `train.py` runs without modification, tests validate multi-resolution support
- [ ] Phase 2: Embedding pyramid generates correct shapes, tests validate dimensions and weights

## Notes for Implementing Agent

- Focus on correctness over optimization
- Use 1×1 convolutions only (no spatial padding issues)
- The pyramid logic should be independent of the existing model code for now
- Each level's 1×1 conv should be a separate parameter, not shared
- Write the simplest implementation that passes the tests
