# Barevision: Non-Semantic Perception Architecture

Barevision is a collection of machine learning models for non-semantic perception tasks, specifically designed to run on cheap inference hardware.

## Project Vision

Create a complete perception pipeline for low-cost robots using only a monocular camera. The system understands the 3D world through geometric reasoning (optical flow → depth → pose → SLAM) without semantic labels ("this is a wall, a person"), focusing instead on low-level "voxel" reconstruction.

## Known Hardware Constraints

- We limit ourselves to using only one camera, so the approach is monocular.
- Most solutions involving images as matrices tend to involve `gatherND` style operations, which serve as a basis for image warping. For example, if we wanted to apply flow to an image, gatherND would be the go-to approach. The NPUs we're targeting don't have support for gatherND. Even if they had it, it'd be inefficient. We don't resort to solutions involving this kind of dynamic memory access (a matrix containing individual memory offsets).
- We aim to be able to reconstruct a 3D scene on very cheap hardware, so optimization doesn't happen after the design, but rather guides the solution search space.

## Component Map

| Package | Status | Purpose | Key Constraint |
|---------|--------|---------|----------------|
| **[`barevision.flow`](src/barevision/flow/ARCHITECTURE.md)** | ✅ Implemented | Self-supervised embedding learning for patch matching | Sharp, unique patch representations |
| `barevision.depth` | 📋 Future | Monocular depth from flow | Depth from flow magnitude + epipolar geometry |
| `barevision.pose` | 📋 Future | Camera pose estimation | Rotation/translation from flow field |
| `barevision.slam` | 📋 Future | Visual SLAM system | Integration of flow, depth, pose |

## Core Design Patterns

- We use simplified cross attention mechanisms to find a patch in one frame inside the other frame.
- We use a small attention window size for being able to run attention efficiently.
- We use simplified self attention mechanism to help guess flow on areas where cross attention didn't match well.
- We process frames in a pyramid of resolutions to be able to capture large flow with a small window size.
- Hierarchies in the pyramid are processed coarse to fine grained, each level becoming a prior for next level flow.
- Attention mechanism use visual similarity and spatial proximity for computing weights.
- The sharpness of the attention vectors is a strong indicator of a good match, we use it as confidence.

### Abstraction Boundaries

```
Embedding <- Window <- Grid <- Image level
```

Clean separation between spatial operations (splitting/stitching) and attention mechanics.

- **Embedding**: A dimensional representation of a patch of the image, computed by a model. An abstract "pixel".
- **Window**: A square contiguous region of embeddings where attention mechanism can run on.
- **Grid**: A rectangular arrangement of non overlapping windows that makes up the entire image being analyzed.
- **Image**: The raw frame, typically in 1 or 3 channels.

Images are analyzed in pairs (as in a pair of consecutive frames in a video).

### Verification Pipeline

1. **Unit tests**: `pytest barevision.flow`
2. **Smoke test**: `python -m barevision.flow.train --smoke-test`
3. **Import validation**: `from barevision.flow import SimpleEmbeddingModel`

## Package Details

### `barevision.flow` - Self-Supervised Embedding Learning

**Goal**: Learn embedding representations optimized for attention-based patch matching.

**Approach**:
- **Loss function**: Self-attention entropy + cross-attention entropy minimization
- **Objective**: Embeddings should identify patches uniquely (sharp self-attention) and find 1-2 matches in next frame (sharp cross-attention)
- **Training data**: "Single-take" video frames (no cuts)
- **Future**: Reconstructive loss will be added to the composite loss

**Rationale**: Training embeddings with simpler, focused objectives that directly optimize for patch matching. This avoids the gradient complexity of end-to-end flow training and enables faster experimentation.

## Integration Strategy

The flow package is designed for incremental integration:

1. **Phase 1**: Standalone embedding training produces optimized representations
2. **Phase 2**: Embeddings integrated into full optical flow estimation pipeline
3. **Phase 3**: Depth estimation from flow outputs
4. **Phase 4**: Integration of pose and SLAM as pipelines stabilize

Each package maintains its own:
- Training loop
- Configuration (`settings.py`)
- Architecture documentation
- Test suite

## Configuration

All packages use tyro CLI with nested dataclass support:

```bash
# Embedding training
python -m barevision.flow.train --model.window-size 16 --dataset.img-size 200 200
```

## Future Directions

### Short-term (v2+)
- Add reconstructive loss to composite objective
- Overlapping windows for better peer propagation
- Confidence calibration across pyramid levels

### Medium-term
- Depth estimation from optical flow (epipolar geometry)
- Camera pose estimation
- Moving object detection via flow consensus

### Long-term
- Full visual SLAM pipeline
- Multi-camera support
- Real-time deployment on target NPU hardware

## Documentation Structure

- **This file**: High-level project architecture
- **Package-specific**: `src/barevision/<package>/ARCHITECTURE.md`
- **Development**: `AGENTS.md` (notes for AI assistants)
- **Getting started**: `README.md`

New contributors should read this document first, then dive into specific package documentation as needed.
