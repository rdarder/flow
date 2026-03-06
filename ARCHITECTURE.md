# Barevision: Non-Semantic Perception Architecture

Barevision is a collection of machine learning models for non-semantic perception tasks, specifically designed to run on cheap inference hardware.

## Project Vision

Create a complete perception pipeline for low-cost robots using only a monocular camera. The system understands the 3D world through geometric reasoning (optical flow → depth → pose → SLAM) without semantic labels ("this is a wall, a person"), focusing instead on low-level "voxel" reconstruction.

## Known hardware Constraints

- We limit ourselves to using only one camera, so the approach is monocular.
- Most solutions involving images as matrices tend to involve `gatherND` style operations, which serve as a basis for image warping. For example, 
if we wanted to apply flow to an image, gatherND would be the go to approach. The NPUs we're targeting don't have support for gatherND. Even if they had it, it'd be inefficient. we don't resort to solutions involving this kind of dynamic memory access (a matrix containing individual memory offsets)
- We aim to be able to reconstruct a 3d scene on very cheap hardware, so optimization doesn't haappen after the design, but rather guide the solution search space. 

## Component Map

| Package | Status | Purpose | Key Constraint |
|---------|--------|---------|----------------|
| **[`barevision.flow`](src/barevision/flow/ARCHITECTURE.md)** | ✅ Implemented | Hierarchical optical flow estimation | No warping → attention-based matching |
| **[`barevision.embeddings`](src/barevision/embeddings/ARCHITECTURE.md)** | 🚧 Planned | Self-supervised embedding learning | Sharp, unique patch representations |
| `barevision.depth` | 📋 Future | Monocular depth from flow | Depth from flow magnitude + epipolar geometry |
| `barevision.pose` | 📋 Future | Camera pose estimation | Rotation/translation from flow field |
| `barevision.slam` | 📋 Future | Visual SLAM system | Integration of flow, depth, pose |

## Core Design Patterns

- We use a simplified cross attention mechanisms to find a patch in one frame inside the other frame. 
- We use a small attention window size for being able to run attention efficiently.
- We use simplified self attention mechanism to help guess flow on areas where cross attention didn't match well.
- We process frames in a pyramid of resolutions to be able to capture large flow with a small window size. 
- Hierarchies in the pyramid are processed coarse to fine grained, each level becoming a prior for next level flow. 
- Attention mechanism use visual similarity and spatial proximity for computing weights. 
- The sharpness of the attention vectors is a strong indicator of a good match, we use it as confidence.

### 4. Abstraction Boundaries
```
Embedding <- Window <- Grid <- Image level
```
Clean separation between spatial operations (splitting/stitching) and attention mechanics.

Embedding: A dimensional representation of a patch of the image, computed by a model. An abstract "pixel".
Window: A square contiguous region of embeddings where attention mechanism can run on.
Grid: A rectangular arrangement of non overlapping windows that makes up the entire image being analyzed. 
Image: the raw rame, typically in 1 or 3 channels.

* Images are analyzed in pairs (as in a pair of consecutive frames in a video

TODO: downsampling and levels is not correctly explained here. perhaps done in the embeddings package.

### Verification Pipeline
1. **Unit tests**: `pytest barevision.flow`
2. **Smoke test**: `python -m barevision.flow.train --smoke-test`
3. **Import validation**: `from barevision.flow import HierarchicalFlowModel`

## Package Details

### `barevision.flow` - Optical Flow Estimation
**Architecture**: Multi-level pyramid with windowed attention
- **Level 0** (coarsest): 16×16 windows, zero prior
- **Level 1** (finest): 32×32 windows, upsampled prior from Level 0
- **TokenCrossAttention**: Cross-frame matching with prior guidance
- **TokenSelfAttention**: Peer propagation for textureless regions
- **PriorBlender**: Confidence-weighted flow combination

**Key innovation**: Pure attention-based matching without warping.

### `barevision.embeddings` - Self-Supervised Embedding Learning
**Goal**: Train embeddings separately from flow model to avoid gradient complexity.

**Approach**:
- **Loss function**: Self-attention entropy + cross-attention peakiness
- **Objective**: Embeddings should identify patches uniquely (sharp self-attention) and find 1-2 matches in next frame (sharp cross-attention)
- **Training data**: "Single-take" video frames (no cuts)
- **Interface**: Compatible with `EmbeddingPyramid` for drop-in replacement

**Rationale**: Flow model gradients are too complex/interconnected; simpler embedding objectives yield better representations.

## Integration Strategy

Packages are designed for incremental integration:

1. **Phase 1**: `embeddings` trains standalone, produces better embeddings
2. **Phase 2**: Swap `EmbeddingPyramid` in `flow` with trained embeddings
3. **Phase 3**: Train `depth` using flow outputs
4. **Phase 4**: Integrate `pose` and `slam` as pipelines stabilize

Each package maintains its own:
- Training loop
- Configuration (`settings.py`)
- Architecture documentation
- Test suite

## Configuration

All packages use tyro CLI with nested dataclass support:

```bash
# Flow training
python -m barevision.flow.train --model.num-levels 3 --dataset.img-size 384 512

# (Future) Embedding training  
python -m barevision.embeddings.train --dataset.video-path /path/to/videos
```

## Future Directions

### Short-term (v2+)
- Shift/crop search windows based on average prior flow
- Overlapping windows for better peer propagation
- Learned upsampling instead of 2× replication
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
