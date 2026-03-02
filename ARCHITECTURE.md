# Barevision: Non-Semantic Perception Architecture

Barevision is a collection of machine learning models for non-semantic perception tasks, specifically designed to run on cheap NPUs with strict hardware constraints.

## Project Vision

Create a complete perception pipeline for low-cost robots using only a monocular camera. The system understands the 3D world through geometric reasoning (optical flow → depth → pose → SLAM) without semantic labels ("this is a wall, a person"), focusing instead on low-level "voxel" reconstruction.

## Hardware Constraints (NPU-Bound)

This project targets NPUs with the following limitations:

- **No `gatherND` operations**: Traditional image warping is prohibited
- **Fixed memory access patterns**: Only convolutions, pooling, element-wise operations
- **Limited memory**: Hierarchical designs keep attention matrices tractable
- **Cheap deployment**: Models must run on sub-$10 hardware

**Design principle**: Every algorithm must respect these constraints. If an operation requires `gatherND`, we find an alternative (e.g., attention-based matching instead of warping).

## Component Map

| Package | Status | Purpose | Key Constraint |
|---------|--------|---------|----------------|
| **[`barevision.flow`](src/barevision/flow/ARCHITECTURE.md)** | ✅ Implemented | Hierarchical optical flow estimation | No warping → attention-based matching |
| **[`barevision.embeddings`](src/barevision/embeddings/ARCHITECTURE.md)** | 🚧 Planned | Self-supervised embedding learning | Sharp, unique patch representations |
| `barevision.depth` | 📋 Future | Monocular depth from flow | Depth from flow magnitude + epipolar geometry |
| `barevision.pose` | 📋 Future | Camera pose estimation | Rotation/translation from flow field |
| `barevision.slam` | 📋 Future | Visual SLAM system | Integration of flow, depth, pose |

## Core Design Patterns

### 1. Hierarchical Processing
- Coarse-to-fine pyramids capture large motions
- Fixed-size attention windows (16×16) regardless of image resolution
- Confidence-weighted blending across levels

### 2. Attention-Based Matching
- Replace `gatherND` warping with attention mechanisms
- Visual similarity + spatial proximity scoring
- Self-attention for peer propagation in textureless regions

### 3. Gradient Isolation
- Each pyramid level trains independently
- Stop-gradient on upsampled priors
- Prevents gradient confusion in deep hierarchies

### 4. Abstraction Boundaries
```
Image → Grid → Window → Token
```
Clean separation between spatial operations (splitting/stitching) and attention mechanics.

## Development Methodology

**"Integrate immediately, verify always"** – Run smoke tests after each architectural change to ensure the system remains functional.

### Verification Pipeline
1. **Unit tests**: `pytest src/barevision/flow`
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