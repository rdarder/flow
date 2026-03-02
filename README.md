# Barevision: Non-Semantic Perception for Cheap Robots

Barevision is a collection of machine learning models for non-semantic perception tasks, specifically designed to run on cheap NPUs with strict hardware constraints.

## Project Vision

Create a complete perception pipeline for low-cost robots using only a monocular camera, with algorithms that respect NPU limitations (no arbitrary memory access patterns like `gatherND`).

## Components

### Current Packages

- **`barevision.flow`**: Hierarchical optical flow estimation using attention-based matching
  - Multi-level pyramid with windowed attention
  - Confidence-based flow blending
  - No warping operations (gatherND-free)

- **`barevision.embeddings`** (Planned): Self-supervised embedding learning for patch matching
  - Sharpness-based loss functions (self/cross-attention entropy)
  - Compatible with flow model's embedding interface

### Future Packages

- **`barevision.depth`**: Monocular depth estimation from optical flow
- **`barevision.pose`**: Camera pose estimation
- **`barevision.slam`**: Visual SLAM system

## Quick Start

### Installation

```bash
# Install from source
pip install -e .
```

### Running Optical Flow Training

```bash
# Smoke test
python -m barevision.flow.train --smoke-test

# Full training with custom settings
python -m barevision.flow.train --model.num-levels 3 --dataset.img-size 384 512 --training.epochs 50
```

## Hardware Constraints

This project targets NPUs with the following limitations:
- **No `gatherND` operations**: Traditional image warping is prohibited
- **Fixed memory access patterns**: Convolutions, pooling, element-wise operations only
- **Limited memory**: Hierarchical design keeps attention matrices tractable (16×16 windows)

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: High-level project architecture and design decisions
- **[src/barevision/flow/ARCHITECTURE.md](src/barevision/flow/ARCHITECTURE.md)**: Optical flow model details
- **[src/barevision/embeddings/ARCHITECTURE.md](src/barevision/embeddings/ARCHITECTURE.md)**: Embedding training approach
- **[AGENTS.md](AGENTS.md)**: Development notes for AI assistants

## Development Philosophy

**"Integrate immediately, verify always"** – Run smoke tests after each architectural change to ensure the system remains functional.

## License

MIT