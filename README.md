# Barevision: Non-Semantic Perception for Cheap Robots

Barevision is a collection of machine learning models for non-semantic perception tasks, specifically designed to run on cheap hardware.

## Project Vision

Create a complete perception pipeline for low-cost robots using only a monocular camera and cheap hardware.

## Components

### Current Packages

- **`barevision.flow`**: Hierarchical optical flow estimation using attention-based matching
  - Multi-level pyramid with windowed attention
  - Confidence-based flow blending
  - No warping operations (gatherND-free)

- **`barevision.embeddings`** (Planned): Self-supervised embedding learning for patch matching
  - Sharpness-based loss functions (self/cross-attention entropy)
  - Compatible with flow model's embedding interface

For more details on each package, look for the ARCHITECTURE.md file within each package's path.

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
## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: High-level project architecture and design decisions
- **[src/barevision/flow/ARCHITECTURE.md](src/barevision/flow/ARCHITECTURE.md)**: Optical flow model details
- **[src/barevision/embeddings/ARCHITECTURE.md](src/barevision/embeddings/ARCHITECTURE.md)**: Embedding training approach
- **[AGENTS.md](AGENTS.md)**: Development notes for AI assistants

## Development Philosophy

**"Integrate immediately, verify always"** – Run smoke tests after each architectural change to ensure the system remains functional.

## License

MIT
