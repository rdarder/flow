# Barevision: Non-Semantic Perception for Cheap Robots

Barevision is a collection of machine learning models for non-semantic perception tasks, specifically designed to run on cheap hardware.

## Project Vision

Create a complete perception pipeline for low-cost robots using only a monocular camera and cheap hardware.

## Components

### Current Packages

- **`barevision.flow`**: Self-supervised embedding learning for patch matching
  - Sharpness-based loss functions (self/cross-attention entropy)
  - Trains embeddings optimized for attention-based matching
  - Foundation for optical flow estimation

- **`barevision.old_flow`** (Deprecated): Hierarchical optical flow estimation
  - Kept for reference during transition
  - Will be removed once new flow pipeline is complete

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

### Running Embedding Training

```bash
# Smoke test
python -m barevision.flow.train --smoke-test

# Full training with custom settings
python -m barevision.flow.train --model.window-size 16 --dataset.img-size 200 200
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: High-level project architecture and design decisions
- **[src/barevision/flow/ARCHITECTURE.md](src/barevision/flow/ARCHITECTURE.md)**: Embedding training approach
- **[AGENTS.md](AGENTS.md)**: Development notes for AI assistants

## Development Philosophy

**"Integrate immediately, verify always"** – Run smoke tests after each architectural change to ensure the system remains functional.

## License

MIT
