# Flow Training Pipeline

Hierarchical optical flow model using multi-level pyramid with windowed attention and confidence-based blending.

## Key Design Decisions

- **Hierarchical Architecture**: Coarse-to-fine pyramid (Level 0 → Level 1 → ...)
- **Windowed Attention**: Each level processes embeddings within attention windows
- **Confidence Blending**: Coarse flow is blended into fine flow using confidence weights
- **Flow Visualization**: All levels displayed at same resolution using pixel-equivalent scaling

For detailed algorithm design, see [design.md](design.md).

## Project Structure

```
src/flow/
├── train.py                  # Main training loop
├── settings.py               # Configuration dataclasses
├── hierarchical_model.py     # Complete flow model
├── checkpoint_manager.py     # Orbax checkpointing
├── visualization.py          # Multi-view diagnostics
├── logging_utils.py          # TensorBoard logging
├── synthetic_dataset.py      # Training data generation
└── window_grid.py            # Resolution validation utilities
```

## Settings

Configuration via tyro CLI with nested dataclass support:

```bash
# Example: Train with custom settings
python -m flow.train --model.num-levels 3 --dataset.img-size 128 --training.epochs 50

# Resume from checkpoint
python -m flow.train --training.resume

# Smoke test
python -m flow.train --smoke-test
```

See `settings.py` for all available options:
- `ModelSettings`: Architecture parameters
- `DatasetSettings`: Training data configuration  
- `TrainingSettings`: Optimization and loop settings
- `LoggingSettings`: TensorBoard configuration
- `VisualizationSettings`: Color scales and display options

## Development Notes

- Methodology: "Integrate immediately, verify always" - smoke test after each change
- Shape-aware loss function handles pyramid output vs input size mismatch
- Fixed color scales for cross-epoch comparison (configurable via `flow_max_percent`)
