# Flow Training Pipeline

Hierarchical optical flow model using multi-level pyramid with windowed attention and confidence-based blending.

## Key Design Decisions

- **Hierarchical Architecture**: Coarse-to-fine pyramid (Level 0 → Level 1 → ...)
- **Windowed Attention**: Each level processes embeddings within attention windows
- **Confidence Blending**: Coarse flow is blended into fine flow using confidence weights
- **Flow Visualization**: All levels displayed at same resolution using pixel-equivalent scaling
- **No Warping**: Pure attention-based matching, no gatherND operations
- **Gradient Isolation**: Each level trains independently (stop-gradient on upsampled priors)
- **Abstraction Layers**: Clean separation: image → grid → window → token

For detailed algorithm design, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Structure

```
src/flow/
├── train.py                  # Main training loop
├── settings.py               # Configuration dataclasses
├── hierarchical_model.py     # Complete flow model (orchestrates pyramid + levels)
├── token_attention.py        # Token-level attention (TokenCrossAttention, TokenSelfAttention)
├── grid_flow.py              # Grid flow estimator (window splitting + attention orchestration)
├── flow_blender.py           # Flow blending utilities (PriorBlender, upsampling)
├── embedding_pyramid.py      # Multi-scale embedding generation
├── window_grid.py            # Spatial utilities (split/stitch windows, coordinates)
├── checkpoint_manager.py     # Orbax checkpointing
├── visualization.py          # Multi-view diagnostics
├── logging_utils.py          # TensorBoard logging
├── synthetic_dataset.py      # Training data generation
└── chairs_dataset.py         # Real training data (FlyingChairs)
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
