# Flow Training Pipeline - Development Log

## Current Status: Phase 1 Complete ✓

### ✅ Phase 1: Settings Infrastructure + Train.py Integration

**Completed:**
- Settings dataclasses (`ModelSettings`, `DatasetSettings`, `TrainingSettings`, `LoggingSettings`)
- Settings container with cross-validation (img_size vs num_levels)
- tyro CLI integration with nested dataclass support
- Refactored `train.py` to use:
  - `HierarchicalFlowModel` instead of `BarebonesFlowModel`
  - Settings-based configuration
  - Shape-aware loss function (handles pyramid output vs input size mismatch)
  - Smoke test mode (`--smoke-test`)

**CLI Usage:**
```bash
# Default settings
python -m flow.train

# Override specific settings
python -m flow.train --model.num-levels 3 --dataset.img-size 128
python -m flow.train --training.epochs 10 --training.steps-per-epoch 50

# Smoke test (fast verification)
python -m flow.train --smoke-test
```

**Verified:**
- ✓ Settings validation works (cross-checks img_size against num_levels)
- ✓ tyro CLI parsing with nested args
- ✓ Training loop runs end-to-end
- ✓ Visualization figures render correctly
- ✓ Loss decreases (5.65 → 5.64 in smoke test)

### Next: Phase 2 - Checkpointing

**Plan:**
1. Add Orbax checkpointing to save/restore model state
2. Add resume-from-checkpoint logic
3. Test checkpoint save/load cycle
4. Smoke test with checkpointing enabled

**Key features:**
- Save checkpoints every N steps (configurable)
- Resume training from checkpoint
- Keep N most recent checkpoints (cleanup)

### Phase 3 - Visualization Enhancement

**Plan:**
- Multi-view hierarchical visualization
- Pyramid level flow visualization
- Confidence map visualization
- Blending weight visualization

### Phase 4 - Subcommands

**Plan:**
- `train` command (default)
- `resume` command with checkpoint path
- `eval` command for inference

## Settings Reference

### ModelSettings
- `num_levels`: Number of pyramid levels (default: 2)
- `embed_dim`: Embedding dimension (default: 16)
- `in_channels`: Input channels 1 or 3 (default: 3)
- `window_size`: Attention window size (default: 16)
- `auto_crop`: Auto-crop to valid resolution (default: True)

### DatasetSettings
- `img_size`: Input image size, square (default: 64)
- `length`: Dataset size (default: 5000)
- `max_flow`: Max flow magnitude (default: 5)
- `batch_size`: Training batch size (default: 4)
- `num_workers`: DataLoader workers (default: 4)
- `blob_size_range`: Synthetic blob size range (default: (2, 6))

### TrainingSettings
- `learning_rate`: Optimizer LR (default: 1e-4)
- `epochs`: Training epochs (default: 100)
- `steps_per_epoch`: Steps per epoch, -1 for full (default: -1)
- `log_every_steps`: Logging frequency (default: 50)
- `checkpoint_freq`: Checkpoint interval, 0 to disable (default: 1000)
- `grad_clip_norm`: Gradient clipping, 0 to disable (default: 0.0)
- `seed`: Random seed (default: 42)

### LoggingSettings
- `log_dir`: TensorBoard log directory (default: "runs")
- `run_name_prefix`: Run name prefix (default: "flow")
- `num_visualization_samples`: Samples to visualize (default: 4)
- `log_views`: Views to log (default: ("overview", "pyramid", "confidence", "blending"))
