# Flow Training Pipeline - Development Log

## Current Status: Phase 2 Complete ✓

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

### ✅ Phase 2: Checkpointing (Refactored)

**Completed:**
- Added Orbax checkpoint dependency
- Implemented **Null Object Pattern** for checkpointing:
  - `AbstractCheckpointManager`: Unified interface
  - `NullCheckpointManager`: No-op implementation when checkpointing disabled
  - `OrbaxCheckpointManager`: Full Orbax integration with `should_save()` API
  - Factory function `create_checkpoint_manager()` for clean instantiation
- Integrated proper Orbax CheckpointManager API:
  - Uses `CheckpointManagerOptions(save_interval_steps, max_to_keep)`
  - Leverages `should_save(step)` method for save decisions
  - Automatic cleanup via Orbax (not manual)
  - No conditionals in training loop - just call methods
- Handles integer key conversion (Orbax/MsgPack limitation)
- Clean API: checkpointing decisions are internal to the manager

**CLI Usage:**
```bash
# Train with checkpointing (default: every 1000 steps)
python -m flow.train

# Train with custom checkpoint frequency
python -m flow.train --training.checkpoint-freq 500

# Disable checkpointing
python -m flow.train --training.checkpoint-freq 0

# Resume from specific checkpoint
python -m flow.train --training.resume-from-checkpoint checkpoints/1000

# Auto-resume from latest checkpoint (automatic)
python -m flow.train  # Will find and resume from latest checkpoint if available

# Smoke test with checkpointing
python -m flow.train --smoke-test
```

**Implementation Details:**
- Training loop now simply calls `checkpoint_manager.should_save(step)` and `checkpoint_manager.save()`
- No `if checkpoint_manager is not None` checks in the training loop
- Null object pattern handles disabled checkpointing transparently
- All checkpointing logic centralized in `checkpoint_manager.py`
- ✓ Integer dict keys properly converted (0, 1 instead of '0', '1')
- ✓ Loss continues from where it left off after resume

### Next: Phase 3 - Visualization Enhancement

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
- `checkpoint_dir`: Directory to save checkpoints (default: "checkpoints")
- `keep_last_n_checkpoints`: Number of recent checkpoints to keep (default: 3, 0 to keep all)
- `resume_from_checkpoint`: Path to checkpoint to resume from (default: "")
- `grad_clip_norm`: Gradient clipping, 0 to disable (default: 0.0)
- `seed`: Random seed (default: 42)

### LoggingSettings
- `log_dir`: TensorBoard log directory (default: "runs")
- `run_name_prefix`: Run name prefix (default: "flow")
- `num_visualization_samples`: Samples to visualize (default: 4)
- `log_views`: Views to log (default: ("overview", "pyramid", "confidence", "blending"))
