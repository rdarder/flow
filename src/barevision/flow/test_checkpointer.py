"""Tests for the unified Checkpointer.

Tests verify that checkpointing logic correctly handles:
- Periodic step checkpoints
- Best model checkpoints based on validation loss
- Metadata preservation
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from flax import nnx

from barevision.flow.checkpointer import Checkpointer
from barevision.flow.settings import (
    CheckpointSettings,
    ValidationSettings,
)
from barevision.embeddings.settings import (
    Settings,
    DatasetSettings,
    ModelSettings,
    LossSettings,
    SpatialVarianceLossSettings,
    TrainingSettings,
    LoggingSettings,
)
from barevision.utils.console import ConsoleLogger


@pytest.fixture
def test_settings(tmp_path):
    """Create test settings with temporary checkpoint directory."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,
            coarse_grid_size=1,
            window_size=16,
            num_levels=2,
        ),
        model=ModelSettings(
            embed_dim=8,
            hidden_dim=16,
            num_groups=2,
            num_levels=2,
        ),
        loss=LossSettings(
            spatial_variance=SpatialVarianceLossSettings(
                window_size=16,
                level_weight_decay=1.0,
                lambda_self=0.5,
                self_temperature=0.3,
                cross_temperature=0.3,
            )
        ),
        training=TrainingSettings(epochs=1, learning_rate=1e-3),
        logging=LoggingSettings(
            tensorboard_dir=str(tmp_path / "runs"),
            run_name_prefix="test",
        ),
        checkpoint=CheckpointSettings(
            every_steps=5,
            location=str(tmp_path / "checkpoints"),
            save_best=True,
        ),
        validation=ValidationSettings(
            every_epochs=1,
        ),
    )


@pytest.fixture
def model_and_logger(test_settings):
    """Create model and logger for testing."""
    rngs = nnx.Rngs(test_settings.training.seed)
    model = nnx.Linear(8, 16, rngs=rngs)  # Simple generic model
    logger = ConsoleLogger()
    return model, logger


def test_maybe_save_step_respects_interval(test_settings, model_and_logger, tmp_path):
    """Checkpointer only saves at step intervals."""
    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    # Steps that shouldn't trigger save (not multiples of 5)
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=1, global_step=1)
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=2, global_step=2)
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=3, global_step=3)

    # Step 5 should trigger save
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=5, global_step=5)

    # Step 10 should trigger save
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=10, global_step=10)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    saved_checkpoints = list(checkpoint_dir.glob("step_*"))

    assert len(saved_checkpoints) == 2
    assert (checkpoint_dir / "step_000005").exists()
    assert (checkpoint_dir / "step_000010").exists()


def test_maybe_save_step_disabled(test_settings, model_and_logger):
    """Checkpointer doesn't save when every_steps is 0."""
    test_settings.checkpoint.every_steps = 0
    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=1, global_step=1)
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=5, global_step=5)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    assert not checkpoint_dir.exists()


def test_maybe_save_best_only_on_improvement(test_settings, model_and_logger, tmp_path):
    """Best checkpoint only saved when validation loss improves."""
    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    # First validation - should save (inf -> 1.0)
    checkpointer.maybe_save_best(model, epoch=1, global_step=10, val_loss=1.0)

    # Worse validation - should NOT save (1.0 -> 1.5)
    checkpointer.maybe_save_best(model, epoch=1, global_step=20, val_loss=1.5)

    # Better validation - should save (1.0 -> 0.8)
    checkpointer.maybe_save_best(model, epoch=1, global_step=30, val_loss=0.8)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    best_checkpoint = checkpoint_dir / "best"

    assert best_checkpoint.exists()

    # Verify best_val_loss is tracked
    assert checkpointer.best_val_loss == 0.8
    assert checkpointer.best_step == 30


def test_maybe_save_best_disabled(test_settings, model_and_logger):
    """Best checkpoint not saved when save_best is False."""
    test_settings.checkpoint.save_best = False
    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    checkpointer.maybe_save_best(model, epoch=1, global_step=10, val_loss=0.5)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    assert not (checkpoint_dir / "best").exists()


def test_checkpoint_contains_metadata(test_settings, model_and_logger, tmp_path):
    """Saved checkpoints contain expected metadata."""
    import orbax.checkpoint as ocp

    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    checkpointer.maybe_save_step(model, epoch=2, step_in_epoch=5, global_step=5)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    checkpoint_path = checkpoint_dir / "step_000005"

    # Load checkpoint and verify metadata
    checkpointer_instance = ocp.PyTreeCheckpointer()
    restored = checkpointer_instance.restore(
        checkpoint_path, item=ocp.args.PyTreeRestore()
    )

    assert restored["step"] == 5
    assert restored["epoch"] == 2
    assert restored["step_in_epoch"] == 5
    assert "model" in restored


def test_best_checkpoint_contains_val_loss(test_settings, model_and_logger, tmp_path):
    """Best checkpoint contains validation loss metadata."""
    import orbax.checkpoint as ocp

    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    checkpointer.maybe_save_best(model, epoch=1, global_step=10, val_loss=0.75)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    checkpoint_path = checkpoint_dir / "best"

    # Load checkpoint and verify metadata
    checkpointer_instance = ocp.PyTreeCheckpointer()
    restored = checkpointer_instance.restore(
        checkpoint_path, item=ocp.args.PyTreeRestore()
    )

    assert restored["best_val_loss"] == 0.75
    assert restored["step"] == 10


def test_close_logs_summary(test_settings, model_and_logger, capsys):
    """Checkpointer.close() logs training summary."""
    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    # Simulate some checkpointing activity
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=5, global_step=5)
    checkpointer.maybe_save_best(model, epoch=1, global_step=10, val_loss=0.5)

    checkpointer.close()

    captured = capsys.readouterr()
    assert "Validation Summary" in captured.out
    assert "Best validation loss: 0.500000" in captured.out
    assert "Achieved at step: 10" in captured.out
    assert "Total checkpoints saved: 2" in captured.out


def test_checkpoint_overwrites_best(test_settings, model_and_logger, tmp_path):
    """Best checkpoint is overwritten when better validation is found."""
    import orbax.checkpoint as ocp

    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    # First best
    checkpointer.maybe_save_best(model, epoch=1, global_step=10, val_loss=1.0)

    # Better best - should overwrite
    checkpointer.maybe_save_best(model, epoch=2, global_step=20, val_loss=0.5)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    checkpoint_path = checkpoint_dir / "best"

    # Load and verify it's the second checkpoint
    checkpointer_instance = ocp.PyTreeCheckpointer()
    restored = checkpointer_instance.restore(
        checkpoint_path, item=ocp.args.PyTreeRestore()
    )

    assert restored["step"] == 20
    assert restored["best_val_loss"] == 0.5
    assert restored["epoch"] == 2


def test_static_methods(test_settings, model_and_logger, tmp_path):
    """Test static methods for loading checkpoints."""
    import orbax.checkpoint as ocp

    model, logger = model_and_logger
    checkpointer = Checkpointer(test_settings.checkpoint, "test_run", logger)

    # Save a checkpoint
    checkpointer.maybe_save_step(model, epoch=1, step_in_epoch=5, global_step=5)

    checkpoint_dir = Path(test_settings.checkpoint.location) / "test_run"
    checkpoint_path = checkpoint_dir / "step_000005"

    # Test load_metadata
    metadata = Checkpointer.load_metadata(checkpoint_path)
    assert metadata["step"] == 5
    assert metadata["epoch"] == 1
    assert metadata["step_in_epoch"] == 5

    # Test restore
    new_model = nnx.Linear(8, 16, rngs=nnx.Rngs(0))  # Different seed
    step = Checkpointer.restore(checkpoint_path, new_model)
    assert step == 5
