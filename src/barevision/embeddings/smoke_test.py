"""Smoke test for embeddings training with spatial variance loss.

This script sets up a minimal but working training setup that does
logging and visualization with epoch limits to run through most branches.

Run: python -m barevision.embeddings.smoke_test
"""

from barevision.embeddings.training import EmbeddingsTrainer
from barevision.embeddings.settings import (
    Settings,
    DatasetSettings,
    ModelSettings,
    LossSettings,
    SpatialVarianceLossSettings,
    TrainingSettings,
    LoggingSettings,
    ValidationSettings,
)
from barevision.embeddings.checkpointer import CheckpointSettings


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,
            coarse_grid_size=1,
            window_size=16,
            num_levels=2,  # Only 2 levels for faster testing
            min_frame_distance=1,
            max_frame_distance=5,
            max_samples=1,  # Only 1 sample per epoch
            num_workers=0,
        ),
        model=ModelSettings(
            embed_dim=8,  # Smaller embedding dim
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
        training=TrainingSettings(
            epochs=2,  # Run 2 epochs to test epoch boundaries
            learning_rate=1e-3,
        ),
        logging=LoggingSettings(
            tensorboard_dir="test_runs",
            run_name_prefix="smoke_test",
            every_steps=1,  # Log everything every step
            visualizations_every_steps=1,  # Visualize every step
        ),
        checkpoint=CheckpointSettings(
            every_steps=2,  # Checkpoint every 2 steps
            location="test_checkpoints",
        ),
        validation=ValidationSettings(
            every_epochs=2,  # Validate every 2 epochs
        ),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDINGS SMOKE TEST")
    print("=" * 60)
    print()
    print("Running full training loop with:")
    print("  - Logging every step")
    print("  - Visualizations every step")
    print("  - Checkpointing every 2 steps")
    print("  - Validation every 2 epochs")
    print()

    settings = create_smoke_test_settings()
    trainer = EmbeddingsTrainer(settings)
    trainer()

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED ✓")
    print("=" * 60)
    print()
    print("The embeddings training pipeline is working correctly.")
    print("You can now run full training with:")
    print("  python -m barevision.flow.embeddings.training")
