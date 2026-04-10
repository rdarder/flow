"""Smoke test for embeddings training with linear attention flow loss.

This script sets up a minimal but working training setup that does
logging and visualization with epoch limits to run through most branches.

Run: python -m barevision.embeddings.smoke_test
"""

from barevision.embeddings.training import EmbeddingsTrainer
from barevision.config import RootConfig, HierarchicalModelConfig, LossConfig
from barevision.embeddings.model import LevelConfig, UIBConfig
from barevision.dataset.video import DatasetConfig
from barevision.embeddings.linear_attention_loss import LinearAttentionFlowLossConfig
from barevision.embeddings.checkpointer import CheckpointConfig
from barevision.config import TrainingConfig, LoggingConfig, ValidationConfig


def create_smoke_test_config() -> RootConfig:
    """Minimal config for quick validation."""
    return RootConfig(
        name="smoke_test",
        model=HierarchicalModelConfig(
            levels=(
                LevelConfig(
                    uib_configs=(
                        UIBConfig(
                            in_channels=3,
                            out_channels=8,
                            expanded_channels=16,
                            use_dw_before_expand=True,
                            use_dw_after_expand=True,
                            downsample_after=True,
                            use_l2_norm=False,
                        ),
                        UIBConfig(
                            in_channels=8,
                            out_channels=16,
                            expanded_channels=32,
                            use_dw_before_expand=True,
                            use_dw_after_expand=True,
                            downsample_after=False,
                            use_l2_norm=True,
                        ),
                    ),
                ),
                LevelConfig(
                    uib_configs=(
                        UIBConfig(
                            in_channels=16,
                            out_channels=16,
                            expanded_channels=64,
                            use_dw_before_expand=True,
                            use_dw_after_expand=True,
                            downsample_after=True,
                            use_l2_norm=True,
                        ),
                    ),
                ),
            ),
        ),
        dataset=DatasetConfig(
            batch_size=1,
            coarse_grid_size=1,
            window_size=16,
            num_levels=2,  # Only 2 levels for faster testing
            min_frame_distance=1,
            max_frame_distance=5,
            max_samples=1,  # Only 1 sample per epoch
            frame_cache_max_mb=100,  # Small limit for smoke test
        ),
        loss=LossConfig(
            linear_attention_flow=LinearAttentionFlowLossConfig(
                window_size=16,
                level_weight_decay=1.0,
                lambda_reconstruction=1.0,
                lambda_diversity=0.1,
                diversity_scope="per_window",
            )
        ),
        training=TrainingConfig(
            epochs=2,  # Run 2 epochs to test epoch boundaries
            learning_rate=1e-3,
        ),
        logging=LoggingConfig(
            tensorboard_dir="test_runs",
            every_steps=1,  # Log everything every step
            visualizations_every_steps=1,  # Visualize every step
        ),
        checkpoint=CheckpointConfig(
            every_epochs=2,  # Checkpoint every 2 epochs
            location="test_checkpoints",
            keep_best_n=2,  # Keep best 2 by validation loss
        ),
        validation=ValidationConfig(
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
    print("  - Checkpointing every 2 epochs")
    print("  - Validation every 2 epochs")
    print()

    config = create_smoke_test_config()
    trainer = EmbeddingsTrainer(config)
    trainer()

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED ✓")
    print("=" * 60)
    print()
    print("The embeddings training pipeline is working correctly.")
    print("You can now run full training with:")
    print("  python -m barevision.embeddings.training --config config.yaml")
