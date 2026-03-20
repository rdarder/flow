from barevision.flow.settings import (
    Settings,
    DatasetSettings,
    ModelSettings,
    TrainingSettings,
    LoggingSettings,
    CheckpointSettings,
    ValidationSettings,
    AugmentationSettings,
)
from barevision.flow.joint.training import train


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,  # Minimum for speed
            coarse_grid_size=1,  # Phase 2: 1×1 grid at coarsest
            window_size=16,
            num_levels=3,  # 3 pyramid levels
            min_frame_distance=1,  # Adjacent frames
            max_frame_distance=5,  # Up to 5 frames apart for motion variety
            max_samples=2,  # Only 2 samples for speed
            num_workers=0,
        ),
        model=ModelSettings(
            window_size=16,
            num_levels=3,
            embed_dim=16,
            level_weight_decay=1.0,
            lambda_entropy=0.5,
            recon_weight=0.1,
            entropy_temperature=1.0,
            flow_temperature=0.3,
            flow_hidden_dim=16,
        ),
        training=TrainingSettings(
            epochs=1,
            learning_rate=1e-4,
        ),
        logging=LoggingSettings(
            tensorboard_dir="runs",
            run_name_prefix="smoke_test",
            every_steps=1,  # Log everything every step in smoke test
            visualizations_every_steps=1,  # Log visualizations every step in smoke test
        ),
        checkpoint=CheckpointSettings(
            every_steps=0,  # Disable periodic checkpointing in smoke test
            location="checkpoints",
            save_final=False,  # Don't save final checkpoint in smoke test
        ),
        validation=ValidationSettings(
            every_epochs=0,  # Disable validation in smoke test for speed
            save_best=False,
        ),
        augmentation=AugmentationSettings(
            horizontal_flip_prob=0.5,  # Enable horizontal flip in smoke test
            vertical_flip_prob=0.0,
            rotation_prob=0.0,
            rotation_max_angle=15.0,
            color_augmentation_prob=0.0,
            color_jitter_strength=0.1,
            swap_frames_prob=0.0,
        ),
    )


if __name__ == "__main__":
    train(create_smoke_test_settings())
