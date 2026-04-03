from barevision.flow.joint import JointEmbeddingFlowModel
from barevision.flow.settings import (
    Settings,
    DatasetSettings,
    ModelSettings,
    TrainingSettings,
    LoggingSettings,
    CheckpointSettings,
    ValidationSettings,
    FlowModelSettings,
    JointEmbeddingFlowModelSettings,
    LossSettings,
    FlowLossSettings,
    JointEmbeddingFlowSettings,
)
from barevision.embeddings.settings import (
    ModelSettings as EmbeddingsModelSettings,
    LossSettings as EmbeddingsLossSettings,
    SpatialVarianceLossSettings,
)
from barevision.flow.joint.training import Trainer


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,
            coarse_grid_size=1,
            window_size=16,
            num_levels=3,
            min_frame_distance=1,
            max_frame_distance=5,
            max_samples=1,
            num_workers=0,
        ),
        model=ModelSettings(
            embedding=EmbeddingsModelSettings(),
            flow=FlowModelSettings(),
            joint=JointEmbeddingFlowModelSettings(),
        ),
        loss=LossSettings(
            embedding=EmbeddingsLossSettings(
                spatial_variance=SpatialVarianceLossSettings(
                    window_size=16,
                )
            ),
            flow=FlowLossSettings(),
            joint=JointEmbeddingFlowSettings(),
        ),
        training=TrainingSettings(
            epochs=2,
            learning_rate=1e-3,
        ),
        logging=LoggingSettings(
            tensorboard_dir="test_runs",
            run_name_prefix="smoke_test",
            every_steps=1,  # Log everything every step in smoke test
            visualizations_every_steps=1,  # Log visualizations every step in smoke test
        ),
        checkpoint=CheckpointSettings(
            every_steps=2,
            location="test_checkpoints",
        ),
        validation=ValidationSettings(
            every_epochs=2,
        ),
    )


if __name__ == "__main__":
    Trainer(create_smoke_test_settings())()
