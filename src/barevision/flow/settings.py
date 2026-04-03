"""Configuration settings for flow estimation and joint training.

This module contains settings for the flow matching package and joint
training (which is kept for reference but is outdated).

For embeddings-only training, see barevision.embeddings.settings.
"""

from dataclasses import dataclass

from barevision.utils.checks import check_value


@dataclass
class DatasetSettings:
    """Dataset configuration.

    Attributes:
        batch_size: Training batch size
        coarse_grid_size: Target coarse-level grid dimension (default 3 for 3×3 grid)
        window_size: Window size at coarse level (default 16)
        num_levels: Number of pyramid levels (used to calculate required input size)
        min_frame_distance: Minimum temporal distance for frame pairs (default 1)
        max_frame_distance: Maximum temporal distance for frame pairs
        max_samples: Maximum samples per epoch (-1 for full dataset)
        num_workers: Number of worker processes for data loading (0 = main process only)
    """

    batch_size: int = 8
    coarse_grid_size: int = 1
    window_size: int = 16
    num_levels: int = 3
    min_frame_distance: int = 1
    max_frame_distance: int = 5
    max_samples: int = -1
    num_workers: int = 4

    def __post_init__(self):
        check_value(
            self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}"
        )
        check_value(
            self.coarse_grid_size >= 1,
            f"coarse_grid_size must be >= 1, got {self.coarse_grid_size}",
        )
        check_value(
            self.window_size >= 1, f"window_size must be >= 1, got {self.window_size}"
        )
        check_value(
            self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        )
        check_value(
            self.min_frame_distance >= 1,
            f"min_frame_distance must be >= 1, got {self.min_frame_distance}",
        )
        check_value(
            self.max_frame_distance >= self.min_frame_distance,
            f"max_frame_distance ({self.max_frame_distance}) must be >= min_frame_distance ({self.min_frame_distance})",
        )
        check_value(
            self.num_workers >= 0, f"num_workers must be >= 0, got {self.num_workers}"
        )


@dataclass
class LoggingSettings:
    """TensorBoard logging configuration.

    Attributes:
        tensorboard_dir: Root directory for TensorBoard logs
        run_name_prefix: Prefix for auto-generated run names
        every_steps: Log metrics, statistics, and console output every N steps
        visualizations_every_steps: Generate and log visualization figures every N steps
    """

    tensorboard_dir: str = "runs"
    run_name_prefix: str = "flow"
    every_steps: int = 100
    visualizations_every_steps: int = 100

    def __post_init__(self):
        check_value(self.tensorboard_dir, "log_dir cannot be empty")
        check_value(
            self.every_steps >= 1,
            f"every_steps must be >= 1, got {self.every_steps}",
        )
        check_value(
            self.visualizations_every_steps >= 1,
            f"visualizations_every_steps must be >= 1, got {self.visualizations_every_steps}",
        )


@dataclass(frozen=True)
class FlowModelSettings:
    """Flow estimation model settings.

    Attributes:
        temperature: Temperature for flow estimation attention
        hidden_dim: Flow estimator hidden layer dimensions
        window_size: Window size for flow estimation
    """

    temperature: float = 0.3
    hidden_dim: int = 16
    window_size: int = 16

    def __post_init__(self):
        check_value(
            self.window_size >= 1, f"window_size must be >= 1, got {self.window_size}"
        )
        check_value(
            self.temperature > 0,
            f"Temperature must be > 0, got {self.temperature}",
        )
        check_value(
            self.hidden_dim >= 1,
            f"hidden dimension must be >= 1, got {self.hidden_dim}",
        )


@dataclass(frozen=True)
class FlowLossSettings:
    """Flow loss settings.

    Attributes:
        level_weight_decay: Loss weight decay factor per level
    """

    level_weight_decay: float = 1.5


@dataclass(frozen=True)
class JointEmbeddingFlowSettings:
    """Joint training loss settings.

    Attributes:
        recon_weight: Reconstruction loss weight
        entropy_weight: Entropy loss weight
    """

    recon_weight: float = 0.2
    entropy_weight: float = 0.8

    def __post_init__(self):
        check_value(self.recon_weight >= 0, "recon_weight cannot be negative")
        check_value(self.entropy_weight >= 0, "entropy_weight cannot be negative")
        check_value(
            self.entropy_weight + self.recon_weight >= 0.1,
            "both entropy_weight and recon_weight are too small.",
        )


@dataclass(frozen=True)
class JointEmbeddingFlowModelSettings:
    """Joint model settings (currently empty)."""

    pass


@dataclass(frozen=True)
class ModelSettings:
    """Combined model settings for joint training.

    Attributes:
        embedding: Embedding model settings
        flow: Flow model settings
        joint: Joint training settings
    """

    embedding: "barevision.embeddings.settings.ModelSettings"
    flow: FlowModelSettings
    joint: JointEmbeddingFlowModelSettings


@dataclass(frozen=True)
class LossSettings:
    """Combined loss settings for joint training.

    Attributes:
        joint: Joint training loss settings
        flow: Flow loss settings
    """

    joint: JointEmbeddingFlowSettings
    flow: FlowLossSettings


@dataclass
class TrainingSettings:
    """Training hyperparameters.

    Attributes:
        seed: Random seed for all randomness
        epochs: Number of training epochs
        learning_rate: Optimizer learning rate
    """

    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-3

    def __post_init__(self):
        check_value(self.seed >= 0, f"seed must be >= 0, got {self.seed}")
        check_value(self.epochs >= 1, f"epochs must be >= 1, got {self.epochs}")
        check_value(
            self.learning_rate > 0,
            f"learning_rate must be > 0, got {self.learning_rate}",
        )


@dataclass
class ValidationSettings:
    """Validation configuration.

    Attributes:
        every_epochs: Run validation every N epochs (0 to disable validation)
    """

    every_epochs: int = 1

    def __post_init__(self):
        check_value(
            self.every_epochs >= 0,
            f"every_epochs must be >= 0, got {self.every_epochs}",
        )


@dataclass
class Settings:
    """Full settings for joint flow training (outdated).

    This is kept for reference but joint training is superseded by
    standalone embeddings training.
    """

    dataset: DatasetSettings
    model: ModelSettings
    loss: LossSettings
    training: TrainingSettings
    logging: LoggingSettings
    checkpoint: "barevision.embeddings.checkpointer.CheckpointSettings"
    validation: ValidationSettings


# Import embeddings settings for backwards compatibility and joint training
import barevision.embeddings.settings as embeddings_settings
