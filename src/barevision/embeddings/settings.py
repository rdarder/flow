"""Configuration settings for embeddings training.

Minimal settings using tyro for CLI parsing. Only includes parameters
currently used by the embeddings training script.
"""

from dataclasses import dataclass

from barevision.embeddings.checkpointer import CheckpointSettings
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
    max_frame_distance: int = 3
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
        every_steps: Log metrics, statistics, and console output every N steps
        visualizations_every_steps: Generate and log visualization figures every N steps
    """

    tensorboard_dir: str = "runs"
    every_steps: int = 200
    visualizations_every_steps: int = 1000

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
class ModelSettings:
    """Model architecture settings for embeddings training.

    Attributes:
        embed_dim: Output embedding dimension per level
        hidden_dim: Hidden feature dimension
        num_groups: Number of groups for grouped convolutions
        num_levels: Number of pyramid levels
    """

    embed_dim: int = 16
    hidden_dim: int = 32
    num_groups: int = 4
    num_levels: int = 3

    def __post_init__(self):
        check_value(
            self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        )
        check_value(
            self.embed_dim >= 1, f"embed_dim must be >= 1, got {self.embed_dim}"
        )


@dataclass(frozen=True)
class SpatialVarianceLossSettings:
    """Settings for spatial variance loss.

    Attributes:
        window_size: Attention window size in pixels (must divide feature map dimensions)
        level_weight_decay: Loss weight decay factor per level (default 1.0 = uniform)
                           Coarser levels get higher weight: level_i weight = decay^i.
                           Set to 1.0 for uniform weighting across levels.
        lambda_self: Self-attention loss weight in [0, 1] (default 0.5 = equal weighting)
                    loss = lambda_self * self_loss + (1 - lambda_self) * cross_loss
        self_temperature: Temperature for self-attention softmax (default 0.3)
                         Lower = sharper attention peaks
        cross_temperature: Temperature for cross-attention softmax (default 0.3)
                          Lower = sharper attention peaks
    """

    window_size: int = 16
    level_weight_decay: float = 1.1
    lambda_self: float = 0.6
    self_temperature: float = 0.25
    cross_temperature: float = 0.25

    def __post_init__(self):
        check_value(
            0 <= self.lambda_self <= 1,
            f"lambda_self must be in [0, 1], got {self.lambda_self}",
        )
        check_value(
            self.level_weight_decay >= 0,
            f"level_weight_decay must be >= 0, got {self.level_weight_decay}",
        )
        check_value(
            self.self_temperature > 0,
            f"self_temperature must be > 0, got {self.self_temperature}",
        )
        check_value(
            self.cross_temperature > 0,
            f"cross_temperature must be > 0, got {self.cross_temperature}",
        )
        check_value(
            self.window_size >= 1, f"window_size must be >= 1, got {self.window_size}"
        )


@dataclass(frozen=True)
class LossSettings:
    """Loss settings for embeddings training.

    Attributes:
        spatial_variance: Spatial variance loss configuration
    """

    spatial_variance: SpatialVarianceLossSettings


@dataclass
class TrainingSettings:
    """Training hyperparameters.

    Attributes:
        seed: Random seed for all randomness: model initialization, data shuffling, train/val split
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
    """Full settings for embeddings training.

    This is the main entry point for CLI configuration.
    run_name_prefix: Prefix for auto-generated run names
    """

    dataset: DatasetSettings
    model: ModelSettings
    loss: LossSettings
    training: TrainingSettings
    logging: LoggingSettings
    checkpoint: CheckpointSettings
    validation: ValidationSettings
    run_name_prefix: str = "embeddings"
