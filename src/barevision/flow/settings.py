"""Configuration settings for embedding model training.

Minimal settings using tyro for CLI parsing. Only includes parameters
currently used by the training script.
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
    min_frame_distance: int = 1  # Minimum k for frame pairs (t, t+k)
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
            f"log_every_steps must be >= 1, got {self.every_steps}",
        )
        check_value(
            self.visualizations_every_steps >= 1,
            f"log_visualizations_every_steps must be >= 1, got {self.visualizations_every_steps}",
        )


@dataclass(frozen=True)
class EmbeddingModelSettings:
    """
    num_levels: Number of pyramid levels (default 3)
    embed_dim: Output embedding dimension per level (default 16)
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
    level_weight_decay: float = 1.0
    lambda_self: float = 0.5
    self_temperature: float = 0.3
    cross_temperature: float = 0.3

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
class EmbeddingLossSettings:
    """
    window_size: Attention window size in pixels (must divide img_size dimensions)
    level_weight_decay: Loss weight decay factor per level (default 1.0 = uniform)
                       Coarser levels get higher weight: level_i weight = decay^i.
                       Set to 1.0 for uniform weighting across levels.
    lambda_entropy: Cross-attention loss weight in [0, 1] (default 0.5 = equal weighting)
                   entropy_loss = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss
    entropy_temperature: Temperature for entropy loss computation (default 1.0)
                        Fixed temperature for entropy calculation. Lower = sharper peaks in loss.
    """

    window_size: int = 16
    level_weight_decay: float = 2.0  # Uniform weighting across levels
    entropy_temperature: float = 0.3  # Fixed temperature for entropy loss
    lambda_entropy: float = 0.6  # Equal weighting between self and cross entropy

    def __post_init__(self):
        check_value(
            0 <= self.lambda_entropy <= 1,
            f"lambda_entropy must be in [0, 1], got {self.lambda_entropy}",
        )
        check_value(
            self.level_weight_decay >= 0,
            f"level_weight_decay must be >= 0, got {self.level_weight_decay}",
        )
        check_value(
            self.entropy_temperature > 0,
            f"entropy_temperature must be > 0, got {self.entropy_temperature}",
        )
        check_value(
            self.window_size >= 1, f"window_size must be >= 1, got {self.window_size}"
        )


@dataclass(frozen=True)
class FlowLossSettings:
    level_weight_decay: float = 1.5


@dataclass(frozen=True)
class JointEmbeddingFlowSettings:
    """
    recon_weight: Reconstruction loss weight (default 0.1)
      total_loss = entropy_loss + recon_weight * reconstruction_loss
      Higher values prioritize tracking accuracy over embedding distinctness.
    """

    recon_weight: float = 0.2  # Reconstruction loss weight (entropy is primary)
    entropy_weight: float = 0.8

    def __post_init__(self):
        check_value(self.recon_weight >= 0, "recon_weight cannot be negative")
        check_value(self.entropy_weight >= 0, "entropy_weight cannot be negative")
        check_value(
            self.entropy_weight + self.recon_weight >= 0.1,
            "both entropy_weight and recon_weight are too small.",
        )


@dataclass(frozen=True)
class LossSetting:
    joint: JointEmbeddingFlowSettings
    flow: FlowLossSettings
    embedding: EmbeddingLossSettings


@dataclass(frozen=True)
class FlowModelSettings:
    """
    flow_temperature: Temperature for flow estimation attention (default 0.3)
                 Used during inference. Lower = sharper attention, higher = smoother.
    """

    temperature: float = 0.3  # Temperature for flow estimation
    hidden_dim: int = 16  # Flow estimator hidden layers dimensions
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
class LossSettings:
    joint: JointEmbeddingFlowSettings
    embedding: EmbeddingLossSettings
    flow: FlowLossSettings


@dataclass(frozen=True)
class JointEmbeddingFlowModelSettings:
    pass


@dataclass(frozen=True)
class ModelSettings:
    embedding: EmbeddingModelSettings
    flow: FlowModelSettings
    joint: JointEmbeddingFlowModelSettings


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
class CheckpointSettings:
    """Checkpoint configuration for model persistence.

    Attributes:
        every_steps: Save checkpoint every N steps (0 to disable periodic checkpointing)
        location: Base directory for checkpoints (default "checkpoints")
                  Final path will be {location}/{run_name}/
        save_best: Whether to save best model checkpoint based on validation loss
        save_final: Whether to save a final checkpoint when training completes
        resume_from: Path to checkpoint to resume training from (optional)
                     If provided, loads model weights and continues from saved step
    """

    every_steps: int = 100
    location: str = "checkpoints"
    save_best: bool = True
    resume_from: str = ""  # Empty string = no resume

    def __post_init__(self):
        check_value(
            self.every_steps >= 0,
            f"every_steps must be >= 0, got {self.every_steps}",
        )
        check_value(self.location, "location cannot be empty")


@dataclass
class Settings:
    """The full settings used for joint the flow model."""

    dataset: DatasetSettings
    model: ModelSettings
    loss: LossSettings
    training: TrainingSettings
    logging: LoggingSettings
    checkpoint: CheckpointSettings
    validation: ValidationSettings


@dataclass(frozen=True)
class EmbeddingsLossSettings:
    """Loss settings for standalone embeddings training.

    Uses spatial variance loss instead of entropy loss.

    Attributes:
        spatial_variance: Spatial variance loss configuration
    """

    spatial_variance: SpatialVarianceLossSettings


@dataclass(frozen=True)
class EmbeddingsModelSettings:
    """Model settings for standalone embeddings training.

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


@dataclass
class EmbeddingsSettings:
    """Settings for standalone embeddings training.

    This is separate from the joint flow training settings to allow
    independent training of embeddings with spatial variance loss.
    """

    dataset: DatasetSettings
    model: EmbeddingsModelSettings
    loss: EmbeddingsLossSettings
    training: TrainingSettings
    logging: LoggingSettings
    checkpoint: CheckpointSettings
    validation: ValidationSettings
