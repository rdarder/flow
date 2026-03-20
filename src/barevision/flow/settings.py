"""Configuration settings for embedding model training.

Minimal settings using tyro for CLI parsing. Only includes parameters
currently used by the training script.
"""

from dataclasses import dataclass
from typing import Tuple

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
        log_dir: Root directory for TensorBoard logs
        run_name_prefix: Prefix for auto-generated run names
        log_every_steps: Log metrics, statistics, and console output every N steps
        log_visualizations_every_steps: Generate and log visualization figures every N steps
    """

    log_dir: str = "runs"
    run_name_prefix: str = "embeddings"
    log_every_steps: int = 10
    log_visualizations_every_steps: int = 20

    def __post_init__(self):
        check_value(self.log_dir, "log_dir cannot be empty")
        check_value(
            self.log_every_steps >= 1,
            f"log_every_steps must be >= 1, got {self.log_every_steps}",
        )
        check_value(
            self.log_visualizations_every_steps >= 1,
            f"log_visualizations_every_steps must be >= 1, got {self.log_visualizations_every_steps}",
        )


@dataclass(frozen=True)
class ModelSettings:
    """Model architecture configuration.

    Attributes:
        window_size: Attention window size in pixels (must divide img_size dimensions)
        num_levels: Number of pyramid levels (default 3)
        embed_dim: Output embedding dimension per level (default 16)
        level_weight_decay: Loss weight decay factor per level (default 1.0 = uniform)
                           Coarser levels get higher weight: level_i weight = decay^i.
                           Set to 1.0 for uniform weighting across levels.
        lambda_entropy: Cross-attention loss weight in [0, 1] (default 0.5 = equal weighting)
                       entropy_loss = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss
        recon_weight: Reconstruction loss weight (default 0.1)
                      total_loss = entropy_loss + recon_weight * reconstruction_loss
                      Higher values prioritize tracking accuracy over embedding distinctness.
        entropy_temperature: Temperature for entropy loss computation (default 1.0)
                            Fixed temperature for entropy calculation. Lower = sharper peaks in loss.
        flow_temperature: Temperature for flow estimation attention (default 0.3)
                         Used during inference. Lower = sharper attention, higher = smoother.
    """

    window_size: int = 16
    num_levels: int = 3
    embed_dim: int = 16
    level_weight_decay: float = 1.0  # Uniform weighting across levels
    lambda_entropy: float = 0.5  # Equal weighting between self and cross entropy
    recon_weight: float = 0.1  # Reconstruction loss weight (entropy is primary)
    entropy_temperature: float = 1.0  # Fixed temperature for entropy loss
    flow_temperature: float = 0.3  # Temperature for flow estimation
    flow_hidden_dim: int = 16  # Flow estimator hidden dimension

    def __post_init__(self):
        check_value(
            self.window_size >= 1, f"window_size must be >= 1, got {self.window_size}"
        )
        check_value(
            self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        )
        check_value(
            self.embed_dim >= 1, f"embed_dim must be >= 1, got {self.embed_dim}"
        )
        check_value(
            self.level_weight_decay >= 0,
            f"level_weight_decay must be >= 0, got {self.level_weight_decay}",
        )
        check_value(
            0 <= self.lambda_entropy <= 1,
            f"lambda_entropy must be in [0, 1], got {self.lambda_entropy}",
        )
        check_value(
            self.recon_weight >= 0,
            f"recon_weight must be >= 0, got {self.recon_weight}",
        )
        check_value(
            self.entropy_temperature > 0,
            f"entropy_temperature must be > 0, got {self.entropy_temperature}",
        )
        check_value(
            self.flow_temperature > 0,
            f"flow_temperature must be > 0, got {self.flow_temperature}",
        )
        check_value(
            self.flow_hidden_dim >= 1,
            f"flow_hidden_dim must be >= 1, got {self.flow_hidden_dim}",
        )


@dataclass
class TrainingSettings:
    """Training hyperparameters.

    Attributes:
        seed: Random seed for all randomness: model initialization, data shuffling, train/val split
        epochs: Number of training epochs
        learning_rate: Optimizer learning rate
    """

    seed: int = 42
    epochs: int = 1
    learning_rate: float = 1e-4

    def __post_init__(self):
        check_value(self.seed >= 0, f"seed must be >= 0, got {self.seed}")
        check_value(self.epochs >= 1, f"epochs must be >= 1, got {self.epochs}")
        check_value(
            self.learning_rate > 0,
            f"learning_rate must be > 0, got {self.learning_rate}",
        )


@dataclass
class AugmentationSettings:
    """Data augmentation configuration.

    Augmentations are applied on-the-fly during training with per-sample deterministic seeding.
    This means the same sample always gets the same augmentation within and across epochs,
    but different samples get different augmentations.

    Attributes:
        horizontal_flip_prob: Probability of horizontal flip (left-right mirror), default 0.5
        vertical_flip_prob: Probability of vertical flip (up-down mirror), default 0.0
        rotation_prob: Probability of random rotation, default 0.0
        rotation_max_angle: Maximum rotation angle in degrees (rotation is uniform in [-angle, +angle])
        color_augmentation_prob: Probability of color augmentation (brightness/contrast), default 0.0
        color_jitter_strength: Strength of color jitter (multiplier for brightness/contrast changes)
        swap_frames_prob: Probability of swapping (img1, img2) order, default 0.0
                          Note: For self-supervised reconstruction loss, no flow negation needed.
    """

    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotation_prob: float = 0.0
    rotation_max_angle: float = 15.0
    color_augmentation_prob: float = 0.0
    color_jitter_strength: float = 0.1
    swap_frames_prob: float = 0.0

    def __post_init__(self):
        check_value(
            0 <= self.horizontal_flip_prob <= 1,
            f"horizontal_flip_prob must be in [0, 1], got {self.horizontal_flip_prob}",
        )
        check_value(
            0 <= self.vertical_flip_prob <= 1,
            f"vertical_flip_prob must be in [0, 1], got {self.vertical_flip_prob}",
        )
        check_value(
            0 <= self.rotation_prob <= 1,
            f"rotation_prob must be in [0, 1], got {self.rotation_prob}",
        )
        check_value(
            self.rotation_max_angle >= 0,
            f"rotation_max_angle must be >= 0, got {self.rotation_max_angle}",
        )
        check_value(
            0 <= self.color_augmentation_prob <= 1,
            f"color_augmentation_prob must be in [0, 1], got {self.color_augmentation_prob}",
        )
        check_value(
            self.color_jitter_strength >= 0,
            f"color_jitter_strength must be >= 0, got {self.color_jitter_strength}",
        )
        check_value(
            0 <= self.swap_frames_prob <= 1,
            f"swap_frames_prob must be in [0, 1], got {self.swap_frames_prob}",
        )


@dataclass
class ValidationSettings:
    """Validation configuration.

    Attributes:
        every_epochs: Run validation every N epochs (0 to disable validation)
        save_best: Whether to save best model checkpoint based on validation loss
    """

    every_epochs: int = 1
    save_best: bool = True

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
        save_final: Whether to save a final checkpoint when training completes
        resume_from: Path to checkpoint to resume training from (optional)
                     If provided, loads model weights and continues from saved step
    """

    every_steps: int = 100
    location: str = "checkpoints"
    save_final: bool = True
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
    training: TrainingSettings
    logging: LoggingSettings
    checkpoint: CheckpointSettings
    validation: ValidationSettings
    augmentation: AugmentationSettings
    smoke_test: bool = False
