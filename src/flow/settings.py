"""Configuration settings for the flow model training pipeline.

Provides typed dataclasses for model, dataset, training, and logging configuration.
Uses window_grid utilities for image size validation against pyramid levels.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from flow.window_grid import compute_valid_resolution, validate_resolution


@dataclass
class ModelSettings:
    """Model architecture settings.

    Attributes:
        num_levels: Number of pyramid levels (coarse to fine)
        embed_dim: Embedding dimension at each pyramid level
        in_channels: Number of input channels (3 for RGB, 1 for grayscale)
        window_size: Size of attention windows for flow processing
        auto_crop: Whether to automatically crop inputs to valid size
    """

    num_levels: int = 2
    embed_dim: int = 16
    in_channels: int = 3
    window_size: int = 16
    auto_crop: bool = True

    def __post_init__(self):
        if self.num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {self.num_levels}")
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {self.embed_dim}")
        if self.in_channels not in [1, 3]:
            raise ValueError(f"in_channels must be 1 or 3, got {self.in_channels}")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")


@dataclass
class DatasetSettings:
    """Dataset configuration settings.

    Attributes:
        img_size: Size of input images (square, H=W)
        length: Number of samples in synthetic dataset
        max_flow: Maximum flow magnitude in pixels
        batch_size: Training batch size
        num_workers: Number of data loading workers
        blob_size_range: Min/max size of synthetic blobs
        num_blobs_range: Min/max number of blobs (auto-scaled if None)
    """

    img_size: int = 64
    length: int = 5000
    max_flow: int = 5
    batch_size: int = 32
    num_workers: int = 4
    blob_size_range: Tuple[int, int] = (2, 6)
    num_blobs_range: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if self.img_size < 8:
            raise ValueError(f"img_size must be >= 8, got {self.img_size}")
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}")
        if self.max_flow < 1:
            raise ValueError(f"max_flow must be >= 1, got {self.max_flow}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.blob_size_range[0] > self.blob_size_range[1]:
            raise ValueError(
                f"blob_size_range min must be <= max, got {self.blob_size_range}"
            )

        # Auto-calculate num_blobs_range based on image size if not provided
        # Base is 1-2 blobs for 32x32, scale proportionally with area
        if self.num_blobs_range is None:
            base_blobs = 2  # Base for 32x32
            scale_factor = (self.img_size / 32) ** 2  # Scale with area
            min_blobs = max(1, int(base_blobs * scale_factor * 0.5))
            max_blobs = max(min_blobs + 1, int(base_blobs * scale_factor))
            self.num_blobs_range = (min_blobs, max_blobs)


@dataclass
class TrainingSettings:
    """Training hyperparameters and loop settings.

    Attributes:
        learning_rate: Optimizer learning rate
        epochs: Number of training epochs
        steps_per_epoch: Steps to run per epoch (-1 for full dataset)
        log_every_steps: Log metrics every N steps
        checkpoint_freq: Save checkpoint every N steps (0 to disable)
        checkpoint_dir: Directory to save checkpoints
        keep_last_n_checkpoints: Number of recent checkpoints to keep (0 to keep all)
        grad_clip_norm: Gradient clipping norm (0 to disable)
        seed: Random seed for reproducibility
        resume: Whether to resume from latest checkpoint in checkpoint_dir
    """

    learning_rate: float = 1e-4
    epochs: int = 100
    steps_per_epoch: int = -1
    log_every_steps: int = 50
    checkpoint_freq: int = 1000
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3
    grad_clip_norm: float = 0.0
    seed: int = 42
    resume: bool = False

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.log_every_steps < 1:
            raise ValueError(
                f"log_every_steps must be >= 1, got {self.log_every_steps}"
            )
        if self.checkpoint_freq < 0:
            raise ValueError(
                f"checkpoint_freq must be >= 0, got {self.checkpoint_freq}"
            )
        if self.keep_last_n_checkpoints < 0:
            raise ValueError(
                f"keep_last_n_checkpoints must be >= 0, got {self.keep_last_n_checkpoints}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(f"grad_clip_norm must be >= 0, got {self.grad_clip_norm}")


@dataclass
class LoggingSettings:
    """TensorBoard logging and visualization settings.

    Attributes:
        log_dir: Root directory for TensorBoard logs
        run_name_prefix: Prefix for auto-generated run names
        num_visualization_samples: Number of samples to visualize per epoch
        log_views: Tuple of view names to log
    """

    log_dir: str = "runs"
    run_name_prefix: str = "flow"
    num_visualization_samples: int = 4
    log_views: Tuple[str, ...] = (
        "overview",
        "pyramid",
        "confidence",
        "blending",
    )

    def __post_init__(self):
        if not self.log_dir:
            raise ValueError("log_dir cannot be empty")
        if self.num_visualization_samples < 1:
            raise ValueError(
                f"num_visualization_samples must be >= 1, got {self.num_visualization_samples}"
            )
        valid_views = {"overview", "pyramid", "confidence", "blending", "window"}
        invalid_views = set(self.log_views) - valid_views
        if invalid_views:
            raise ValueError(
                f"Invalid log_views: {invalid_views}. Valid: {valid_views}"
            )


@dataclass
class VisualizationSettings:
    """Visualization color and display settings.

    Attributes:
        flow_max_percent: Percentage of image resolution to use as max flow
                          magnitude for color scaling (0.1 = 10% of image size)
    """

    flow_max_percent: float = 0.1

    def __post_init__(self):
        if self.flow_max_percent <= 0 or self.flow_max_percent > 1.0:
            raise ValueError(
                f"flow_max_percent must be in (0, 1.0], got {self.flow_max_percent}"
            )


@dataclass
class Settings:
    """Complete experiment configuration container.

    Passed as single parameter to training functions.
    Uses tyro for CLI parsing with nested dataclass support.

    Attributes:
        model: ModelSettings instance
        dataset: DatasetSettings instance
        training: TrainingSettings instance
        logging: LoggingSettings instance
        visualization: VisualizationSettings instance
    """

    model: ModelSettings
    dataset: DatasetSettings
    training: TrainingSettings
    logging: LoggingSettings
    visualization: VisualizationSettings

    def validate(self) -> Tuple[bool, str]:
        """Validate cross-compatibility between model and dataset.

        Returns:
            (is_valid, message): Boolean validity and descriptive message
        """
        return validate_resolution(
            self.dataset.img_size,
            self.model.num_levels,
            self.model.window_size
        )

    def get_required_image_size(self) -> int:
        """Get minimum required image size for current model configuration."""
        return compute_valid_resolution(
            self.model.num_levels,
            self.model.window_size
        )
