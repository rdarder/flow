"""Configuration settings for embedding model training.

Minimal settings using tyro for CLI parsing. Only includes parameters
currently used by the training script.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DatasetSettings:
    """Dataset configuration.

    Attributes:
        batch_size: Training batch size
        img_size: Input image size as (height, width) tuple.
                  Must result in embeddings divisible by window_size (16).
                  Model uses 5×5 valid conv, so output is (H-4, W-4).
                  Recommended: (200, 200) -> (196, 196) embeddings, not divisible
                  Better: (196, 196) -> (192, 192) embeddings = 12x12 windows
        max_frame_distance: Maximum temporal distance for frame pairs
        num_workers: Number of worker processes for data loading (0 = main process only)
    """

    batch_size: int = 4
    img_size: Tuple[int, int] = (196, 196)
    max_frame_distance: int = 5
    num_workers: int = 4

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if len(self.img_size) != 2:
            raise ValueError(f"img_size must be (height, width), got {self.img_size}")
        if self.max_frame_distance < 1:
            raise ValueError(
                f"max_frame_distance must be >= 1, got {self.max_frame_distance}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")


@dataclass
class LoggingSettings:
    """TensorBoard logging configuration.

    Attributes:
        log_dir: Root directory for TensorBoard logs
        run_name_prefix: Prefix for auto-generated run names
        log_metrics_every_steps: Log scalar/histogram metrics every N steps
        log_visualizations_every_steps: Generate and log visualization figures every N steps (0 to disable)
        log_statistics_every_steps: Log embedding/attention statistics every N steps (0 to disable)
    """

    log_dir: str = "runs"
    run_name_prefix: str = "embeddings"
    log_metrics_every_steps: int = 10
    log_visualizations_every_steps: int = 20
    log_statistics_every_steps: int = 50

    def __post_init__(self):
        if not self.log_dir:
            raise ValueError("log_dir cannot be empty")
        if self.log_metrics_every_steps < 1:
            raise ValueError(
                f"log_metrics_every_steps must be >= 1, got {self.log_metrics_every_steps}"
            )
        if self.log_visualizations_every_steps < 0:
            raise ValueError(
                f"log_visualizations_every_steps must be >= 0, got {self.log_visualizations_every_steps}"
            )
        if self.log_statistics_every_steps < 0:
            raise ValueError(
                f"log_statistics_every_steps must be >= 0, got {self.log_statistics_every_steps}"
            )


@dataclass
class LossSettings:
    """Loss function weights.

    Attributes:
        self_entropy_weight: Weight for self-attention entropy loss (alpha)
        cross_entropy_weight: Weight for cross-attention entropy loss (beta)
    """

    self_entropy_weight: float = 1.0
    cross_entropy_weight: float = 0.1

    def __post_init__(self):
        if self.self_entropy_weight < 0:
            raise ValueError(
                f"self_entropy_weight must be >= 0, got {self.self_entropy_weight}"
            )
        if self.cross_entropy_weight < 0:
            raise ValueError(
                f"cross_entropy_weight must be >= 0, got {self.cross_entropy_weight}"
            )


@dataclass
class TrainingSettings:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch (-1 for full dataset)
        learning_rate: Optimizer learning rate
        smoke_test: Run minimal smoke test (overrides epochs=1, steps_per_epoch=10, batch_size=2)
        checkpoint_freq: Save checkpoint every N steps (0 to disable)
        checkpoint_dir: Directory to save checkpoints
        keep_last_n_checkpoints: Number of recent checkpoints to keep (0 to keep all)
        resume: Whether to resume from latest checkpoint in checkpoint_dir
    """

    epochs: int = 1
    steps_per_epoch: int = -1
    learning_rate: float = 1e-4
    smoke_test: bool = False
    checkpoint_freq: int = 50
    checkpoint_dir: str = "checkpoints/embeddings"
    keep_last_n_checkpoints: int = 3
    resume: bool = False

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.checkpoint_freq < 0:
            raise ValueError(
                f"checkpoint_freq must be >= 0, got {self.checkpoint_freq}"
            )
        if self.keep_last_n_checkpoints < 0:
            raise ValueError(
                f"keep_last_n_checkpoints must be >= 0, got {self.keep_last_n_checkpoints}"
            )


@dataclass
class Settings:
    """Complete experiment configuration.

    Passed as single parameter to training functions.
    Uses tyro for CLI parsing with nested dataclass support.
    """

    dataset: DatasetSettings
    training: TrainingSettings
    logging: LoggingSettings
    loss: LossSettings = field(default_factory=LossSettings)


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,  # Minimum for speed
            img_size=(196, 196),  # 196-4=192, divisible by 16
            max_frame_distance=5,
            num_workers=0,
        ),
        training=TrainingSettings(
            epochs=1,
            steps_per_epoch=2,  # Only 2 steps for speed
            learning_rate=1e-4,
            checkpoint_freq=1,  # Save every step for testing
            checkpoint_dir="test_checkpoints/embeddings",
            keep_last_n_checkpoints=2,
        ),
        logging=LoggingSettings(
            log_dir="runs",
            run_name_prefix="smoke_test",
            log_metrics_every_steps=1,  # Log metrics every step in smoke test
            log_visualizations_every_steps=1,  # Log visualizations every step in smoke test
        ),
    )
