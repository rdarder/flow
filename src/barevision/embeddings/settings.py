"""Configuration settings for embedding model training.

Minimal settings using tyro for CLI parsing. Only includes parameters
currently used by the training script.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DatasetSettings:
    """Dataset configuration.

    Attributes:
        batch_size: Training batch size
        img_size: Input image size as (height, width) tuple.
                  Must result in embeddings divisible by window_size (16).
                  Model uses 5×5 valid conv, so output is (H-4, W-4).
                  Recommended: (196, 196) -> (192, 192) embeddings = 12x12 windows
        max_frame_distance: Maximum temporal distance for frame pairs
        max_samples: Maximum samples per epoch (-1 for full dataset)
        num_workers: Number of worker processes for data loading (0 = main process only)
    """

    batch_size: int = 4
    img_size: Tuple[int, int] = (196, 196)
    max_frame_distance: int = 5
    max_samples: int = -1
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
        log_every_steps: Log metrics, statistics, and console output every N steps
        log_visualizations_every_steps: Generate and log visualization figures every N steps
    """

    log_dir: str = "runs"
    run_name_prefix: str = "embeddings"
    log_every_steps: int = 10
    log_visualizations_every_steps: int = 20

    def __post_init__(self):
        if not self.log_dir:
            raise ValueError("log_dir cannot be empty")
        if self.log_every_steps < 1:
            raise ValueError(
                f"log_every_steps must be >= 1, got {self.log_every_steps}"
            )
        if self.log_visualizations_every_steps < 1:
            raise ValueError(
                f"log_visualizations_every_steps must be >= 1, got {self.log_visualizations_every_steps}"
            )


@dataclass
class TrainingSettings:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Optimizer learning rate
    """

    epochs: int = 1
    learning_rate: float = 1e-4

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")


@dataclass
class Settings:
    """Complete experiment configuration.

    Passed as single parameter to training functions.
    Uses tyro for CLI parsing with nested dataclass support.

    Attributes:
        smoke_test: Run minimal smoke test (overrides other settings for quick validation)
    """

    dataset: DatasetSettings
    training: TrainingSettings
    logging: LoggingSettings
    smoke_test: bool = False


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,  # Minimum for speed
            img_size=(196, 196),  # 196-4=192, divisible by 16
            max_frame_distance=5,
            max_samples=2,  # Only 2 samples for speed
            num_workers=0,
        ),
        training=TrainingSettings(
            epochs=1,
            learning_rate=1e-4,
        ),
        logging=LoggingSettings(
            log_dir="runs",
            run_name_prefix="smoke_test",
            log_every_steps=1,  # Log everything every step in smoke test
            log_visualizations_every_steps=1,  # Log visualizations every step in smoke test
        ),
        smoke_test=False,  # Already applied by caller
    )
