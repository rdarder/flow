"""Configuration settings for embedding model training.

Minimal settings using tyro for CLI parsing. Only includes parameters
currently used by the training script.
"""

from dataclasses import dataclass


@dataclass
class DatasetSettings:
    """Dataset configuration.

    Attributes:
        batch_size: Training batch size
        img_size: Input image size as (height, width) tuple.
                 Must result in embeddings divisible by window_size (16).
                 Model uses 3x3 valid conv, so output is (H-2, W-2).
                 Recommended: (194, 194) -> (192, 192) embeddings = 12x12 windows
        max_frame_distance: Maximum temporal distance for frame pairs
        num_workers: Number of worker processes for data loading (0 = main process only)
    """

    batch_size: int = 4
    img_size: tuple[int, int] = (194, 194)
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
class TrainingSettings:
    """Training hyperparameters.

    Attributes:
        epochs: Number of training epochs
        steps_per_epoch: Steps per epoch (-1 for full dataset)
        learning_rate: Optimizer learning rate
        smoke_test: Run minimal smoke test (overrides epochs=1, steps_per_epoch=10, batch_size=2)
    """

    epochs: int = 1
    steps_per_epoch: int = -1
    learning_rate: float = 1e-4
    smoke_test: bool = False

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
    """

    dataset: DatasetSettings
    training: TrainingSettings


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=2,
            img_size=(194, 194),  # Results in 192x192 embeddings (12x12 windows)
            max_frame_distance=5,
            num_workers=0,  # No multiprocessing in smoke tests (simpler debugging)
        ),
        training=TrainingSettings(
            epochs=1,
            steps_per_epoch=10,
            learning_rate=1e-4,
        ),
    )
