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
        coarse_grid_size: Target coarse-level grid dimension (default 3 for 3×3 grid)
        window_size: Window size at coarse level (default 16)
        num_levels: Number of pyramid levels (used to calculate required input size)
        max_frame_distance: Maximum temporal distance for frame pairs
        max_samples: Maximum samples per epoch (-1 for full dataset)
        num_workers: Number of worker processes for data loading (0 = main process only)
        seed: Random seed for data shuffling and train/val split
    """

    batch_size: int = 4
    coarse_grid_size: int = 1  # Phase 2: 1×1 grid at coarsest level (16×16 window)
    window_size: int = 16
    num_levels: int = 3
    max_frame_distance: int = (
        2  # Phase 2: Restrict to adjacent frames for deep supervision
    )
    max_samples: int = -1
    num_workers: int = 4
    seed: int = 42

    @property
    def img_size(self) -> Tuple[int, int]:
        """Calculate required input image size based on pyramid configuration.

        Returns:
            (height, width) tuple for input images
        """
        from barevision.flow.embeddings.model import calculate_required_input_size

        # Target coarse dimension: grid_size × window_size
        target_coarse_dim = self.coarse_grid_size * self.window_size

        # Calculate required input size
        input_size = calculate_required_input_size(
            target_coarse_dim=target_coarse_dim,
            num_levels=self.num_levels,
        )

        return (input_size, input_size)

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.coarse_grid_size < 1:
            raise ValueError(
                f"coarse_grid_size must be >= 1, got {self.coarse_grid_size}"
            )
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {self.num_levels}")
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

    def should_log_something(self, step: int):
        return (
            step % self.log_visualizations_every_steps == 0
            or step % self.log_every_steps == 0
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
    flow_hidden_dim: int = 24  # Flow estimator hidden dimension

    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {self.num_levels}")
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {self.embed_dim}")
        if self.level_weight_decay < 0:
            raise ValueError(
                f"level_weight_decay must be >= 0, got {self.level_weight_decay}"
            )
        if not 0 <= self.lambda_entropy <= 1:
            raise ValueError(
                f"lambda_entropy must be in [0, 1], got {self.lambda_entropy}"
            )
        if self.recon_weight < 0:
            raise ValueError(f"recon_weight must be >= 0, got {self.recon_weight}")
        if self.entropy_temperature <= 0:
            raise ValueError(
                f"entropy_temperature must be > 0, got {self.entropy_temperature}"
            )
        if self.flow_temperature <= 0:
            raise ValueError(
                f"flow_temperature must be > 0, got {self.flow_temperature}"
            )
        if self.flow_hidden_dim < 1:
            raise ValueError(
                f"flow_hidden_dim must be >= 1, got {self.flow_hidden_dim}"
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
class ValidationSettings:
    """Validation configuration.

    Attributes:
        every_epochs: Run validation every N epochs (0 to disable validation)
        save_best: Whether to save best model checkpoint based on validation loss
    """

    every_epochs: int = 1
    save_best: bool = True

    def __post_init__(self):
        if self.every_epochs < 0:
            raise ValueError(f"every_epochs must be >= 0, got {self.every_epochs}")


@dataclass
class CheckpointSettings:
    """Checkpoint configuration for model persistence.

    Attributes:
        every_steps: Save checkpoint every N steps (0 to disable periodic checkpointing)
        location: Base directory for checkpoints (default "checkpoints")
                  Final path will be {location}/{run_name}/
        save_final: Whether to save a final checkpoint when training completes
    """

    every_steps: int = 100
    location: str = "checkpoints"
    save_final: bool = True

    def __post_init__(self):
        if self.every_steps < 0:
            raise ValueError(f"every_steps must be >= 0, got {self.every_steps}")
        if not self.location:
            raise ValueError("location cannot be empty")


@dataclass
class Settings:
    """Complete experiment configuration.

    Passed as single parameter to training functions.
    Uses tyro for CLI parsing with nested dataclass support.

    Attributes:
        smoke_test: Run minimal smoke test (overrides other settings for quick validation)
    """

    dataset: DatasetSettings
    model: ModelSettings
    training: TrainingSettings
    logging: LoggingSettings
    checkpoint: CheckpointSettings
    validation: ValidationSettings
    smoke_test: bool = False


def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        dataset=DatasetSettings(
            batch_size=1,  # Minimum for speed
            coarse_grid_size=1,  # Phase 2: 1×1 grid at coarsest
            window_size=16,
            num_levels=3,  # 3 pyramid levels
            max_frame_distance=2,  # Phase 2: Adjacent frames only
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
            flow_hidden_dim=24,
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
        checkpoint=CheckpointSettings(
            every_steps=0,  # Disable periodic checkpointing in smoke test
            location="checkpoints",
            save_final=False,  # Don't save final checkpoint in smoke test
        ),
        validation=ValidationSettings(
            every_epochs=0,  # Disable validation in smoke test for speed
            save_best=False,
        ),
        smoke_test=False,  # Already applied by caller
    )
