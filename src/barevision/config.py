"""Configuration loading for barevision training.

This module provides the RootConfig class that ties together all configuration
subsections, and the load_config() function for loading from YAML files.

Config classes are defined in their respective business modules:
- model.py: UIBConfig, LevelConfig, HierarchicalModelConfig
- dataset/video.py: DatasetConfig
- embeddings/spatial_losses.py: SpatialVarianceLossConfig, LossConfig
- embeddings/checkpointer.py: CheckpointConfig

Training-related configs (TrainingConfig, LoggingConfig, ValidationConfig)
are defined here to avoid circular imports.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from barevision.embeddings.model import HierarchicalModelConfig
from barevision.dataset.video import DatasetConfig
from barevision.embeddings.spatial_losses import SpatialVarianceLossConfig
from barevision.embeddings.checkpointer import CheckpointConfig


class TrainingConfig(BaseModel):
    """Training hyperparameters.

    Attributes:
        seed: Random seed for all randomness: model initialization, data shuffling, train/val split
        epochs: Number of training epochs
        learning_rate: Optimizer learning rate
    """

    model_config = ConfigDict(frozen=True)

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    learning_rate: float = Field(
        default=1e-3, gt=0, description="Optimizer learning rate"
    )


class LoggingConfig(BaseModel):
    """TensorBoard logging configuration.

    Attributes:
        tensorboard_dir: Root directory for TensorBoard logs
        every_steps: Log metrics, statistics, and console output every N steps
        visualizations_every_steps: Generate and log visualization figures every N steps
    """

    model_config = ConfigDict(frozen=True)

    tensorboard_dir: str = Field(
        default="runs", min_length=1, description="Root directory for TensorBoard logs"
    )
    every_steps: int = Field(default=200, ge=1, description="Log metrics every N steps")
    visualizations_every_steps: int = Field(
        default=1000, ge=1, description="Generate visualizations every N steps"
    )


class ValidationConfig(BaseModel):
    """Validation configuration.

    Attributes:
        every_epochs: Run validation every N epochs (0 to disable validation)
    """

    model_config = ConfigDict(frozen=True)

    every_epochs: int = Field(
        default=1, ge=0, description="Run validation every N epochs (0 to disable)"
    )


class LossConfig(BaseModel):
    """Loss configuration wrapper.

    Attributes:
        spatial_variance: Spatial variance loss configuration
    """

    model_config = ConfigDict(frozen=True)

    spatial_variance: SpatialVarianceLossConfig = Field(
        description="Spatial variance loss configuration"
    )


class RootConfig(BaseModel):
    """Root configuration for embeddings training.

    This is the main configuration object passed to EmbeddingsTrainer.
    All subsections are frozen for JAX/JIT compatibility.

    Attributes:
        name: Experiment name identifier
        model: Hierarchical embedding model configuration
        dataset: Dataset loading configuration
        loss: Loss function configuration
        training: Training hyperparameters
        logging: TensorBoard logging configuration
        checkpoint: Checkpointing configuration
        validation: Validation configuration
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        default="experiment", min_length=1, description="Experiment name identifier"
    )
    model: HierarchicalModelConfig = Field(
        description="Hierarchical embedding model configuration"
    )
    dataset: DatasetConfig = Field(description="Dataset loading configuration")
    loss: LossConfig = Field(description="Loss function configuration")
    training: TrainingConfig = Field(description="Training hyperparameters")
    logging: LoggingConfig = Field(description="TensorBoard logging configuration")
    checkpoint: CheckpointConfig = Field(description="Checkpointing configuration")
    validation: ValidationConfig = Field(description="Validation configuration")


def load_config(path: Path = Path("config.yaml")) -> RootConfig:
    """Load and validate configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated RootConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If config validation fails
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return RootConfig(**data)
