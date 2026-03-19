"""Checkpoint utilities for model persistence.

Uses Orbax for saving/loading model state with configuration metadata.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import orbax.checkpoint as ocp
from flax import nnx

from barevision.flow.settings import Settings


def _convert_string_keys_to_int(obj: Any) -> Any:
    """Recursively convert string keys that look like integers back to int.

    Orbax/JSON serialization converts integer keys to strings. This function
    reverses that for proper NNX state restoration.

    Args:
        obj: Dictionary or value to process

    Returns:
        Object with integer-like string keys converted to int
    """
    if isinstance(obj, dict):
        return {
            (
                int(k) if isinstance(k, str) and k.isdigit() else k
            ): _convert_string_keys_to_int(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_convert_string_keys_to_int(item) for item in obj]
    else:
        return obj


def generate_run_name(prefix: str = "flow") -> str:
    """Generate a unique run name with timestamp.

    Used consistently across logging and checkpointing.

    Args:
        prefix: Run name prefix (default "flow")

    Returns:
        Run name in format "{prefix}_{YYYYMMDD}_{HHMMSS}"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def get_checkpoint_path(
    checkpoint_location: str, run_name: str, step: int | str
) -> Path:
    """Get the directory path for a checkpoint.

    Args:
        checkpoint_location: Base directory for checkpoints
        run_name: Unique run identifier
        step: Global step number (use -1 for "final", or "best" for best model)

    Returns:
        Absolute Path to checkpoint directory
    """
    if step == -1 or step == "final":
        step_str = "final"
    elif step == "best":
        step_str = "best"
    else:
        step_str = f"step_{step:06d}"

    return Path(checkpoint_location).resolve() / run_name / step_str


def save_checkpoint(
    model: nnx.Module,
    step: int,
    settings: Settings,
    run_name: str,
    save_final: bool = False,
) -> Path:
    """Save model checkpoint with configuration metadata.

    Args:
        model: NNX model to save
        step: Current global step
        settings: Full settings object (for config metadata)
        run_name: Run identifier for directory naming
        save_final: If True, save to "final" subdirectory

    Returns:
        Path to saved checkpoint directory
    """
    # Determine checkpoint path
    if save_final:
        checkpoint_path = get_checkpoint_path(
            settings.checkpoint.location, run_name, -1
        )
    else:
        checkpoint_path = get_checkpoint_path(
            settings.checkpoint.location, run_name, step
        )

    # Prepare checkpoint data
    checkpoint_data = {
        "model": nnx.state(model).to_pure_dict(),
        "step": step,
        "config": _settings_to_config_dict(settings),
    }

    # Save using Orbax
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(checkpoint_path, ocp.args.PyTreeSave(checkpoint_data))

    return checkpoint_path


def save_best_checkpoint(
    model: nnx.Module,
    step: int,
    val_loss: float,
    settings: Settings,
    run_name: str,
) -> Path:
    """Save best model checkpoint based on validation loss.

    Args:
        model: NNX model to save
        step: Current global step
        val_loss: Validation loss that triggered this checkpoint
        settings: Full settings object (for config metadata)
        run_name: Run identifier for directory naming

    Returns:
        Path to saved checkpoint directory
    """
    import shutil

    checkpoint_path = get_checkpoint_path(
        settings.checkpoint.location, run_name, "best"
    )

    # Remove existing best checkpoint if it exists (overwrite)
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)

    # Prepare checkpoint data with additional validation loss info
    checkpoint_data = {
        "model": nnx.state(model).to_pure_dict(),
        "step": step,
        "config": _settings_to_config_dict(settings),
        "best_val_loss": val_loss,
    }

    # Save using Orbax
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(checkpoint_path, ocp.args.PyTreeSave(checkpoint_data))

    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dictionary with keys: model, step, config
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path, item=ocp.args.PyTreeRestore())

    # Convert string keys back to integers for NNX compatibility
    return _convert_string_keys_to_int(restored)


def restore_model_from_checkpoint(
    checkpoint_path: Path,
    model: nnx.Module,
) -> int:
    """Restore model state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model instance to restore into (must have compatible structure)

    Returns:
        Global step from checkpoint
    """
    checkpoint_data = load_checkpoint(checkpoint_path)

    # Restore model state
    nnx.update(model, nnx.State(checkpoint_data["model"]))

    return checkpoint_data["step"]


def _settings_to_config_dict(settings: Settings) -> Dict[str, Any]:
    """Convert settings to serializable dictionary.

    Extracts only the fields needed for inference and model reconstruction.

    Args:
        settings: Full settings object

    Returns:
        Dictionary with model architecture and preprocessing config
    """
    # Convert to dict
    config = asdict(settings)
    return config


def config_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Extract configuration from a checkpoint without loading full model.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Configuration dictionary
    """
    checkpoint_data = load_checkpoint(checkpoint_path)
    return checkpoint_data.get("config", {})
