"""Unified checkpoint management for Barevision.

Provides both stateful checkpointing for training and stateless
loading for inference/resumption.
"""

import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import orbax.checkpoint as ocp
from flax import nnx

from barevision.flow.settings import EmbeddingsSettings
from barevision.utils.console import ConsoleLogger


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


class Checkpointer:
    """Manages checkpoint saving and loading for training and inference.

    Provides both stateful checkpointing for training (tracking best models,
    periodic saves) and stateless loading for inference and resumption.

    Attributes:
        logger: Console logger for messages
        best_val_loss: Best validation loss seen so far (training only)
        best_step: Global step where best validation was achieved (training only)

    Example (training):
        ```python
        checkpointer = Checkpointer(settings, run_name, logger)
        for epoch, batch in enumerate(loader):
            checkpointer.maybe_save_step(model, epoch, step, global_step)
        checkpointer.maybe_save_best(model, epoch, global_step, val_loss)
        ```

    Example (inference/resume):
        ```python
        config = Checkpointer.load_config(checkpoint_path)
        step = Checkpointer.restore(checkpoint_path, model)
        ```
    """

    def __init__(
        self,
        settings: EmbeddingsSettings,
        run_name: str,
        logger: ConsoleLogger,
    ):
        """Initialize checkpointer with constant configuration.

        Args:
            settings: Embeddings settings (constant for training session)
            run_name: Run identifier for checkpoint directory naming
            logger: Console logger for messages
        """
        self.settings = settings
        self.run_name = run_name
        self.logger = logger
        self.best_val_loss = float("inf")
        self.best_step = 0
        self._checkpoints_saved = 0
        self._orbax_checkpointer = ocp.PyTreeCheckpointer()

    def maybe_save_step(
        self,
        model: nnx.Module,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
    ):
        """Consider saving a periodic checkpoint.

        Saves if global_step is a multiple of checkpoint.every_steps.
        No-op if checkpointing is disabled (every_steps <= 0).

        Args:
            model: Model to checkpoint
            epoch: Current epoch number (1-indexed)
            step_in_epoch: Step number within current epoch (1-indexed)
            global_step: Global step number (1-indexed)
        """
        every_steps = self.settings.checkpoint.every_steps
        if every_steps <= 0:
            return

        if global_step % every_steps == 0:
            self._save_checkpoint(
                model=model,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                checkpoint_type="step",
            )

    def maybe_save_best(
        self,
        model: nnx.Module,
        epoch: int,
        global_step: int,
        val_loss: float,
    ):
        """Consider saving the best model checkpoint.

        Saves if val_loss is better than the previous best.
        No-op if validation.save_best is False.

        Args:
            model: Model to checkpoint
            epoch: Current epoch number (1-indexed)
            global_step: Global step number (1-indexed)
            val_loss: Current validation loss
        """
        if not self.settings.validation.save_best:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_step = global_step
            self._save_checkpoint(
                model=model,
                epoch=epoch,
                step_in_epoch=-1,  # Not applicable for best checkpoint
                global_step=global_step,
                checkpoint_type="best",
                val_loss=val_loss,
            )

    def _save_checkpoint(
        self,
        model: nnx.Module,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        checkpoint_type: str = "step",
        val_loss: float | None = None,
    ):
        """Save checkpoint using Orbax.

        Args:
            model: Model to save
            epoch: Current epoch number
            step_in_epoch: Step within epoch (-1 if not applicable)
            global_step: Global step number
            checkpoint_type: "step" or "best"
            val_loss: Validation loss (only for best checkpoints)
        """
        if checkpoint_type == "best":
            checkpoint_label = "best"
        else:
            checkpoint_label = global_step

        checkpoint_path = self.get_checkpoint_path(
            self.settings.checkpoint.location,
            self.run_name,
            checkpoint_label,
        )

        # Remove existing checkpoint if overwriting (for "best")
        if checkpoint_path.exists() and checkpoint_type == "best":
            shutil.rmtree(checkpoint_path)

        checkpoint_data = {
            "model": nnx.state(model).to_pure_dict(),
            "step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "config": asdict(self.settings),
        }

        if val_loss is not None:
            checkpoint_data["best_val_loss"] = val_loss

        self._orbax_checkpointer.save(
            checkpoint_path, ocp.args.PyTreeSave(checkpoint_data)
        )

        self._checkpoints_saved += 1

        if checkpoint_type == "best":
            self.logger.log(
                f"New best model saved at step {global_step} "
                f"(val_loss: {val_loss:.6f}): {checkpoint_path}"
            )
        else:
            self.logger.log(
                f"Checkpoint saved at step {global_step}: {checkpoint_path}"
            )

    def close(self):
        """Log training summary."""
        if self.settings.validation.save_best and self.best_val_loss < float("inf"):
            self.logger.log(f"\n{'=' * 60}")
            self.logger.log("Validation Summary:")
            self.logger.log(f"  Best validation loss: {self.best_val_loss:.6f}")
            self.logger.log(f"  Achieved at step: {self.best_step}")
            self.logger.log(f"  Total checkpoints saved: {self._checkpoints_saved}")
            self.logger.log(f"{'=' * 60}")

    @staticmethod
    def generate_run_name(prefix: str = "barevision") -> str:
        """Generate a unique run name with timestamp.

        Used consistently across logging and checkpointing.

        Args:
            prefix: Run name prefix (default "barevision")

        Returns:
            Run name in format "{prefix}_{YYYYMMDD}_{HHMMSS}"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    @staticmethod
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

    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with keys: model, step, config, and optional metadata
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(checkpoint_path, item=ocp.args.PyTreeRestore())

        # Convert string keys back to integers for NNX compatibility
        return _convert_string_keys_to_int(restored)

    @staticmethod
    def restore(checkpoint_path: Path, model: nnx.Module) -> int:
        """Restore model state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model instance to restore into (must have compatible structure)

        Returns:
            Global step from checkpoint
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = Checkpointer.load_checkpoint(checkpoint_path)

        # Restore model state
        nnx.update(model, nnx.State(checkpoint_data["model"]))

        return checkpoint_data["step"]

    @staticmethod
    def load_config(checkpoint_path: Path) -> Dict[str, Any]:
        """Extract configuration from a checkpoint without loading full model.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Configuration dictionary
        """
        checkpoint_data = Checkpointer.load_checkpoint(checkpoint_path)
        return checkpoint_data.get("config", {})

    @staticmethod
    def load_metadata(checkpoint_path: Path) -> Dict[str, Any]:
        """Extract all metadata from a checkpoint without loading model state.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with step, epoch, step_in_epoch, and any other metadata
        """
        checkpoint_data = Checkpointer.load_checkpoint(checkpoint_path)
        return {
            "step": checkpoint_data.get("step"),
            "epoch": checkpoint_data.get("epoch"),
            "step_in_epoch": checkpoint_data.get("step_in_epoch"),
            "best_val_loss": checkpoint_data.get("best_val_loss"),
        }
