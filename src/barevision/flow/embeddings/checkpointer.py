"""Checkpoint management for embeddings training.

Encapsulates checkpoint saving logic with Orbax backend.
"""

import shutil
from dataclasses import asdict
from pathlib import Path

import orbax.checkpoint as ocp
from flax import nnx

from barevision.flow.checkpoint_utils import get_checkpoint_path
from barevision.flow.settings import EmbeddingsSettings
from barevision.utils.console import ConsoleLogger


class Checkpointer:
    """Manages checkpoint saving for embeddings training.

    Handles both periodic step checkpoints and best model checkpoints
    based on validation loss. All conditional logic is encapsulated here.

    Attributes:
        logger: Console logger for messages
        best_val_loss: Best validation loss seen so far
        best_step: Global step where best validation was achieved
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

        checkpoint_path = get_checkpoint_path(
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
