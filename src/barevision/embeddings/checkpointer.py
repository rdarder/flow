"""Checkpoint utilities using Orbax CheckpointManager.

Provides a simple wrapper around CheckpointManager for embeddings training.
Uses training loss for both periodic saving and best checkpoint preservation.
"""

from pathlib import Path
from typing import Optional

import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as cm
from flax import nnx
from orbax.checkpoint.args import StandardSave, StandardRestore
from pydantic import BaseModel, ConfigDict, Field

from orbax.checkpoint.checkpoint_managers import (
    PreservationPolicy,
    PolicyCheckpointInfo,
    PreservationContext,
)

from barevision.utils.console import ConsoleLogger


class CheckpointConfig(BaseModel):
    """Checkpoint configuration for model persistence.

    Attributes:
        every_epochs: Save checkpoint every N epochs (0 to disable)
        location: Base directory for checkpoints (default "checkpoints")
        keep_best_n: Number of best checkpoints to keep based on validation loss (0 to disable)
        resume_from: Path to checkpoint to resume training from (optional)
                     If provided, loads model weights only (training always starts from epoch 1)
    """

    model_config = ConfigDict(frozen=True)

    every_epochs: int = Field(default=1, ge=0, description="Save checkpoint every N epochs (0 to disable)")
    location: str = Field(default="checkpoints", min_length=1, description="Base directory for checkpoints")
    keep_best_n: int = Field(default=3, ge=1, description="Number of best checkpoints to keep based on validation loss")
    resume_from: Optional[Path] = Field(default=None, description="Path to checkpoint to resume from")


class CheckpointManagerWrapper:
    def __init__(
        self,
        config: CheckpointConfig,
        run_name: str,
        logger: ConsoleLogger,
    ):
        self.config = config
        self.run_name = run_name
        self.logger = logger

        # Create checkpoint directory
        self._checkpoint_dir = Path(config.location).resolve() / run_name

        # Setup preservation policy: keep best N by validation loss
        preservation_policy = KeepBestLossN(
            n=config.keep_best_n,
            mode="min",  # Lower loss is better
        )

        # Create CheckpointManager
        self._manager = ocp.CheckpointManager(
            directory=self._checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                preservation_policy=preservation_policy,
                create=True,
                enable_async_checkpointing=False,
            ),
        )

    def save_epoch(
        self,
        model: nnx.Module,
        epoch: int,
        val_loss: float,
    ):
        """Save a checkpoint with validation loss.

        Validation loss is used for preservation decisions (keeping the best N checkpoints).

        Args:
            model: Model to checkpoint
            epoch: Epoch number (1-indexed)
            val_loss: Validation loss (used for preservation decisions)
        """
        graphdef, tree = nnx.split(model)
        self._manager.save(
            epoch,
            args=StandardSave(tree),
            metrics={"val_loss": val_loss},
        )
        self.logger.log(f"Checkpoint saved at epoch {epoch} (val_loss: {val_loss:.6f})")



    def close(self):
        pass

    def maybe_restore(self, model: nnx.Module):
        """Restore model state from the latest checkpoint.

        Args:
            model: Model instance to restore into (must have compatible structure)

        Note:
            Training always starts from epoch 1, step 1. Only model weights are restored.
        """
        if self.config.resume_from is None:
            return
        if not self.config.resume_from.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.config.resume_from}"
            )
        graphdef, tree = nnx.split(model)
        restored = self._manager.restore(step=None, args=StandardRestore(tree))
        nnx.update(model, restored)




class KeepBestLossN(PreservationPolicy):
    """Preservation policy that keeps the N checkpoints with best loss.

    Uses training loss from metrics to determine which checkpoints to keep.
    """

    def __init__(self, n: int, mode: str = "min"):
        self._n = n
        self._mode = mode

    def should_preserve(
        self,
        checkpoints: list[cm.PolicyCheckpointInfo],
        *,
        context: cm.PreservationContext,
    ) -> list[bool]:
        """Decide which checkpoints to preserve based on loss.

        Keeps the N checkpoints with the best (lowest) training loss.
        """
        # Keep all if we haven't exceeded the limit
        if len(checkpoints) <= self._n:
            return [True] * len(checkpoints)

        # Sort by score (best first) and keep top N
        scored = [(i, self._get_score(c)) for i, c in enumerate(checkpoints)]
        scored.sort(key=lambda x: x[1])
        keep_indices = set(idx for idx, _ in scored[: self._n])
        return [i in keep_indices for i in range(len(checkpoints))]

    # Score each checkpoint by validation loss
    def _get_score(self, ckpt: cm.PolicyCheckpointInfo) -> float:
        if ckpt.metrics is None or "val_loss" not in ckpt.metrics:
            # No loss = worst score (will be deleted first)
            return float("inf")
        return ckpt.metrics["val_loss"]



