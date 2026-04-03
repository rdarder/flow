"""Checkpoint utilities using Orbax CheckpointManager.

Provides a simple wrapper around CheckpointManager for embeddings training.
Uses training loss for both periodic saving and best checkpoint preservation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Optional

import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as cm
from flax import nnx
from orbax.checkpoint.args import StandardSave, StandardRestore

from orbax.checkpoint.checkpoint_managers import (
    PreservationPolicy,
    SaveDecisionPolicy,
    PolicyCheckpointInfo,
    DecisionContext,
    FixedIntervalPolicy,
)

from barevision.utils.checks import check_value
from barevision.utils.console import ConsoleLogger


@dataclass
class CheckpointSettings:
    """Checkpoint configuration for model persistence.

    Attributes:
        every_steps: Save checkpoint every N steps (0 to disable periodic checkpointing)
        location: Base directory for checkpoints (default "checkpoints")
        keep_best_n: Number of best checkpoints to keep based on training loss (0 to disable)
        resume_from: Path to checkpoint to resume training from (optional)
                     If provided, loads model weights and continues from saved step
    """

    every_steps: int = 100
    location: str = "checkpoints"
    keep_best_n: int = 3
    resume_from: Optional[Path] = None

    def __post_init__(self):
        check_value(
            self.every_steps >= 0,
            f"every_steps must be >= 0, got {self.every_steps}",
        )
        check_value(self.location, "location cannot be empty")
        check_value(
            self.keep_best_n > 0,
            f"keep_best_n must be > 0, got {self.keep_best_n}",
        )


class CheckpointManagerWrapper:
    def __init__(
        self,
        settings: CheckpointSettings,
        run_name: str,
        logger: ConsoleLogger,
    ):
        self.settings = settings
        self.run_name = run_name
        self.logger = logger

        # Create checkpoint directory
        self._checkpoint_dir = Path(settings.location).resolve() / run_name

        # Setup preservation policy: keep best N by training loss
        preservation_policy = KeepBestLossN(
            n=self.settings.keep_best_n,
            mode="min",  # Lower loss is better
        )

        save_policy = self.get_save_policy()

        # Create CheckpointManager
        self._manager = ocp.CheckpointManager(
            directory=self._checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                save_decision_policy=save_policy,
                preservation_policy=preservation_policy,
                create=True,
                enable_async_checkpointing=False,
            ),
        )

    def get_save_policy(self) -> SaveDecisionPolicy:
        if self.settings.every_steps > 0:
            return FixedIntervalPolicy(interval=self.settings.every_steps)
        return NeverSavePolicy()

    def save_step(
        self,
        model: nnx.Module,
        step: int,
        train_loss: float,
    ):
        """Save a checkpoint with training loss.

        The CheckpointManager decides internally whether to actually save
        based on the save policy. Training loss is used for preservation
        decisions (keeping the best N checkpoints).

        Args:
            model: Model to checkpoint
            step: Global step number (1-indexed)
            train_loss: Current training loss (used for preservation decisions)
        """
        graphdef, tree = nnx.split(model)
        saved = self._manager.save(
            step,
            args=StandardSave(tree),
            metrics={"train_loss": train_loss},
        )

        if saved:
            self.logger.log(f"Checkpoint saved at step {step} ")

    def _to_tree(
        self, model: nnx.Module, step: int = 1, epoch: int = 1, step_in_epoch: int = 1
    ):
        graphdef, model_state = nnx.split(model)
        data = {
            "model": model_state,
            "step": step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
        }

    def close(self):
        pass

    @staticmethod
    def generate_run_name(prefix: str = "embeddings") -> str:
        """Generate a unique run name with timestamp.

        Used consistently across logging and checkpointing.

        Args:
            prefix: Run name prefix (default "embeddings")

        Returns:
            Run name in format "{prefix}_{YYYYMMDD}_{HHMMSS}"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def maybe_restore(self, model: nnx.Module) -> int:
        """Restore model state from the latest checkpointed state..

        Args:
            model: Model instance to restore into (must have compatible structure)

        Returns:
            Global step from checkpoint
        """
        if self.settings.resume_from is None:
            return 1
        if not self.settings.resume_from.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.settings.resume_from}"
            )
        step = self._manager.latest_step()
        graphdef, tree = nnx.split(model)
        restored = self._manager.restore(step=None, args=StandardRestore(tree))
        nnx.update(model, restored)

        return step

    @staticmethod
    def load_metadata(checkpoint_path: Path) -> dict[str, Any]:
        """Extract metadata from a checkpoint without loading model state.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with step, epoch, step_in_epoch
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_path = Path(checkpoint_path).resolve()
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_data = checkpointer.restore(
            checkpoint_path, item=ocp.args.PyTreeRestore()
        )

        return {
            "step": checkpoint_data.get("step"),
            "epoch": checkpoint_data.get("epoch"),
            "step_in_epoch": checkpoint_data.get("step_in_epoch"),
        }


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

    # Score each checkpoint by loss
    def _get_score(self, ckpt: cm.PolicyCheckpointInfo) -> float:
        if ckpt.metrics is None or "train_loss" not in ckpt.metrics:
            # No loss = worst score (will be deleted first)
            return float("inf")
        return ckpt.metrics["train_loss"]


class NeverSavePolicy(SaveDecisionPolicy):
    """Save policy that never saves. Used when every_steps=0."""

    def should_save(
        self,
        step: PolicyCheckpointInfo,
        previous_steps: Sequence[PolicyCheckpointInfo],
        *,
        context: DecisionContext,
    ) -> bool:
        return False
