"""Checkpoint utilities using Orbax CheckpointManager.

Provides a simple wrapper around CheckpointManager for embeddings training.
Uses training loss for both periodic saving and best checkpoint preservation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_managers as cm
from flax import nnx
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
    resume_from: str = ""

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
    """Wrapper around Orbax CheckpointManager for embeddings training.

    Uses training loss for checkpoint preservation decisions.
    All checkpoints are saved with training loss, and the best N are kept.

    Attributes:
        logger: Console logger for messages

    Example:
        ```python
        checkpointer = CheckpointManagerWrapper(checkpoint_settings, run_name, logger)
        for epoch, batch in enumerate(loader):
            loss = train_step()
            checkpointer.save_step(model, epoch, step, global_step, loss)
        ```
    """

    def __init__(
        self,
        checkpoint_settings: CheckpointSettings,
        run_name: str,
        logger: ConsoleLogger,
    ):
        """Initialize checkpointer with checkpoint configuration.

        Args:
            checkpoint_settings: Checkpoint configuration
            run_name: Run identifier for checkpoint directory naming
            logger: Console logger for messages
        """
        self.checkpoint_settings = checkpoint_settings
        self.run_name = run_name
        self.logger = logger

        # Create checkpoint directory
        self._checkpoint_dir = Path(checkpoint_settings.location).resolve() / run_name

        # Setup preservation policy: keep best N by training loss
        preservation_policy = KeepBestLossN(
            n=self.checkpoint_settings.keep_best_n,
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

        self._checkpoints_saved = 0

    def get_save_policy(self) -> SaveDecisionPolicy:
        if self.checkpoint_settings.every_steps > 0:
            return FixedIntervalPolicy(interval=self.checkpoint_settings.every_steps)
        return NeverSavePolicy()

    def save_step(
        self,
        model: nnx.Module,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        train_loss: float,
    ):
        """Save a checkpoint with training loss.

        The CheckpointManager decides internally whether to actually save
        based on the save policy. Training loss is used for preservation
        decisions (keeping the best N checkpoints).

        Args:
            model: Model to checkpoint
            epoch: Current epoch number (1-indexed)
            step_in_epoch: Step number within current epoch (1-indexed)
            global_step: Global step number (1-indexed)
            train_loss: Current training loss (used for preservation decisions)
        """
        data = {
            "model": nnx.to_pure_dict(nnx.state(model)),
            "step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
        }

        saved = self._manager.save(
            global_step,
            args=ocp.args.StandardSave(data),
            metrics={"train_loss": train_loss},
        )

        if saved:
            self._checkpoints_saved += 1
            self.logger.log(
                f"Checkpoint saved at step {global_step} "
                f"(train_loss: {train_loss:.6f}): {self._get_step_path(global_step)}"
            )

    def close(self):
        """Log training summary."""
        if self.checkpoint_settings.keep_best_n > 0:
            self.logger.log(f"\n{'=' * 60}")
            self.logger.log("Checkpoint Summary:")
            self.logger.log(
                f"  Keeping best {self.checkpoint_settings.keep_best_n} "
                f"checkpoints by training loss"
            )
            self.logger.log(f"  Total checkpoints saved: {self._checkpoints_saved}")
            self.logger.log(f"  Current checkpoints: {list(self._manager.all_steps())}")
            self.logger.log(f"{'=' * 60}")

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

    def _get_step_path(self, step: int) -> Path:
        """Get the path for a step checkpoint."""
        return self._checkpoint_dir / str(step)

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

        checkpoint_path = Path(checkpoint_path).resolve()
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint_data = checkpointer.restore(
            checkpoint_path, item=ocp.args.PyTreeRestore()
        )

        # Restore model state
        nnx.update(model, nnx.State(checkpoint_data["model"]))

        return checkpoint_data["step"]

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
