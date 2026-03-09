"""Checkpoint management with Orbax for Flow model training.

Provides a unified interface for checkpointing with Null Object pattern support.
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from .hierarchical_model import HierarchicalFlowModel


class AbstractCheckpointManager(ABC):
    """Abstract base class for checkpoint managers.

    Provides a unified interface with Null Object pattern support.
    """

    @abstractmethod
    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step."""
        pass

    @abstractmethod
    def save(
        self,
        step: int,
        model: HierarchicalFlowModel,
        optimizer: nnx.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save a checkpoint."""
        pass

    @abstractmethod
    def restore(
        self,
        model: HierarchicalFlowModel,
        optimizer: nnx.Optimizer,
        step: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Restore from checkpoint. Returns (epoch, global_step)."""
        pass

    @abstractmethod
    def latest_step(self) -> Optional[int]:
        """Get the latest checkpoint step."""
        pass

    @abstractmethod
    def close(self):
        """Close the checkpoint manager."""
        pass


class NullCheckpointManager(AbstractCheckpointManager):
    """Null implementation - does nothing when checkpointing is disabled."""

    def should_save(self, step: int) -> bool:
        return False

    def save(
        self,
        step: int,
        model: HierarchicalFlowModel,
        optimizer: nnx.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return False

    def restore(
        self,
        model: HierarchicalFlowModel,
        optimizer: nnx.Optimizer,
        step: Optional[int] = None,
    ) -> Tuple[int, int]:
        return 0, 0

    def latest_step(self) -> Optional[int]:
        return None

    def close(self):
        pass


class OrbaxCheckpointManager(AbstractCheckpointManager):
    """Orbax-based checkpoint manager with should_save() support."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_interval_steps: int = 1000,
        max_to_keep: int = 3,
    ):
        """Initialize Orbax checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval_steps: Save checkpoint every N steps
            max_to_keep: Maximum number of checkpoints to keep
        """
        # Use absolute path for Orbax
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Store save interval for manual checking (Orbax's should_save doesn't work as expected)
        self.save_interval_steps = save_interval_steps
        self._last_saved_step = -1

        # Configure Orbax CheckpointManager options
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
        )

        # Initialize Orbax CheckpointManager
        self._manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=options,
        )

        # Track current epoch for metadata
        self._current_epoch = 0

    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step.

        Implements proper interval checking:
        - Never save at step 0 (nothing to save yet)
        - Save when step > 0 and step % save_interval_steps == 0
        - Don't save if we've already saved at this step
        """
        # Never save at step 0
        if step == 0:
            return False

        # Don't save if we've already saved at this step
        if step == self._last_saved_step:
            return False

        # Save at interval boundaries
        return step % self.save_interval_steps == 0

    def save(
        self,
        step: int,
        model: HierarchicalFlowModel,
        optimizer: nnx.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save a checkpoint.

        Args:
            step: Current global step
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch number
            metrics: Optional metrics to save

        Returns:
            True if save was successful
        """
        self._current_epoch = epoch

        # Prepare checkpoint data
        checkpoint = {
            "model": nnx.state(model),
            "optimizer_state": optimizer.opt_state,
            "optimizer_step": optimizer.step,
            "epoch": epoch,
        }

        # Save using Orbax
        try:
            self._manager.save(
                step,
                args=ocp.args.StandardSave(checkpoint),
            )
            self._last_saved_step = step
            print(f"Checkpoint saved at step {step}")
            return True
        except Exception as e:
            print(f"Warning: Failed to save checkpoint at step {step}: {e}")
            return False

    def restore(
        self,
        model: HierarchicalFlowModel,
        optimizer: nnx.Optimizer,
        step: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Restore from checkpoint.

        Args:
            model: Model to restore state into
            optimizer: Optimizer to restore state into
            step: Step to restore from (None = latest)

        Returns:
            Tuple of (epoch, global_step)
        """
        if step is None:
            step = self._manager.latest_step()

        if step is None:
            raise ValueError("No checkpoint found to restore from")

        # Restore using Orbax
        checkpoint = self._manager.restore(step, args=ocp.args.StandardRestore())

        # Handle integer key conversion (Orbax/MsgPack limitation)
        model_state_dict = self._fix_state_keys(checkpoint["model"])

        # Restore model state
        current_state = nnx.state(model)
        self._update_state_from_dict(current_state, model_state_dict)
        nnx.update(model, current_state)

        # Restore optimizer state
        optimizer.opt_state = checkpoint["optimizer_state"]  # type: ignore[attr-defined]
        optimizer.step = checkpoint["optimizer_step"]  # type: ignore[attr-defined]

        epoch = int(checkpoint.get("epoch", 0))
        global_step = step

        print(f"Checkpoint restored from step {step}")
        print(f"  Resuming from epoch {epoch}, step {global_step}")

        return epoch, global_step

    def latest_step(self) -> Optional[int]:
        """Get the latest checkpoint step."""
        return self._manager.latest_step()

    def close(self):
        """Close the checkpoint manager."""
        self._manager.close()

    def _fix_state_keys(self, state):
        """Recursively convert string keys back to integers."""
        if isinstance(state, dict):
            new_state = {}
            for key, value in state.items():
                if isinstance(key, str) and key.isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_state[new_key] = self._fix_state_keys(value)
            return new_state
        elif isinstance(state, (list, tuple)):
            return type(state)(self._fix_state_keys(item) for item in state)
        else:
            return state

    def _update_state_from_dict(self, state_obj, state_dict):
        """Recursively update State object with values from loaded dict."""
        if isinstance(state_obj, nnx.State):
            for key in state_dict:
                if key in state_obj:
                    self._update_state_from_dict(state_obj[key], state_dict[key])
        elif (
            isinstance(state_dict, dict)
            and hasattr(state_obj, "__getitem__")
            and not hasattr(state_obj, "value")
        ):
            for key in state_dict:
                if key in state_obj:
                    self._update_state_from_dict(state_obj[key], state_dict[key])
        elif hasattr(state_obj, "value"):
            if isinstance(state_dict, dict) and "value" in state_dict:
                state_obj.value = state_dict["value"]
            else:
                state_obj.value = state_dict


def create_checkpoint_manager(
    checkpoint_dir: str,
    save_interval_steps: int = 1000,
    max_to_keep: int = 3,
    enabled: bool = True,
) -> AbstractCheckpointManager:
    """Factory function to create appropriate checkpoint manager.

    Args:
        checkpoint_dir: Directory for checkpoints
        save_interval_steps: Save every N steps
        max_to_keep: Max checkpoints to keep
        enabled: If False, returns NullCheckpointManager

    Returns:
        Checkpoint manager instance
    """
    if not enabled or save_interval_steps <= 0:
        return NullCheckpointManager()

    return OrbaxCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_interval_steps=save_interval_steps,
        max_to_keep=max_to_keep,
    )
