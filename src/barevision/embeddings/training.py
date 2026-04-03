"""Training script for standalone embeddings model with spatial variance loss.

This script trains the hierarchical embedding model independently from flow estimation,
using spatial variance loss to encourage spatially concentrated attention patterns.

Entry point: python -m barevision.embeddings.training
"""

import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.embeddings.checkpointer import (
    CheckpointManagerWrapper,
    CheckpointSettings,
)
from barevision.dataset.video import create_dataloader
from barevision.embeddings.model import (
    count_parameters,
    HierarchicalEmbeddingModel,
)
from barevision.embeddings.spatial_losses import HierarchicalSpatialVarianceLoss
from barevision.embeddings.visualization import log_visualizations
from barevision.embeddings.logging_utils import (
    should_log_something,
)
from barevision.embeddings.settings import Settings
from barevision.utils.console import ConsoleLogger
from barevision.utils.logging import TensorboardLogger


def loss_fn(model, img_pair, loss_fn_obj, need_aux: bool):
    """Compute spatial variance loss for embeddings training.

    Args:
        model: HierarchicalEmbeddingModel
        img_pair: Tuple of (img1, img2)
        loss_fn_obj: HierarchicalSpatialVarianceLoss instance
        need_aux: Whether to return auxiliary data

    Returns:
        Tuple of (loss, aux_dict)
    """
    pyramid1 = model(img_pair[0])
    pyramid2 = model(img_pair[1])
    total_loss, aux = loss_fn_obj((pyramid1, pyramid2), need_aux=need_aux)

    # Add pyramids to aux for visualization when needed
    if need_aux:
        aux["pyramids"] = (pyramid1, pyramid2)

    return total_loss, aux


@partial(nnx.jit, static_argnames=("loss_fn_obj", "return_aux"))
def _compute_loss_and_grads(
    model: HierarchicalEmbeddingModel,
    loss_fn_obj: HierarchicalSpatialVarianceLoss,
    optimizer: nnx.Optimizer,
    img_pair: tuple[jnp.ndarray, jnp.ndarray],
    return_aux: bool,
):
    """Single training step with gradient computation and update.

    Args:
        model: Embedding model
        loss_fn_obj: Loss function instance
        optimizer: NNX optimizer
        img_pair: Tuple of (img1, img2)
        return_aux: Whether to return auxiliary data

    Returns:
        Tuple of (loss, aux_dict)
    """
    loss_derivative = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = loss_derivative(model, img_pair, loss_fn_obj, return_aux)
    optimizer.update(model, grads)
    if not return_aux:
        return total_loss, {}
    aux["img_pair"] = img_pair
    return total_loss, aux


@partial(nnx.jit, static_argnames="loss_fn_obj")
def _validation_step(
    model: HierarchicalEmbeddingModel,
    loss_fn_obj: HierarchicalSpatialVarianceLoss,
    img_pair: tuple[jnp.ndarray, jnp.ndarray],
):
    """Validation step without gradient computation.

    Args:
        model: Embedding model
        loss_fn_obj: Loss function instance
        img_pair: Tuple of (img1, img2)

    Returns:
        Scalar loss value
    """
    total_loss, _ = loss_fn(model, img_pair, loss_fn_obj, need_aux=False)
    return total_loss


class EmbeddingsTrainer:
    """Trainer for standalone embeddings model.

    Trains hierarchical embedding model with spatial variance loss.
    Checkpointing uses training loss for preservation decisions.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.run_name = CheckpointManagerWrapper.generate_run_name(
            prefix=settings.logging.run_name_prefix
        )
        self.logger = ConsoleLogger()
        self.tensorboard = TensorboardLogger(
            log_dir=settings.logging.tensorboard_dir,
            run_name=self.run_name,
            strict=True,
        )

        rngs = nnx.Rngs(settings.training.seed)
        embeddings_model = HierarchicalEmbeddingModel(settings.model, rngs=rngs)
        self.loss_fn_obj = HierarchicalSpatialVarianceLoss(
            settings.loss.spatial_variance
        )

        # Optimizer for embeddings model only
        self.optimizer = nnx.Optimizer(
            embeddings_model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(settings.training.learning_rate),
            ),
            wrt=nnx.Param,
        )
        self.model = embeddings_model
        self.checkpointer = CheckpointManagerWrapper(
            settings.checkpoint, self.run_name, self.logger
        )

    def _maybe_restore_from_checkpoint(self) -> int:
        """Load model parameters from checkpoint depending on settings.

        Returns the global step, either the restored one or 1.
        """
        if not self.settings.checkpoint.resume_from:
            return 1

        resume_path = Path(self.settings.checkpoint.resume_from)
        with self.logger.task(f"Resuming from checkpoint: {resume_path}"):
            return CheckpointManagerWrapper.restore(resume_path, self.model)

    def _report_model_params(self):
        """Report model parameter counts."""
        total_params = count_parameters(self.model)
        self.logger.log(f"Embedding model parameters: {total_params}\n")

    def _should_validate(self, epoch: int) -> bool:
        """Check if validation should run this epoch."""
        return (
            self.settings.validation.every_epochs > 0
            and (epoch % self.settings.validation.every_epochs) == 0
        )

    def _maybe_run_validation(self, epoch: int, global_step: int):
        """Run validation if due and log results.

        Validation is for monitoring only - checkpointing uses training loss.

        Args:
            epoch: Current epoch number
            global_step: Current global step
        """
        if not self._should_validate(epoch):
            return

        self.logger.log(f"\nRunning validation at epoch {epoch}...")
        val_loss = self._run_validation()
        self.logger.log(f"Validation loss: {val_loss:.6f}")
        self.tensorboard.log_scalar("Loss/validation", val_loss, step=global_step)
        # Note: No checkpoint save here - checkpoints use training loss

    def __call__(self):
        """Main embeddings training loop."""
        self._log_header()
        self.logger.log("")

        # Resume from checkpoint if requested
        global_step = self._maybe_restore_from_checkpoint()
        self._report_model_params()

        for epoch in range(1, self.settings.training.epochs + 1):
            global_step = self._train_epoch(epoch, global_step)
            self._maybe_run_validation(epoch, global_step)

        self.checkpointer.close()
        self.tensorboard.close()
        self.logger.log("=" * 60)
        self.logger.log("TRAINING COMPLETE")
        self.logger.log("=" * 60)

    def _train_epoch(self, epoch: int, global_step: int):
        """Train for one epoch.

        Args:
            epoch: Current epoch number
            global_step: Current global step

        Returns:
            Updated global step
        """
        epoch_seed = self.settings.training.seed + epoch

        loader = create_dataloader(
            self.settings.dataset,
            split="train",
            shuffle=True,
            random_seed=epoch_seed,
        )
        epoch_start = time.time()

        for step, (img1, img2, metadata) in enumerate(loader):
            train_loss = self._train_step_and_maybe_log(
                epoch=epoch,
                step=step,
                global_step=global_step,
                img1=img1,
                img2=img2,
                metadata=metadata,
                epoch_start=epoch_start,
            )
            self.checkpointer.save_step(
                model=self.model,
                step=global_step,
                train_loss=train_loss,
            )
            global_step += 1

        return global_step

    def _run_validation(self) -> float:
        """Run validation on validation dataset.

        Returns:
            Average validation loss
        """
        loader = create_dataloader(
            self.settings.dataset,
            split="val",
            shuffle=False,
            random_seed=self.settings.training.seed,
        )

        total_loss = 0.0
        num_batches = 0

        for img1, img2, _ in loader:
            loss = _validation_step(
                self.model,
                self.loss_fn_obj,
                (img1, img2),
            )
            total_loss += float(loss)
            num_batches += 1

        if num_batches == 0:
            return 0.0

        avg_loss = total_loss / num_batches
        return avg_loss

    def _train_step_and_maybe_log(
        self,
        epoch: int,
        step: int,
        global_step: int,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        metadata: list,
        epoch_start: float,
    ) -> float:
        """Execute training step and optionally log/visualize.

        Encapsulates the need_aux decision and all logging logic.

        Args:
            epoch: Current epoch number
            step: Step within epoch
            global_step: Global step number
            img1: First frame
            img2: Second frame
            metadata: Batch metadata
            epoch_start: Epoch start time for speed calculation

        Returns:
            Loss value
        """
        # Determine if we should compute aux data for logging/visualization
        need_aux = should_log_something(self.settings.logging, global_step)

        # Execute training step
        loss, aux = _compute_loss_and_grads(
            self.model, self.loss_fn_obj, self.optimizer, (img1, img2), need_aux
        )

        # Log progress if due
        if global_step % self.settings.logging.every_steps == 0:
            self._log_progress(epoch, global_step, loss, aux, epoch_start, img1)

        # Log visualizations if due (requires aux data)
        if (
            global_step % self.settings.logging.visualizations_every_steps == 0
            and need_aux
        ):
            self._log_visualizations(img1, img2, aux, metadata[0], global_step, epoch)

        return loss

    def _log_progress(
        self,
        epoch: int,
        global_step: int,
        loss: float,
        aux: dict,
        epoch_start: float,
        img1: jnp.ndarray,
    ):
        """Log training progress metrics.

        Args:
            epoch: Current epoch number
            global_step: Global step number
            loss: Combined loss value
            aux: Auxiliary data from training step
            epoch_start: Epoch start time
            img1: Input frame for diagnostics
        """
        # Log metrics
        self.tensorboard.log_scalar("Loss/total", float(loss), global_step)
        self.tensorboard.log_scalar(
            "Loss/spatial_variance/self", float(aux["self_loss"]), global_step
        )
        self.tensorboard.log_scalar(
            "Loss/spatial_variance/cross", float(aux["cross_loss"]), global_step
        )

        # Log embedding statistics
        from barevision.embeddings.logging_utils import log_diagnostics

        log_diagnostics(
            self.tensorboard,
            self.model,
            img1,
            global_step,
            self.settings.loss.spatial_variance.window_size,
        )

        # Print progress line
        steps_per_sec = global_step / (time.time() - epoch_start)
        self_var = float(aux["self_loss"])
        cross_var = float(aux["cross_loss"])
        self.logger.log(
            f"Epoch {epoch} | Step {global_step} | Loss: {float(loss):.4f} "
            f"(self: {self_var:.2f} | cross: {cross_var:.2f}) | "
            f"{steps_per_sec:.1f} steps/sec"
        )

    def _log_visualizations(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        aux: dict,
        metadata: dict,
        global_step: int,
        epoch: int,
    ):
        """Log training visualizations.

        Args:
            img1: First frame
            img2: Second frame
            aux: Auxiliary data from training step
            metadata: Batch metadata
            global_step: Global step number
            epoch: Current epoch number
        """
        pyramid1, pyramid2 = aux["pyramids"]

        log_visualizations(
            self.tensorboard,
            img1[0:1],
            img2[0:1],
            pyramid1,
            pyramid2,
            aux_data=aux,
            metadata=metadata,
            step=global_step,
            window_size=self.settings.loss.spatial_variance.window_size,
            num_levels=self.settings.model.num_levels,
        )

    def _log_header(self):
        """Log training configuration header."""
        from barevision.utils import image

        self.logger.log("=" * 60)
        self.logger.log("EMBEDDINGS TRAINING (Spatial Variance Loss)")
        self.logger.log("=" * 60)
        self.logger.log("")
        self.logger.log(f"Pyramid levels: {self.settings.model.num_levels}")
        self.logger.log(f"Embedding dim: {self.settings.model.embed_dim}")
        self.logger.log(
            f"Window size: {self.settings.loss.spatial_variance.window_size}×{self.settings.loss.spatial_variance.window_size}"
        )

        image_size = image.image_size(
            self.settings.dataset.coarse_grid_size,
            self.settings.dataset.window_size,
            self.settings.dataset.num_levels,
        )
        self.logger.log(f"Image size: {image_size}")
        self.logger.log("")
        self.logger.log(f"Epochs: {self.settings.training.epochs}")
        self.logger.log(f"Batch size: {self.settings.dataset.batch_size}")
        if self.settings.dataset.max_samples > 0:
            self.logger.log(
                f"Max samples per epoch: {self.settings.dataset.max_samples}"
            )
        self.logger.log("")
        self.logger.log(f"Loss: Spatial Variance")
        self.logger.log(
            f"  - Lambda self: {self.settings.loss.spatial_variance.lambda_self}"
        )
        self.logger.log(
            f"  - Self temperature: {self.settings.loss.spatial_variance.self_temperature}"
        )
        self.logger.log(
            f"  - Cross temperature: {self.settings.loss.spatial_variance.cross_temperature}"
        )
        self.logger.log(
            f"  - Level weight decay: {self.settings.loss.spatial_variance.level_weight_decay}"
        )
        self.logger.log("")


if __name__ == "__main__":
    parsed_settings = tyro.cli(Settings)
    EmbeddingsTrainer(parsed_settings)()
