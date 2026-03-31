"""Training script for standalone embeddings model with spatial variance loss.

This script trains the hierarchical embedding model independently from flow estimation,
using spatial variance loss to encourage spatially concentrated attention patterns.

Entry point: python -m barevision.flow.embeddings.training
"""

import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.flow.checkpointer import Checkpointer
from barevision.flow.dataset.video import create_dataloader
from barevision.flow.embeddings.model import (
    count_parameters,
    HierarchicalEmbeddingModel,
)
from barevision.flow.embeddings.spatial_losses import HierarchicalSpatialVarianceLoss
from barevision.flow.embeddings.visualization_train import log_visualizations
from barevision.flow.logging_utils import (
    log_progress,
    should_log_something,
)
from barevision.flow.settings import EmbeddingsSettings
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
def train_step(
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
def validation_step(
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
    """

    def __init__(self, settings: EmbeddingsSettings):
        self.settings = settings
        self.run_name = Checkpointer.generate_run_name(prefix=settings.logging.run_name_prefix)
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
        self.checkpointer = Checkpointer(settings, self.run_name, self.logger)

    def _maybe_restore_from_checkpoint(self) -> int:
        """Load model parameters from checkpoint depending on settings.

        Returns the global step, either the restored one or 1.
        """
        if not self.settings.checkpoint.resume_from:
            return 1

        resume_path = Path(self.settings.checkpoint.resume_from)
        with self.logger.task(f"Resuming from checkpoint: {resume_path}"):
            return Checkpointer.restore(resume_path, self.model)

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

        Args:
            epoch: Current epoch number
            global_step: Current global step
        """
        if not self._should_validate(epoch):
            return

        self.logger.log(f"\nRunning validation at epoch {epoch}...")
        val_loss = self._run_validation()
        self.logger.log(f"Validation loss: {val_loss:.6f}")
        self.tensorboard.log_scalar(
            "Loss/validation", val_loss, step=global_step
        )
        self.checkpointer.maybe_save_best(
            model=self.model,
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
        )

    def __call__(self):
        """Main embeddings training loop."""
        settings = self.settings
        print_header_embeddings(settings, self.logger)
        self.logger.log("")

        # Resume from checkpoint if requested
        global_step = self._maybe_restore_from_checkpoint()
        self._report_model_params()

        for epoch in range(1, settings.training.epochs + 1):
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
            # Determine if we should return aux data (for logging or visualization)
            need_aux = should_log_something(self.settings.logging, global_step)

            loss, aux = train_step(
                self.model, self.loss_fn_obj, self.optimizer, (img1, img2), need_aux
            )

            if global_step % self.settings.logging.every_steps == 0:
                log_progress_embeddings(
                    self.tensorboard,
                    self.logger,
                    self.model,
                    img1,
                    epoch,
                    global_step,
                    loss,
                    aux,
                    epoch_start,
                    self.settings.loss.spatial_variance.window_size,
                )

            if (
                global_step % self.settings.logging.visualizations_every_steps == 0
                and need_aux
            ):
                # Get pyramids from aux_data
                pyramid1, pyramid2 = aux["pyramids"]

                # Pass original images for visualization
                log_visualizations(
                    self.tensorboard,
                    img1[0:1],  # Original RGB frame 1
                    img2[0:1],  # Original RGB frame 2
                    pyramid1,
                    pyramid2,
                    aux_data=aux,
                    metadata=metadata[0],
                    step=global_step,
                    window_size=self.settings.loss.spatial_variance.window_size,
                    num_levels=self.settings.model.num_levels,
                )

            self.checkpointer.maybe_save_step(
                model=self.model,
                epoch=epoch,
                step_in_epoch=step + 1,
                global_step=global_step,
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
            loss = validation_step(
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


def print_header_embeddings(settings: EmbeddingsSettings, logger):
    """Print training header for embeddings training."""
    from barevision.utils import image

    logger.log("=" * 60)
    logger.log("EMBEDDINGS TRAINING (Spatial Variance Loss)")
    logger.log("=" * 60)
    logger.log("")
    logger.log(f"Pyramid levels: {settings.model.num_levels}")
    logger.log(f"Embedding dim: {settings.model.embed_dim}")
    logger.log(
        f"Window size: {settings.loss.spatial_variance.window_size}×{settings.loss.spatial_variance.window_size}"
    )

    image_size = image.image_size(
        settings.dataset.coarse_grid_size,
        settings.dataset.window_size,
        settings.dataset.num_levels,
    )
    logger.log(f"Image size: {image_size}")
    logger.log("")
    logger.log(f"Epochs: {settings.training.epochs}")
    logger.log(f"Batch size: {settings.dataset.batch_size}")
    if settings.dataset.max_samples > 0:
        logger.log(f"Max samples per epoch: {settings.dataset.max_samples}")
    logger.log("")
    logger.log(f"Loss: Spatial Variance")
    logger.log(f"  - Lambda self: {settings.loss.spatial_variance.lambda_self}")
    logger.log(f"  - Self temperature: {settings.loss.spatial_variance.self_temperature}")
    logger.log(f"  - Cross temperature: {settings.loss.spatial_variance.cross_temperature}")
    logger.log(
        f"  - Level weight decay: {settings.loss.spatial_variance.level_weight_decay}"
    )
    logger.log("")


def log_progress_embeddings(
    tensorboard: TensorboardLogger,
    console_logger,
    model,
    img1,
    epoch: int,
    step: int,
    loss,
    aux,
    epoch_start: float,
    window_size: int = 16,
):
    """Log progress for embeddings training.

    Args:
        tensorboard: TensorBoard logger
        console_logger: Console logger for progress output
        model: Embedding model
        img1: Input frame for diagnostics
        epoch: Current epoch number
        step: Current step within epoch
        loss: Combined loss value
        aux: Auxiliary loss information
        epoch_start: Time when epoch started (for speed calculation)
        window_size: Attention window size
    """
    # Log metrics
    tensorboard.log_scalar("Loss/total", float(loss), step)
    tensorboard.log_scalar("Loss/spatial_variance/self", float(aux["self_loss"]), step)
    tensorboard.log_scalar("Loss/spatial_variance/cross", float(aux["cross_loss"]), step)

    # Log embedding statistics
    from barevision.flow.logging_utils import log_diagnostics

    log_diagnostics(tensorboard, model, img1, step, window_size)

    # Print progress line
    steps_per_sec = (step + 1) / (time.time() - epoch_start)
    self_var = float(aux["self_loss"])
    cross_var = float(aux["cross_loss"])
    console_logger.log(
        f"Epoch {epoch} | Step {step} | Loss: {float(loss):.4f} "
        f"(self: {self_var:.2f} | cross: {cross_var:.2f}) | "
        f"{steps_per_sec:.1f} steps/sec"
    )


if __name__ == "__main__":
    settings = tyro.cli(EmbeddingsSettings)
    EmbeddingsTrainer(settings)()
