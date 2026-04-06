import datetime
import time
from functools import partial

import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.embeddings.checkpointer import (
    CheckpointManagerWrapper,
)
from barevision.dataset.video import create_dataloader
from barevision.embeddings.model import (
    count_parameters,
    HierarchicalEmbeddingModel,
)
from barevision.embeddings.spatial_losses import HierarchicalSpatialVarianceLoss
from barevision.embeddings.visualization import log_visualizations
from barevision.embeddings.settings import Settings
from barevision.utils.console import ConsoleLogger
from barevision.utils.logging import TensorboardLogger
from barevision.embeddings.logging_utils import log_diagnostics
from barevision.utils import image


class EmbeddingsTrainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.run_name = self._generate_run_name(prefix=settings.run_name_prefix)
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

        self.optimizer = nnx.Optimizer(
            embeddings_model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(settings.training.learning_rate),
            ),
            wrt=nnx.Param,
        )
        self.model = embeddings_model
        self.checkpointer = CheckpointManagerWrapper(
            settings.checkpoint, self.run_name, self.logger
        )

    @staticmethod
    def _generate_run_name(prefix: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def _report_model_params(self):
        total_params = count_parameters(self.model)
        self.logger.log(f"Embedding model parameters: {total_params}\n")

    def _should_validate(self, epoch: int) -> bool:
        return (
            self.settings.validation.every_epochs > 0
            and (epoch % self.settings.validation.every_epochs) == 0
        )

    def _maybe_validate_and_checkpoint(self, epoch: int, global_step: int):
        should_validate = (
            self.settings.validation.every_epochs > 0
            and epoch % self.settings.validation.every_epochs == 0
        )
        should_checkpoint = (
            self.settings.checkpoint.every_epochs > 0
            and epoch % self.settings.checkpoint.every_epochs == 0
        )

        if should_checkpoint:
            self.logger.log(f"\nRunning validation at epoch {epoch}...")
            # we always run a validation before checkpointing so we know the validation loss.
            val_loss = self._run_validation(global_step)
            self.checkpointer.save_epoch(self.model, epoch, val_loss)
        elif should_validate:
            # if we needed to both checkpoint and validate this was dealt with on the checkpoint case.
            self.logger.log(f"\nRunning validation at epoch {epoch}...")
            self._run_validation(global_step)

    def __call__(self):
        self._log_header()
        self.logger.log("")

        self.checkpointer.maybe_restore(self.model)
        self._report_model_params()

        global_step = 0
        for epoch in range(1, self.settings.training.epochs + 1):
            global_step = self._train_epoch(epoch, global_step)
            self._maybe_validate_and_checkpoint(epoch, global_step)

        self.checkpointer.close()
        self.tensorboard.close()
        self.logger.log("=" * 60)
        self.logger.log("TRAINING COMPLETE")
        self.logger.log("=" * 60)

    def _train_epoch(self, epoch: int, global_step: int):
        epoch_seed = self.settings.training.seed + epoch

        loader = create_dataloader(
            self.settings.dataset,
            split="train",
            shuffle=True,
            random_seed=epoch_seed,
        )
        epoch_start = time.time()

        for batch_idx, (img1, img2, metadata) in enumerate(loader):
            global_step += 1
            step_in_epoch = batch_idx + 1  # 1-indexed
            self._train_step_and_maybe_log(
                epoch=epoch,
                global_step=global_step,
                step_in_epoch=step_in_epoch,
                img1=img1,
                img2=img2,
                metadata=metadata,
                epoch_start=epoch_start,
            )

        return global_step

    def _run_validation(self, step: int) -> float:
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

        val_loss = total_loss / num_batches

        self.logger.log(f"Validation loss: {val_loss:.6f}")
        self.tensorboard.log_scalar("Loss/validation", val_loss, step=step)

        return val_loss

    def _train_step_and_maybe_log(
        self,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        metadata: list,
        epoch_start: float,
    ):
        should_log = self._log_this_step(global_step)
        should_visualize = self._should_log_visualizations(global_step)

        loss, aux = _compute_loss_and_grads(
            self.model,
            self.loss_fn_obj,
            self.optimizer,
            (img1, img2),
            return_aux=should_log or should_visualize,
        )

        if should_log:
            self._log_progress(epoch, global_step, step_in_epoch, loss, aux, epoch_start, img1)
        if should_visualize:
            self._log_visualizations(img1, img2, aux, metadata[0], global_step)

    def _log_this_step(self, step: int):
        return (
            step % self.settings.logging.visualizations_every_steps == 0
            or self._should_log_visualizations(step)
        )

    def _should_log_visualizations(self, step: int):
        return step % self.settings.logging.every_steps == 0

    def _log_progress(
        self,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        loss: float,
        aux: dict,
        epoch_start: float,
        img1: jnp.ndarray,
    ):
        self.tensorboard.log_scalar("Loss/total", float(loss), global_step)
        self.tensorboard.log_scalar(
            "Loss/spatial_variance/self", float(aux["self_loss"]), global_step
        )
        self.tensorboard.log_scalar(
            "Loss/spatial_variance/cross", float(aux["cross_loss"]), global_step
        )

        log_diagnostics(
            self.tensorboard,
            self.model,
            img1,
            global_step,
            self.settings.loss.spatial_variance.window_size,
        )

        steps_per_sec = step_in_epoch / (time.time() - epoch_start)
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
    ):
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

        self.logger.log("=" * 60)
        self.logger.log("EMBEDDINGS TRAINING (Spatial Variance Loss)")
        self.logger.log("=" * 60)
        self.logger.log("")
        self.logger.log(f"Pyramid levels: {self.settings.model.num_levels}")
        self.logger.log(f"Embedding dim: {self.settings.model.embed_dim}")
        self.logger.log(f"Hidden dim: {self.settings.model.hidden_dim}")
        self.logger.log(
            f"Window size: {self.settings.loss.spatial_variance.window_size}×{self.settings.loss.spatial_variance.window_size}"
        )
        self.logger.log("")
        self.logger.log("Normalization configuration:")
        self.logger.log(f"  - GroupNorm: {self.settings.model.use_group_norm}")
        self.logger.log(f"  - Mean subtraction: {self.settings.model.use_mean_subtraction}")
        self.logger.log(f"  - L2 norm: {self.settings.model.use_l2_norm}")

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


if __name__ == "__main__":
    parsed_settings = tyro.cli(Settings)
    EmbeddingsTrainer(parsed_settings)()
