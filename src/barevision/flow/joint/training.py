"""Training script for joint embeddings + flow estimation model."""

import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.embeddings.checkpointer import Checkpointer
from barevision.dataset.video import create_dataloader
from barevision.embeddings import HierarchicalSpatialVarianceLoss
from barevision.embeddings.model import (
    count_parameters,
    HierarchicalEmbeddingModel,
)
from barevision.flow.joint import JointEmbeddingFlowModel
from barevision.flow.joint.losses import (
    JointEmbeddingReconstructionLoss,
)
from barevision.flow.joint.visualization import log_visualizations
from barevision.embeddings.logging_utils import (
    log_progress,
    print_footer,
    print_header,
    should_log_something,
)
from barevision.flow.matching import HierarchicalFlowEstimator
from barevision.flow.matching.losses import HierarchicalReconstructionLoss
from barevision.flow.settings import (
    Settings,
)
from barevision.utils.console import ConsoleLogger
from barevision.utils.logging import TensorboardLogger


def loss_fn(model, img_pair, loss, need_aux: bool):
    embeddings_pair, flows = model(img_pair)
    total_loss, aux = loss(embeddings_pair, flows, need_aux=need_aux)
    return total_loss, aux


@partial(nnx.jit, static_argnames=("loss_calculator", "return_aux"))
def train_step(
    model: JointEmbeddingFlowModel,
    loss_calculator: JointEmbeddingReconstructionLoss,
    optimizer: nnx.Optimizer,
    img_pair: tuple[jnp.ndarray, jnp.ndarray],
    return_aux: bool,
):
    loss_derivative = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = loss_derivative(
        model, img_pair, loss_calculator, return_aux
    )
    optimizer.update(model, grads)
    if not return_aux:
        return total_loss, {}
    aux["img_pair"] = img_pair
    return total_loss, aux


@partial(nnx.jit, static_argnames="loss_calculator")
def validation_step(
    model: JointEmbeddingFlowModel,
    loss_calculator: JointEmbeddingReconstructionLoss,
    img_pair: tuple[jnp.ndarray, jnp.ndarray],
):
    total_loss, _ = loss_fn(model, img_pair, loss_calculator, need_aux=False)
    return total_loss


class Trainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.run_name = Checkpointer.generate_run_name(
            prefix=settings.logging.run_name_prefix
        )
        self.logger = ConsoleLogger()
        self.tensorboard = TensorboardLogger(
            log_dir=settings.logging.tensorboard_dir,
            run_name=self.run_name,
            strict=True,
        )

        rngs = nnx.Rngs(settings.training.seed)
        embeddings_model = HierarchicalEmbeddingModel(
            settings.model.embedding, rngs=rngs
        )
        flow_model = HierarchicalFlowEstimator(settings.model.flow, rngs=rngs)
        joint_model = JointEmbeddingFlowModel(
            embeddings_model=embeddings_model,
            flow_estimator=flow_model,
            settings=settings.model.joint,
            rngs=rngs,
        )
        # Use spatial variance loss for embeddings (replaced entropy loss)
        from barevision.embeddings.settings import SpatialVarianceLossSettings

        embeddings_loss = HierarchicalSpatialVarianceLoss(
            SpatialVarianceLossSettings(
                window_size=settings.loss.embedding.window_size,
                level_weight_decay=settings.loss.embedding.level_weight_decay,
                lambda_self=1.0
                - settings.loss.embedding.lambda_entropy,  # Map lambda_entropy to lambda_self
                self_temperature=settings.loss.embedding.entropy_temperature,
                cross_temperature=settings.loss.embedding.entropy_temperature,
            )
        )
        reconstruction_loss = HierarchicalReconstructionLoss(self.settings.loss.flow)
        self.loss_calculator = JointEmbeddingReconstructionLoss(
            embeddings_loss, reconstruction_loss, self.settings.loss.joint
        )
        # Single optimizer for all parameters
        self.optimizer = nnx.Optimizer(
            joint_model,
            optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(settings.training.learning_rate),
            ),
            wrt=nnx.Param,
        )
        self.model = joint_model
        self.checkpointer = Checkpointer(
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
            return Checkpointer.restore(resume_path, self.model)

    def _report_model_params(self):
        # Count parameters
        total_params = count_parameters(self.model)
        embed_params = count_parameters(self.model.embedding_model)
        flow_params = count_parameters(self.model.flow_estimator)
        print(f"Embedding model parameters: {embed_params}")
        print(f"Flow estimator parameters: {flow_params}")
        print(f"Total parameters: {total_params}\n")

    def _should_validate(self, epoch: int) -> bool:
        return (
            self.settings.validation.every_epochs > 0
            and (epoch % self.settings.validation.every_epochs) == 0
        )

    def __call__(self):
        """Main joint training loop."""
        settings = self.settings
        print_header(self.settings)

        # Resume from checkpoint if requested
        global_step = self._maybe_restore_from_checkpoint()
        self._report_model_params()

        for epoch in range(1, settings.training.epochs + 1):
            global_step = self._train_epoch(epoch, global_step)
            if self._should_validate(epoch):
                print(f"\nRunning validation at epoch {epoch}...")
                val_loss = self._run_validation()
                print(f"Validation loss: {val_loss:.6f}")
                self.tensorboard.log_scalar(
                    "Loss/validation", val_loss, step=global_step
                )
                self.checkpointer.maybe_save_best(
                    model=self.model,
                    epoch=epoch,
                    global_step=global_step,
                    val_loss=val_loss,
                )

        self.checkpointer.close()
        self.tensorboard.close()
        print_footer()

    def _train_epoch(self, epoch: int, global_step: int):
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
                self.model, self.loss_calculator, self.optimizer, (img1, img2), need_aux
            )

            if global_step % self.settings.logging.every_steps == 0:
                log_progress(
                    self.tensorboard,
                    self.model,
                    img1,
                    epoch,
                    global_step,
                    loss,
                    aux,
                    epoch_start,
                    self.settings.model.flow.window_size,
                )

            if global_step % self.settings.logging.visualizations_every_steps == 0:
                # Get pyramids and flows from aux_data
                pyramid1, pyramid2 = aux["embeddings"]
                flows = aux.get("flows", None)

                # Pass original images for visualization (not embeddings)
                log_visualizations(
                    self.tensorboard,
                    img1[0:1],  # Original RGB frame 1
                    img2[0:1],  # Original RGB frame 2
                    pyramid1,
                    pyramid2,
                    flows,
                    aux_data=aux,
                    metadata=metadata[0],
                    step=global_step,
                    window_size=self.settings.model.flow.window_size,
                    # warning, embeddings loss/flow might have different window sizes!
                    num_levels=self.settings.model.embedding.num_levels,
                )

            # Checkpoint periodically (skip step 0)
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
                self.loss_calculator,
                (img1, img2),
            )
            total_loss += float(loss)
            num_batches += 1

        if num_batches == 0:
            return 0.0

        avg_loss = total_loss / num_batches
        return avg_loss


if __name__ == "__main__":
    settings = tyro.cli(Settings)
    Trainer(settings)()
