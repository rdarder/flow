"""Training script for joint embeddings + flow estimation model."""

import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.flow.checkpoint_utils import (
    generate_run_name,
    restore_model_from_checkpoint,
    save_checkpoint,
    save_best_checkpoint,
)
from barevision.flow.dataset.video import create_dataloader
from barevision.flow.embeddings.losses import HierarchicalEmbeddingLoss
from barevision.flow.embeddings.model import (
    count_parameters,
    HierarchicalEmbeddingModel,
)
from barevision.flow.joint import JointEmbeddingFlowModel
from barevision.flow.joint.losses import (
    JointEmbeddingReconstructionLoss,
)
from barevision.flow.joint.visualization import log_visualizations
from barevision.flow.logging_utils import (
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


def loss_fn(img_pair, model, loss):
    embeddings_pair, flows = model(img_pair)
    total_loss, aux = loss(embeddings_pair, flows)
    return total_loss, aux


@partial(nnx.jit, static_argnames=("loss", "return_aux"))
def train_step(
    model: JointEmbeddingFlowModel,
    loss: JointEmbeddingReconstructionLoss,
    optimizer: nnx.Optimizer,
    img_pair: tuple[jnp.ndarray, jnp.ndarray],
    return_aux: bool,
):
    loss_derivative = nnx.value_and_grad(loss_fn, has_aux=True)
    (total_loss, aux), grads = loss_derivative(img_pair, model, loss)
    optimizer.update(grads)
    if not return_aux:
        return loss, {}
    aux["img_pair"] = img_pair
    return loss, aux


@partial(nnx.jit)
def validation_step(
    model: JointEmbeddingFlowModel,
    loss: JointEmbeddingReconstructionLoss,
    img_pair: tuple[jnp.ndarray, jnp.ndarray],
):
    total_loss, _ = loss_fn(img_pair, model, loss)
    return total_loss


class Trainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.run_name = generate_run_name(prefix=settings.logging.run_name_prefix)
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
        embeddings_loss = HierarchicalEmbeddingLoss(self.settings.loss.embedding)
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

    def _maybe_restore_from_checkpoint(self) -> int:
        """Load model parameters from checkpoint depending on settings.

        Returns the global step, either the restored one or 1.
        """
        if not self.settings.checkpoint.resume_from:
            return 1

        resume_path = Path(self.settings.checkpoint.resume_from)
        with self.logger.task(f"Resuming from checkpoint: {resume_path}"):
            return restore_model_from_checkpoint(resume_path, self.model)

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

        best_val_loss = float("inf")
        best_val_step = 0

        for epoch in range(1, settings.training.epochs + 1):
            global_step = self._train_epoch(epoch, global_step)
            if self._should_validate(epoch):
                print(f"\nRunning validation at epoch {epoch}...")
                val_loss = self._run_validation()
                print(f"Validation loss: {val_loss:.6f}")
                self.tensorboard.log_scalar(
                    "Loss/validation", val_loss, step=global_step
                )
                if settings.validation.save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = global_step
                    checkpoint_path = save_best_checkpoint(
                        model=self.model,
                        step=global_step,
                        val_loss=val_loss,
                        settings=settings,
                        run_name=self.run_name,
                    )
                    print(
                        f"New best model saved at step {global_step} "
                        f"(val_loss: {val_loss:.6f}): {checkpoint_path}"
                    )

        if settings.validation.every_epochs > 0 and best_val_loss < float("inf"):
            print(f"\n{'=' * 60}")
            print(f"Validation Summary:")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            print(f"  Achieved at step: {best_val_step}")
            print(f"{'=' * 60}")

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
                pyramid1 = aux["model"]["pyramid1"]
                pyramid2 = aux["model"]["pyramid2"]
                flows = aux["model"].get("flows", None)

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
            if self.settings.checkpoint.every_steps > 0:
                if (
                    global_step > 0
                    and global_step % self.settings.checkpoint.every_steps == 0
                ):
                    checkpoint_path = save_checkpoint(
                        model=self.model,
                        step=global_step,
                        settings=self.settings,
                        run_name=self.run_name,
                    )
                    print(f"Checkpoint saved at step {global_step}: {checkpoint_path}")

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
                img1,
                img2,
            )
            total_loss += float(loss)
            num_batches += 1

        if num_batches == 0:
            return 0.0

        avg_loss = total_loss / num_batches
        return avg_loss


if __name__ == "__main__":
    trainer = tyro.cli(Trainer)
    trainer()
