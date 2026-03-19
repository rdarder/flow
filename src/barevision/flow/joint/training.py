"""Training script for joint embeddings + flow estimation model."""

import time
from functools import partial
from pathlib import Path

import optax
import tyro
from flax import nnx

from barevision.flow.joint.model import Model as OpticalFlowModel
from barevision.flow.joint.losses import compute_loss
from barevision.flow.embeddings.model import count_parameters
from barevision.flow.logging_utils import (
    log_progress,
    print_footer,
    print_header,
    should_log_something,
)
from barevision.flow.settings import (
    ModelSettings,
    Settings,
)
from barevision.flow.dataset.video_dataset import create_dataloader
from barevision.flow.joint.visualization import log_visualizations
from barevision.flow.checkpoint_utils import (
    generate_run_name,
    restore_model_from_checkpoint,
    save_checkpoint,
    save_best_checkpoint,
)
from barevision.utils.logging import JaxLogger


@partial(
    nnx.jit,
    static_argnames=("return_aux", "model_settings"),
)
def train_step(
    model: OpticalFlowModel,
    optimizer,
    img1,
    img2,
    model_settings: ModelSettings,
    return_aux: bool = False,
):
    """Execute single training step with gradient update.

    Args:
        model: OpticalFlowModel (combines embeddings + flow estimation)
        optimizer: Single optimizer for all model parameters
        img1: Frame 1 (B, H, W, 3)
        img2: Frame 2 (B, H, W, 3)
        model_settings: Model configuration
        return_aux: If True, return comprehensive auxiliary data

    Returns:
        Tuple of (loss, aux_dict)
    """

    def loss_fn(model):
        # Forward pass (uses flow_temperature for attention)
        flows, pyramid1, pyramid2 = model(
            img1, img2, temperature=model_settings.flow_temperature
        )

        # Compute loss (uses entropy_temperature for entropy loss)
        loss, loss_aux = compute_loss(
            pyramid1,
            pyramid2,
            flows=flows,
            window_size=model_settings.window_size,
            lambda_entropy=model_settings.lambda_entropy,
            level_weight_decay=model_settings.level_weight_decay,
            recon_weight=model_settings.recon_weight,
            entropy_temperature=model_settings.entropy_temperature,
        )

        # Build aux structure
        aux = {}
        if return_aux:
            aux = {
                "model": {
                    "pyramid1": pyramid1,
                    "pyramid2": pyramid2,
                    "flows": flows,
                },
                "loss": loss_aux,
            }

        return loss, aux

    # Compute loss and gradients
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    # Update model
    optimizer.update(model, grads)

    return loss, aux


@partial(
    nnx.jit,
    static_argnames=("model_settings",),
)
def validation_step(
    model: OpticalFlowModel,
    img1,
    img2,
    model_settings: ModelSettings,
):
    """Execute single validation step (no gradients).

    Args:
        model: OpticalFlowModel (combines embeddings + flow estimation)
        img1: Frame 1 (B, H, W, 3)
        img2: Frame 2 (B, H, W, 3)
        model_settings: Model configuration

    Returns:
        Validation loss (scalar)
    """
    # Forward pass (uses flow_temperature for attention)
    flows, pyramid1, pyramid2 = model(
        img1, img2, temperature=model_settings.flow_temperature
    )

    # Compute loss (uses entropy_temperature for entropy loss)
    loss, _ = compute_loss(
        pyramid1,
        pyramid2,
        flows=flows,
        window_size=model_settings.window_size,
        lambda_entropy=model_settings.lambda_entropy,
        level_weight_decay=model_settings.level_weight_decay,
        recon_weight=model_settings.recon_weight,
        entropy_temperature=model_settings.entropy_temperature,
    )

    return loss


def run_epoch(
    epoch,
    global_step,
    model: OpticalFlowModel,
    optimizer,
    logger,
    settings: Settings,
    run_name: str,
):
    # Compute deterministic seed for this epoch
    epoch_seed = settings.training.seed + epoch

    loader = create_dataloader(
        settings.dataset,
        split="train",
        shuffle=True,
        random_seed=epoch_seed,
        augmentation_settings=settings.augmentation,
    )
    epoch_start = time.time()

    for step, (img1, img2, metadata) in enumerate(loader):
        # Determine if we should return aux data (for logging or visualization)
        should_return_aux = should_log_something(settings.logging, global_step)

        loss, aux = train_step(
            model,
            optimizer,
            img1,
            img2,
            settings.model,
            return_aux=should_return_aux,
        )

        if global_step % settings.logging.log_every_steps == 0:
            log_progress(
                logger,
                model,
                img1,
                epoch,
                global_step,
                loss,
                aux,
                epoch_start,
                settings.model.window_size,
            )

        if global_step % settings.logging.log_visualizations_every_steps == 0:
            # Get pyramids and flows from aux_data
            pyramid1 = aux["model"]["pyramid1"]
            pyramid2 = aux["model"]["pyramid2"]
            flows = aux["model"].get("flows", None)

            # Pass original images for visualization (not embeddings)
            log_visualizations(
                logger,
                img1[0:1],  # Original RGB frame 1
                img2[0:1],  # Original RGB frame 2
                pyramid1,
                pyramid2,
                flows,
                aux_data=aux,
                metadata=metadata[0],
                step=global_step,
                window_size=settings.model.window_size,
                num_levels=settings.model.num_levels,
            )

        # Checkpoint periodically (skip step 0)
        if settings.checkpoint.every_steps > 0:
            if global_step > 0 and global_step % settings.checkpoint.every_steps == 0:
                checkpoint_path = save_checkpoint(
                    model=model,
                    step=global_step,
                    settings=settings,
                    run_name=run_name,
                )
                print(f"Checkpoint saved at step {global_step}: {checkpoint_path}")

        global_step += 1

    return global_step


def run_validation(
    model: OpticalFlowModel,
    settings: Settings,
) -> float:
    """Run validation on validation dataset.

    Args:
        model: OpticalFlowModel in evaluation mode
        settings: Full settings configuration

    Returns:
        Average validation loss
    """
    # Create validation dataloader (no shuffle, deterministic, no augmentation)
    loader = create_dataloader(
        settings.dataset,
        split="val",
        shuffle=False,
        random_seed=settings.training.seed,
        augmentation_settings=None,  # No augmentation for validation
    )

    total_loss = 0.0
    num_batches = 0

    for img1, img2, _ in loader:
        loss = validation_step(
            model,
            img1,
            img2,
            settings.model,
        )
        total_loss += float(loss)
        num_batches += 1

    if num_batches == 0:
        return 0.0

    avg_loss = total_loss / num_batches
    return avg_loss


def train(settings: Settings):
    """Main joint training loop."""
    print_header(settings)

    # Generate run name for consistent logging and checkpointing
    run_name = generate_run_name(prefix=settings.logging.run_name_prefix)

    logger = JaxLogger(
        log_dir=settings.logging.log_dir,
        run_name=run_name,
        strict=True,
    )

    # Create optical flow model (combines embeddings + flow estimation)
    # Architecture constants:
    # - hidden_dim=32: intermediate feature dimension in convolution blocks
    # - num_groups=8: grouped convolution groups (32/8 = 4 channels per group)
    # - rngs initialized from training.seed for reproducibility across runs
    model = OpticalFlowModel(
        hidden_dim=32,
        embed_dim=settings.model.embed_dim,
        num_groups=8,
        num_levels=settings.model.num_levels,
        flow_hidden_dim=settings.model.flow_hidden_dim,
        window_size=settings.model.window_size,
        rngs=nnx.Rngs(settings.training.seed),
    )

    # Single optimizer for all parameters
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(settings.training.learning_rate),
        ),
        wrt=nnx.Param,
    )

    # Resume from checkpoint if requested
    if settings.checkpoint.resume_from:
        resume_path = Path(settings.checkpoint.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        print(f"\nResuming from checkpoint: {resume_path}")
        loaded_step = restore_model_from_checkpoint(resume_path, model)
        print(f"Restored model from step {loaded_step}")

        # Note: This is a simplified resume - we don't restore optimizer state
        # For production use, you'd want to save/restore optimizer state too
        global_step = loaded_step
        print(f"Continuing training from step {global_step}\n")
    else:
        global_step = 0

    # Count parameters
    total_params = count_parameters(model)
    embed_params = count_parameters(model.embedding_model)
    flow_params = count_parameters(model.flow_estimator)
    print(f"Embedding model parameters: {embed_params}")
    print(f"Flow estimator parameters: {flow_params}")
    print(f"Total parameters: {total_params}\n")

    best_val_loss = float("inf")
    best_val_step = 0

    for epoch in range(settings.training.epochs):
        global_step = run_epoch(
            epoch,
            global_step,
            model,
            optimizer,
            logger,
            settings,
            run_name,
        )

        # Run validation at end of each epoch
        if settings.validation.every_epochs > 0:
            if (epoch + 1) % settings.validation.every_epochs == 0:
                print(f"\nRunning validation at epoch {epoch + 1}...")
                val_loss = run_validation(model, settings)
                print(f"Validation loss: {val_loss:.6f}")

                # Log validation loss to TensorBoard
                logger.log_scalar("Loss/validation", val_loss, step=global_step)

                # Save best model if validation improved
                if settings.validation.save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = global_step
                    checkpoint_path = save_best_checkpoint(
                        model=model,
                        step=global_step,
                        val_loss=val_loss,
                        settings=settings,
                        run_name=run_name,
                    )
                    print(
                        f"New best model saved at step {global_step} "
                        f"(val_loss: {val_loss:.6f}): {checkpoint_path}"
                    )

    # Save final checkpoint
    if settings.checkpoint.save_final:
        checkpoint_path = save_checkpoint(
            model=model,
            step=global_step,
            settings=settings,
            run_name=run_name,
            save_final=True,
        )
        print(f"Final checkpoint saved: {checkpoint_path}")

    # Print validation summary
    if settings.validation.every_epochs > 0 and best_val_loss < float("inf"):
        print(f"\n{'=' * 60}")
        print(f"Validation Summary:")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Achieved at step: {best_val_step}")
        print(f"{'=' * 60}")

    logger.close()
    print_footer()
    return model


if __name__ == "__main__":
    tyro.cli(train)
