"""Training script for optical flow model."""

import time
from functools import partial

import optax
import tyro
from flax import nnx

from barevision.flow.training.model import Model as OpticalFlowModel
from barevision.flow.training.losses import compute_loss
from barevision.flow.embeddings.model import count_parameters
from barevision.flow.logging_utils import (
    log_progress,
    print_footer,
    print_header,
)
from barevision.flow.settings import (
    ModelSettings,
    Settings,
    create_smoke_test_settings,
)
from barevision.flow.dataset.video_dataset import create_dataloader
from barevision.flow.training.visualization import log_visualizations
from barevision.flow.checkpoint_utils import (
    generate_run_name,
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
        flow, pyramid1, pyramid2 = model(
            img1, img2, temperature=model_settings.flow_temperature
        )

        # Get coarsest level embeddings
        emb1_coarse = pyramid1[-1]
        emb2_coarse = pyramid2[-1]

        # Warp Frame 1 embeddings
        from barevision.flow.matching.losses import warp_embeddings

        warped = warp_embeddings(emb1_coarse, flow)

        # Compute loss (uses entropy_temperature for entropy loss)
        loss, loss_aux = compute_loss(
            pyramid1,
            pyramid2,
            warped_embeddings=warped,
            target_embeddings=emb2_coarse,
            window_size=model_settings.window_size,
            lambda_entropy=model_settings.lambda_entropy,
            level_weight_decay=model_settings.level_weight_decay,
            recon_weight=model_settings.recon_weight,
            entropy_temperature=model_settings.entropy_temperature,
            return_attention_weights=return_aux,
        )

        # Build aux structure
        aux = {}
        if return_aux:
            aux = {
                "model": {
                    "pyramid1": pyramid1,
                    "pyramid2": pyramid2,
                    "flow": flow,
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
    flow, pyramid1, pyramid2 = model(
        img1, img2, temperature=model_settings.flow_temperature
    )

    # Get coarsest level embeddings
    emb1_coarse = pyramid1[-1]
    emb2_coarse = pyramid2[-1]

    # Warp Frame 1 embeddings
    from barevision.flow.matching.losses import warp_embeddings

    warped = warp_embeddings(emb1_coarse, flow)

    # Compute loss (uses entropy_temperature for entropy loss)
    loss, _ = compute_loss(
        pyramid1,
        pyramid2,
        warped_embeddings=warped,
        target_embeddings=emb2_coarse,
        window_size=model_settings.window_size,
        lambda_entropy=model_settings.lambda_entropy,
        level_weight_decay=model_settings.level_weight_decay,
        recon_weight=model_settings.recon_weight,
        entropy_temperature=model_settings.entropy_temperature,
        return_attention_weights=False,
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
        should_return_aux = settings.logging.should_log_something(global_step)

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
            # Get pyramids and flow from aux_data
            pyramid1 = aux["model"]["pyramid1"]
            pyramid2 = aux["model"]["pyramid2"]
            flow = aux["model"].get("flow", None)

            # Pass original images for visualization (not embeddings)
            log_visualizations(
                logger,
                img1[0:1],  # Original RGB frame 1
                img2[0:1],  # Original RGB frame 2
                pyramid1,
                pyramid2,
                flow,
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
    """Main training loop."""
    is_smoke_test = settings.smoke_test
    if is_smoke_test:
        settings = create_smoke_test_settings()

    print_header(settings)

    # Generate run name for consistent logging and checkpointing
    run_name = generate_run_name(prefix=settings.logging.run_name_prefix)

    # Use strict mode for smoke tests to catch visualization errors
    logger = JaxLogger(
        log_dir=settings.logging.log_dir,
        run_name=run_name,
        strict=is_smoke_test,
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

    # Count parameters
    total_params = count_parameters(model)
    embed_params = count_parameters(model.embedding_model)
    flow_params = count_parameters(model.flow_estimator)
    print(f"Embedding model parameters: {embed_params}")
    print(f"Flow estimator parameters: {flow_params}")
    print(f"Total parameters: {total_params}\n")

    global_step = 0
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

    # Validate smoke test: ensure visualizations were logged
    if is_smoke_test:
        _validate_smoke_test_visualizations(logger.log_dir)

    print_footer()
    return model


def _validate_smoke_test_visualizations(log_dir: str):
    """Validate that expected visualizations were logged during smoke test.

    Raises RuntimeError if expected image tags are missing.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"images": 0})
        ea.Reload()

        image_tags = ea.Tags().get("images", [])

        # Expected visualization tags
        expected_tags = [
            "Flow/Predicted_Colorwheel",
            "Flow/Predicted_Arrows",
            "Level0/Frame_Grid",
            "Level0/Attention_Maps",
        ]

        missing_tags = [tag for tag in expected_tags if tag not in image_tags]

        if missing_tags:
            raise RuntimeError(
                f"Smoke test validation failed: Missing expected image tags: {missing_tags}. "
                f"Found tags: {image_tags}"
            )

        # Verify images were actually logged (not just tags)
        for tag in expected_tags:
            images = ea.Images(tag)
            if len(images) == 0:
                raise RuntimeError(
                    f"Smoke test validation failed: Tag '{tag}' has no images logged"
                )

        print(
            f"✓ Smoke test validation passed: {len(expected_tags)} image tags verified"
        )

    except ImportError:
        print("Warning: tensorboard not available for smoke test validation")
    except Exception as e:
        raise RuntimeError(f"Smoke test validation error: {e}") from e


def main():
    """Entry point."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
