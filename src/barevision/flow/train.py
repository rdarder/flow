"""Training script for optical flow model."""

import time
from functools import partial

import optax
import tyro
from flax import nnx

from barevision.flow.loss import compute_combined_loss
from barevision.flow.logging_utils import log_progress, print_footer, print_header
from barevision.flow.optical_flow_model import OpticalFlowModel
from barevision.flow.model import count_parameters
from barevision.flow.reconstruction_loss import warp_embeddings
from barevision.flow.settings import (
    ModelSettings,
    Settings,
    create_smoke_test_settings,
)
from barevision.flow.video_dataset import create_dataloader
from barevision.flow.visualization import log_visualizations
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

    Uses combined entropy + reconstruction loss across all pyramid levels.

    Args:
        model: OpticalFlowModel (combines embeddings + flow estimation)
        optimizer: Single optimizer for all model parameters
        model_settings: Model configuration (window_size, temperatures, loss weights, etc.)
        return_aux: If True, return comprehensive auxiliary data for debugging/visualization.
                   When False, XLA eliminates aux computation as dead code.
    """
    # Extract values from settings group
    window_size = model_settings.window_size
    level_weight_decay = model_settings.level_weight_decay
    lambda_entropy = model_settings.lambda_entropy
    lambda_recon = model_settings.lambda_recon
    temperature = model_settings.temperature

    def loss_fn(model):
        # Get embeddings and flow in single forward pass
        flow, pyramid1, pyramid2 = model(img1, img2, temperature=temperature)

        # Get coarsest level embeddings for reconstruction
        emb1_coarse = pyramid1[-1]
        emb2_coarse = pyramid2[-1]

        # Warp Frame 1 embeddings using predicted flow
        warped = warp_embeddings(emb1_coarse, flow)

        # Compute combined loss
        loss, loss_aux = compute_combined_loss(
            pyramid1,
            pyramid2,
            warped_embeddings=warped,
            target_embeddings=emb2_coarse,
            window_size=window_size,
            lambda_entropy=lambda_entropy,
            level_weight_decay=level_weight_decay,
            lambda_recon=lambda_recon,
            temperature=temperature,
            return_attention_weights=return_aux,
        )

        # Build aux structure
        aux = {}
        if return_aux:
            aux = {
                "model": {
                    "pyramid1": pyramid1,
                    "pyramid2": pyramid2,
                },
                "loss": loss_aux,
            }

        return loss, aux

    # Get gradients
    (loss, aux), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(model)

    # Update model (single optimizer for all parameters)
    optimizer.update(model, grads)

    return loss, aux


def run_epoch(
    epoch,
    global_step,
    model: OpticalFlowModel,
    optimizer,
    logger,
    dataset_settings,
    model_settings,
    logging_settings,
):
    # Compute deterministic seed for this epoch
    epoch_seed = dataset_settings.seed + epoch

    loader = create_dataloader(
        dataset_settings,
        split="train",
        shuffle=True,
        random_seed=epoch_seed,
    )
    epoch_start = time.time()

    for step, (img1, img2, metadata) in enumerate(loader):
        # Determine if we should return aux data (for logging or visualization)
        should_return_aux = logging_settings.should_log_something(global_step)
        
        loss, aux = train_step(
            model,
            optimizer,
            img1,
            img2,
            model_settings,
            return_aux=should_return_aux,
        )

        if global_step % logging_settings.log_every_steps == 0:
            log_progress(
                logger,
                model,
                img1,
                epoch,
                global_step,
                loss,
                aux,
                epoch_start,
                model_settings.window_size,
            )

        if global_step % logging_settings.log_visualizations_every_steps == 0:
            log_visualizations(
                logger,
                model,
                img1[0:1],
                img2[0:1],
                metadata[0],  # Take first batch element
                global_step,
                model_settings.window_size,
                model_settings.num_levels,
                aux_data=aux,
            )

        global_step += 1

    return global_step


def train(settings: Settings):
    """Main training loop."""
    if settings.smoke_test:
        settings = create_smoke_test_settings()

    print_header(settings)

    logger = JaxLogger(
        log_dir=settings.logging.log_dir,
        run_name_prefix=settings.logging.run_name_prefix,
    )

    # Create optical flow model (combines embeddings + flow estimation)
    model = OpticalFlowModel(
        embed_dim=settings.model.embed_dim,
        num_levels=settings.model.num_levels,
        flow_hidden_dim=settings.model.flow_hidden_dim,
        window_size=settings.model.window_size,
        rngs=nnx.Rngs(0),
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
    for epoch in range(settings.training.epochs):
        global_step = run_epoch(
            epoch,
            global_step,
            model,
            optimizer,
            logger,
            settings.dataset,
            settings.model,
            settings.logging,
        )

    logger.close()
    print_footer()
    return model


def main():
    """Entry point."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
