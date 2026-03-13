"""Training script for embedding model with flow estimation."""

import time
from functools import partial

import optax
import tyro
from flax import nnx

from barevision.flow.loss import compute_combined_loss
from barevision.flow.logging_utils import log_progress, print_footer, print_header
from barevision.flow.model import HierarchicalEmbeddingModel, count_parameters
from barevision.flow.flow_estimator import FlowEstimator
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
    model,
    flow_estimator,
    optimizer,
    flow_optimizer,
    img1,
    img2,
    model_settings: ModelSettings,
    return_aux: bool = False,
):
    """Execute single training step with gradient update.

    Uses combined entropy + reconstruction loss across all pyramid levels.

    Args:
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

    def loss_fn(model, flow_estimator):
        # Get pyramid from both frames
        pyramid1 = model(img1)
        pyramid2 = model(img2)

        # Compute flow at coarsest level
        flow = model.compute_flow(
            img1, img2, flow_estimator, temperature=temperature
        )

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

    # Get gradients for both model and flow_estimator
    (loss, aux), (model_grads, flow_grads) = nnx.value_and_grad(
        loss_fn, has_aux=True, argnums=(0, 1)
    )(model, flow_estimator)

    # Update both model and flow estimator
    optimizer.update(model, model_grads)
    flow_optimizer.update(flow_estimator, flow_grads)

    return loss, aux


def run_epoch(
    epoch,
    global_step,
    model,
    flow_estimator,
    optimizer,
    flow_optimizer,
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
            flow_estimator,
            optimizer,
            flow_optimizer,
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

    # Create embedding model
    model = HierarchicalEmbeddingModel(
        embed_dim=settings.model.embed_dim,
        in_channels=3,
        num_levels=settings.model.num_levels,
        rngs=nnx.Rngs(0),
    )

    # Create flow estimator
    flow_estimator = FlowEstimator(
        window_size=settings.model.window_size,
        hidden_dim=settings.model.flow_hidden_dim,
        rngs=nnx.Rngs(0),
    )

    # Create optimizers for both models
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(settings.training.learning_rate),
        ),
        wrt=nnx.Param,
    )

    flow_optimizer = nnx.Optimizer(
        flow_estimator,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(settings.training.learning_rate),
        ),
        wrt=nnx.Param,
    )

    # Count parameters
    model_params = count_parameters(model)
    flow_params = count_parameters(flow_estimator)
    print(f"Embedding model parameters: {model_params}")
    print(f"Flow estimator parameters: {flow_params}")
    print(f"Total parameters: {model_params + flow_params}\n")

    global_step = 0
    for epoch in range(settings.training.epochs):
        global_step = run_epoch(
            epoch,
            global_step,
            model,
            flow_estimator,
            optimizer,
            flow_optimizer,
            logger,
            settings.dataset,
            settings.model,
            settings.logging,
        )

    logger.close()
    print_footer()
    return model, flow_estimator


def main():
    """Entry point."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
