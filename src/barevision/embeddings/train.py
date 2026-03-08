"""Training script for embedding model."""

import time
from functools import partial

import optax
import tyro
from flax import nnx

from barevision.embeddings.loss import compute_embedding_losses
from barevision.embeddings.logging_utils import log_progress, print_footer, print_header
from barevision.embeddings.model import SimpleEmbeddingModel, count_parameters
from barevision.embeddings.settings import (
    ModelSettings,
    Settings,
    create_smoke_test_settings,
)
from barevision.embeddings.video_dataset import create_dataloader
from barevision.embeddings.visualization import log_visualizations
from barevision.utils.logging import JaxLogger


@partial(nnx.jit, static_argnames=("logging", "window_size"))
def train_step(
    model, optimizer, img1, img2, logging: bool = False, window_size: int = 16
):
    """Execute single training step with gradient update."""

    def loss_fn(model):
        return compute_embedding_losses(
            model(img1), model(img2), window_size=window_size
        )

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    if not logging:
        aux = {}  # drop aux so jit can trace it as not being used.
    optimizer.update(model, grads)
    return loss, aux


def run_epoch(
    epoch,
    model,
    optimizer,
    logger,
    dataset_settings,
    model_settings,
    logging_settings,
):
    loader = create_dataloader(dataset_settings, split="train")
    epoch_start = time.time()

    for step, (img1, img2, metadata) in enumerate(loader):
        global_step = step

        loss, aux = train_step(
            model,
            optimizer,
            img1,
            img2,
            logging_settings.should_log_something(global_step),
            model_settings.window_size,
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
                {},
                global_step,
                model_settings.window_size,
            )


def train(settings: Settings):
    """Main training loop."""
    if settings.smoke_test:
        settings = create_smoke_test_settings()

    print_header(settings)

    logger = JaxLogger(
        log_dir=settings.logging.log_dir,
        run_name_prefix=settings.logging.run_name_prefix,
    )

    model = SimpleEmbeddingModel(
        embed_dim=16,
        in_channels=3,
        rngs=nnx.Rngs(0),
    )

    optimizer = nnx.Optimizer(
        model, optax.adam(settings.training.learning_rate), wrt=nnx.Param
    )

    print(f"Model parameters: {count_parameters(model)}\n")

    for epoch in range(settings.training.epochs):
        run_epoch(
            epoch,
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
