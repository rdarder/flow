"""Training script for embedding model."""

import time
from functools import partial

import optax
import tyro
from flax import nnx

from barevision.embeddings.loss import compute_embedding_losses
from barevision.embeddings.logging_utils import log_progress
from barevision.embeddings.model import SimpleEmbeddingModel, count_parameters
from barevision.embeddings.settings import Settings, create_smoke_test_settings
from barevision.embeddings.video_dataset import create_dataloader
from barevision.embeddings.visualization import log_visualizations
from barevision.utils.logging import JaxLogger


@partial(nnx.jit, static_argnames=("logging"))
def train_step(model, optimizer, img1, img2, logging: bool = False):
    """Execute single training step with gradient update."""

    def loss_fn(model):
        emb1 = model(img1)
        emb2 = model(img2)
        combined, self_loss, cross_loss = compute_embedding_losses(emb1, emb2)
        return combined.mean(), (self_loss.mean(), cross_loss.mean())

    (loss, (self_loss, cross_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model
    )
    aux = {}
    if logging:
        aux["self_loss"] = self_loss
        aux["cross_loss"] = cross_loss
    optimizer.update(model, grads)
    return loss, aux


def _run_epoch(
    epoch,
    model,
    optimizer,
    logger,
    dataset_settings,
    logging_settings,
):
    """Run single epoch and return average loss."""
    loader = create_dataloader(dataset_settings, split="train")

    epoch_start = time.time()
    epoch_losses = []

    for step, (img1, img2, metadata) in enumerate(loader):
        global_step = step

        loss, aux = train_step(
            model,
            optimizer,
            img1,
            img2,
            logging_settings.should_log_something(global_step),
        )
        epoch_losses.append(float(loss))

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
            )

        if global_step % logging_settings.log_visualizations_every_steps == 0:
            log_visualizations(logger, model, img1[0:1], img1[0:1], {}, global_step)

    return sum(epoch_losses) / len(epoch_losses)


def train(settings: Settings):
    """Main training loop."""
    if settings.smoke_test:
        settings = create_smoke_test_settings()

    _print_header(settings)

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
        avg_loss = _run_epoch(
            epoch,
            model,
            optimizer,
            logger,
            settings.dataset,
            settings.logging,
        )
        print(f"Epoch {epoch} complete | Avg loss: {avg_loss:.4f}\n")

    logger.close()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model


def _print_header(settings):
    """Print training configuration header."""
    print("=" * 60)
    print("EMBEDDING TRAINING")
    print("=" * 60)
    print()
    print(f"Epochs: {settings.training.epochs}")
    print(f"Batch size: {settings.dataset.batch_size}")
    if settings.dataset.max_samples > 0:
        print(f"Max samples per epoch: {settings.dataset.max_samples}")
    print()


def main():
    """Entry point."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
