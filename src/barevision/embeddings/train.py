"""Training script for embedding model."""

import time

import optax
import tyro
from flax import nnx

from barevision.embeddings.loss import compute_embedding_losses
from barevision.embeddings.logging_utils import (
    log_attention_statistics,
    log_embedding_statistics,
    log_gradient_statistics,
)
from barevision.embeddings.model import SimpleEmbeddingModel, count_parameters
from barevision.embeddings.settings import (
    DatasetSettings,
    LoggingSettings,
    Settings,
    TrainingSettings,
    create_smoke_test_settings,
)
from barevision.embeddings.video_dataset import create_dataloader
from barevision.embeddings.visualization import log_visualizations
from barevision.utils.logging import JaxLogger


def train_step(model, optimizer, img1, img2):
    """Execute single training step with gradient update."""
    def loss_fn(model):
        emb1 = model(img1)
        emb2 = model(img2)
        combined, self_loss, cross_loss = compute_embedding_losses(emb1, emb2)
        return combined.mean(), (self_loss.mean(), cross_loss.mean())

    (loss, (self_loss, cross_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss, self_loss, cross_loss


def _run_epoch(epoch, model, optimizer, logger, dataset_settings, training_settings, logging_settings):
    """Run single epoch and return average loss."""
    loader = create_dataloader(dataset_settings, split="train")

    epoch_start = time.time()
    epoch_losses = []

    for step, (img1, img2, metadata) in enumerate(loader):
        global_step = step

        loss, self_loss, cross_loss = train_step(model, optimizer, img1, img2)
        epoch_losses.append(float(loss))

        if global_step % logging_settings.log_every_steps == 0:
            logger.log_scalar("Loss/train_step", float(loss), global_step)
            logger.log_scalar("Loss/self_entropy", float(self_loss), global_step)
            logger.log_scalar("Loss/cross_entropy", float(cross_loss), global_step)

            _log_diagnostics(logger, model, img1, global_step)

            elapsed = time.time() - epoch_start
            steps_per_sec = (step + 1) / elapsed
            print(f"Epoch {epoch} | Step {global_step} | Loss: {float(loss):.4f} | {steps_per_sec:.1f} steps/sec")

        if logging_settings.log_visualizations_every_steps > 0 and global_step % logging_settings.log_visualizations_every_steps == 0:
            _log_visualizations(logger, model, img1, global_step)

    return sum(epoch_losses) / len(epoch_losses)


def _log_diagnostics(logger, model, img1, step):
    """Log gradient and embedding statistics."""
    log_gradient_statistics(logger, None, model, step)

    embeddings = model(img1)
    log_embedding_statistics(logger, embeddings, step)
    log_attention_statistics(logger, embeddings, step)


def _log_visualizations(logger, model, img1, step):
    """Generate and log visualization figures."""
    log_visualizations(logger, model, img1[0:1], img1[0:1], {}, step)


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

    optimizer = nnx.Optimizer(model, optax.adam(settings.training.learning_rate), wrt=nnx.Param)

    print(f"Model parameters: {count_parameters(model)}\n")

    for epoch in range(settings.training.epochs):
        avg_loss = _run_epoch(
            epoch,
            model,
            optimizer,
            logger,
            settings.dataset,
            settings.training,
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
