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
from barevision.embeddings.settings import Settings, create_smoke_test_settings
from barevision.embeddings.video_dataset import create_dataloader
from barevision.embeddings.visualization import log_visualizations
from barevision.utils.logging import JaxLogger


def train_step(graphdef, state, tx, opt_state, img1, img2):
    """Execute single training step with gradient update."""
    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        emb1 = model(img1)
        emb2 = model(img2)
        combined, self_loss, cross_loss = compute_embedding_losses(emb1, emb2)
        return combined.mean(), self_loss.mean(), cross_loss.mean()

    combined, self_loss, cross_loss = loss_fn(state)
    grads = nnx.grad(lambda s: loss_fn(s)[0])(state)
    updates, opt_state = tx.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)

    return state, opt_state, float(combined), float(self_loss), float(cross_loss)


def _run_epoch(epoch, graphdef, state, tx, opt_state, loader, logger, log_every_steps, log_viz_every_steps):
    """Run single epoch and return average loss."""
    epoch_start = time.time()
    epoch_losses = []

    for step, (img1, img2, metadata) in enumerate(loader):
        global_step = step  # Simple step counting within epoch

        # Training step
        state, opt_state, loss, self_loss, cross_loss = train_step(
            graphdef, state, tx, opt_state, img1, img2
        )
        epoch_losses.append(loss)

        # Log metrics
        logger.log_scalar("Loss/train_step", loss, global_step)
        logger.log_scalar("Loss/self_entropy", self_loss, global_step)
        logger.log_scalar("Loss/cross_entropy", cross_loss, global_step)

        # Periodic logging
        if global_step % log_every_steps == 0:
            _log_diagnostics(logger, graphdef, state, img1, metadata[0] if metadata else {}, global_step, log_viz_every_steps)

            elapsed = time.time() - epoch_start
            steps_per_sec = (step + 1) / elapsed
            print(f"Epoch {epoch} | Step {global_step} | Loss: {loss:.4f} | {steps_per_sec:.1f} steps/sec")

    return sum(epoch_losses) / len(epoch_losses)


def _log_diagnostics(logger, graphdef, state, img1, metadata, step, log_viz_every_steps):
    """Log gradient statistics, embeddings, and visualizations."""
    temp_model = nnx.merge(graphdef, state)

    # Gradient statistics
    log_gradient_statistics(logger, None, temp_model, step)

    # Embedding statistics
    embeddings = temp_model(img1)
    log_embedding_statistics(logger, embeddings, step)
    log_attention_statistics(logger, embeddings, step)

    # Visualizations
    if log_viz_every_steps > 0 and step % log_viz_every_steps == 0:
        log_visualizations(logger, temp_model, img1[None], img1[None], metadata, step)


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

    graphdef, state = nnx.split(model)
    tx = optax.adam(settings.training.learning_rate)
    opt_state = tx.init(state)

    print(f"Model parameters: {count_parameters(model)}\n")

    for epoch in range(settings.training.epochs):
        loader = create_dataloader(
            split="train",
            batch_size=settings.dataset.batch_size,
            img_size=settings.dataset.img_size,
            steps_per_epoch=settings.training.steps_per_epoch,
        )
        avg_loss = _run_epoch(
            epoch, 
            graphdef, state, tx, opt_state, 
            loader, 
            logger,
            settings.logging.log_every_steps,
            settings.logging.log_visualizations_every_steps,
        )
        print(f"Epoch {epoch} complete | Avg loss: {avg_loss:.4f}\n")

    logger.close()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return graphdef, state


def _print_header(settings):
    """Print training configuration header."""
    print("=" * 60)
    print("EMBEDDING TRAINING")
    print("=" * 60)
    print()
    print(f"Epochs: {settings.training.epochs}")
    print(f"Batch size: {settings.dataset.batch_size}")
    if settings.training.steps_per_epoch > 0:
        print(f"Steps per epoch: {settings.training.steps_per_epoch}")
    print()


def main():
    """Entry point."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
