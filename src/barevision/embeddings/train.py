"""Minimal training script for embedding model.

Bare-bones training loop with just enough to verify the pipeline works.
Uses tyro for CLI configuration.

Run:
    python -m barevision.embeddings.train
    python -m barevision.embeddings.train --training.epochs 5 --dataset.batch-size 8
    python -m barevision.embeddings.train --smoke-test
"""

import time

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import (
    self_attention_entropy_loss,
    cross_attention_entropy_loss,
    combined_loss,
)
from barevision.embeddings.video_dataset import VideoFrameDataset
from barevision.embeddings.settings import (
    Settings,
    DatasetSettings,
    TrainingSettings,
    create_smoke_test_settings,
)


def create_dataloader(split: str, batch_size: int, max_frames: int | None = None):
    """Simple data loader that yields batches.

    Args:
        split: 'train' or 'val'
        batch_size: Number of samples per batch
        max_frames: Maximum number of frames to load (for smoke tests)

    Yields:
        Tuple of (img1_batch, img2_batch) each of shape (B, H, W, 3)
    """
    dataset = VideoFrameDataset(
        split=split,
        max_frame_distance=5,
        img_size=(190, 190),
    )

    # For smoke tests, limit dataset size
    if max_frames is not None:
        indices = list(range(min(max_frames, len(dataset))))
    else:
        indices = list(range(len(dataset)))

    # Shuffle training data
    if split == "train":
        import random
        random.shuffle(indices)

    # Yield batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Skip incomplete batches

        imgs1 = []
        imgs2 = []

        for idx in batch_indices:
            img1, img2, _ = dataset[idx]
            imgs1.append(img1)
            imgs2.append(img2)

        img1_batch = jnp.stack(imgs1)
        img2_batch = jnp.stack(imgs2)

        yield img1_batch, img2_batch


def train_step(model, optimizer, img1, img2, alpha=1.0, beta=1.0):
    """Single training step.

    Args:
        model: Embedding model
        optimizer: NNX optimizer
        img1: Batch of frame 1 images (B, H, W, 3)
        img2: Batch of frame 2 images (B, H, W, 3)
        alpha: Weight for self-attention loss
        beta: Weight for cross-attention loss

    Returns:
        Loss value (scalar)
    """

    def loss_fn(m):
        emb1 = m(img1)
        emb2 = m(img2)

        self_loss = self_attention_entropy_loss(emb1)
        cross_loss = cross_attention_entropy_loss(emb1, emb2)
        total = combined_loss(self_loss, cross_loss, alpha=alpha, beta=beta)

        return total.mean()

    # Compute gradients
    grads = nnx.grad(loss_fn)(model)

    # Apply gradients
    nnx.update(optimizer, grads)

    # Return loss value
    loss = loss_fn(model)
    return loss


def train(settings: Settings):
    """Main training loop.

    Args:
        settings: Training configuration
    """
    if settings.training.smoke_test:
        settings = create_smoke_test_settings()

    print("=" * 60)
    print("EMBEDDING TRAINING (Minimal)")
    print("=" * 60)
    print()

    # Initialize model
    print("Initializing model...")
    model = SimpleEmbeddingModel(
        embed_dim=16,
        in_channels=3,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    optimizer = nnx.Optimizer(model, optax.adam(settings.training.learning_rate), wrt=nnx.Param)
    
    # Count parameters
    state = nnx.state(model)
    param_count = 0
    for module_state in state.values():
        for param_value in module_state.values():
            if hasattr(param_value, "size"):
                param_count += param_value.size
    print(f"Model parameters: {param_count}")
    print()

    # Training loop
    print(f"Training for {settings.training.epochs} epochs with batch_size={settings.dataset.batch_size}")
    if settings.training.steps_per_epoch > 0:
        print(f"Steps per epoch: {settings.training.steps_per_epoch}")
    print()

    for epoch in range(settings.training.epochs):
        epoch_start = time.time()
        epoch_losses = []

        # Create data loader
        loader = create_dataloader(
            split="train",
            batch_size=settings.dataset.batch_size,
            max_frames=settings.training.steps_per_epoch * settings.dataset.batch_size 
                       if settings.training.steps_per_epoch > 0 else None,
        )

        for step, (img1, img2) in enumerate(loader):
            if settings.training.steps_per_epoch > 0 and step >= settings.training.steps_per_epoch:
                break

            # Training step
            loss = train_step(model, optimizer, img1, img2)
            epoch_losses.append(float(loss))

            # Log every few steps
            if step % 5 == 0:
                elapsed = time.time() - epoch_start
                steps_per_sec = (step + 1) / elapsed
                print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss: {loss:.4f} | {steps_per_sec:.1f} steps/sec"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} complete | Avg loss: {avg_loss:.4f} | {epoch_time:.1f}s")
        print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return model, optimizer


def main():
    """Entry point with tyro CLI support."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
