"""Minimal training script for embedding model.

Bare-bones training loop with just enough to verify the pipeline works.
Uses tyro for CLI configuration.

Run:
    python -m barevision.embeddings.train
    python -m barevision.embeddings.train --training.epochs 5 --dataset.batch-size 8
    python -m barevision.embeddings.train --training.smoke-test
"""

import random
import time
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import combined_loss
from barevision.embeddings.video_dataset import VideoFrameDataset
from barevision.embeddings.settings import (
    Settings,
    DatasetSettings,
    TrainingSettings,
    create_smoke_test_settings,
)
from barevision.utils.logging import JaxLogger
from barevision.embeddings.checkpoint_manager import create_checkpoint_manager
from barevision.embeddings.logging_utils import (
    log_attention_statistics,
    log_embedding_statistics,
    log_gradient_statistics,
)
from barevision.embeddings.visualization import log_visualizations


def create_dataloader(
    split: str,
    batch_size: int,
    img_size: tuple[int, int],
    max_frames: int | None = None,
    shuffle: bool = True,
    random_seed: int | None = None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, list[dict]]]:
    """Simple data loader that yields batches.

    Note: Currently uses single-process loading. For multiprocessing,
    integrate with PyTorch DataLoader or equivalent.
    See dataset.num_workers setting for configuration.

    Args:
        split: 'train' or 'val'
        batch_size: Number of samples per batch
        img_size: Image size (height, width) - must produce embeddings divisible by 16
        max_frames: Maximum number of frames to load (for smoke tests).
                   If used with shuffle=True, randomly samples this many frames.
        shuffle: Whether to shuffle the dataset (default True for train)
        random_seed: Random seed for shuffling (for reproducibility)

    Yields:
        Tuple of (img1_batch, img2_batch, metadata_batch) where:
            - img1_batch: (B, H, W, 3)
            - img2_batch: (B, H, W, 3)
            - metadata_batch: dict with video_name, frame_t, frame_tk, distance
    """
    dataset = VideoFrameDataset(
        split=split,
        max_frame_distance=5,
        img_size=img_size,
    )

    # Get all indices
    indices = list(range(len(dataset)))

    # For smoke tests, sample a subset
    if max_frames is not None:
        if shuffle:
            # Randomly sample max_frames indices
            rng = random.Random(random_seed)
            indices = rng.sample(indices, min(max_frames, len(indices)))
        else:
            # Take first max_frames
            indices = indices[: min(max_frames, len(indices))]
    elif shuffle and split == "train":
        # Shuffle all indices for full dataset
        if random_seed is not None:
            rng = random.Random(random_seed)
            rng.shuffle(indices)
        else:
            random.shuffle(indices)

    # Yield batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Skip incomplete batches

        imgs1: List[jnp.ndarray] = []
        imgs2: List[jnp.ndarray] = []
        metadata_list: List[dict] = []

        for idx in batch_indices:
            img1, img2, meta = dataset[idx]
            imgs1.append(img1)
            imgs2.append(img2)
            metadata_list.append(meta)

        img1_batch = jnp.stack(imgs1)
        img2_batch = jnp.stack(imgs2)

        yield img1_batch, img2_batch, metadata_list


def create_train_state(model, learning_rate: float):
    """Create training state with optimizer.
    
    Uses NNX split/merge pattern for proper gradient updates.
    
    Args:
        model: NNX model
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Tuple of (graphdef, state, tx, opt_state)
    """
    # Split model into graphdef and state
    graphdef, state = nnx.split(model)
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    opt_state = tx.init(state)
    
    return graphdef, state, tx, opt_state


def train_step(graphdef, state, tx, opt_state, img1, img2, alpha=1.0, beta=1.0):
    """Single training step using split/merge pattern.
    
    Args:
        graphdef: Model graph definition
        state: Model state
        tx: Optimizer
        opt_state: Optimizer state
        img1: Batch of frame 1 images (B, H, W, 3)
        img2: Batch of frame 2 images (B, H, W, 3)
        alpha: Weight for self-attention loss
        beta: Weight for cross-attention loss
        
    Returns:
        Tuple of (new_state, new_opt_state, loss, self_loss_mean, cross_loss_mean)
    """

    def loss_fn(state):
        model = nnx.merge(graphdef, state)
        emb1 = model(img1)
        emb2 = model(img2)

        # Compute losses separately for logging
        from barevision.embeddings.loss import (
            self_attention_entropy_loss_core,
            cross_attention_entropy_loss_core,
        )
        from barevision.utils.grid import WindowGrid
        
        window_size = 16
        grid = WindowGrid(window_size=window_size)
        windows1 = grid.split(emb1)
        windows2 = grid.split(emb2)
        
        B, num_windows, wh, ww, D = windows1.shape
        flat_windows1 = windows1.reshape(B * num_windows, wh, ww, D)
        flat_windows2 = windows2.reshape(B * num_windows, wh, ww, D)
        
        self_loss = self_attention_entropy_loss_core(flat_windows1)
        cross_loss = cross_attention_entropy_loss_core(flat_windows1, flat_windows2)
        
        # Combined loss
        combined = alpha * self_loss + beta * cross_loss
        
        return combined.mean(), self_loss.mean(), cross_loss.mean()

    # Compute gradients
    loss_value, self_loss_mean, cross_loss_mean = loss_fn(state)
    grads = nnx.grad(lambda s: loss_fn(s)[0])(state)

    # Apply gradients
    updates, opt_state = tx.update(grads, opt_state, state)
    state = optax.apply_updates(state, updates)

    return state, opt_state, float(loss_value), float(self_loss_mean), float(cross_loss_mean)


def train(settings: Settings):
    """Main training loop.

    Args:
        settings: Training configuration
    """
    if settings.training.smoke_test:
        settings = create_smoke_test_settings()

    print("=" * 60)
    print("EMBEDDING TRAINING")
    print("=" * 60)
    print()

    # Initialize logger
    logger = JaxLogger(
        log_dir=settings.logging.log_dir,
        run_name_prefix=settings.logging.run_name_prefix,
    )

    # Initialize model
    print("Initializing model...")
    model = SimpleEmbeddingModel(
        embed_dim=16,
        in_channels=3,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )

    # Create training state (graphdef, state, tx, opt_state)
    graphdef, state, tx, opt_state = create_train_state(
        model, settings.training.learning_rate
    )

    # Count parameters
    param_count = 0
    for module_state in state.values():
        for param_value in module_state.values():
            if hasattr(param_value, "size"):
                param_count += param_value.size
    print(f"Model parameters: {param_count}")
    print()

    # Initialize checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=settings.training.checkpoint_dir,
        save_interval_steps=settings.training.checkpoint_freq,
        max_to_keep=settings.training.keep_last_n_checkpoints,
        enabled=settings.training.checkpoint_freq > 0,
    )

    # Handle resume (simplified - just load state for now)
    start_epoch = 0
    global_step = 0

    if settings.training.resume:
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Resuming from checkpoint at step {latest_step}")
            start_epoch, global_step, state = checkpoint_manager.restore(
                state=state,
            )
            print(f"Resumed at epoch {start_epoch}, step {global_step}")
        else:
            print("Warning: No checkpoint found to resume from")
            print("Starting fresh training...")

    # Training loop
    print(
        f"Training for {settings.training.epochs} epochs with batch_size={settings.dataset.batch_size}"
    )
    if settings.training.steps_per_epoch > 0:
        print(f"Steps per epoch: {settings.training.steps_per_epoch}")
    print()

    for epoch in range(start_epoch, settings.training.epochs):
        epoch_start = time.time()
        epoch_losses = []

        # Create data loader
        loader = create_dataloader(
            split="train",
            batch_size=settings.dataset.batch_size,
            img_size=settings.dataset.img_size,
            max_frames=(
                settings.training.steps_per_epoch * settings.dataset.batch_size
                if settings.training.steps_per_epoch > 0
                else None
            ),
        )

        for step, (img1, img2, metadata) in enumerate(loader):
            if (
                settings.training.steps_per_epoch > 0
                and step >= settings.training.steps_per_epoch
            ):
                break

            # Training step
            state, opt_state, loss, self_loss, cross_loss = train_step(
                graphdef, state, tx, opt_state, img1, img2
            )
            epoch_losses.append(loss)
            global_step += 1

            # Log loss components to TensorBoard
            logger.log_scalar("Loss/train_step", loss, global_step)
            logger.log_scalar("Loss/self_entropy", self_loss, global_step)
            logger.log_scalar("Loss/cross_entropy", cross_loss, global_step)

            # Save checkpoint if needed
            if checkpoint_manager.should_save(global_step):
                checkpoint_manager.save(
                    step=global_step,
                    state=state,
                    opt_state=opt_state,
                    epoch=epoch,
                )

            # Log gradient statistics periodically (every 10 steps)
            if global_step % 10 == 0:
                # Create temporary model for gradient logging
                temp_model = nnx.merge(graphdef, state)
                log_gradient_statistics(logger, None, temp_model, global_step)
            
            # Log embedding and attention statistics periodically
            if settings.logging.log_statistics_every_steps > 0 and \
               global_step % settings.logging.log_statistics_every_steps == 0:
                # Get a sample for statistics
                stats_loader = create_dataloader(
                    split="train",
                    batch_size=1,
                    img_size=settings.dataset.img_size,
                    max_frames=1,
                    shuffle=True,
                    random_seed=global_step,
                )
                stats_img1, _, _ = next(stats_loader)
                temp_model = nnx.merge(graphdef, state)
                embeddings = temp_model(stats_img1)
                
                log_embedding_statistics(logger, embeddings, global_step)
                log_attention_statistics(logger, embeddings, global_step)

            # Log visualizations periodically
            if (
                settings.logging.log_visualizations_every_steps > 0
                and global_step % settings.logging.log_visualizations_every_steps == 0
            ):
                # Get a fresh random sample batch for visualization
                # Use global_step as seed to get different frames each time
                viz_loader = create_dataloader(
                    split="train",
                    batch_size=1,
                    img_size=settings.dataset.img_size,
                    max_frames=1,
                    shuffle=True,
                    random_seed=global_step,
                )
                viz_img1, viz_img2, viz_metadata = next(viz_loader)
                # Create temporary model for visualization
                temp_model = nnx.merge(graphdef, state)
                log_visualizations(
                    logger, temp_model, viz_img1, viz_img2, viz_metadata[0], global_step, settings
                )

            # Log every few steps to console
            if step % 5 == 0:
                elapsed = time.time() - epoch_start
                steps_per_sec = (step + 1) / elapsed
                print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss: {loss:.4f} (self={self_loss:.4f}, cross={cross_loss:.4f}) | "
                    f"{steps_per_sec:.1f} steps/sec"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.log_scalar("Loss/train_epoch", avg_loss, epoch)

        # Log embedding and attention statistics once per epoch
        # Get a sample batch for diagnostics
        sample_loader = create_dataloader(
            split="train",
            batch_size=1,
            img_size=settings.dataset.img_size,
            max_frames=1,
        )
        sample_img1, sample_img2, _ = next(sample_loader)
        # Create temporary model for statistics
        temp_model = nnx.merge(graphdef, state)
        embeddings = temp_model(sample_img1)

        log_embedding_statistics(logger, embeddings, epoch)
        log_attention_statistics(logger, embeddings, epoch)

        print(f"Epoch {epoch} complete | Avg loss: {avg_loss:.4f} | {epoch_time:.1f}s")
        print()

    checkpoint_manager.close()
    logger.close()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return graphdef, state


def main():
    """Entry point with tyro CLI support."""
    settings = tyro.cli(Settings)
    train(settings)


if __name__ == "__main__":
    main()
