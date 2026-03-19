"""Embedding-specific logging utilities.

Builds on barevision.utils.logging.JaxLogger with embedding-specific diagnostics.
"""

import time

import jax.numpy as jnp
import numpy as np

from barevision.flow.embeddings.losses import (
    self_attention_entropy_loss,
    cross_attention_entropy_loss,
)
from barevision.flow.settings import LoggingSettings
from barevision.utils import image
from barevision.utils.grid import WindowGrid
from barevision.utils.logging import JaxLogger


def log_attention_statistics(
    logger: JaxLogger,
    embeddings: jnp.ndarray,
    step: int,
    window_size: int = 16,
    prefix: str = "Attention",
):
    """Log attention entropy distributions for diagnostic monitoring.

    Tracks:
    - Self-attention entropy: training minimizes this (encourage unique embeddings where only self dominates)
    - Cross-attention entropy: training minimizes this (encourage confident cross-frame matching)

    Args:
        logger: JaxLogger instance
        embeddings: (B, H, W, D) embeddings from a single frame
        step: Global step
        window_size: Attention window size
        prefix: Tag prefix for organization
    """
    B, H, W, D = embeddings.shape

    # Split into windows
    grid = WindowGrid(window_size=window_size)
    windows = grid.split(embeddings)
    num_windows = (H // window_size) * (W // window_size)
    flat_windows = windows.reshape(B * num_windows, window_size, window_size, D)

    # Compute entropies (both return tuple of (loss, aux))
    # Use temperature=1.0 for diagnostic logging (standard entropy measurement)
    self_entropy, self_aux = self_attention_entropy_loss(flat_windows, temperature=1.0)
    cross_entropy, cross_aux = cross_attention_entropy_loss(
        flat_windows, flat_windows, temperature=1.0
    )

    # Log histograms
    logger.log_histogram(
        f"{prefix}/self_entropy", np.array(self_entropy.flatten()), step
    )
    logger.log_histogram(
        f"{prefix}/cross_entropy", np.array(cross_entropy.flatten()), step
    )

    # Log histograms
    logger.log_histogram(
        f"{prefix}/self_entropy", np.array(self_entropy.flatten()), step
    )
    logger.log_histogram(
        f"{prefix}/cross_entropy", np.array(cross_entropy.flatten()), step
    )

    # Log summary statistics
    logger.log_scalar(f"{prefix}/self_entropy_mean", float(np.mean(self_entropy)), step)
    logger.log_scalar(f"{prefix}/self_entropy_std", float(np.std(self_entropy)), step)
    logger.log_scalar(
        f"{prefix}/cross_entropy_mean", float(np.mean(cross_entropy)), step
    )
    logger.log_scalar(f"{prefix}/cross_entropy_std", float(np.std(cross_entropy)), step)


def log_embedding_statistics(
    logger: JaxLogger,
    embeddings: jnp.ndarray,
    step: int,
    prefix: str = "Embeddings",
):
    """Log embedding value distributions.

    Useful for detecting:
    - Embedding collapse (all values converge)
    - Exploding activations
    - Dead neurons (always zero)

    Args:
        logger: JaxLogger instance
        embeddings: (B, H, W, D) embeddings
        step: Global step
        prefix: Tag prefix
    """
    emb = np.array(embeddings)

    # Overall statistics
    logger.log_histogram(f"{prefix}/values", emb.flatten(), step)
    logger.log_scalar(f"{prefix}/mean", float(np.mean(emb)), step)
    logger.log_scalar(f"{prefix}/std", float(np.std(emb)), step)
    logger.log_scalar(f"{prefix}/max", float(np.max(emb)), step)
    logger.log_scalar(f"{prefix}/min", float(np.min(emb)), step)

    # Per-channel statistics (sample a few channels)
    D = embeddings.shape[-1]
    sample_channels = [0, D // 2, D - 1] if D > 3 else list(range(D))
    for ch in sample_channels:
        logger.log_scalar(f"{prefix}/ch{ch}_mean", float(np.mean(emb[..., ch])), step)


def log_gradient_statistics(
    logger: JaxLogger,
    optimizer,
    model,
    step: int,
    prefix: str = "Gradients",
):
    """Log gradient statistics from optimizer state with per-layer breakdown.

    Recursively handles nested structures including nnx.List.

    Args:
        logger: JaxLogger instance
        optimizer: NNX optimizer (gradients are in optimizer state)
        model: NNX model (for parameter statistics)
        step: Global step
        prefix: Tag prefix
    """
    from flax import nnx
    from flax.nnx import State

    try:
        # Get model state for parameter statistics
        model_state = nnx.state(model, nnx.Param)

        param_stats = {}

        def process_state_recursive(state_obj, path=""):
            """Recursively process state, handling nnx.List and nested structures."""
            if not isinstance(state_obj, State):
                return

            for key, value in state_obj.items():
                # Build full path for this parameter
                current_path = f"{path}.{key}" if path else str(key)

                if isinstance(value, State):
                    # Recurse into nested state (e.g., nnx.List items or nested modules)
                    process_state_recursive(value, current_path)
                elif hasattr(value, "size"):
                    # This is an actual parameter array
                    param_array = np.array(value)
                    param_key = current_path

                    # Log parameter statistics
                    param_stats[param_key] = {
                        "mean": float(np.mean(param_array)),
                        "std": float(np.std(param_array)),
                        "abs_max": float(np.max(np.abs(param_array))),
                    }

        # Process all parameters recursively
        process_state_recursive(model_state)

        # Log parameter statistics
        for param_key, stats in param_stats.items():
            logger.log_scalar(f"{prefix}/param_mean/{param_key}", stats["mean"], step)
            logger.log_scalar(f"{prefix}/param_std/{param_key}", stats["std"], step)

        # Log overall summary
        if param_stats:
            all_abs_max = [s["abs_max"] for s in param_stats.values()]
            logger.log_scalar(
                f"{prefix}/max_param_abs", float(np.max(all_abs_max)), step
            )
            logger.log_scalar(
                f"{prefix}/min_param_abs", float(np.min(all_abs_max)), step
            )

    except Exception as e:
        import traceback

        print(f"Warning: Could not log gradient statistics: {e}")
        traceback.print_exc()


def log_metrics(logger: JaxLogger, loss, aux, step: int):
    """Log loss metrics to TensorBoard."""
    logger.log_scalar("Loss/train_step", float(loss), step)

    # Handle nested aux structure (from return_aux=True) or flat structure
    loss_aux = aux.get("loss", aux) if isinstance(aux, dict) else {}

    if "self_loss" in loss_aux:
        logger.log_scalar("Loss/self_entropy", float(loss_aux["self_loss"]), step)
    if "cross_loss" in loss_aux:
        logger.log_scalar("Loss/cross_entropy", float(loss_aux["cross_loss"]), step)

    # Log reconstruction loss if available (flow estimation)
    if "reconstruction_loss" in loss_aux:
        logger.log_scalar(
            "Loss/reconstruction", float(loss_aux["reconstruction_loss"]), step
        )
        logger.log_scalar("Loss/entropy", float(loss_aux["entropy_loss"]), step)


def log_diagnostics(logger: JaxLogger, model, img1, step: int, window_size: int = 16):
    """Log gradient statistics, embeddings, and attention statistics.

    For hierarchical models, uses coarsest pyramid level.
    """
    log_gradient_statistics(logger, None, model, step)

    # Get pyramid and use coarsest level
    # Support both OpticalFlowModel and HierarchicalEmbeddingModel
    if hasattr(model, "extract_embeddings"):
        pyramid = model.extract_embeddings(img1)
    else:
        pyramid = model(img1)

    embeddings = pyramid[-1]  # Coarsest level
    log_embedding_statistics(logger, embeddings, step)
    log_attention_statistics(logger, embeddings, step, window_size)


def format_progress_line(
    epoch: int, step: int, loss: float, elapsed: float, aux: dict | None = None
) -> str:
    """Format training progress line for console output."""
    steps_per_sec = (step + 1) / elapsed

    # Start with basic info
    parts = [f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}"]

    # Handle nested aux structure
    loss_aux = aux.get("loss", aux) if isinstance(aux, dict) else {}

    # Add loss breakdown if available
    if loss_aux and "entropy_loss" in loss_aux and "reconstruction_loss" in loss_aux:
        entropy = float(loss_aux["entropy_loss"])
        recon = float(loss_aux["reconstruction_loss"])
        parts.append(f"Entropy: {entropy:.4f} | Recon: {recon:.4f}")

    parts.append(f"{steps_per_sec:.1f} steps/sec")

    return " | ".join(parts)


def log_progress(
    logger: JaxLogger,
    model,
    img1,
    epoch: int,
    step: int,
    loss,
    aux,
    epoch_start: float,
    window_size: int = 16,
):
    """Log all standard training diagnostics and print progress.

    This function orchestrates regular training logging:
    1. Log loss metrics
    2. Log gradient and embedding diagnostics
    3. Print progress line to console

    Args:
        logger: JaxLogger instance
        model: NNX model
        img1: Input frame for diagnostics
        epoch: Current epoch number
        step: Current step within epoch
        loss: Combined loss value
        aux: auxiliary loss information (like self/cross attention loss components.
        epoch_start: Time when epoch started (for speed calculation)
        window_size: Attention window size
    """
    log_metrics(logger, loss, aux, step)
    log_diagnostics(logger, model, img1, step, window_size)
    print(
        format_progress_line(epoch, step, float(loss), time.time() - epoch_start, aux)
    )


def print_footer():
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def print_header(settings):
    print("=" * 60)
    print("OPTICAL FLOW TRAINING")
    print("=" * 60)
    print()
    print(f"Pyramid levels: {settings.model.num_levels}")
    print(
        f"Coarse grid: {settings.dataset.coarse_grid_size}×{settings.dataset.coarse_grid_size}"
    )
    print(f"Window size: {settings.model.window_size}×{settings.model.window_size}")
    print(f"Embedding dim: {settings.model.embed_dim}")
    image_size = image.image_size(
        settings.dataset.coarse_grid_size,
        settings.dataset.window_size,
        settings.dataset.num_levels,
    )
    print(f"Image size: {image_size}")
    print()
    print(f"Epochs: {settings.training.epochs}")
    print(f"Batch size: {settings.dataset.batch_size}")
    if settings.dataset.max_samples > 0:
        print(f"Max samples per epoch: {settings.dataset.max_samples}")
    print()


def should_log_something(settings: LoggingSettings, step: int):
    return (
        step % settings.log_visualizations_every_steps == 0
        or step % settings.log_every_steps == 0
    )
