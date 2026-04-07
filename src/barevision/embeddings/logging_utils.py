"""Embedding-specific logging utilities.

Builds on barevision.utils.logging.JaxLogger with embedding-specific diagnostics.
"""

import time

import jax.numpy as jnp
import numpy as np

from barevision.embeddings.settings import Settings
from barevision.embeddings.spatial_losses import (
    self_attention_spatial_variance,
    cross_attention_spatial_variance,
)
from barevision.embeddings.model import HierarchicalEmbeddingModel
from barevision.utils.grid import (
    WindowGrid,
    generate_normalized_coordinates,
    crop_to_grid_aligned,
)
from barevision.utils.logging import TensorboardLogger


def log_attention_statistics(
    logger: TensorboardLogger,
    embeddings: jnp.ndarray,
    step: int,
    window_size: int = 16,
    temperature: float = 0.25,
    prefix: str = "Attention",
):
    """Log attention spatial variance distributions for diagnostic monitoring.

    Tracks:
    - Self-attention spatial variance: training minimizes this (encourage concentrated attention)
    - Cross-attention spatial variance: training minimizes this (encourage confident matching)

    Args:
        logger: JaxLogger instance
        embeddings: (B, H, W, D) embeddings from a single frame
        step: Global step
        window_size: Attention window size
        temperature: Softmax temperature (should match training setting)
        prefix: Tag prefix for organization
    """
    B, H, W, D = embeddings.shape

    # Crop to grid-aligned dimensions (same as loss calculation)
    embeddings_cropped = crop_to_grid_aligned(embeddings, window_size)
    B_crop, H_crop, W_crop, D_crop = embeddings_cropped.shape

    # Split into windows
    grid = WindowGrid(window_size=window_size)
    windows = grid.split(embeddings_cropped)
    num_windows = (H_crop // window_size) * (W_crop // window_size)
    flat_windows = windows.reshape(B_crop * num_windows, window_size, window_size, D_crop)

    # Precompute coordinates
    coords = generate_normalized_coordinates(window_size)

    # Compute spatial variances (both return tuple of (loss, aux))
    self_variance, self_aux = self_attention_spatial_variance(
        flat_windows, temperature=temperature, coords=coords
    )
    cross_variance, cross_aux = cross_attention_spatial_variance(
        flat_windows, flat_windows, temperature=temperature, coords=coords
    )

    # Log histograms
    logger.log_histogram(
        f"{prefix}/self_variance", np.array(self_variance.flatten()), step
    )
    logger.log_histogram(
        f"{prefix}/cross_variance", np.array(cross_variance.flatten()), step
    )

    # Log summary statistics
    logger.log_scalar(
        f"{prefix}/self_variance_mean", float(np.mean(self_variance)), step
    )
    logger.log_scalar(f"{prefix}/self_variance_std", float(np.std(self_variance)), step)
    logger.log_scalar(
        f"{prefix}/cross_variance_mean", float(np.mean(cross_variance)), step
    )
    logger.log_scalar(
        f"{prefix}/cross_variance_std", float(np.std(cross_variance)), step
    )


def log_embedding_statistics(
    logger: TensorboardLogger,
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
    logger: TensorboardLogger,
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


def log_metrics(logger: TensorboardLogger, loss, aux, step: int):
    """Log loss metrics to TensorBoard.

    For spatial variance loss, logs:
    - Total loss
    - Self and cross loss components
    - Per-level breakdown (if available in aux)
    """
    logger.log_scalar("Loss/total", float(loss), step)

    # Spatial variance loss structure
    if "self_loss" in aux:
        logger.log_scalar("Loss/spatial_variance/self", float(aux["self_loss"]), step)
        logger.log_scalar("Loss/spatial_variance/cross", float(aux["cross_loss"]), step)

        # Log per-level breakdown if available
        if "level_losses" in aux and "level_weights" in aux:
            for i, (level_loss, level_weight) in enumerate(
                zip(aux["level_losses"], aux["level_weights"])
            ):
                logger.log_scalar(
                    f"Loss/level_{i}/weighted_loss", float(level_loss), step
                )
                logger.log_scalar(
                    f"Loss/level_{i}/weight", float(level_weight), step
                )


def log_diagnostics(
    logger: TensorboardLogger,
    model,
    img1,
    step: int,
    window_size: int = 16,
    temperature: float = 0.25,
):
    """Log gradient statistics, embeddings, and attention statistics.

    Works with both:
    - Joint model (has model.embedding_model attribute)
    - Standalone embeddings model (model IS the embedding model)

    For hierarchical models, logs all pyramid levels.

    Args:
        logger: TensorBoard logger
        model: NNX model
        img1: Input frame for diagnostics
        step: Global step
        window_size: Attention window size
        temperature: Softmax temperature (should match training setting)
    """
    log_gradient_statistics(logger, None, model, step)

    # Get embeddings - handle both joint and standalone models
    if hasattr(model, "embedding_model"):
        # Joint model
        pyramid = model.embedding_model(img1)
    else:
        # Standalone embeddings model
        pyramid = model(img1)

    # Log statistics for all pyramid levels
    for i, embeddings in enumerate(pyramid):
        prefix = f"Embeddings/level{i}"
        log_embedding_statistics(logger, embeddings, step, prefix=prefix)
        log_attention_statistics(
            logger, embeddings, step, window_size, temperature, prefix=f"Attention/level{i}"
        )


def format_progress_line(
    epoch: int, step: int, loss: float, elapsed: float, aux: dict | None = None
) -> str:
    """Format training progress line for console output.

    Works with spatial variance loss structure.
    """
    steps_per_sec = (step + 1) / elapsed

    # Start with basic info
    parts = [f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}"]

    # Add loss breakdown if available (spatial variance loss structure)
    if aux and "self_loss" in aux:
        self_var = float(aux["self_loss"])
        cross_var = float(aux["cross_loss"])
        parts.append(
            f"Spatial Var: {self_var + cross_var:.4f} "
            f"(self: {self_var:.2f} | cross: {cross_var:.2f})"
        )

    # Fallback to old entropy structure for backward compatibility
    elif aux and "entropy" in aux:
        self_entropy = float(aux["entropy"]["self_loss"])
        cross_entropy = float(aux["entropy"]["cross_loss"])
        total_entropy = float(aux["entropy"]["loss"])
        if "reconstruction" in aux:
            reconstruction = float(aux["reconstruction"]["loss"])
            parts.append(
                f"Entropy: {total_entropy:.4f} "
                f"(self: {self_entropy:.2f} | cross: {cross_entropy:.2f}) "
                f"| Recon: {reconstruction:.4f}"
            )
        else:
            parts.append(
                f"Entropy: {total_entropy:.4f} "
                f"(self: {self_entropy:.2f} | cross: {cross_entropy:.2f})"
            )

    parts.append(f"{steps_per_sec:.1f} steps/sec")

    return " | ".join(parts)


def log_progress(
    logger: TensorboardLogger,
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


def print_header(settings: Settings):
    print("=" * 60)
    print("OPTICAL FLOW TRAINING")
    print("=" * 60)
    print()
    print(f"Pyramid levels: {settings.model.num_levels}")
    print(
        f"Coarse grid: {settings.dataset.coarse_grid_size}×{settings.dataset.coarse_grid_size}"
    )
    print(f"Window size: {settings.dataset.window_size}×{settings.dataset.window_size}")
    print(f"Embedding dim: {settings.model.embed_dim}")
    image_size = settings.model.target_to_input(
        settings.dataset.coarse_grid_size,
        settings.dataset.window_size,
    )
    print(f"Image size: {image_size}")
    print()
    print(f"Epochs: {settings.training.epochs}")
    print(f"Batch size: {settings.dataset.batch_size}")
    if settings.dataset.max_samples > 0:
        print(f"Max samples per epoch: {settings.dataset.max_samples}")
    print()
