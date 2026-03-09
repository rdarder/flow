"""Embedding-specific logging utilities.

Builds on barevision.utils.logging.JaxLogger with embedding-specific diagnostics.
"""

import time

import jax.numpy as jnp
import numpy as np

from barevision.utils.logging import JaxLogger
from barevision.utils.grid import WindowGrid
from barevision.flow.loss import (
    self_attention_entropy_loss_core,
    cross_attention_entropy_loss_core,
)


def log_attention_statistics(
    logger: JaxLogger,
    embeddings: jnp.ndarray,
    step: int,
    window_size: int = 16,
    prefix: str = "Attention",
):
    """Log attention entropy distributions for diagnostic monitoring.

    Tracks:
    - Self-attention entropy: should be HIGH (tolerate ambiguity within frame)
    - Cross-attention entropy: should be LOW (find precise matches across frames)

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

    # Compute entropies
    self_entropy = -self_attention_entropy_loss_core(
        flat_windows
    )  # Positive = higher entropy
    cross_entropy = cross_attention_entropy_loss_core(flat_windows, flat_windows)

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

    Useful for detecting:
    - Vanishing gradients (all gradients → 0)
    - Exploding gradients (gradients too large)
    - Dead parameters (gradients always zero)
    - Layer-specific issues (some layers learning, others not)

    Args:
        logger: JaxLogger instance
        optimizer: NNX optimizer (gradients are in optimizer state)
        model: NNX model (for parameter statistics)
        step: Global step
        prefix: Tag prefix
    """
    from flax import nnx

    try:
        # Get optimizer state which contains gradients
        opt_state = nnx.state(optimizer)
        model_state = nnx.state(model, nnx.Param)

        grad_stats = {}
        param_stats = {}

        # Iterate through model parameters to find corresponding gradients
        for module_path, module_state in model_state.items():
            module_name = str(module_path).replace("/", ".")

            for param_name, param_value in module_state.items():
                param_array = np.array(param_value)
                param_key = f"{module_name}.{param_name}"

                # Log parameter statistics
                param_stats[param_key] = {
                    "mean": float(np.mean(param_array)),
                    "std": float(np.std(param_array)),
                    "abs_max": float(np.max(np.abs(param_array))),
                }

                # Try to find corresponding gradient in optimizer state
                try:
                    # Gradient might be in opt_state[module_path][param_name] or nested
                    if module_path in opt_state:
                        opt_module = opt_state[module_path]
                        if param_name in opt_module:
                            grad_value = opt_module[param_name]
                            if hasattr(grad_value, "value"):
                                grad_array = np.array(grad_value.value)
                            else:
                                grad_array = np.array(grad_value)

                            if np.issubdtype(grad_array.dtype, np.number):
                                grad_norm = float(np.linalg.norm(grad_array))
                                grad_mean = float(np.mean(np.abs(grad_array)))

                                grad_stats[param_key] = {
                                    "norm": grad_norm,
                                    "mean_abs": grad_mean,
                                    "ratio": grad_norm
                                    / (param_stats[param_key]["abs_max"] + 1e-10),
                                }
                except (KeyError, AttributeError, TypeError):
                    # Gradient not found or not accessible
                    pass

        # Log per-layer gradient statistics
        for param_key, stats in grad_stats.items():
            logger.log_scalar(f"{prefix}/norm/{param_key}", stats["norm"], step)
            logger.log_scalar(f"{prefix}/mean_abs/{param_key}", stats["mean_abs"], step)
            logger.log_scalar(f"{prefix}/param_ratio/{param_key}", stats["ratio"], step)

        # Log parameter statistics
        for param_key, stats in param_stats.items():
            logger.log_scalar(f"{prefix}/param_mean/{param_key}", stats["mean"], step)
            logger.log_scalar(f"{prefix}/param_std/{param_key}", stats["std"], step)

        # Log overall summary
        if grad_stats:
            all_norms = [s["norm"] for s in grad_stats.values()]
            logger.log_scalar(
                f"{prefix}/total_norm", float(np.linalg.norm(all_norms)), step
            )
            logger.log_scalar(f"{prefix}/max_norm", float(np.max(all_norms)), step)
            logger.log_scalar(f"{prefix}/min_norm", float(np.min(all_norms)), step)

    except Exception as e:
        import traceback

        print(f"Warning: Could not log gradient statistics: {e}")
        traceback.print_exc()


def log_metrics(logger: JaxLogger, loss, aux, step: int):
    """Log loss metrics to TensorBoard."""
    logger.log_scalar("Loss/train_step", float(loss), step)
    logger.log_scalar("Loss/self_entropy", float(aux["self_loss"]), step)
    logger.log_scalar("Loss/cross_entropy", float(aux["cross_loss"]), step)


def log_diagnostics(logger: JaxLogger, model, img1, step: int, window_size: int = 16):
    """Log gradient statistics, embeddings, and attention statistics.
    
    For hierarchical models, uses coarsest pyramid level.
    """
    log_gradient_statistics(logger, None, model, step)
    
    # Get pyramid and use coarsest level
    pyramid = model(img1)
    embeddings = pyramid[-1]  # Coarsest level
    log_embedding_statistics(logger, embeddings, step)
    log_attention_statistics(logger, embeddings, step, window_size)


def format_progress_line(epoch: int, step: int, loss: float, elapsed: float) -> str:
    """Format training progress line for console output."""
    steps_per_sec = (step + 1) / elapsed
    return f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | {steps_per_sec:.1f} steps/sec"


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
    print(format_progress_line(epoch, step, float(loss), time.time() - epoch_start))


def print_footer():
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def print_header(settings):
    """Print training configuration header."""
    print("=" * 60)
    print("HIERARCHICAL EMBEDDING TRAINING")
    print("=" * 60)
    print()
    print(f"Pyramid levels: {settings.model.num_levels}")
    print(f"Coarse grid: {settings.dataset.coarse_grid_size}×{settings.dataset.coarse_grid_size}")
    print(f"Window size: {settings.model.window_size}×{settings.model.window_size}")
    print(f"Embedding dim: {settings.model.embed_dim}")
    print(f"Input size: {settings.dataset.img_size}")
    print()
    print(f"Epochs: {settings.training.epochs}")
    print(f"Batch size: {settings.dataset.batch_size}")
    if settings.dataset.max_samples > 0:
        print(f"Max samples per epoch: {settings.dataset.max_samples}")
    print()
