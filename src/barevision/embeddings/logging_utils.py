"""Embedding-specific logging utilities.

Builds on barevision.utils.logging.JaxLogger with embedding-specific diagnostics.
"""

import jax.numpy as jnp
import numpy as np

from barevision.utils.logging import JaxLogger
from barevision.utils.grid import WindowGrid
from barevision.embeddings.loss import (
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
    step: int,
    prefix: str = "Gradients",
):
    """Log gradient statistics from optimizer state.

    Useful for detecting:
    - Vanishing gradients (all gradients → 0)
    - Exploding gradients (gradients too large)
    - Dead parameters (gradients always zero)

    Args:
        logger: JaxLogger instance
        optimizer: NNX optimizer (gradients are in optimizer state)
        step: Global step
        prefix: Tag prefix
    """
    from flax import nnx

    try:
        # Get optimizer state which contains gradients
        opt_state = nnx.state(optimizer)
        
        all_grads = []
        
        # Iterate through optimizer state to find gradients
        for module_path, module_state in opt_state.items():
            for param_name, param_value in module_state.items():
                # Look for gradient arrays in optimizer state
                if hasattr(param_value, "value") and hasattr(param_value.value, "shape"):
                    grad = np.array(param_value.value)
                    if np.issubdtype(grad.dtype, np.number):
                        all_grads.append(grad.flatten())
        
        if all_grads:
            all_grads_flat = np.concatenate(all_grads)
            
            # Log overall gradient statistics
            logger.log_histogram(f"{prefix}/all", all_grads_flat, step)
            logger.log_scalar(f"{prefix}/norm", float(np.linalg.norm(all_grads_flat)), step)
            logger.log_scalar(f"{prefix}/mean", float(np.mean(np.abs(all_grads_flat))), step)
            logger.log_scalar(f"{prefix}/max", float(np.max(np.abs(all_grads_flat))), step)
            
    except Exception as e:
        print(f"Warning: Could not log gradient statistics: {e}")
