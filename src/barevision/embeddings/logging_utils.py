"""Embedding-specific logging utilities.

Builds on barevision.utils.logging.JaxLogger with embedding-specific diagnostics.
"""

import time

import jax.numpy as jnp
import numpy as np

from barevision.utils.logging import TensorboardLogger


def log_flow_statistics(
    logger: TensorboardLogger,
    flow: jnp.ndarray,
    confidence: jnp.ndarray,
    step: int,
    prefix: str = "Flow",
):
    """Log flow and confidence statistics for diagnostic monitoring.

    Tracks:
    - Flow magnitude distribution
    - Confidence distribution (negative variance)

    Args:
        logger: TensorBoard logger instance
        flow: (B, H, W, 2) flow vectors
        confidence: (B, H, W) confidence scores
        step: Global step
        prefix: Tag prefix for organization
    """
    flow_array = np.array(flow)
    confidence_array = np.array(confidence)

    # Flow magnitude
    flow_magnitude = np.linalg.norm(flow_array, axis=-1)
    logger.log_histogram(f"{prefix}/magnitude", flow_magnitude.flatten(), step)
    logger.log_scalar(f"{prefix}/magnitude_mean", float(np.mean(flow_magnitude)), step)
    logger.log_scalar(f"{prefix}/magnitude_std", float(np.std(flow_magnitude)), step)
    logger.log_scalar(f"{prefix}/magnitude_max", float(np.max(flow_magnitude)), step)

    # Flow components
    logger.log_histogram(f"{prefix}/dx", flow_array[..., 0].flatten(), step)
    logger.log_histogram(f"{prefix}/dy", flow_array[..., 1].flatten(), step)

    # Confidence (negative variance, so more negative = less confident)
    logger.log_histogram(f"{prefix}/confidence", confidence_array.flatten(), step)
    logger.log_scalar(f"{prefix}/confidence_mean", float(np.mean(confidence_array)), step)
    logger.log_scalar(f"{prefix}/confidence_std", float(np.std(confidence_array)), step)


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

    For linear attention flow loss, logs:
    - Total loss
    - Reconstruction, diversity, and concordance loss components
    - Per-level breakdown (if available in aux)
    """
    logger.log_scalar("Loss/total", float(loss), step)

    # Linear attention flow loss structure
    if "reconstruction_loss" in aux:
        logger.log_scalar(
            "Loss/reconstruction", float(aux["reconstruction_loss"]), step
        )
        div_loss = float(aux["diversity_loss"])  # Already normalized: 0=perfect, 1=collapse
        div_variance = float(aux.get("diversity_variance", 0.0))  # Raw variance
        logger.log_scalar("Loss/diversity", div_loss, step)
        logger.log_scalar("Loss/diversity_variance", div_variance, step)

        if "concordance_loss" in aux:
            logger.log_scalar(
                "Loss/concordance", float(aux["concordance_loss"]), step
            )

        # Log per-level breakdown if available
        if "level_losses" in aux and "level_weights" in aux:
            for i, (level_loss, level_weight) in enumerate(
                zip(aux["level_losses"], aux["level_weights"])
            ):
                logger.log_scalar(
                    f"Loss/level_{i}/weighted_loss", float(level_loss), step
                )
                logger.log_scalar(f"Loss/level_{i}/weight", float(level_weight), step)

            # Log per-level loss components
            if "level_reconstruction_losses" in aux:
                for i, level_recon in enumerate(aux["level_reconstruction_losses"]):
                    logger.log_scalar(
                        f"Loss/level_{i}/reconstruction", float(level_recon), step
                    )
            if "level_diversity_losses" in aux:
                for i, level_div in enumerate(aux["level_diversity_losses"]):
                    logger.log_scalar(
                        f"Loss/level_{i}/diversity", float(level_div), step
                    )
            if "level_concordance_losses" in aux:
                for i, level_conc in enumerate(aux["level_concordance_losses"]):
                    logger.log_scalar(
                        f"Loss/level_{i}/concordance", float(level_conc), step
                    )


def log_diagnostics(
    logger: TensorboardLogger,
    model,
    optimizer,
    pyramid,
    step: int,
    aux: dict,
):
    """Log gradient statistics, embeddings, and flow statistics.

    Args:
        logger: TensorBoard logger
        model: NNX model (for gradient statistics)
        optimizer: NNX optimizer (for gradient statistics)
        pyramid: List of embedding tensors from forward pass (one per level)
        step: Global step
        aux: Auxiliary data from loss (contains flow, confidence per level)
    """
    log_gradient_statistics(logger, optimizer, model, step)

    # Log flow_stats diagnostics if available (from linear attention)
    if "flow_stats" in aux:
        stats = aux["flow_stats"]
        logger.log_scalar("Flow/self_com_min", float(stats["self_com_min"]), step)
        logger.log_scalar("Flow/self_com_max", float(stats["self_com_max"]), step)
        logger.log_scalar("Flow/cross_com_min", float(stats["cross_com_min"]), step)
        logger.log_scalar("Flow/cross_com_max", float(stats["cross_com_max"]), step)
        logger.log_scalar("Flow/flow_min", float(stats["flow_min"]), step)
        logger.log_scalar("Flow/flow_max", float(stats["flow_max"]), step)
        logger.log_scalar(
            "Flow/weight_sum_self_mean", float(stats["weight_sum_self_mean"]), step
        )
        logger.log_scalar(
            "Flow/weight_sum_cross_mean", float(stats["weight_sum_cross_mean"]), step
        )

    # Log statistics for all pyramid levels
    for i, embeddings in enumerate(pyramid):
        prefix = f"Embeddings/level{i}"
        log_embedding_statistics(logger, embeddings, step, prefix=prefix)
        
        # Log flow statistics if available
        if "flow" in aux and i < len(aux["flow"]):
            flow_stats_prefix = f"Flow/level{i}"
            log_flow_statistics(
                logger,
                aux["flow"][i],
                aux["confidence"][i],
                step,
                prefix=flow_stats_prefix,
            )


def format_progress_line(
    epoch: int, step: int, loss: float, elapsed: float, aux: dict | None = None
) -> str:
    """Format training progress line for console output.

    Works with linear attention flow loss structure.
    """
    steps_per_sec = (step + 1) / elapsed

    # Start with basic info
    parts = [f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}"]

    # Linear attention flow loss structure
    if aux and "reconstruction_loss" in aux:
        recon = float(aux["reconstruction_loss"])
        div = float(aux["diversity_loss"])  # Normalized: 0=perfect, 1=collapse
        div_var = float(aux.get("diversity_variance", 0.0))  # Raw variance
        parts.append(f"Recon: {recon:.4f} | Div: {div:.4f}")
        
        if "concordance_loss" in aux:
            conc = float(aux["concordance_loss"])
            parts.append(f"Conc: {conc:.4f}")
        
        parts.append(f"Var: {div_var:.4f}")

    parts.append(f"{steps_per_sec:.1f} steps/sec")

    return " | ".join(parts)


def log_progress(
    logger: TensorboardLogger,
    model,
    optimizer,
    pyramid,
    epoch: int,
    step: int,
    loss,
    aux,
    epoch_start: float,
):
    """Log all standard training diagnostics and print progress.

    This function orchestrates regular training logging:
    1. Log loss metrics
    2. Log gradient and embedding diagnostics
    3. Print progress line to console

    Args:
        logger: JaxLogger instance
        model: NNX model
        optimizer: NNX optimizer
        pyramid: List of embedding tensors from forward pass
        epoch: Current epoch number
        step: Current step within epoch
        loss: Combined loss value
        aux: Auxiliary loss information
        epoch_start: Time when epoch started (for speed calculation)
    """
    log_metrics(logger, loss, aux, step)
    log_diagnostics(logger, model, optimizer, pyramid, step, aux)
    print(
        format_progress_line(epoch, step, float(loss), time.time() - epoch_start, aux)
    )


def print_footer():
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


def print_header(config):
    """Print training header.

    Args:
        config: RootConfig or dict with dataset and training settings
    """
    print("=" * 60)
    print("OPTICAL FLOW TRAINING")
    print("=" * 60)
    print()
    print(
        f"Coarse grid: {config.dataset.coarse_grid_size}×{config.dataset.coarse_grid_size}"
    )
    print(f"Window size: {config.dataset.window_size}×{config.dataset.window_size}")
    print()
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.dataset.batch_size}")
    if config.dataset.max_samples > 0:
        print(f"Max samples per epoch: {config.dataset.max_samples}")
    print()
