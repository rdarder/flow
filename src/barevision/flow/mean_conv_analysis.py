"""Mean convolution kernel analysis for monitoring Local Contrast Normalization.

Provides diagnostics to understand what the depthwise mean_conv kernels are learning
during training. Focuses on behavioral properties rather than raw weight statistics.

Use Case: Monitor mean_conv kernels to verify they maintain low-pass averaging behavior
while adapting to feature statistics.

Example:
    from barevision.flow.mean_conv_analysis import analyze_mean_conv_kernels

    # During training validation
    model_state = nnx.state(model)
    analysis = analyze_mean_conv_kernels(model_state, num_levels=3)

    # Log to TensorBoard
    for level, stats in analysis.items():
        for metric, value in stats["scalars"].items():
            logger.scalar(f"mean_conv_behavior/{level}/{metric}", value)
"""

from typing import Any, Dict

import jax.numpy as jnp
from flax import nnx


def gaussian_kernel_2d(sigma: float = 1.0, size: int = 3) -> jnp.ndarray:
    """Create a reference 2D Gaussian kernel.

    Args:
        sigma: Standard deviation
        size: Kernel size (default 3)

    Returns:
        Normalized kernel (sums to 1.0)
    """
    ax = jnp.arange(-(size // 2), size // 2 + 1, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / jnp.sum(kernel)


def analyze_mean_conv_kernels(
    model_state: nnx.State, num_levels: int, hidden_dim: int = 32
) -> Dict[str, Dict[str, Any]]:
    """Analyze mean_conv kernels across all pyramid levels.

    Computes behavioral diagnostics that reveal what the kernels are learning:
    - Are they still performing averaging? (weight sums)
    - Are they still low-pass filters? (center vs surround)
    - Are channels specializing? (variance across channels)
    - How much adaptation from initialization? (drift magnitude)

    Args:
        model_state: Full model state from nnx.state(model)
        num_levels: Number of pyramid levels
        hidden_dim: Hidden dimension (channels per kernel)

    Returns:
        Dictionary mapping level name to analysis results:
        {
            "level_0": {
                "scalars": {metric_name: value, ...},
                "histograms": {metric_name: array, ...},
                "kernels": kernel_array  # For image visualization
            },
            ...
        }
    """
    results = {}

    for level_idx in range(num_levels):
        level_name = f"level_{level_idx}"

        # Try to find mean_conv in the model state
        try:
            if level_idx == 0:
                # StemBlock
                kernel = model_state["blocks"][0]["mean_conv"]["kernel"]
            else:
                # StandardBlock
                kernel = model_state["blocks"][level_idx]["mean_conv"]["kernel"]
        except (KeyError, IndexError):
            # mean_conv not found at this level (may not exist yet)
            continue

        # kernel shape: (3, 3, hidden_dim, hidden_dim) - depthwise (diagonal)
        # Extract the diagonal kernels: (3, 3, hidden_dim)
        diagonal_kernels = jnp.diagonal(kernel, axis1=2, axis2=3)

        # Compute diagnostics
        scalars = {}
        histograms = {}

        # 1. Weight sum distribution (should be ≈ 1.0 for averaging)
        weight_sums = jnp.sum(diagonal_kernels, axis=(0, 1))  # (hidden_dim,)
        scalars["weight_sum_mean"] = jnp.mean(weight_sums)
        scalars["weight_sum_std"] = jnp.std(weight_sums)
        scalars["weight_sum_min"] = jnp.min(weight_sums)
        scalars["weight_sum_max"] = jnp.max(weight_sums)
        histograms["weight_sums"] = weight_sums

        # 2. Center vs surround ratio (should be > 1 for low-pass)
        center_weights = diagonal_kernels[1, 1, :]  # (hidden_dim,)
        corner_weights = jnp.stack(
            [
                diagonal_kernels[0, 0, :],
                diagonal_kernels[0, 2, :],
                diagonal_kernels[2, 0, :],
                diagonal_kernels[2, 2, :],
            ]
        )  # (4, hidden_dim)
        corner_mean = jnp.mean(corner_weights, axis=0)  # (hidden_dim,)

        # Add epsilon to avoid division by zero
        center_surround_ratio = center_weights / (corner_mean + 1e-8)

        scalars["center_surround_ratio_mean"] = jnp.mean(center_surround_ratio)
        scalars["center_surround_ratio_std"] = jnp.std(center_surround_ratio)
        scalars["center_surround_ratio_min"] = jnp.min(center_surround_ratio)
        scalars["center_surround_ratio_max"] = jnp.max(center_surround_ratio)
        histograms["center_surround_ratios"] = center_surround_ratio
        histograms["center_weights"] = center_weights
        histograms["corner_weights_mean"] = corner_mean

        # 3. Deviation from Gaussian initialization (sigma=1.0)
        reference_gaussian = gaussian_kernel_2d(sigma=1.0, size=3)
        drift_per_channel = jnp.sqrt(
            jnp.sum(
                (diagonal_kernels - reference_gaussian[:, :, None]) ** 2, axis=(0, 1)
            )
        )  # (hidden_dim,)

        scalars["drift_from_init_mean"] = jnp.mean(drift_per_channel)
        scalars["drift_from_init_std"] = jnp.std(drift_per_channel)
        scalars["drift_from_init_max"] = jnp.max(drift_per_channel)
        histograms["drift_from_init"] = drift_per_channel

        # 4. Effective sigma estimation (fit Gaussian to each kernel)
        # Approximate by measuring spread: higher variance = larger sigma
        ax = jnp.arange(-1, 2, dtype=jnp.float32)
        xx, yy = jnp.meshgrid(ax, ax)
        squared_distances = xx**2 + yy**2

        # For each channel, compute weighted average of squared distance
        # Higher values = more spread out = larger effective sigma
        spread_per_channel = jnp.sum(
            diagonal_kernels * squared_distances[:, :, None], axis=(0, 1)
        )  # (hidden_dim,)

        # Convert to sigma estimate: for Gaussian, E[r²] = 2*sigma²
        effective_sigma = jnp.sqrt(spread_per_channel / 2)

        scalars["effective_sigma_mean"] = jnp.mean(effective_sigma)
        scalars["effective_sigma_std"] = jnp.std(effective_sigma)
        scalars["effective_sigma_min"] = jnp.min(effective_sigma)
        scalars["effective_sigma_max"] = jnp.max(effective_sigma)
        histograms["effective_sigma"] = effective_sigma

        # 5. Channel specialization metric
        # High variance in weight sums = channels are doing different things
        scalars["channel_specialization"] = jnp.var(weight_sums)

        # 6. Sign consistency (are weights all positive like a true average?)
        positive_ratio = jnp.mean((diagonal_kernels > 0).astype(jnp.float32))
        scalars["positive_weight_ratio"] = positive_ratio

        results[level_name] = {
            "scalars": scalars,
            "histograms": histograms,
            "kernels": diagonal_kernels,  # (3, 3, hidden_dim) for visualization
        }

    return results


def log_mean_conv_analysis(
    logger: Any,
    model_state: nnx.State,
    num_levels: int,
    global_step: int,
    hidden_dim: int = 32,
    log_histograms: bool = True,
    log_images: bool = True,
    prefix: str = "mean_conv_behavior",
) -> None:
    """Log mean_conv kernel analysis to TensorBoard.

    Args:
        logger: TensorBoardLogger instance
        model_state: Model state from nnx.state(model)
        num_levels: Number of pyramid levels
        global_step: Current training step
        hidden_dim: Hidden dimension
        log_histograms: Whether to log histogram data
        log_images: Whether to log kernel grid images
        prefix: Prefix for TensorBoard tags
    """
    analysis = analyze_mean_conv_kernels(model_state, num_levels, hidden_dim)

    for level_name, results in analysis.items():
        # Log scalars (fast, log every step)
        for metric, value in results["scalars"].items():
            tag = f"{prefix}/{level_name}/{metric}"
            logger.scalar(tag, float(value), step=global_step)

        if log_histograms:
            # Log histograms (slower, log less frequently)
            for metric, values in results["histograms"].items():
                tag = f"{prefix}/{level_name}/{metric}"
                logger.histogram(tag, values, step=global_step)

        if log_images:
            # Log kernel grid visualization (slowest, log rarely)
            kernels = results["kernels"]  # (3, 3, hidden_dim)

            # Reshape to grid: arrange 32 kernels as 8×4 grid
            # Each kernel is 3×3, so grid is (8*3) × (4*3) = 24×12
            hidden_dim = kernels.shape[-1]
            kernels_reshaped = kernels.reshape(3, 3, 8, 4, hidden_dim // (8 * 4))

            # This is tricky, let's do a simpler approach
            # Just reshape to (3*8, 3*4) = (24, 12) for 32 kernels
            if hidden_dim == 32:
                # Rearrange: (3, 3, 32) → (3, 3, 8, 4) → (24, 12)
                kernels_grid = kernels.reshape(3, 3, 8, 4)
                kernels_grid = kernels_grid.transpose(2, 0, 3, 1)  # (8, 3, 4, 3)
                kernels_grid = kernels_grid.reshape(24, 12)

                # Normalize to [0, 1] for visualization
                k_min = kernels_grid.min()
                k_max = kernels_grid.max()
                kernels_grid = (kernels_grid - k_min) / (k_max - k_min + 1e-8)

                tag = f"{prefix}/{level_name}/kernels_grid"
                logger.image(tag, kernels_grid[None, :, :, None], step=global_step)
