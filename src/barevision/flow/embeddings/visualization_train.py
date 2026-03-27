"""Training visualizations for embeddings with spatial variance loss.

Generates figures for embedding statistics and attention variance maps.
All functions return RGB numpy arrays suitable for TensorBoard logging.
"""

from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from barevision.flow.embeddings.visualization import (
    create_attention_maps_figure,
    create_frame_with_grid_figure,
)


def log_visualizations(
    logger,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    aux_data: Dict,
    metadata: dict,
    step: int,
    window_size: int = 16,
    num_levels: int = 3,
):
    """Generate and log visualization figures for embeddings training.

    For each pyramid level:
    - Frame grid overlay (showing window boundaries)
    - Variance maps (spatial variance of attention)
    - Attention maps for selected pixels

    Args:
        logger: TensorBoard logger instance
        img1: Frame 1 (1, H, W, 3) - original RGB
        img2: Frame 2 (1, H, W, 3) - original RGB
        pyramid1: List of embedding pyramids for frame 1
        pyramid2: List of embedding pyramids for frame 2
        aux_data: Auxiliary data from training step (contains variance maps, attention weights)
        metadata: dict with video_name, frame_t, frame_tk, distance
        step: Global step for logging
        window_size: Attention window size in pixels
        num_levels: Number of pyramid levels
    """
    import gc

    # Log visualizations for each pyramid level
    for level_idx in range(num_levels):
        level_emb = pyramid1[level_idx]
        B, H_emb, W_emb, _ = level_emb.shape

        # Crop to grid-aligned dimensions
        crop_h = (H_emb // window_size) * window_size
        crop_w = (W_emb // window_size) * window_size

        # Calculate number of windows at this level (after cropping)
        num_windows_h = crop_h // window_size
        num_windows_w = crop_w // window_size

        # Skip levels that are too small
        if num_windows_h == 0 or num_windows_w == 0:
            continue

        # Select random window at this level
        rng = np.random.default_rng(seed=step + level_idx * 1000)
        window_row = int(rng.integers(0, num_windows_h))
        window_col = int(rng.integers(0, num_windows_w))
        window_indices = (window_row, window_col)

        # Downscale original images to match CROPPED dimensions
        import jax

        img1_downscaled = jax.image.resize(
            img1[0], (crop_h, crop_w, 3), method="bilinear"
        )
        img2_downscaled = jax.image.resize(
            img2[0], (crop_h, crop_w, 3), method="bilinear"
        )

        # Extract attention data from aux if available
        self_attention_weights = aux_data["level_self_attention_weights"][level_idx]
        cross_attention_weights = aux_data["level_cross_attention_weights"][level_idx]

        # Extract window-specific attention data
        viz_data = _extract_window_data_for_viz(
            self_attention_weights=self_attention_weights,
            cross_attention_weights=cross_attention_weights,
            window_indices=window_indices,
            num_windows_h=num_windows_h,
            num_windows_w=num_windows_w,
            window_size=window_size,
            pixel_selection_seed=step + level_idx * 1000,
        )

        # Extract window crop from downscaled images
        emb_h_start = window_row * window_size
        emb_w_start = window_col * window_size
        window_crop1 = np.array(
            img1_downscaled[
                emb_h_start : emb_h_start + window_size,
                emb_w_start : emb_w_start + window_size,
                :,
            ]
        )
        window_crop2 = np.array(
            img2_downscaled[
                emb_h_start : emb_h_start + window_size,
                emb_w_start : emb_w_start + window_size,
                :,
            ]
        )

        # 1. Frame with grid (showing downscaled frames with level-specific grid)
        fig_frame = create_frame_with_grid_figure(
            np.array(img1_downscaled),
            np.array(img2_downscaled),
            metadata,
            window_size,
            highlighted_window=window_indices,
        )
        logger.log_figure(f"Level{level_idx}/Frame_Grid", fig_frame, step)

        # 2. Attention maps for selected pixels
        fig_attn = create_attention_maps_figure(
            window_crop1=window_crop1,
            window_crop2=window_crop2,
            self_attn_maps=viz_data["self_attention_maps"],
            cross_attn_maps=viz_data["cross_attention_maps"],
            pixel_positions=viz_data["pixel_positions"],
            window_size=window_size,
            window_indices=window_indices,
            frame_t=metadata.get("frame_t", 0),
            frame_tk=metadata.get("frame_tk", 0),
            distance=metadata.get("distance", 0),
        )
        logger.log_figure(f"Level{level_idx}/Attention_Maps", fig_attn, step)

        # 3. Log variance maps (as heatmaps)
        self_var_maps = aux_data["level_self_variance_maps"][level_idx]
        cross_var_maps = aux_data["level_cross_variance_maps"][level_idx]

        # Visualize variance maps as heatmaps
        self_var_viz = _variance_map_to_heatmap(self_var_maps[0])  # First batch item
        cross_var_viz = _variance_map_to_heatmap(cross_var_maps[0])

        logger.log_image(f"Level{level_idx}/self_variance_map", self_var_viz, step)
        logger.log_image(f"Level{level_idx}/cross_variance_map", cross_var_viz, step)

    # Clean up
    gc.collect()


def _variance_map_to_heatmap(variance_map: np.ndarray) -> np.ndarray:
    """Convert variance map to heatmap visualization.

    Args:
        variance_map: (H, W) spatial variance values

    Returns:
        (H, W, 3) RGB heatmap
    """
    # Normalize to [0, 1] for visualization
    var_min = variance_map.min()
    var_max = variance_map.max()

    if var_max - var_min > 1e-6:
        normalized = (variance_map - var_min) / (var_max - var_min)
    else:
        normalized = np.zeros_like(variance_map)

    # Flip vertically to match image coordinate system (y=0 at top)
    # The variance map comes from array indexing where row 0 is top,
    # but we need to flip for correct visual alignment with frames
    normalized = np.flipud(normalized)

    # Use viridis colormap (matplotlib)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(normalized, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set_title("Spatial Variance (lower = more concentrated)", fontsize=12)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # Convert to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    buffer = np.frombuffer(buf, dtype=np.uint8)
    image_array = buffer.reshape(height, width, 4)[:, :, :3]
    plt.close(fig)

    return image_array


def _extract_window_data_for_viz(
    self_attention_weights: jnp.ndarray,
    cross_attention_weights: jnp.ndarray,
    window_indices: tuple[int, int],
    num_windows_h: int,
    num_windows_w: int,
    window_size: int = 16,
    pixel_selection_seed: int = 0,
    num_pixels: int = 4,
) -> dict:
    """Extract all attention data needed for visualizing a specific window.

    This is adapted from barevision.flow.visualization_attention to work with
    embeddings-only training.

    Args:
        self_attention_weights: (num_windows, window_size^2, window_size^2) self-attention per window
        cross_attention_weights: (num_windows, window_size^2, window_size^2) cross-attention per window
        window_indices: (row, col) of window to visualize
        num_windows_h: Number of windows vertically at this level
        num_windows_w: Number of windows horizontally at this level
        window_size: Size of each window in pixels (default 16)
        pixel_selection_seed: Random seed for pixel selection
        num_pixels: Number of pixels to show within window (default 4)

    Returns:
        Dictionary containing:
            - self_attention_maps: (num_pixels, window_size, window_size)
            - cross_attention_maps: (num_pixels, window_size, window_size)
            - pixel_positions: (num_pixels, 2) (y, x) positions
    """
    # Select random pixels within the window
    import jax

    pixel_indices = _select_random_pixels(
        window_size=window_size,
        num_pixels=num_pixels,
        seed=pixel_selection_seed,
    )

    # Calculate flat window index
    window_idx = window_indices[0] * num_windows_w + window_indices[1]

    # Extract this window's attention (shape: window_size^2 x window_size^2)
    window_self_attn = self_attention_weights[window_idx]
    window_cross_attn = cross_attention_weights[window_idx]

    # Extract pixel-specific attention maps
    self_attn_maps = _extract_pixel_attention_maps(
        window_self_attn,
        pixel_indices,
        window_size,
    )
    cross_attn_maps = _extract_pixel_attention_maps(
        window_cross_attn,
        pixel_indices,
        window_size,
    )

    # Compute pixel positions
    pixel_positions = _compute_pixel_positions(pixel_indices, window_size)

    return {
        "self_attention_maps": np.array(self_attn_maps),
        "cross_attention_maps": np.array(cross_attn_maps),
        "pixel_positions": np.array(pixel_positions),
    }


def _select_random_pixels(
    window_size: int,
    num_pixels: int = 4,
    seed: int = 0,
) -> jnp.ndarray:
    """Select random pixel indices within a window."""
    N = window_size * window_size
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, N, shape=(num_pixels,), replace=False)


def _extract_pixel_attention_maps(
    attention_weights: jnp.ndarray,
    pixel_indices: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Extract attention maps for specific pixels within a window."""
    if attention_weights.ndim == 2:
        # No batch dimension
        N, _ = attention_weights.shape
        selected_attn = attention_weights[pixel_indices, :]  # (num_pixels, N)
        selected_attn = selected_attn.reshape(-1, window_size, window_size)
    else:
        # Has batch dimension
        B, N, _ = attention_weights.shape
        selected_attn = attention_weights[:, pixel_indices, :]  # (B, num_pixels, N)
        selected_attn = selected_attn.reshape(B, -1, window_size, window_size)

    return selected_attn


def _compute_pixel_positions(
    pixel_indices: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Compute (y, x) positions for pixel indices within a window."""
    pixel_y = pixel_indices // window_size
    pixel_x = pixel_indices % window_size
    return jnp.stack([pixel_y, pixel_x], axis=-1)
