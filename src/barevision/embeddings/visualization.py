"""Visualization utilities for embedding model.

Generates figures for embedding statistics and attention maps.
All functions return RGB numpy arrays suitable for TensorBoard logging.

Organization:
- Figure creation functions (create_*_figure): Return RGB arrays
- Data extraction helpers (_extract_window_data_for_viz, etc.)
- Training orchestration (log_visualizations): Logs to TensorBoard
"""

from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Set non-interactive backend for headless environments
matplotlib.use("Agg")

# Figure sizes (width, height) in inches
FIGURE_DPI = 100
FRAME_WITH_GRID_SIZE = (16, 10)
ATTENTION_MAPS_SIZE = (20, 10)


def _figure_to_array(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to RGB numpy array for TensorBoard.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB image array (H, W, 3) as uint8
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Get RGBA buffer from canvas (modern matplotlib API)
    buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
    buffer = np.frombuffer(buf, dtype=np.uint8)

    # RGBA to RGB - skip alpha channel (4th byte of each pixel)
    image_array = buffer.reshape(height, width, 4)[:, :, :3]

    plt.close(fig)

    return image_array


def _add_grid_overlay(
    ax: Axes,
    window_size: int,
    H: int,
    W: int,
    highlighted_window: tuple[int, int] | None = None,
):
    """Add grid overlay to an axis showing an image.

    Args:
        ax: Matplotlib axis
        window_size: Size of windows (typically 16)
        H: Image height in pixels (already cropped to grid-aligned)
        W: Image width in pixels (already cropped to grid-aligned)
        highlighted_window: (row, col) of window to highlight with colored border
    """
    # Draw grid lines at pixel EDGES, not centers.
    # When imshow displays an H×W image, pixels are centered at (0, 1, 2, ... H-1)
    # So pixel edges are at (-0.5, 0.5, 1.5, ... H-0.5)
    # Grid lines should be at: -0.5, 15.5, 31.5, 47.5, 63.5 for 16-pixel windows
    # Use bright white for better visibility
    for i in range(0, H, window_size):
        line_pos = i - 0.5  # Edge of pixel i (not center)
        ax.axhline(line_pos, color="white", linestyle="-", linewidth=1, alpha=0.8)

    # Right and bottom borders
    ax.axhline(H - 0.5, color="white", linestyle="-", linewidth=1, alpha=0.8)

    for j in range(0, W, window_size):
        line_pos = j - 0.5  # Edge of pixel j (not center)
        ax.axvline(line_pos, color="white", linestyle="-", linewidth=1, alpha=0.8)

    # Right and bottom borders
    ax.axvline(W - 0.5, color="white", linestyle="-", linewidth=1, alpha=0.8)

    # Highlight specific window if requested
    if highlighted_window is not None:
        row, col = highlighted_window
        y0 = row * window_size - 0.5  # Edge alignment
        x0 = col * window_size - 0.5  # Edge alignment

        rect = Rectangle(
            (x0, y0),
            window_size,
            window_size,
            linewidth=3,
            edgecolor="lime",
            facecolor="none",
            label="Selected window",
        )
        ax.add_patch(rect)


def create_frame_with_grid_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    metadata: dict,
    window_size: int,
    highlighted_window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Create figure showing both frames with grid overlay.

    Args:
        img1: (H, W, 3) RGB frame 1
        img2: (H, W, 3) RGB frame 2
        metadata: dict with video_name, frame_t, frame_tk, distance
        window_size: Size of grid cells
        highlighted_window: (row, col) of window to highlight

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    H, W = img1.shape[:2]
    video_name = metadata.get("video_name", "unknown")
    frame_t = metadata.get("frame_t", 0)
    frame_tk = metadata.get("frame_tk", 0)
    distance = metadata.get("distance", 0)

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=FRAME_WITH_GRID_SIZE, dpi=FIGURE_DPI)

    # Frame 1 (left)
    axes[0].imshow(img1)
    _add_grid_overlay(axes[0], window_size, H, W, highlighted_window)
    title1 = f"Frame {frame_t} | Video: {video_name}"
    row, col = highlighted_window if highlighted_window else (None, None)
    if highlighted_window is not None:
        title1 += f" | Window ({row}, {col})"
    axes[0].set_title(title1, fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Frame 2 (right)
    axes[1].imshow(img2)
    _add_grid_overlay(axes[1], window_size, H, W, highlighted_window)
    title2 = f"Frame {frame_tk} (t+{distance}) | Video: {video_name}"
    if highlighted_window is not None:
        title2 += f" | Window ({row}, {col})"
    axes[1].set_title(title2, fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    return _figure_to_array(fig)


def create_attention_maps_figure(
    window_crop1: np.ndarray,
    window_crop2: np.ndarray,
    self_attn_maps: np.ndarray,
    cross_attn_maps: np.ndarray,
    pixel_positions: np.ndarray,
    window_size: int = 16,
    window_indices: tuple[int, int] | None = None,
    frame_t: int = 0,
    frame_tk: int = 0,
    distance: int = 0,
) -> np.ndarray:
    """Create figure showing attention maps for selected query pixels.

    Layout:
    - Left column: Frame 1 and Frame 2 crops with pixel markers
    - Right columns: One column per query pixel showing self and cross attention

    Args:
        window_crop1: (16, 16, 3) image crop for frame 1
        window_crop2: (16, 16, 3) image crop for frame 2
        self_attn_maps: (N, 16, 16) self-attention weights for N query pixels
        cross_attn_maps: (N, 16, 16) cross-attention weights for N query pixels
        pixel_positions: (N, 2) (y, x) positions of query pixels
        window_size: Window dimension (default 16)
        window_indices: (row, col) of window in grid (for title)
        frame_t: Frame number for crop 1
        frame_tk: Frame number for crop 2
        distance: Temporal distance between frames

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    n_pixels = self_attn_maps.shape[0]

    # Create layout: 2 rows, (n_pixels + 1) columns
    fig, axes = plt.subplots(
        2, n_pixels + 1, figsize=ATTENTION_MAPS_SIZE, dpi=FIGURE_DPI
    )

    # Ensure axes is 2D array even for single pixel
    if n_pixels == 1:
        axes = axes.reshape(2, 1)

    # Build title suffix with window coordinates
    if window_indices is not None:
        row, col = window_indices
        title_suffix = f" | Window ({row}, {col})"
    else:
        title_suffix = ""

    # Row 0: Frame 1 crop
    axes[0, 0].imshow(window_crop1)
    axes[0, 0].set_title(
        f"Frame {frame_t}{title_suffix}", fontsize=12, fontweight="bold"
    )
    axes[0, 0].axis("off")

    # Row 1: Frame 2 crop
    axes[1, 0].imshow(window_crop2)
    axes[1, 0].set_title(
        f"Frame {frame_tk} (t+{distance})", fontsize=12, fontweight="bold"
    )
    axes[1, 0].axis("off")

    # Define colors for pixel markers
    colors = ["red", "blue", "green", "orange"]

    # Mark pixel positions on Frame 1 (row 0)
    for i, (y, x) in enumerate(pixel_positions):
        color = colors[i % len(colors)]

        axes[0, 0].scatter(
            [x + 0.5],
            [y + 0.5],
            c=color,
            s=100,
            marker="x",
            linewidths=3,
            label=f"Pixel {i}" if i == 0 else "",
        )
    axes[0, 0].legend(loc="upper right", fontsize=9)

    # Auto-scale across ALL attention maps for better contrast
    all_attn_values = np.concatenate([self_attn_maps.ravel(), cross_attn_maps.ravel()])
    attn_min = float(all_attn_values.min())
    attn_max = float(all_attn_values.max())

    # Apply minimum scale floor to avoid over-amplifying noise
    SCALE_FLOOR = 0.01
    if attn_max - attn_min < SCALE_FLOOR:
        attn_max = attn_min + SCALE_FLOOR

    # For better spatial pattern visibility, use percentile-based scaling
    # This enhances contrast by ignoring extreme outliers
    p5 = float(np.percentile(all_attn_values, 5))
    p95 = float(np.percentile(all_attn_values, 95))
    if p95 - p5 < SCALE_FLOOR:
        p95 = p5 + SCALE_FLOOR

    # Columns 1..n: Attention maps with AUTO-SCALED colors
    for i in range(n_pixels):
        col = i + 1
        y, x = pixel_positions[i]
        color = colors[i % 4]

        # Row 0: Self-attention (use percentile scaling for better contrast)
        im_self = axes[0, col].imshow(
            self_attn_maps[i], cmap="viridis", vmin=p5, vmax=p95
        )
        axes[0, col].set_title(
            f"Self-Attn (Pixel {i})\nPos: ({y}, {x})", fontsize=10, fontweight="bold"
        )
        axes[0, col].axis("off")
        plt.colorbar(im_self, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Mark source pixel position
        axes[0, col].scatter(
            [x + 0.5], [y + 0.5], c=color, s=80, marker="x", linewidths=2
        )

        # Row 1: Cross-attention (use percentile scaling for better contrast)
        im_cross = axes[1, col].imshow(
            cross_attn_maps[i], cmap="viridis", vmin=p5, vmax=p95
        )
        axes[1, col].set_title(
            f"Cross-Attn (Pixel {i})", fontsize=10, fontweight="bold"
        )
        axes[1, col].axis("off")
        plt.colorbar(im_cross, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Mark source pixel position (where query comes from)
        axes[1, col].scatter(
            [x + 0.5], [y + 0.5], c=color, s=80, marker="x", linewidths=2
        )

        # Mark best match position in frame 2
        best_match = np.unravel_index(
            np.argmax(cross_attn_maps[i]), cross_attn_maps[i].shape
        )
        axes[1, col].scatter(
            [best_match[1] + 0.5],
            [best_match[0] + 0.5],
            c=color,  # Same color as source pixel
            s=60,
            marker="+",
            linewidths=2,
            label="Best match" if i == 0 else "",
        )

    # Mark best match positions on Frame 2 crop (row 1, column 0)
    for i in range(n_pixels):
        color = colors[i % 4]
        best_match = np.unravel_index(
            np.argmax(cross_attn_maps[i]), cross_attn_maps[i].shape
        )
        axes[1, 0].scatter(
            [best_match[1] + 0.5],
            [best_match[0] + 0.5],
            c=color,
            s=80,
            marker="+",
            linewidths=2,
            label=f"Match {i}" if i == 0 else "",
        )
    axes[1, 0].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return _figure_to_array(fig)


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

    # Clean up
    gc.collect()


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
