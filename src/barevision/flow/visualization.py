"""Visualization functions for embedding training diagnostics.

Generates matplotlib figures for TensorBoard logging. All functions return
RGB numpy arrays suitable for direct logging.

Design:
- Figures are large (~1600x1000 for detail figures) for clear inspection
- Color scales are explicit with colorbars or fixed ranges
- Random window/pixel selection provides variety across logging steps
"""

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from barevision.flow.training.model import Model as OpticalFlowModel
from barevision.utils.grid import WindowGrid
from barevision.utils.logging import JaxLogger

# Set non-interactive backend for headless environments
matplotlib.use("Agg")

# =============================================================================
# Constants for Figure Layouts
# =============================================================================

FIGURE_DPI = 100

# Figure sizes (width, height) in inches
FRAME_WITH_GRID_SIZE = (16, 10)  # Large overview with grid
ATTENTION_MAPS_SIZE = (20, 10)  # Multiple attention maps


# =============================================================================
# Utility Functions
# =============================================================================


def _figure_to_array(fig: Figure) -> np.ndarray:  # type: ignore[assignment]
    """Convert matplotlib figure to RGB numpy array for TensorBoard.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB image array (H, W, 3) as uint8
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Get RGBA buffer from canvas (modern matplotlib API)
    buf = fig.canvas.buffer_rgba()
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
    """Add 16x16 grid overlay to an axis showing an image.

    Args:
        ax: Matplotlib axis
        window_size: Size of windows (typically 16)
        H: Image height in pixels
        W: Image width in pixels
        highlighted_window: (row, col) of window to highlight with colored border
    """
    # Draw grid lines
    for i in range(0, H + 1, window_size):
        ax.axhline(i, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    for j in range(0, W + 1, window_size):
        ax.axvline(j, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Highlight specific window if requested
    if highlighted_window is not None:
        row, col = highlighted_window
        y0 = row * window_size
        x0 = col * window_size

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


def _select_random_pixels(
    window_size: int, num_pixels: int = 4, seed: int | None = None
) -> np.ndarray:
    """Select random pixel indices within a window.

    Args:
        window_size: Window dimension (typically 16)
        num_pixels: Number of pixels to select
        seed: Random seed for reproducibility

    Returns:
        (num_pixels,) array of flat pixel indices
    """
    rng = np.random.default_rng(seed)
    N = window_size * window_size
    return rng.choice(N, size=num_pixels, replace=False)


# =============================================================================
# Visualization Functions
# =============================================================================


def create_frame_with_grid_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    metadata: dict,
    window_size: int,
    highlighted_window: Optional[Tuple[int, int]] = None,
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
    window_indices: Optional[Tuple[int, int]] = None,
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
    logger: JaxLogger,
    model,  # OpticalFlowModel or HierarchicalEmbeddingModel
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    metadata: dict,
    step: int,
    window_size: int = 16,
    num_levels: int = 3,
    aux_data: dict | None = None,
):
    """Generate and log visualization figures for hierarchical model.

    Phase 2 (Deep Supervision):
    - Generates visualizations for ALL pyramid levels
    - Each level shows attention maps at its native resolution
    - Levels are logged separately for inspection

    Pipeline:
    1. Get pyramid from aux_data (computed during training)
    2. For each level: select random window, extract attention from aux
    3. Generate frame/grid and attention maps figures per level
    4. Log to TensorBoard with level-specific tags

    Args:
        logger: JaxLogger instance for TensorBoard logging
        model: OpticalFlowModel or HierarchicalEmbeddingModel (only used if aux_data is None)
        img1: Frame 1 (1, H, W, 3)
        img2: Frame 2 (1, H, W, 3)
        metadata: dict with video_name, frame_t, frame_tk, distance
        step: Global step for logging
        window_size: Attention window size in pixels
        num_levels: Number of pyramid levels
        aux_data: Auxiliary data from train_step containing pyramid and attention maps
    """
    import gc

    # Get pyramid from aux_data
    if aux_data is None or "model" not in aux_data:
        # Fallback: compute pyramid if aux not provided
        # Support both OpticalFlowModel and HierarchicalEmbeddingModel
        if hasattr(model, 'extract_embeddings'):
            pyramid1 = model.extract_embeddings(img1)
            pyramid2 = model.extract_embeddings(img2)
        else:
            pyramid1 = model(img1)
            pyramid2 = model(img2)
    else:
        pyramid1 = aux_data["model"]["pyramid1"]
        pyramid2 = aux_data["model"]["pyramid2"]

    # Log flow visualization if available in aux_data
    if aux_data is not None and "flow" in aux_data:
        try:
            from barevision.flow.visualization_flow import (
                flow_to_colorwheel,
                flow_to_arrows,
            )

            flow = aux_data["flow"]

            # Get coarsest level dimensions
            flow_coarse = pyramid1[-1]
            B, H_flow, W_flow, _ = flow_coarse.shape

            # Reshape flow if needed (batch size 1)
            flow_viz = flow[0] if flow.shape[0] == 1 else flow.mean(axis=0)

            # Convert to colorwheel with enhanced visibility
            # max_flow=0.3 makes small motions visible (flows >= 0.3 are fully saturated)
            flow_rgb = flow_to_colorwheel(flow_viz, max_flow=0.3)

            # Log flow colorwheel visualization
            logger.log_image("Flow/Predicted_Colorwheel", flow_rgb, step)

            # Create and log arrow visualization
            arrows_rgb = flow_to_arrows(flow_viz, max_flow=0.3, scale=2.0, grid_density=8)
            logger.log_image("Flow/Predicted_Arrows", arrows_rgb, step)
        except Exception as e:
            # Silently skip flow visualization if it fails
            pass

    # Log visualizations for each level
    for level_idx in range(num_levels):
        level_emb = pyramid1[level_idx]
        B, H_emb, W_emb, _ = level_emb.shape

        # Calculate number of windows at this level
        num_windows_h = H_emb // window_size
        num_windows_w = W_emb // window_size

        # Skip levels that are too small for even one window
        if num_windows_h == 0 or num_windows_w == 0:
            continue

        # Select random window at this level
        rng = np.random.default_rng(seed=step + level_idx * 1000)
        window_row = int(rng.integers(0, num_windows_h))
        window_col = int(rng.integers(0, num_windows_w))
        window_indices = (window_row, window_col)

        # Downscale original images to match this level's embedding dimensions
        img1_downscaled = jax.image.resize(
            img1[0], (H_emb, W_emb, 3), method="bilinear"
        )
        img2_downscaled = jax.image.resize(
            img2[0], (H_emb, W_emb, 3), method="bilinear"
        )

        # Extract attention data from aux if available
        if aux_data is not None and "loss" in aux_data:
            loss_aux = aux_data["loss"]
            # Get level-specific attention data
            self_attn_list = loss_aux.get("level_self_attention_weights", None)
            cross_attn_list = loss_aux.get("level_cross_attention_weights", None)
            
            if self_attn_list is not None and len(self_attn_list) > level_idx:
                # Extract window data using visualization_attention module
                from barevision.flow.visualization_attention import (
                    extract_window_data_for_viz,
                )
                
                viz_data = extract_window_data_for_viz(
                    self_attention_weights=self_attn_list[level_idx],
                    cross_attention_weights=cross_attn_list[level_idx],
                    self_entropy_map=loss_aux.get("level_self_entropy_maps", [None])[level_idx],
                    cross_entropy_map=loss_aux.get("level_cross_entropy_maps", [None])[level_idx],
                    window_indices=window_indices,
                    num_windows_h=num_windows_h,
                    num_windows_w=num_windows_w,
                    window_size=window_size,
                    pixel_selection_seed=step + level_idx * 1000,
                )
                
                self_attn_np = viz_data["self_attention_maps"]
                cross_attn_np = viz_data["cross_attention_maps"]
                pixel_positions_np = viz_data["pixel_positions"]
            else:
                # Fallback: skip attention maps if not available
                continue
        else:
            # No aux data available, skip this level
            continue

        # Convert JAX arrays to numpy for visualization
        img1_np = np.array(img1_downscaled)
        img2_np = np.array(img2_downscaled)

        # Extract window crop from downscaled images
        emb_h_start = window_row * window_size
        emb_w_start = window_col * window_size

        window_crop1_np = np.array(
            img1_np[
                emb_h_start : emb_h_start + window_size,
                emb_w_start : emb_w_start + window_size,
                :,
            ]
        )
        window_crop2_np = np.array(
            img2_np[
                emb_h_start : emb_h_start + window_size,
                emb_w_start : emb_w_start + window_size,
                :,
            ]
        )

        # 1. Frame with grid (showing downscaled frames with level-specific grid)
        fig_frame = create_frame_with_grid_figure(
            img1_np, img2_np, metadata, window_size, highlighted_window=window_indices
        )
        logger.log_figure(f"Level{level_idx}/Frame_Grid", fig_frame, step)

        # 2. Attention maps for selected pixels
        fig_attn = create_attention_maps_figure(
            window_crop1_np,
            window_crop2_np,
            self_attn_np,
            cross_attn_np,
            pixel_positions_np,
            window_size,
            window_indices=window_indices,
            frame_t=metadata.get("frame_t", 0),
            frame_tk=metadata.get("frame_tk", 0),
            distance=metadata.get("distance", 0),
        )
        logger.log_figure(f"Level{level_idx}/Attention_Maps", fig_attn, step)

    # Clean up pyramid references
    del pyramid1, pyramid2
    gc.collect()
