"""Visualization utilities for embedding model.

Generates figures for embedding diagnostics and per-dimension activations.
All functions return RGB numpy arrays suitable for TensorBoard logging.

Organization:
- Figure creation functions (create_*_figure): Return RGB arrays
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
PER_CHANNEL_HEATMAPS_SIZE = (12, 8)


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


def create_dimension_activations_figure(
    window_crop1: np.ndarray,
    window_crop2: np.ndarray,
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    window_size: int = 16,
    window_indices: tuple[int, int] | None = None,
    frame_t: int = 0,
    frame_tk: int = 0,
    distance: int = 0,
) -> np.ndarray:
    """Create figure showing per-dimension activation patterns.

    For each embedding dimension, show mean activation across the window.
    This visualizes "which dimensions fire" without coordinate weighting.

    Layout:
    - Left: Frame 1 and Frame 2 crops
    - Right: Bar charts of dimension activations for both frames

    Args:
        window_crop1: (16, 16, 3) image crop for frame 1
        window_crop2: (16, 16, 3) image crop for frame 2
        embeddings1: (16, 16, D) embeddings for frame 1
        embeddings2: (16, 16, D) embeddings for frame 2
        window_size: Window dimension (default 16)
        window_indices: (row, col) of window in grid (for title)
        frame_t: Frame number for crop 1
        frame_tk: Frame number for crop 2
        distance: Temporal distance between frames

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    D = embeddings1.shape[-1]

    # Build title suffix with window coordinates
    if window_indices is not None:
        row, col = window_indices
        title_suffix = f" | Window ({row}, {col})"
    else:
        title_suffix = ""

    # Compute mean activation per dimension
    mean_act1 = embeddings1.mean(axis=(0, 1))
    mean_act2 = embeddings2.mean(axis=(0, 1))

    # Create layout: 2 rows, 3 columns (frame1, frame2, activations)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=FIGURE_DPI)

    # Row 0: Frame 1
    axes[0, 0].imshow(window_crop1)
    axes[0, 0].set_title(f"Frame {frame_t}{title_suffix}", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Row 1: Frame 2
    axes[1, 0].imshow(window_crop2)
    axes[1, 0].set_title(
        f"Frame {frame_tk} (t+{distance})", fontsize=12, fontweight="bold"
    )
    axes[1, 0].axis("off")

    # Row 0: Dimension activations for frame 1
    axes[0, 1].bar(range(D), mean_act1, color="steelblue")
    axes[0, 1].set_title(f"Frame {frame_t}: Mean Activation per Dimension", fontsize=10)
    axes[0, 1].set_xlabel("Dimension")
    axes[0, 1].set_ylabel("Mean Activation")
    axes[0, 1].set_ylim(-1.0, 1.0)  # L2-normalized embeddings
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Row 1: Dimension activations for frame 2
    axes[1, 1].bar(range(D), mean_act2, color="coral")
    axes[1, 1].set_title(
        f"Frame {frame_tk}: Mean Activation per Dimension", fontsize=10
    )
    axes[1, 1].set_xlabel("Dimension")
    axes[1, 1].set_ylabel("Mean Activation")
    axes[1, 1].set_ylim(-1.0, 1.0)
    axes[1, 1].grid(axis="y", alpha=0.3)

    # Hide third column (reserved for future visualizations)
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return _figure_to_array(fig)


def create_per_channel_heatmaps_figure(
    embeddings: np.ndarray,
    window_crop: np.ndarray,
    window_size: int = 16,
    window_indices: tuple[int, int] | None = None,
    frame_label: str = "Frame",
) -> np.ndarray:
    """Create figure showing per-channel activation heatmaps.

    For each embedding dimension, show a heatmap of spatial activations
    within the window. This visualizes "where in space does each channel fire".

    Layout:
    - Left: Original window crop (spatial reference)
    - Right: Grid of D small heatmaps (one per dimension)

    Args:
        embeddings: (window_size, window_size, D) embeddings for one window
        window_crop: (window_size, window_size, 3) RGB image crop
        window_size: Window dimension (default 16)
        window_indices: (row, col) of window in grid (for title)
        frame_label: Label for the frame (e.g., "Frame 1", "Frame 42")

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    D = embeddings.shape[-1]

    # Build title suffix with window coordinates
    if window_indices is not None:
        row, col = window_indices
        title_suffix = f" | Window ({row}, {col})"
    else:
        title_suffix = ""

    # Determine grid layout for D channels
    # Aim for roughly square-ish grid
    heatmap_cols = int(np.ceil(np.sqrt(D)))
    heatmap_rows = int(np.ceil(D / heatmap_cols))

    # Create figure using GridSpec for flexible layout
    # Column 0: image (wider), Columns 1+: heatmaps
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(PER_CHANNEL_HEATMAPS_SIZE[0] + 3, PER_CHANNEL_HEATMAPS_SIZE[1]), dpi=FIGURE_DPI)
    gs = GridSpec(
        heatmap_rows, heatmap_cols + 1,
        figure=fig,
        width_ratios=[1.2] + [1.0] * heatmap_cols,
        wspace=0.3,
        hspace=0.3
    )

    # First column: show the window crop (span all rows)
    ax_image = fig.add_subplot(gs[:, 0])
    ax_image.imshow(window_crop)
    ax_image.set_title(f"{frame_label}{title_suffix}", fontsize=10, fontweight="bold")
    ax_image.axis("off")

    # Remaining columns: heatmaps for each dimension
    axes_flat = []
    for row in range(heatmap_rows):
        for col in range(1, heatmap_cols + 1):
            axes_flat.append(fig.add_subplot(gs[row, col]))

    # Plot each dimension
    for d in range(D):
        ax = axes_flat[d]
        channel_data = embeddings[:, :, d]
        
        # Show heatmap with sequential colormap (embeddings are in [0, 1])
        im = ax.imshow(
            channel_data,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest"
        )
        ax.set_title(f"Dim {d}", fontsize=8)
        ax.axis("off")

    # Hide unused subplots if D doesn't fill the grid
    for d in range(D, len(axes_flat)):
        axes_flat[d].axis("off")

    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=0.0, vmax=1.0)
        ),
        cax=cbar_ax,
        label="Activation"
    )
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{frame_label}{title_suffix} - Per-Channel Activations",
        fontsize=12,
        fontweight="bold",
        y=0.98
    )

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
    - Per-dimension activation patterns

    Args:
        logger: TensorBoard logger instance
        img1: Frame 1 (1, H, W, 3) - original RGB
        img2: Frame 2 (1, H, W, 3) - original RGB
        pyramid1: List of embedding pyramids for frame 1
        pyramid2: List of embedding pyramids for frame 2
        aux_data: Auxiliary data from training step (contains flow, confidence)
        metadata: dict with video_name, frame_t, frame_tk, distance
        step: Global step for logging
        window_size: Window size in pixels
        num_levels: Number of pyramid levels
    """
    import gc

    # Log visualizations for each pyramid level
    for level_idx in range(num_levels):
        level_emb1 = pyramid1[level_idx]
        level_emb2 = pyramid2[level_idx]
        B, H_emb, W_emb, _ = level_emb1.shape

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

        # Extract window crop from embeddings
        emb_h_start = window_row * window_size
        emb_w_start = window_col * window_size
        window_emb1 = np.array(
            level_emb1[0, emb_h_start : emb_h_start + window_size, emb_w_start : emb_w_start + window_size, :]
        )
        window_emb2 = np.array(
            level_emb2[0, emb_h_start : emb_h_start + window_size, emb_w_start : emb_w_start + window_size, :]
        )

        # Extract window crop from downscaled images
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

        # 2. Per-dimension activation patterns (bar charts)
        fig_dim = create_dimension_activations_figure(
            window_crop1=window_crop1,
            window_crop2=window_crop2,
            embeddings1=window_emb1,
            embeddings2=window_emb2,
            window_size=window_size,
            window_indices=window_indices,
            frame_t=metadata.get("frame_t", 0),
            frame_tk=metadata.get("frame_tk", 0),
            distance=metadata.get("distance", 0),
        )
        logger.log_figure(f"Level{level_idx}/Dimension_Activations", fig_dim, step)

        # 3. Per-channel spatial heatmaps with reference image
        frame_t_label = f"Frame {metadata.get('frame_t', 0)}"
        frame_tk_label = f"Frame {metadata.get('frame_tk', 0)} (t+{metadata.get('distance', 0)})"
        
        fig_heatmap1 = create_per_channel_heatmaps_figure(
            embeddings=window_emb1,
            window_crop=window_crop1,
            window_size=window_size,
            window_indices=window_indices,
            frame_label=frame_t_label,
        )
        logger.log_figure(f"Level{level_idx}/Per_Channel_Heatmaps_F1", fig_heatmap1, step)

        fig_heatmap2 = create_per_channel_heatmaps_figure(
            embeddings=window_emb2,
            window_crop=window_crop2,
            window_size=window_size,
            window_indices=window_indices,
            frame_label=frame_tk_label,
        )
        logger.log_figure(f"Level{level_idx}/Per_Channel_Heatmaps_F2", fig_heatmap2, step)

    # Clean up
    gc.collect()



