"""Visualization functions for embedding training diagnostics.

Generates matplotlib figures for TensorBoard logging. All functions return
RGB numpy arrays suitable for direct logging.

Design:
- Figures are large (~1600x1000 for detail figures) for clear inspection
- Color scales are explicit with colorbars or fixed ranges
- Random window/pixel selection provides variety across logging steps
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from barevision.embeddings.model import AttentionMaps, SimpleEmbeddingModel
from barevision.embeddings.settings import Settings
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
LOSS_HEATMAP_SIZE = (16, 8)  # Two columns: frame + heatmap
ATTENTION_MAPS_SIZE = (20, 10)  # Multiple attention maps
SIMILARITY_MATRIX_SIZE = (16, 8)  # 256x256 matrices


# =============================================================================
# Utility Functions
# =============================================================================


def _figure_to_array(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Convert matplotlib figure to RGB numpy array for TensorBoard.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB image array (H, W, 3) as uint8
    """
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Get ARGB buffer from canvas
    buf = fig.canvas.tostring_argb()
    buffer = np.frombuffer(buf, dtype=np.uint8)

    # ARGB to RGB - skip alpha channel (first byte of each pixel)
    image_array = buffer.reshape(height, width, 4)[:, :, 1:]

    plt.close(fig)

    return image_array


def _add_grid_overlay(
    ax: plt.Axes,
    window_size: int,
    H: int,
    W: int,
    highlighted_window: Optional[Tuple[int, int]] = None,
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

        rect = plt.Rectangle(
            (x0, y0),
            window_size,
            window_size,
            fill=False,
            color="lime",
            linewidth=3,
            alpha=0.8,
        )
        ax.add_patch(rect)


def _select_random_pixels(
    n_pixels: int, window_size: int = 16, seed: Optional[int] = None
) -> List[int]:
    """Select random pixel indices within a window.

    Args:
        n_pixels: Number of pixels to select
        window_size: Window dimension (default 16)
        seed: Random seed for reproducibility

    Returns:
        List of pixel indices in [0, window_size²)
    """
    N = window_size * window_size
    if n_pixels > N:
        raise ValueError(f"Cannot select {n_pixels} pixels from {N} total")

    rng = np.random.default_rng(seed)
    indices = rng.choice(N, size=n_pixels, replace=False)
    return sorted(indices.tolist())


# =============================================================================
# Visualization Figure Functions
# =============================================================================


def create_frame_with_grid_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    metadata: dict,
    window_size: int,
    highlighted_window: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Display both frames with 16x16 grid overlay and highlighted window.
    
    Shows frame 1 and frame 2 side by side with grid overlays and frame info.

    Args:
        img1: (H, W, 3) RGB image, values in [0, 1] - first frame
        img2: (H, W, 3) RGB image, values in [0, 1] - second frame
        metadata: dict with video_name, frame_t, frame_tk, distance
        window_size: Size of grid cells (typically 16)
        highlighted_window: (row, col) of window to highlight with colored border

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    H, W = img1.shape[:2]
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=FIGURE_DPI)
    
    # Extract frame info from metadata
    video_name = metadata.get("video_name", "unknown")
    frame_t = metadata.get("frame_t", 0)
    frame_tk = metadata.get("frame_tk", 0)
    distance = metadata.get("distance", 0)
    
    # Frame 1 (left)
    axes[0].imshow(img1)
    _add_grid_overlay(axes[0], window_size, H, W, highlighted_window)
    title1 = f"Frame {frame_t} | Video: {video_name}"
    if highlighted_window is not None:
        row, col = highlighted_window
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


def create_loss_heatmap_figures(
    img: np.ndarray,
    self_loss: np.ndarray,
    cross_loss: np.ndarray,
    window_size: int,
    highlighted_window: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create loss heatmap visualizations for self and cross entropy.

    Args:
        img: (H, W, 3) RGB image
        self_loss: (H, W) per-pixel self-attention loss
        cross_loss: (H, W) per-pixel cross-attention loss
        window_size: Size of grid cells
        highlighted_window: (row, col) of window to highlight

    Returns:
        Tuple of (self_entropy_fig, cross_entropy_fig) as RGB uint8 arrays
    """
    H, W = img.shape[:2]

    # Ensure loss maps match image dimensions
    assert self_loss.shape == (H, W), f"self_loss shape {self_loss.shape} != ({H}, {W})"
    assert cross_loss.shape == (
        H,
        W,
    ), f"cross_loss shape {cross_loss.shape} != ({H}, {W})"

    # === Self Entropy Figure ===
    fig_self, axes_self = plt.subplots(1, 2, figsize=LOSS_HEATMAP_SIZE, dpi=FIGURE_DPI)

    # Left: Frame with grid
    axes_self[0].imshow(img)
    _add_grid_overlay(axes_self[0], window_size, H, W, highlighted_window)
    axes_self[0].set_title("Frame with Grid", fontsize=12, fontweight="bold")
    axes_self[0].axis("off")

    # Right: Self-entropy heatmap
    vmin_self = float(np.min(self_loss))
    vmax_self = float(np.max(self_loss))
    im_self = axes_self[1].imshow(
        self_loss, cmap="YlOrRd", vmin=vmin_self, vmax=vmax_self
    )
    _add_grid_overlay(axes_self[1], window_size, H, W, highlighted_window)

    mean_self = float(np.mean(self_loss))
    axes_self[1].set_title(
        f"Self-Entropy Loss\nMean: {mean_self:.4f} | Range: [{vmin_self:.4f}, {vmax_self:.4f}]",
        fontsize=12,
        fontweight="bold",
    )
    axes_self[1].axis("off")
    plt.colorbar(im_self, ax=axes_self[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_self_array = _figure_to_array(fig_self)

    # === Cross Entropy Figure ===
    fig_cross, axes_cross = plt.subplots(
        1, 2, figsize=LOSS_HEATMAP_SIZE, dpi=FIGURE_DPI
    )

    # Left: Frame with grid
    axes_cross[0].imshow(img)
    _add_grid_overlay(axes_cross[0], window_size, H, W, highlighted_window)
    axes_cross[0].set_title("Frame with Grid", fontsize=12, fontweight="bold")
    axes_cross[0].axis("off")

    # Right: Cross-entropy heatmap
    vmin_cross = float(np.min(cross_loss))
    vmax_cross = float(np.max(cross_loss))
    im_cross = axes_cross[1].imshow(
        cross_loss, cmap="YlOrRd", vmin=vmin_cross, vmax=vmax_cross
    )
    _add_grid_overlay(axes_cross[1], window_size, H, W, highlighted_window)

    mean_cross = float(np.mean(cross_loss))
    axes_cross[1].set_title(
        f"Cross-Entropy Loss\nMean: {mean_cross:.4f} | Range: [{vmin_cross:.4f}, {vmax_cross:.4f}]",
        fontsize=12,
        fontweight="bold",
    )
    axes_cross[1].axis("off")
    plt.colorbar(im_cross, ax=axes_cross[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_cross_array = _figure_to_array(fig_cross)

    return fig_self_array, fig_cross_array


def create_attention_maps_figure(
    window_crop: np.ndarray,
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
    """Create figure showing attention maps for selected pixels.

    Args:
        window_crop: (16, 16, 3) image crop for frame 1
        window_crop2: (16, 16, 3) image crop for frame 2
        self_attn_maps: (N, 16, 16) self-attention weights for N pixels
        cross_attn_maps: (N, 16, 16) cross-attention weights for N pixels
        pixel_positions: (N, 2) (y, x) positions of queried pixels
        window_size: Window dimension (default 16)
        window_indices: (row, col) of window in grid (for title)
        frame_t: Frame number for crop 1
        frame_tk: Frame number for crop 2
        distance: Temporal distance between frames

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    n_pixels = len(pixel_positions)

    # Layout: 2 rows × (n_pixels + 1) columns
    # Column 0: Both frame crops stacked vertically
    # Columns 1..n: Attention maps for each pixel
    n_cols = n_pixels + 1
    fig, axes = plt.subplots(2, n_cols, figsize=ATTENTION_MAPS_SIZE, dpi=FIGURE_DPI)
    
    # Build title with window coordinates if provided
    if window_indices is not None:
        row, col = window_indices
        title_suffix = f" | Window ({row}, {col})"
    else:
        title_suffix = ""

    # Ensure axes is 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    elif not isinstance(axes, np.ndarray):
        axes = np.array(axes).reshape(2, n_cols)

    # Column 0: Show BOTH frame crops stacked vertically
    # Row 0, Col 0: Frame 1 crop
    axes[0, 0].imshow(window_crop)
    axes[0, 0].set_title(f"Frame {frame_t}{title_suffix}", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
    
    # Row 1, Col 0: Frame 2 crop
    axes[1, 0].imshow(window_crop2)
    axes[1, 0].set_title(f"Frame {frame_tk} (t+{distance})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Mark pixel positions on BOTH crops
    for i, (y, x) in enumerate(pixel_positions):
        colors = ["red", "blue", "green", "orange"]
        color = colors[i % len(colors)]
        
        # Mark on Frame 1 (row 0)
        axes[0, 0].scatter(
            [x + 0.5],
            [y + 0.5],
            c=color,
            s=100,
            marker="x",
            linewidths=3,
            label=f"Pixel {i}" if i == 0 else "",
        )
        # Mark on Frame 2 (row 1)
        axes[1, 0].scatter(
            [x + 0.5],
            [y + 0.5],
            c=color,
            s=100,
            marker="x",
            linewidths=3,
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
            f"Cross-Attn (Pixel {i})\nPos: ({y}, {x})", fontsize=10, fontweight="bold"
        )
        axes[1, col].axis("off")
        plt.colorbar(im_cross, ax=axes[1, col], fraction=0.046, pad=0.04)

        # Mark source pixel position
        axes[1, col].scatter(
            [x + 0.5], [y + 0.5], c=color, s=80, marker="x", linewidths=2
        )

    plt.tight_layout()
    return _figure_to_array(fig)


def create_similarity_matrix_figure(
    similarity_matrix: np.ndarray,
    attention_weights: np.ndarray,
    window_size: int = 16,
    window_indices: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Create figure showing similarity matrix and attention weights as 256x256 images.

    Args:
        similarity_matrix: (256, 256) raw dot-product similarities
        attention_weights: (256, 256) softmax attention weights
        window_size: Window dimension (for title info)
        window_indices: (row, col) of window in grid (for title)

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    assert similarity_matrix.shape == (
        256,
        256,
    ), f"Expected (256, 256), got {similarity_matrix.shape}"
    assert attention_weights.shape == (
        256,
        256,
    ), f"Expected (256, 256), got {attention_weights.shape}"
    
    # Build title suffix with window coordinates
    if window_indices is not None:
        row, col = window_indices
        title_suffix = f" | Window ({row}, {col})"
    else:
        title_suffix = ""

    fig, axes = plt.subplots(1, 2, figsize=SIMILARITY_MATRIX_SIZE, dpi=FIGURE_DPI)

    # Left: Similarity matrix
    vmin_sim = float(np.min(similarity_matrix))
    vmax_sim = float(np.max(similarity_matrix))
    im_sim = axes[0].imshow(
        similarity_matrix, cmap="coolwarm", vmin=vmin_sim, vmax=vmax_sim
    )
    axes[0].set_title(
        f"Similarity Matrix (256×256){title_suffix}\nRange: [{vmin_sim:.3f}, {vmax_sim:.3f}]",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Key Index")
    axes[0].set_ylabel("Query Index")
    plt.colorbar(im_sim, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: Attention weights
    vmin_attn = 0.0
    vmax_attn = 1.0
    im_attn = axes[1].imshow(
        attention_weights, cmap="viridis", vmin=vmin_attn, vmax=vmax_attn
    )

    # Show average attention sparsity
    avg_max_attn = float(np.mean(np.max(attention_weights, axis=-1)))
    axes[1].set_title(
        f"Attention Weights (256×{256})\nAvg Max: {avg_max_attn:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Key Index")
    axes[1].set_ylabel("Query Index")
    plt.colorbar(im_attn, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return _figure_to_array(fig)


def create_entropy_maps_figure(
    window_crop1: np.ndarray,
    window_crop2: np.ndarray,
    self_entropy: np.ndarray,
    cross_entropy: np.ndarray,
    window_size: int = 16,
    window_indices: Optional[Tuple[int, int]] = None,
    frame_t: int = 0,
    frame_tk: int = 0,
    distance: int = 0,
) -> np.ndarray:
    """Create figure showing per-pixel entropy maps with frame crops.
    
    Layout: 2×2 grid showing both frames and their entropy maps.
    This shows the entropy at EVERY pixel position (not specific query pixels).

    Args:
        window_crop1: (16, 16, 3) image crop for frame 1
        window_crop2: (16, 16, 3) image crop for frame 2
        self_entropy: (16, 16) per-pixel self-attention entropy
        cross_entropy: (16, 16) per-pixel cross-attention entropy
        window_size: Window dimension (default 16)
        window_indices: (row, col) of window in grid (for title)
        frame_t: Frame number for crop 1
        frame_tk: Frame number for crop 2
        distance: Temporal distance between frames

    Returns:
        RGB image array (H_fig, W_fig, 3) as uint8
    """
    # Build title with window coordinates
    if window_indices is not None:
        row, col = window_indices
        title_suffix = f" | Window ({row}, {col})"
    else:
        title_suffix = ""
    
    # Create 2×2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=FIGURE_DPI)

    # Top-Left: Frame 1 crop
    axes[0, 0].imshow(window_crop1)
    axes[0, 0].set_title(f"Frame {frame_t}{title_suffix}", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Top-Right: Self-entropy heatmap
    vmin_self = float(np.min(self_entropy))
    vmax_self = float(np.max(self_entropy))
    im_self = axes[0, 1].imshow(
        self_entropy, cmap="viridis", vmin=vmin_self, vmax=vmax_self
    )
    mean_self = float(np.mean(self_entropy))
    axes[0, 1].set_title(
        f"Self-Attention Entropy\nMean: {mean_self:.4f} | Range: [{vmin_self:.4f}, {vmax_self:.4f}]",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 1].set_xlabel("X Position")
    axes[0, 1].set_ylabel("Y Position")
    plt.colorbar(im_self, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Bottom-Left: Frame 2 crop
    axes[1, 0].imshow(window_crop2)
    axes[1, 0].set_title(f"Frame {frame_tk} (t+{distance})", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Bottom-Right: Cross-entropy heatmap
    vmin_cross = float(np.min(cross_entropy))
    vmax_cross = float(np.max(cross_entropy))
    im_cross = axes[1, 1].imshow(
        cross_entropy, cmap="viridis", vmin=vmin_cross, vmax=vmax_cross
    )
    mean_cross = float(np.mean(cross_entropy))
    axes[1, 1].set_title(
        f"Cross-Attention Entropy\nMean: {mean_cross:.4f} | Range: [{vmin_cross:.4f}, {vmax_cross:.4f}]",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("X Position")
    axes[1, 1].set_ylabel("Y Position")
    plt.colorbar(im_cross, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return _figure_to_array(fig)


def log_visualizations(
    logger: JaxLogger,
    model: SimpleEmbeddingModel,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    metadata: dict,
    step: int,
    settings: Settings,
):
    """Generate and log all visualization figures.

    This function orchestrates the entire visualization pipeline:
    1. Select random window for detailed analysis
    2. Call model.compute_attention_maps() to get aux data
    3. Generate all figures
    4. Log to TensorBoard

    Note: Computes losses only for the selected window to save memory.

    Args:
        logger: JaxLogger instance for TensorBoard logging
        model: Embedding model
        img1: Frame 1 (1, H, W, 3)
        img2: Frame 2 (1, H, W, 3)
        metadata: dict with video_name, frame_t, frame_tk, distance
        step: Global step for logging
        settings: Settings object (for window_size, etc.)
    """
    import gc

    window_size = 16  # Fixed for now

    # Get image dimensions
    H, W = img1.shape[1:3]
    H_emb = H - 4  # Account for valid convolutions
    W_emb = W - 4

    # Calculate number of windows
    num_windows_h = H_emb // window_size
    num_windows_w = W_emb // window_size

    # Select random window
    rng = np.random.default_rng(seed=step)
    window_row = int(rng.integers(0, num_windows_h))
    window_col = int(rng.integers(0, num_windows_w))
    window_indices = (window_row, window_col)

    # Call model to get attention maps (separate from training computation)
    attention_data = model.compute_attention_maps(
        img1=img1,
        img2=img2,
        window_indices=window_indices,
        window_size=window_size,
    )

    # Convert JAX arrays to numpy for visualization
    img1_np = np.array(img1[0])  # (H, W, 3)
    img2_np = np.array(img2[0])  # (H, W, 3)
    
    # Extract window crop from BOTH frames
    emb_h_start = window_row * window_size
    emb_w_start = window_col * window_size
    
    # Account for 2-pixel border from valid convolutions
    img_h_start = emb_h_start
    img_h_end = img_h_start + window_size
    img_w_start = emb_w_start
    img_w_end = img_w_start + window_size
    
    window_crop1_np = np.array(img1[0, img_h_start:img_h_end, img_w_start:img_w_end, :])
    window_crop2_np = np.array(img2[0, img_h_start:img_h_end, img_w_start:img_w_end, :])
    
    self_attn_np = np.array(attention_data.self_attention)  # (N, 16, 16)
    cross_attn_np = np.array(attention_data.cross_attention)  # (N, 16, 16)
    pixel_positions_np = np.array(attention_data.pixel_positions)  # (N, 2)
    self_entropy_np = np.array(attention_data.self_entropy)  # (16, 16)
    cross_entropy_np = np.array(attention_data.cross_entropy)  # (16, 16)

    # Compute full-frame loss maps for ALL windows (not just the selected one)
    from barevision.embeddings.loss import (cross_attention_entropy_loss_core,
                                            self_attention_entropy_loss_core)
    from barevision.utils.grid import WindowGrid
    
    D = attention_data.embeddings1.shape[-1]
    emb1 = attention_data.embeddings1[None, ...]  # Add batch dim for splitting
    emb2 = attention_data.embeddings2[None, ...]
    
    # Split into windows
    grid = WindowGrid(window_size=window_size)
    windows1 = grid.split(emb1)  # (B, num_windows, 16, 16, D)
    windows2 = grid.split(emb2)
    
    B, num_windows, wh, ww, D = windows1.shape
    flat_windows1 = windows1.reshape(B * num_windows, wh, ww, D)
    flat_windows2 = windows2.reshape(B * num_windows, wh, ww, D)
    
    # Compute losses for all windows
    self_loss_flat = self_attention_entropy_loss_core(flat_windows1)  # (B*num_windows, 16, 16)
    cross_loss_flat = cross_attention_entropy_loss_core(flat_windows1, flat_windows2)
    
    # Reshape back to full grid: (B, num_windows, 16, 16) -> (B, num_h, num_w, 16, 16) -> (B, H, W)
    num_h = H_emb // window_size
    num_w = W_emb // window_size
    
    self_loss_flat = self_loss_flat.reshape(B, num_h, num_w, window_size, window_size)
    self_loss_flat = self_loss_flat.transpose(0, 1, 3, 2, 4).reshape(B, H_emb, W_emb)
    
    cross_loss_flat = cross_loss_flat.reshape(B, num_h, num_w, window_size, window_size)
    cross_loss_flat = cross_loss_flat.transpose(0, 1, 3, 2, 4).reshape(B, H_emb, W_emb)
    
    # Remove batch dim and pad to match image dimensions
    # Embeddings are (H-4, W-4) due to 5×5 valid conv
    # Image is (H, W), so pad by 2 on each side to match
    H_emb, W_emb = self_loss_flat[0].shape
    H_img, W_img = img1_np.shape[:2]
    pad_h = H_img - H_emb
    pad_w = W_img - W_emb
    assert pad_h % 2 == 0 and pad_w % 2 == 0, f"Padding must be even, got {pad_h}x{pad_w}"
    
    self_loss_display = np.pad(np.array(self_loss_flat[0]), ((pad_h//2, pad_h//2), (pad_w//2, pad_w//2)), mode='edge')
    cross_loss_display = np.pad(np.array(cross_loss_flat[0]), ((pad_h//2, pad_h//2), (pad_w//2, pad_w//2)), mode='edge')
    
    # Extract the selected window embeddings for detailed visualizations
    emb_h_start = window_row * window_size
    emb_w_start = window_col * window_size
    window_emb1 = attention_data.embeddings1[
        emb_h_start : emb_h_start + window_size,
        emb_w_start : emb_w_start + window_size,
        :,
    ]  # (16, 16, D)

    # 1. Frame with grid (showing both frames)
    fig_frame = create_frame_with_grid_figure(
        img1_np, img2_np, metadata, window_size, highlighted_window=window_indices
    )
    logger.log_figure("Frame/Grid", fig_frame, step)

    # 2. Loss heatmaps (showing full-frame entropy for all windows)
    fig_self_loss, fig_cross_loss = create_loss_heatmap_figures(
        img1_np,
        self_loss_display,
        cross_loss_display,
        window_size,
        highlighted_window=window_indices,
    )
    logger.log_figure("Loss/SelfEntropy", fig_self_loss, step)
    logger.log_figure("Loss/CrossEntropy", fig_cross_loss, step)

    # 3. Attention maps for selected pixels (with both frame crops and auto-scaling)
    fig_attn = create_attention_maps_figure(
        window_crop1_np,
        window_crop2_np,  # Both frame crops
        self_attn_np,
        cross_attn_np,
        pixel_positions_np,
        window_size,
        window_indices=(window_row, window_col),
        frame_t=metadata.get("frame_t", 0),
        frame_tk=metadata.get("frame_tk", 0),
        distance=metadata.get("distance", 0),
    )
    logger.log_figure("Attention/Maps", fig_attn, step)

    # 4. Similarity matrix for the window
    flat_window_emb1 = window_emb1.reshape(-1, D)
    similarity_matrix = np.array(flat_window_emb1 @ flat_window_emb1.T)  # (256, 256)


    self_logits = flat_window_emb1 @ flat_window_emb1.T

    # Softmax to get attention weights (memory-efficient)
    self_logits_max = np.max(self_logits, axis=-1, keepdims=True)
    self_logits_shifted = self_logits - self_logits_max
    exp_logits = np.exp(self_logits_shifted)
    self_attn_full = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    fig_sim = create_similarity_matrix_figure(
        similarity_matrix, self_attn_full, window_size,
        window_indices=(window_row, window_col)
    )
    logger.log_figure("Similarity/Matrix", fig_sim, step)
    
    # 5. Entropy maps (2×2 layout with both frame crops)
    fig_entropy = create_entropy_maps_figure(
        window_crop1_np,
        window_crop2_np,
        self_entropy_np,
        cross_entropy_np,
        window_size,
        window_indices=(window_row, window_col),
        frame_t=metadata.get("frame_t", 0),
        frame_tk=metadata.get("frame_tk", 0),
        distance=metadata.get("distance", 0),
    )
    logger.log_figure("Entropy/Maps", fig_entropy, step)

    # Clean up
    del attention_data, window_emb1, flat_window_emb1, similarity_matrix, self_attn_full
    gc.collect()
