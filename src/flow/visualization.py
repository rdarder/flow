"""Visualization utilities for hierarchical optical flow training.

Provides multi-view diagnostic figures with fixed layouts and color scales
for consistent comparison across epochs and runs.
"""

from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set non-interactive backend for headless environments
matplotlib.use("Agg")

# =============================================================================
# Constants for Fixed Layouts and Scales
# =============================================================================

FIGURE_DPI = 100

# Figure sizes (width, height) in inches - fixed for consistency
OVERVIEW_SIZE = (15, 10)  # 2 rows base + level rows
PYRAMID_SIZE_BASE = (16, 4)  # Per level, scales with num_levels
BLENDING_SIZE = (18, 8)
COMPONENTS_SIZE = (15, 10)
CONFIDENCE_SIZE = (14, 10)

# Fixed color scales (can be overridden)
FLOW_MAX_MAGNITUDE = 10.0  # Default maximum flow magnitude for color coding
CONFIDENCE_VMIN = 0.0
CONFIDENCE_VMAX = 1.0
ERROR_VMIN = 0.0
ERROR_VMAX = 5.0

# Colormaps
FLOW_CMAP = "hsv"
CONFIDENCE_CMAP = "viridis"
ERROR_CMAP = "hot"


# =============================================================================
# Core Flow Color Conversion
# =============================================================================


def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """Convert optical flow to RGB color representation.

    Uses HSV color space where:
    - Hue represents flow direction (angle)
    - Saturation is fixed at 1.0
    - Value represents flow magnitude (normalized)

    Args:
        flow: Flow field (H, W, 2) with (dx, dy) components
        max_flow: Maximum flow magnitude for normalization. If None, uses
                  FLOW_MAX_MAGNITUDE constant for consistent scaling.

    Returns:
        RGB image (H, W, 3) with values in [0, 1]
    """
    H, W, C = flow.shape
    assert C == 2, f"Flow must have 2 channels, got {C}"

    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # Normalize angle to [0, 1] for hue
    h = (angle + np.pi) / (2 * np.pi)
    s = np.ones_like(h)

    # Normalize magnitude
    max_mag = max_flow if max_flow is not None else FLOW_MAX_MAGNITUDE
    v = np.clip(magnitude / (max_mag + 1e-8), 0, 1)

    # Convert HSV to RGB
    hsv = np.stack([h, s, v], axis=-1)

    # Use matplotlib's HSV colormap
    cmap = matplotlib.colormaps.get_cmap(FLOW_CMAP)
    rgb = cmap(hsv[..., 0])

    # Apply value as brightness
    rgb[..., :3] *= v[..., np.newaxis]

    return np.clip(rgb[..., :3], 0, 1)


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


def _upsample_flow(flow: np.ndarray, target_h: int, target_w: int, original_resolution: int) -> np.ndarray:
    """Upsample flow field to target resolution and convert to pixel-equivalent.

    Pyramid level flows are in normalized coordinates (0-1 range relative to finest level).
    This function upsamples them to target resolution and converts to pixel-equivalent
    values by multiplying by original image resolution.
    
    Args:
        flow: Flow field (H, W, 2) in normalized coordinates
        target_h: Target height (resolution to visualize at)
        target_w: Target width
        original_resolution: Original image resolution (for pixel-equivalent conversion)
        
    Returns:
        Upsampled flow in pixel-equivalent coordinates (target_h, target_w, 2)
    """
    import jax.numpy as jnp
    from jax.image import resize
    
    src_h, src_w = flow.shape[:2]
    
    # Upsample using bilinear interpolation (normalized coords stay the same during upsampling)
    flow_upsampled = np.array(
        resize(
            jnp.array(flow),
            (target_h, target_w, flow.shape[-1]),
            method="bilinear",
        )
    )
    
    # Convert to pixel-equivalent by scaling by original image resolution
    # (all flows represent movement in the original image space)
    flow_pixel_equivalent = flow_upsampled * np.array([original_resolution, original_resolution])
    
    return flow_pixel_equivalent


# =============================================================================
# Visualization Figure Functions
# =============================================================================


def create_overview_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    flow_gt: np.ndarray,
    flow_pred: np.ndarray,
    level_flows: Optional[Dict[str, np.ndarray]] = None,
    flow_max_percent: float = 0.1,
) -> np.ndarray:
    """Create overview figure showing inputs, GT, predictions, and pyramid levels.

    Layout:
    - Row 1: Frame 1, Frame 2, Ground Truth Flow
    - Row 2: Predicted Flow, Error Heatmap, Error Histogram
    - Additional rows: Individual pyramid level flows (2 per row)

    Args:
        img1: First frame (H, W, C)
        img2: Second frame (H, W, C)
        flow_gt: Ground truth flow (H, W, 2)
        flow_pred: Predicted flow (H, W, 2)
        level_flows: Dictionary of level_name -> flow for pyramid levels
        flow_max_percent: Percentage of original image resolution for max flow color scale

    Returns:
        RGB image array for TensorBoard
    """
    # Target resolution: use flow_pred (finest level) as reference
    target_h, target_w = flow_pred.shape[:2]
    
    # Original image resolution: finest pyramid level is half the original
    # (e.g., 64x64 image -> 32x32 flow, so original = 2 * target)
    original_resolution = 2 * max(target_h, target_w)
    
    # Calculate max_flow as percentage of original image resolution
    max_flow = flow_max_percent * original_resolution
    
    # Handle shape mismatch between GT and prediction
    if flow_pred.shape[:2] != flow_gt.shape[:2]:
        # Downsample GT to match prediction
        import jax.numpy as jnp
        from jax.image import resize

        scale_h = target_h / flow_gt.shape[0]
        scale_w = target_w / flow_gt.shape[1]

        # Scale flow values
        flow_gt_scaled = flow_gt * np.array([scale_w, scale_h])

        # Downsample using interpolation
        flow_gt = np.array(
            resize(
                jnp.array(flow_gt_scaled),
                (target_h, target_w, flow_gt.shape[-1]),
                method="bilinear",
            )
        )

    # Determine grid size
    num_levels = len(level_flows) if level_flows else 0
    n_rows = 2 + (num_levels + 1) // 2  # Base 2 rows + level rows
    n_cols = 3

    # Calculate height based on number of rows
    height = OVERVIEW_SIZE[1] + (n_rows - 2) * 4  # Add 4 inches per level row
    figsize = (OVERVIEW_SIZE[0], height)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=FIGURE_DPI)

    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif not isinstance(axes, np.ndarray):
        axes = np.array(axes).reshape(n_rows, n_cols)

    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # Row 1: Inputs and GT
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("Frame 1", fontsize=12, fontweight="bold")

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("Frame 2", fontsize=12, fontweight="bold")

    axes[0, 2].imshow(flow_to_color(flow_gt, max_flow=max_flow))
    axes[0, 2].set_title(f"Ground Truth Flow (max={max_flow:.1f}px)", fontsize=12, fontweight="bold")

    # Row 2: Predictions
    axes[1, 0].imshow(flow_to_color(flow_pred, max_flow=max_flow))
    axes[1, 0].set_title(f"Predicted Flow (max={max_flow:.1f}px)", fontsize=12, fontweight="bold")

    # Error magnitude
    error = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=-1))
    im_err = axes[1, 1].imshow(error, cmap=ERROR_CMAP, vmin=ERROR_VMIN, vmax=ERROR_VMAX)
    axes[1, 1].set_title("Flow Error", fontsize=12, fontweight="bold")
    plt.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Error histogram
    axes[1, 2].hist(
        error.flatten(), bins=50, color="blue", alpha=0.7, range=(0, ERROR_VMAX)
    )
    axes[1, 2].set_title(
        f"Error Distribution (mean={np.mean(error):.2f})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 2].set_xlabel("Error magnitude")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_xlim(0, ERROR_VMAX)

    # Additional rows: Pyramid levels
    if level_flows:
        level_items = list(level_flows.items())
        for i, (level_name, level_flow) in enumerate(level_items):
            row = 2 + i // 3
            col = i % 3
            if row < n_rows and col < n_cols:
                # Convert to pixel-equivalent using original image resolution
                # (all levels represent movement in the original image space)
                src_h, src_w = level_flow.shape[:2]
                if (src_h, src_w) != (target_h, target_w):
                    # Upsample and convert to pixel-equivalent
                    level_flow_vis = _upsample_flow(level_flow, target_h, target_w, original_resolution)
                    upsample_info = f"({src_h}→{target_h})"
                else:
                    # Already at target res, just convert to pixel-equivalent
                    level_flow_vis = level_flow * np.array([original_resolution, original_resolution])
                    upsample_info = f"({target_h})"
                
                axes[row, col].imshow(flow_to_color(level_flow_vis, max_flow=max_flow))
                axes[row, col].set_title(f"{level_name} Flow {upsample_info}", fontsize=10)

    # Clean up axes
    for ax in axes.flat:
        ax.axis("off")

    return _figure_to_array(fig)


def create_pyramid_detail_figure(
    level_flows: Dict[str, np.ndarray],
    level_confidences: Dict[str, np.ndarray],
    max_flow: Optional[float] = None,
) -> np.ndarray:
    """Create detailed pyramid figure with flows and confidences.

    Layout: One row per pyramid level, each row shows:
    - Flow (color-coded)
    - Confidence map (0-1 scale)
    - Flow magnitude
    - Confidence histogram

    Args:
        level_flows: Dictionary of level_name -> flow (H, W, 2)
        level_confidences: Dictionary of level_name -> confidence (H, W, 1)
        max_flow: Maximum flow magnitude for color coding. If None, uses FLOW_MAX_MAGNITUDE.

    Returns:
        RGB image array for TensorBoard
    """
    num_levels = len(level_flows)
    if num_levels == 0:
        # Return empty figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=FIGURE_DPI)
        ax.text(0.5, 0.5, "No pyramid levels available", ha="center", va="center")
        ax.axis("off")
        return _figure_to_array(fig)

    n_rows = num_levels
    n_cols = 4

    # Calculate figure size - 4 inches per level
    figsize = (PYRAMID_SIZE_BASE[0], PYRAMID_SIZE_BASE[1] * num_levels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=FIGURE_DPI)

    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    level_names = sorted(level_flows.keys())

    for row, level_name in enumerate(level_names):
        flow = level_flows[level_name]
        conf = level_confidences[level_name]

        # Remove batch dimension if present
        if flow.ndim == 4:
            flow = flow[0]
        if conf.ndim == 4:
            conf = conf[0]
        if conf.ndim == 3 and conf.shape[-1] == 1:
            conf = conf[..., 0]

        # Column 1: Flow
        axes[row, 0].imshow(flow_to_color(flow, max_flow=max_flow))
        axes[row, 0].set_title(f"{level_name} Flow", fontsize=10, fontweight="bold")

        # Column 2: Confidence map
        im_conf = axes[row, 1].imshow(
            conf, cmap=CONFIDENCE_CMAP, vmin=CONFIDENCE_VMIN, vmax=CONFIDENCE_VMAX
        )
        axes[row, 1].set_title(
            f"{level_name} Confidence", fontsize=10, fontweight="bold"
        )
        plt.colorbar(im_conf, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Column 3: Flow magnitude
        magnitude = np.sqrt(np.sum(flow**2, axis=-1))
        im_mag = axes[row, 2].imshow(
            magnitude, cmap="plasma", vmin=0, vmax=FLOW_MAX_MAGNITUDE
        )
        axes[row, 2].set_title(
            f"{level_name} Magnitude", fontsize=10, fontweight="bold"
        )
        plt.colorbar(im_mag, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Column 4: Confidence histogram
        axes[row, 3].hist(
            conf.flatten(), bins=30, color="green", alpha=0.7, range=(0, 1)
        )
        axes[row, 3].set_title(
            f"Conf Dist (mean={np.mean(conf):.2f})", fontsize=10, fontweight="bold"
        )
        axes[row, 3].set_xlabel("Confidence")
        axes[row, 3].set_ylabel("Count")
        axes[row, 3].set_xlim(0, 1)

    # Clean up axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return _figure_to_array(fig)


def create_blending_figure(
    flow_fine: np.ndarray,
    conf_fine: np.ndarray,
    flow_coarse_upsampled: np.ndarray,
    conf_coarse_upsampled: np.ndarray,
    weight_fine: np.ndarray,
    weight_coarse: np.ndarray,
    flow_final: np.ndarray,
    original_resolution: int,
    level_name: str = "Level",
    flow_max_percent: float = 0.1,
    flow_gt: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create blending analysis figure.

    Layout (2 rows × 4 columns):
    Row 1: Coarse Upsampled Flow | Fine Flow | Weight Fine | Weight Coarse
    Row 2: Coarse Upsampled Conf | Fine Conf | Blended Result | Error (if GT)

    Args:
        flow_fine: Fine level flow (H, W, 2) in normalized coordinates
        conf_fine: Fine level confidence (H, W, 1)
        flow_coarse_upsampled: Upsampled coarse flow (H, W, 2) in normalized coordinates
        conf_coarse_upsampled: Upsampled coarse confidence (H, W, 1)
        weight_fine: Blending weight for fine (H, W, 1)
        weight_coarse: Blending weight for coarse (H, W, 1)
        flow_final: Final blended flow (H, W, 2) in normalized coordinates
        original_resolution: Original image resolution (for pixel-equivalent conversion)
        level_name: Name of the fine pyramid level (e.g., "Level 1")
        flow_max_percent: Percentage of original resolution for max flow color scale
        flow_gt: Optional ground truth for error visualization

    Returns:
        RGB image array for TensorBoard
    """

    # Remove batch dimensions if present
    def squeeze_batch(arr):
        if arr.ndim == 4:
            return arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr

    flow_fine = squeeze_batch(flow_fine)
    conf_fine = squeeze_batch(conf_fine)
    flow_coarse_upsampled = squeeze_batch(flow_coarse_upsampled)
    conf_coarse_upsampled = squeeze_batch(conf_coarse_upsampled)
    weight_fine = squeeze_batch(weight_fine)
    weight_coarse = squeeze_batch(weight_coarse)
    flow_final = squeeze_batch(flow_final)

    # Convert flows to pixel-equivalent coordinates
    # All flows represent movement in the original image space
    resolution_scale = np.array([original_resolution, original_resolution])
    flow_fine_px = flow_fine * resolution_scale
    flow_coarse_px = flow_coarse_upsampled * resolution_scale
    flow_final_px = flow_final * resolution_scale

    # Calculate max_flow as percentage of original image resolution
    max_flow = flow_max_percent * original_resolution

    fig, axes = plt.subplots(2, 4, figsize=BLENDING_SIZE, dpi=FIGURE_DPI)
    fig.suptitle(f"{level_name} Blending (max={max_flow:.1f}px)", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.2, top=0.93)

    # Row 1
    axes[0, 0].imshow(flow_to_color(flow_coarse_px, max_flow=max_flow))
    axes[0, 0].set_title("Coarse Flow (upsampled)", fontsize=11, fontweight="bold")

    axes[0, 1].imshow(flow_to_color(flow_fine_px, max_flow=max_flow))
    axes[0, 1].set_title("Fine Flow", fontsize=11, fontweight="bold")

    im_wf = axes[0, 2].imshow(weight_fine, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0, 2].set_title("Weight Fine", fontsize=11, fontweight="bold")
    plt.colorbar(im_wf, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_wc = axes[0, 3].imshow(weight_coarse, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0, 3].set_title("Weight Coarse", fontsize=11, fontweight="bold")
    plt.colorbar(im_wc, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 2
    im_cc = axes[1, 0].imshow(
        conf_coarse_upsampled,
        cmap=CONFIDENCE_CMAP,
        vmin=CONFIDENCE_VMIN,
        vmax=CONFIDENCE_VMAX,
    )
    axes[1, 0].set_title("Coarse Confidence", fontsize=11, fontweight="bold")
    plt.colorbar(im_cc, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_cf = axes[1, 1].imshow(
        conf_fine, cmap=CONFIDENCE_CMAP, vmin=CONFIDENCE_VMIN, vmax=CONFIDENCE_VMAX
    )
    axes[1, 1].set_title("Fine Confidence", fontsize=11, fontweight="bold")
    plt.colorbar(im_cf, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(flow_to_color(flow_final_px, max_flow=max_flow))
    axes[1, 2].set_title("Blended Result", fontsize=11, fontweight="bold")

    # Error column
    if flow_gt is not None:
        # Ensure shapes match
        if flow_final.shape[:2] != flow_gt.shape[:2]:
            # Downsample GT
            import jax.numpy as jnp
            from jax.image import resize

            target_h, target_w = flow_final.shape[:2]
            scale_h = target_h / flow_gt.shape[0]
            scale_w = target_w / flow_gt.shape[1]
            flow_gt_scaled = flow_gt * np.array([scale_w, scale_h])
            flow_gt = np.array(
                resize(
                    jnp.array(flow_gt_scaled),
                    (target_h, target_w, flow_gt.shape[-1]),
                    method="bilinear",
                )
            )

        error = np.sqrt(np.sum((flow_final - flow_gt) ** 2, axis=-1))
        im_err = axes[1, 3].imshow(
            error, cmap=ERROR_CMAP, vmin=ERROR_VMIN, vmax=ERROR_VMAX
        )
        axes[1, 3].set_title("Final Error", fontsize=11, fontweight="bold")
        plt.colorbar(im_err, ax=axes[1, 3], fraction=0.046, pad=0.04)
    else:
        axes[1, 3].text(
            0.5, 0.5, "No GT\nAvailable", ha="center", va="center", fontsize=12
        )
        axes[1, 3].set_title("Error (N/A)", fontsize=11, fontweight="bold")

    # Clean up axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return _figure_to_array(fig)


def create_components_figure(
    flow_lookup: np.ndarray,
    flow_peer: np.ndarray,
    conf_lookup: np.ndarray,
    conf_peer: np.ndarray,
    flow_blended: np.ndarray,
    conf_blended: np.ndarray,
    original_resolution: int,
    level_name: str = "Level",
    flow_max_percent: float = 0.1,
) -> np.ndarray:
    """Create component comparison figure (PatchLookup vs PeerPropagation).

    Layout (2 rows × 3 columns):
    Row 1: PatchLookup Flow | PeerPropagation Flow | Blended Flow
    Row 2: PatchLookup Conf | PeerPropagation Conf | Blended Conf

    Args:
        flow_lookup: Flow from PatchLookup (H, W, 2) in normalized coordinates
        flow_peer: Flow from PeerPropagation (H, W, 2) in normalized coordinates
        conf_lookup: Confidence from PatchLookup (H, W, 1)
        conf_peer: Confidence from PeerPropagation (H, W, 1)
        flow_blended: Final blended flow (H, W, 2) in normalized coordinates
        conf_blended: Final blended confidence (H, W, 1)
        original_resolution: Original image resolution (for pixel-equivalent conversion)
        level_name: Name of the pyramid level (e.g., "Level 0", "Level 1")
        flow_max_percent: Percentage of original resolution for max flow color scale

    Returns:
        RGB image array for TensorBoard
    """

    # Remove batch dimensions if present
    def squeeze_batch(arr):
        if arr.ndim == 4:
            return arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr

    flow_lookup = squeeze_batch(flow_lookup)
    flow_peer = squeeze_batch(flow_peer)
    conf_lookup = squeeze_batch(conf_lookup)
    conf_peer = squeeze_batch(conf_peer)
    flow_blended = squeeze_batch(flow_blended)
    conf_blended = squeeze_batch(conf_blended)

    # Convert flows to pixel-equivalent coordinates
    # All flows represent movement in the original image space
    resolution_scale = np.array([original_resolution, original_resolution])
    flow_lookup_px = flow_lookup * resolution_scale
    flow_peer_px = flow_peer * resolution_scale
    flow_blended_px = flow_blended * resolution_scale

    # Calculate max_flow as percentage of original image resolution
    max_flow = flow_max_percent * original_resolution

    fig, axes = plt.subplots(2, 3, figsize=COMPONENTS_SIZE, dpi=FIGURE_DPI)
    fig.suptitle(f"{level_name} Components (max={max_flow:.1f}px)", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.2, top=0.93)

    # Row 1: Flows
    axes[0, 0].imshow(flow_to_color(flow_lookup_px, max_flow=max_flow))
    axes[0, 0].set_title(
        "PatchLookup Flow\n(Cross-Attention)", fontsize=11, fontweight="bold"
    )

    axes[0, 1].imshow(flow_to_color(flow_peer_px, max_flow=max_flow))
    axes[0, 1].set_title(
        "PeerPropagation Flow\n(Self-Attention)", fontsize=11, fontweight="bold"
    )

    axes[0, 2].imshow(flow_to_color(flow_blended_px, max_flow=max_flow))
    axes[0, 2].set_title(
        "Blended Flow\n(Confidence-Weighted)", fontsize=11, fontweight="bold"
    )

    # Row 2: Confidences
    im_cl = axes[1, 0].imshow(
        conf_lookup, cmap=CONFIDENCE_CMAP, vmin=CONFIDENCE_VMIN, vmax=CONFIDENCE_VMAX
    )
    axes[1, 0].set_title("PatchLookup Conf", fontsize=11, fontweight="bold")
    plt.colorbar(im_cl, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_cp = axes[1, 1].imshow(
        conf_peer, cmap=CONFIDENCE_CMAP, vmin=CONFIDENCE_VMIN, vmax=CONFIDENCE_VMAX
    )
    axes[1, 1].set_title("PeerPropagation Conf", fontsize=11, fontweight="bold")
    plt.colorbar(im_cp, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_cb = axes[1, 2].imshow(
        conf_blended, cmap=CONFIDENCE_CMAP, vmin=CONFIDENCE_VMIN, vmax=CONFIDENCE_VMAX
    )
    axes[1, 2].set_title("Blended Conf", fontsize=11, fontweight="bold")
    plt.colorbar(im_cb, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Clean up axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    return _figure_to_array(fig)


def create_confidence_analysis_figure(
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Create confidence vs error analysis figure.

    Layout (2 rows × 2 columns):
    Top-Left: Scatter plot - confidence vs error
    Top-Right: Binned analysis - mean error per confidence decile
    Bottom-Left: Confidence map (spatial visualization)
    Bottom-Right: Error map (spatial visualization)

    Args:
        flow_pred: Predicted flow (H, W, 2)
        flow_gt: Ground truth flow (H, W, 2)
        confidence: Confidence scores (H, W, 1) or (H, W)

    Returns:
        RGB image array for TensorBoard
    """
    # Remove batch dimensions if present
    if flow_pred.ndim == 4:
        flow_pred = flow_pred[0]
    if flow_gt.ndim == 4:
        flow_gt = flow_gt[0]
    if confidence.ndim == 4:
        confidence = confidence[0]
    if confidence.ndim == 3 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]

    # Handle shape mismatch
    if flow_pred.shape[:2] != flow_gt.shape[:2]:
        import jax.numpy as jnp
        from jax.image import resize

        target_h, target_w = flow_pred.shape[:2]
        scale_h = target_h / flow_gt.shape[0]
        scale_w = target_w / flow_gt.shape[1]
        flow_gt_scaled = flow_gt * np.array([scale_w, scale_h])
        flow_gt = np.array(
            resize(
                jnp.array(flow_gt_scaled),
                (target_h, target_w, flow_gt.shape[-1]),
                method="bilinear",
            )
        )

    # Compute error
    error = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=-1))

    fig, axes = plt.subplots(2, 2, figsize=CONFIDENCE_SIZE, dpi=FIGURE_DPI)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Flatten for analysis
    conf_flat = confidence.flatten()
    error_flat = error.flatten()

    # Top-Left: Scatter plot
    # Sample for performance if too many points
    n_points = len(conf_flat)
    if n_points > 5000:
        indices = np.random.choice(n_points, 5000, replace=False)
        conf_sample = conf_flat[indices]
        error_sample = error_flat[indices]
    else:
        conf_sample = conf_flat
        error_sample = error_flat

    axes[0, 0].scatter(conf_sample, error_sample, alpha=0.3, s=1)
    axes[0, 0].set_xlabel("Confidence", fontsize=10)
    axes[0, 0].set_ylabel("Error", fontsize=10)
    axes[0, 0].set_title("Confidence vs Error", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, ERROR_VMAX)
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line
    if len(conf_sample) > 1:
        z = np.polyfit(conf_sample, error_sample, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Trend")
        axes[0, 0].legend()

    # Compute correlation
    if len(conf_flat) > 1:
        corr, p_value = stats.pearsonr(conf_flat, error_flat)
        axes[0, 0].text(
            0.05,
            0.95,
            f"Corr: {corr:.3f}\np={p_value:.2e}",
            transform=axes[0, 0].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Top-Right: Binned analysis
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(n_bins):
        mask = (conf_flat >= bin_edges[i]) & (conf_flat < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (conf_flat >= bin_edges[i]) & (conf_flat <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_means.append(np.mean(error_flat[mask]))
            bin_stds.append(np.std(error_flat[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)

    axes[0, 1].bar(
        bin_centers,
        bin_means,
        width=0.08,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )
    axes[0, 1].errorbar(
        bin_centers, bin_means, yerr=bin_stds, fmt="none", color="red", capsize=3
    )
    axes[0, 1].set_xlabel("Confidence Bin", fontsize=10)
    axes[0, 1].set_ylabel("Mean Error", fontsize=10)
    axes[0, 1].set_title("Error by Confidence Decile", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, ERROR_VMAX)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Bottom-Left: Confidence map
    im_conf = axes[1, 0].imshow(
        confidence, cmap=CONFIDENCE_CMAP, vmin=CONFIDENCE_VMIN, vmax=CONFIDENCE_VMAX
    )
    axes[1, 0].set_title("Confidence Map", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    plt.colorbar(im_conf, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Bottom-Right: Error map
    im_err = axes[1, 1].imshow(error, cmap=ERROR_CMAP, vmin=ERROR_VMIN, vmax=ERROR_VMAX)
    axes[1, 1].set_title("Error Map", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    plt.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)

    return _figure_to_array(fig)
