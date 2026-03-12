"""Flow visualization utilities.

Converts optical flow fields to colorwheel images for visualization.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

matplotlib.use("Agg")


def make_colorwheel() -> np.ndarray:
    """Create standard colorwheel for flow visualization.

    Based on the Middlebury flow color encoding:
    - Red: right (0°)
    - Yellow: down-right (45°)
    - Green: down (90°)
    - Cyan: down-left (135°)
    - Blue: left (180°)
    - Magenta: up-left (225°)
    - Red: up (270°)
    - Yellow: up-right (315°)

    Returns:
        colorwheel: (55, 3) RGB color table
    """
    # Colorwheel parameters from Middlebury
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    colorwheel = np.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=np.uint8)

    col = 0

    # Red to Yellow
    colorwheel[:RY, 0] = 255
    colorwheel[:RY, 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY

    # Yellow to Green
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col += YG

    # Green to Cyan
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC

    # Cyan to Blue
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col += CB

    # Blue to Magenta
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(BM) / BM)
    colorwheel[col : col + BM, 2] = 255
    col += BM

    # Magenta to Red
    colorwheel[col : col + MR, 0] = 255
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)

    return colorwheel


def flow_to_colorwheel(flow: jnp.ndarray, max_flow: float = 1.0) -> np.ndarray:
    """Convert flow field to RGB color image using standard colorwheel encoding.

    Flow convention: (u, v) = displacement in normalized coordinates [0, 1]
    where u = x displacement, v = y displacement.

    Args:
        flow: (H, W, 2) flow field in normalized coordinates
              Positive u = motion right, Positive v = motion down
        max_flow: Maximum magnitude for saturation scaling (default 1.0)

    Returns:
        rgb: (H, W, 3) RGB image (uint8)
        - Hue encodes direction (0-360°)
        - Saturation encodes magnitude (clamped to max_flow)
        - Value = 1.0 for all pixels
    """
    # Convert JAX array to numpy
    flow_np = np.array(flow)
    H, W, _ = flow_np.shape

    # Get colorwheel
    colorwheel = make_colorwheel()
    num_bins = colorwheel.shape[0]

    # Compute flow magnitude and direction
    # u = x component (column 0), v = y component (column 1)
    u = flow_np[..., 0]  # Positive = right
    v = flow_np[..., 1]  # Positive = down

    # Compute magnitude
    mag = np.sqrt(u**2 + v**2)

    # Compute direction (angle in radians)
    # atan2(v, u) gives angle from positive x axis, counter-clockwise
    # We want: 0° = right, 90° = down, 180° = left, 270° = up
    # Standard atan2: 0 = right, +π/2 = up, π = left, -π/2 = down
    # So we negate v to flip y-axis (image coordinates: y increases downward)
    angle = np.arctan2(-v, u)  # Now: 0 = right, +π/2 = down, π = left, -π/2 = up

    # Convert to [0, 2π)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    # Map angle to colorwheel bin [0, num_bins)
    bin_idx = np.floor(angle / (2 * np.pi) * num_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    # Compute saturation based on magnitude
    # Clamp magnitude to max_flow for saturation
    sat = np.clip(mag / max_flow, 0, 1)

    # Look up color from colorwheel and apply saturation
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            rgb[i, j] = colorwheel[bin_idx[i, j]].astype(np.float32) / 255.0

    # Apply saturation (multiply by sat)
    rgb = rgb * sat[..., np.newaxis]

    # Apply value (we use V=1.0 always for now)
    # Could darken for very small magnitudes if desired

    # Convert to uint8
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def flow_components_to_image(flow: jnp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert flow field to separate X and Y component heatmaps.

    Flow convention: (u, v) = displacement in normalized coordinates [0, 1]
    where u = x displacement (positive = right), v = y displacement (positive = down).

    Args:
        flow: (H, W, 2) flow field in normalized coordinates

    Returns:
        Tuple of (x_component_rgb, y_component_rgb):
            - x_component_rgb: (H, W, 3) RGB heatmap of u component (red = right, blue = left)
            - y_component_rgb: (H, W, 3) RGB heatmap of v component (red = down, blue = up)

    Each component is auto-scaled to its own min/max range for visibility.
    """
    # Convert JAX array to numpy
    flow_np = np.array(flow)

    u = flow_np[..., 0]  # X component (positive = right)
    v = flow_np[..., 1]  # Y component (positive = down)

    def component_to_rgb(component: np.ndarray) -> np.ndarray:
        """Convert single component to RGB with red/blue diverging colormap."""
        H, W = component.shape

        # Find symmetric max for balanced color scale
        abs_max = np.max(np.abs(component))
        if abs_max < 1e-10:
            # No motion - return neutral gray
            return np.ones((H, W, 3), dtype=np.uint8) * 128

        # Normalize to [-1, 1]
        normalized = component / abs_max

        # Create RGB image with clean diverging colormap:
        # Red = positive (right/down), Blue = negative (left/up), White = zero
        rgb = np.ones((H, W, 3), dtype=np.float32)  # Start with white

        # For positive values: increase red, keep blue at 1
        # For negative values: increase blue, keep red at 1
        # This creates: red->white->blue gradient

        pos_mask = normalized > 0
        neg_mask = normalized < 0

        # Positive: white -> red (decrease green and blue)
        rgb[pos_mask, 1] = 1 - normalized[pos_mask]  # Green decreases
        rgb[pos_mask, 2] = 1 - normalized[pos_mask]  # Blue decreases

        # Negative: white -> blue (decrease red and green)
        rgb[neg_mask, 0] = (
            1 + normalized[neg_mask]
        )  # Red decreases (normalized is negative)
        rgb[neg_mask, 1] = 1 + normalized[neg_mask]  # Green decreases

        # Convert to uint8
        return (rgb * 255).astype(np.uint8)

    x_rgb = component_to_rgb(u)
    y_rgb = component_to_rgb(v)

    return x_rgb, y_rgb


def create_flow_component_figure(
    flow: jnp.ndarray,
    metadata: dict,
    step: int,
) -> np.ndarray:
    """Create figure showing flow X and Y components as heatmaps.

    Args:
        flow: (H, W, 2) flow field in normalized coordinates
        metadata: dict with frame_t, frame_tk, distance
        step: Global step (used for title)

    Returns:
        rgb: (H_fig, W_fig, 3) RGB figure array (uint8)
    """
    flow_np = np.array(flow)
    H, W, _ = flow_np.shape

    u = flow_np[..., 0]  # X component
    v = flow_np[..., 1]  # Y component

    # Get component images
    x_rgb, y_rgb = flow_components_to_image(flow)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    frame_t = metadata.get("frame_t", 0)
    frame_tk = metadata.get("frame_tk", 0)
    distance = metadata.get("distance", 0)

    # X component (left)
    axes[0].imshow(x_rgb)
    u_min, u_max = float(u.min()), float(u.max())
    axes[0].set_title(
        f"Flow X (U) | Frame {frame_t}→{frame_tk} (Δ{distance})\n"
        f"Range: [{u_min:.3f}, {u_max:.3f}]",
        fontsize=11,
    )
    axes[0].axis("off")

    # Y component (right)
    axes[1].imshow(y_rgb)
    v_min, v_max = float(v.min()), float(v.max())
    axes[1].set_title(
        f"Flow Y (V) | Frame {frame_t}→{frame_tk} (Δ{distance})\n"
        f"Range: [{v_min:.3f}, {v_max:.3f}]",
        fontsize=11,
    )
    axes[1].axis("off")

    plt.tight_layout()

    # Convert to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    buffer = np.frombuffer(buf, dtype=np.uint8)
    image_array = buffer.reshape(height, width, 4)[:, :, :3]
    plt.close(fig)

    return image_array
