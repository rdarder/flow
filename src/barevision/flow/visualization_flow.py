"""Flow visualization utilities.

Converts optical flow fields to colorwheel images and arrow visualizations.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

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


def flow_to_colorwheel(flow: jnp.ndarray, max_flow: float = 0.3) -> np.ndarray:
    """Convert flow field to RGB color image using standard colorwheel encoding.

    Flow convention: (u, v) = displacement in normalized coordinates [0, 1]
    where u = x displacement, v = y displacement.

    Uses a non-linear saturation curve (square root) to make small motions
    visible quickly while leveraging the full dynamic range.

    Args:
        flow: (H, W, 2) flow field in normalized coordinates
              Positive u = motion right, Positive v = motion down
        max_flow: Maximum magnitude for saturation scaling (default 0.3)
                  Flows >= max_flow will be fully saturated.

    Returns:
        rgb: (H, W, 3) RGB image (uint8)
        - Hue encodes direction (0-360°)
        - Saturation encodes magnitude with non-linear scaling for visibility
        - Full contrast: black for no motion, full color for max motion
    """
    # Convert JAX array to numpy
    flow_np = np.array(flow)
    H, W, _ = flow_np.shape

    # Get colorwheel
    colorwheel = make_colorwheel()
    num_bins = colorwheel.shape[0]

    # Compute flow magnitude and direction
    u = flow_np[..., 0]  # Positive = right
    v = flow_np[..., 1]  # Positive = down

    # Compute magnitude
    mag = np.sqrt(u**2 + v**2)

    # Compute direction (angle in radians)
    angle = np.arctan2(-v, u)  # Negate v for image coordinates (y increases downward)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)  # Convert to [0, 2π)

    # Map angle to colorwheel bin [0, num_bins)
    bin_idx = np.floor(angle / (2 * np.pi) * num_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    # Compute saturation with non-linear scaling
    # Square root: fast initial growth, then diminishing returns
    # This makes small motions visible while using full contrast range
    sat = np.clip(mag / max_flow, 0, 1)
    sat = np.sqrt(sat)

    # Apply saturation: black (no motion) to full color (max motion)
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            base_color = colorwheel[bin_idx[i, j]].astype(np.float32) / 255.0
            rgb[i, j] = base_color * sat[i, j]

    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


# Removed: flow_components_to_image and create_flow_component_figure
# The X/Y component heatmaps didn't provide clear visualization


def flow_to_arrows(
    flow: jnp.ndarray,
    max_flow: float = 0.3,
    scale: float = 2.0,
    grid_density: int = 8,
) -> np.ndarray:
    """Create arrow visualization of flow field.
    
    Draws arrows on a high-resolution canvas where:
    - Arrow direction = flow direction
    - Arrow length/thickness = flow magnitude (with non-linear scaling)
    
    Args:
        flow: (H, W, 2) flow field in normalized coordinates
        max_flow: Flow magnitude that produces maximum arrow length (default 0.3)
        scale: Multiplier for arrow size (default 2.0)
        grid_density: Number of arrows per dimension (default 8 for 8x8 grid)
    
    Returns:
        rgb: (512, 512, 3) RGB image (uint8) showing flow arrows
    """
    # Convert JAX array to numpy
    flow_np = np.array(flow)
    H, W, _ = flow_np.shape
    
    # Create high-resolution canvas
    canvas_size = 512
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=100)
    fig.set_size_inches(canvas_size / 100, canvas_size / 100, forward=True)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Invert y to match image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    
    # Create grid for arrows (subsample if flow field is large)
    step_y = max(1, H // grid_density)
    step_x = max(1, W // grid_density)
    
    y_positions = np.arange(0, H, step_y)
    x_positions = np.arange(0, W, step_x)
    X, Y = np.meshgrid(x_positions, y_positions)
    
    # Sample flow at grid positions
    U = flow_np[y_positions[:, np.newaxis], x_positions[np.newaxis, :], 0]  # x component
    V = flow_np[y_positions[:, np.newaxis], x_positions[np.newaxis, :], 1]  # y component
    
    # Compute magnitude for scaling
    mag = np.sqrt(U**2 + V**2)
    
    # Non-linear scaling for arrow length (sqrt for fast initial growth)
    length_scale = scale * np.sqrt(np.clip(mag / max_flow, 0, 1))
    
    # Normalize direction vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        U_norm = U / (mag + 1e-10)
        V_norm = V / (mag + 1e-10)
    
    # Scale by length
    U_scaled = U_norm * length_scale
    V_scaled = V_norm * length_scale
    
    # Draw arrows using quiver
    # Note: quiver in matplotlib uses (X, Y, U, V) where U,V are the arrow components
    q = ax.quiver(
        X, Y, U_scaled, V_scaled,
        angles='xy', scale_units='xy', scale=1,
        color='black', width=0.003, headwidth=4, headlength=5
    )
    
    # Convert to RGB array
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    buffer = np.frombuffer(buf, dtype=np.uint8)
    rgb = buffer.reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    
    return rgb
