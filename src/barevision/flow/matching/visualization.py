"""Visualization utilities for flow matching.

Generates figures for flow fields and centroid positions.
"""

import matplotlib.pyplot as plt
import numpy as np


def flow_to_colorwheel(flow: np.ndarray, max_flow: float = 0.3) -> np.ndarray:
    """Convert flow field to colorwheel visualization.

    Flow direction is encoded as hue, magnitude as saturation.

    Args:
        flow: (H, W, 2) flow field where (u, v) = displacement
        max_flow: Maximum flow magnitude for full saturation (default 0.3)

    Returns:
        (H, W, 3) RGB image with flow colorwheel
    """
    H, W, _ = flow.shape

    # Convert to polar coordinates
    magnitude = np.linalg.norm(flow, axis=-1)
    angle = np.arctan2(flow[..., 1], flow[..., 0])

    # Normalize angle to [0, 1] for hue
    hue = (angle + np.pi) / (2 * np.pi)

    # Normalize magnitude to [0, 1] for saturation (capped at max_flow)
    saturation = np.clip(magnitude / max_flow, 0, 1)

    # Value is always 1 (bright colors)
    value = np.ones_like(magnitude)

    # Convert HSV to RGB
    def hsv_to_rgb(h, s, v):
        """Convert HSV to RGB."""
        i = np.floor(h * 6).astype(int) % 6
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        rgb = np.stack([
            np.where(i == 0, v, np.where(i == 1, t, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v))))),
            np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, p, np.where(i == 4, p, q))))),
            np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, q, v))))),
        ], axis=-1)

        return rgb

    rgb = hsv_to_rgb(hue, saturation, value)
    return rgb


def flow_to_arrows(
    flow: np.ndarray,
    max_flow: float = 0.3,
    scale: float = 2.0,
    grid_density: int = 8,
) -> np.ndarray:
    """Create arrow visualization of flow field.

    Args:
        flow: (H, W, 2) flow field
        max_flow: Maximum flow magnitude for scaling (default 0.3)
        scale: Arrow length multiplier
        grid_density: Number of arrows along each axis

    Returns:
        (H, W, 3) RGB image with arrows
    """
    H, W, _ = flow.shape

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Show background as grayscale of flow magnitude
    magnitude = np.linalg.norm(flow, axis=-1)
    ax.imshow(magnitude, cmap='gray', vmin=0, vmax=max_flow)

    # Create grid for arrows
    step_y = H // grid_density
    step_x = W // grid_density
    y_grid, x_grid = np.meshgrid(
        np.arange(step_y // 2, H, step_y),
        np.arange(step_x // 2, W, step_x),
        indexing='ij'
    )

    # Sample flow at grid points
    u = flow[y_grid, x_grid, 0] * scale * W
    v = -flow[y_grid, x_grid, 1] * scale * H  # Negative because y is inverted in images

    # Normalize arrow colors by magnitude
    mag = np.sqrt(u**2 + v**2)
    mag_max = mag.max()
    colors = mag / (mag_max + 1e-8)

    # Plot arrows
    ax.quiver(
        x_grid, y_grid, u, v,
        angles='xy', scale_units='xy', scale=1,
        width=0.003,
        headwidth=5,
    )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Invert y axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Flow Field (arrows scaled {scale}x)')

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    from PIL import Image
    img = Image.open(buf)
    rgb = np.array(img)[:, :, :3]  # Drop alpha if present
    
    plt.close(fig)
    buf.close()
    return rgb
