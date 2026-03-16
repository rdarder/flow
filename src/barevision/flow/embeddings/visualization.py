"""Visualization utilities for embedding model.

Generates figures for attention maps.
"""

import matplotlib.pyplot as plt
import numpy as np

from barevision.flow.embeddings.losses import (
    self_attention_entropy_loss_core,
    cross_attention_entropy_loss_core,
)
from barevision.utils.grid import WindowGrid


def create_attention_maps_figure(
    self_attention_maps: np.ndarray,
    cross_attention_maps: np.ndarray,
    pixel_positions: np.ndarray,
    window_crop: np.ndarray,
    seed_used: int,
) -> plt.Figure:
    """Create figure showing attention maps for selected pixels.

    Args:
        self_attention_maps: (N, H, W) self-attention for N query pixels
        cross_attention_maps: (N, H, W) cross-attention for N query pixels
        pixel_positions: (N, 2) (y, x) positions of queried pixels
        window_crop: (H, W, 3) image crop for the analyzed window
        seed_used: Random seed used for pixel selection

    Returns:
        Matplotlib figure
    """
    n_pixels = len(pixel_positions)
    fig, axes = plt.subplots(2, n_pixels + 1, figsize=(20, 10))

    # Show window crop in first column
    axes[0, 0].imshow(window_crop)
    axes[0, 0].set_title(f'Window (seed={seed_used})')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Show attention maps for each pixel
    for i in range(n_pixels):
        py, px = pixel_positions[i]

        # Mark pixel position on crop (top row)
        axes[0, 0].plot(px, py, 'r+', markersize=15, markeredgewidth=2)

        # Self attention
        axes[0, i + 1].imshow(self_attention_maps[i], cmap='hot', vmin=0, vmax=1)
        axes[0, i + 1].set_title(f'Self Attn\nPixel ({py},{px})')
        axes[0, i + 1].axis('off')

        # Cross attention
        axes[1, i + 1].imshow(cross_attention_maps[i], cmap='hot', vmin=0, vmax=1)
        axes[1, i + 1].set_title(f'Cross Attn\nPixel ({py},{px})')
        axes[1, i + 1].axis('off')

    plt.tight_layout()
    return fig
