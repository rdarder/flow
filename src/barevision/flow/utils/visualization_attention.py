"""Attention map extraction for visualization.

This module extracts window-specific attention data from full attention maps
computed during training. Used exclusively for generating visualization figures.
"""

import jax
import jax.numpy as jnp
import numpy as np


def extract_pixel_attention_maps(
    attention_weights: jnp.ndarray,
    pixel_indices: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Extract attention maps for specific pixels within a window.

    Args:
        attention_weights: (B, window_size^2, window_size^2) attention matrix for batch of windows
        pixel_indices: (num_pixels,) indices of pixels to extract maps for
        window_size: Size of window in pixels

    Returns:
        (B, num_pixels, window_size, window_size) attention maps

    Raises:
        AssertionError: If attention_weights is not batched (must be 3D)
    """
    assert attention_weights.ndim == 3, f"Expected batched input (3D), got {attention_weights.ndim}D"
    
    B, N, _ = attention_weights.shape
    selected_attn = attention_weights[:, pixel_indices, :]  # (B, num_pixels, N)
    selected_attn = selected_attn.reshape(B, -1, window_size, window_size)

    return selected_attn


def select_random_pixels(
    window_size: int,
    num_pixels: int = 4,
    seed: int = 0,
) -> jnp.ndarray:
    """Select random pixel indices within a window.

    Args:
        window_size: Size of window in pixels
        num_pixels: Number of pixels to select
        seed: Random seed for reproducibility

    Returns:
        (num_pixels,) array of pixel indices
    """
    N = window_size * window_size
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, N, shape=(num_pixels,), replace=False)


def compute_pixel_positions(
    pixel_indices: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Compute (y, x) positions for pixel indices within a window.

    Args:
        pixel_indices: (num_pixels,) array of pixel indices
        window_size: Size of window in pixels

    Returns:
        (num_pixels, 2) array of (y, x) positions
    """
    pixel_y = pixel_indices // window_size
    pixel_x = pixel_indices % window_size
    return jnp.stack([pixel_y, pixel_x], axis=-1)


def extract_window_data_for_viz(
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

    This is the main entry point for visualization. It extracts attention maps
    for a selected window and selected pixels within that window.

    Args:
        self_attention_weights: (B*num_windows, window_size^2, window_size^2) self-attention per window
        cross_attention_weights: (B*num_windows, window_size^2, window_size^2) cross-attention per window
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
            - seed_used: seed used for pixel selection
    """
    # Select random pixels within the window
    pixel_indices = select_random_pixels(
        window_size=window_size,
        num_pixels=num_pixels,
        seed=pixel_selection_seed,
    )

    # Calculate flat window index
    window_idx = window_indices[0] * num_windows_w + window_indices[1]
    
    # Extract this window's attention (shape: window_size^2 x window_size^2)
    window_self_attn = self_attention_weights[window_idx]
    window_cross_attn = cross_attention_weights[window_idx]

    # Add batch dimension for extract_pixel_attention_maps (requires 3D input)
    window_self_attn = window_self_attn[jnp.newaxis, :, :]  # (1, window_size^2, window_size^2)
    window_cross_attn = window_cross_attn[jnp.newaxis, :, :]

    # Extract pixel-specific attention maps
    self_attn_maps = extract_pixel_attention_maps(
        window_self_attn,
        pixel_indices,
        window_size,
    )
    cross_attn_maps = extract_pixel_attention_maps(
        window_cross_attn,
        pixel_indices,
        window_size,
    )

    # Remove batch dimension from results
    self_attn_maps = self_attn_maps[0]  # (num_pixels, window_size, window_size)
    cross_attn_maps = cross_attn_maps[0]

    # Compute pixel positions
    pixel_positions = compute_pixel_positions(pixel_indices, window_size)

    return {
        "self_attention_maps": np.array(self_attn_maps),
        "cross_attention_maps": np.array(cross_attn_maps),
        "pixel_positions": np.array(pixel_positions),
        "seed_used": pixel_selection_seed,
    }
