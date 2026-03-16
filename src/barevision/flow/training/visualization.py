"""Visualization for optical flow training.

Orchestrates visualization from embeddings and matching packages.
"""

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from barevision.flow.embeddings.visualization import (
    create_attention_maps_figure,
)
from barevision.flow.matching.visualization import flow_to_arrows, flow_to_colorwheel


def _select_random_pixels(
    window_size: int,
    num_pixels: int = 4,
    seed: int = 0,
) -> jnp.ndarray:
    """Select random pixel indices within a window."""
    N = window_size * window_size
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, N, shape=(num_pixels,), replace=False)


def _compute_pixel_positions(
    pixel_indices: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Compute (y, x) positions for pixel indices within a window."""
    pixel_y = pixel_indices // window_size
    pixel_x = pixel_indices % window_size
    return jnp.stack([pixel_y, pixel_x], axis=-1)


def _extract_pixel_attention_maps(
    attention_weights: jnp.ndarray,
    pixel_indices: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Extract attention maps for specific pixels within a window.

    Requires batched input (3D). Fails if given 2D input.
    """
    assert attention_weights.ndim == 3, f"Expected batched input (3D), got {attention_weights.ndim}D"
    
    B, N, _ = attention_weights.shape
    selected_attn = attention_weights[:, pixel_indices, :]
    selected_attn = selected_attn.reshape(B, -1, window_size, window_size)
    return selected_attn


def _extract_window_attention_data(
    self_attention_weights: jnp.ndarray,
    cross_attention_weights: jnp.ndarray,
    window_indices: tuple[int, int],
    num_windows_h: int,
    num_windows_w: int,
    window_size: int = 16,
    pixel_selection_seed: int = 0,
    num_pixels: int = 4,
) -> dict:
    """Extract attention data for visualizing a specific window.

    Validates window indices are in bounds.
    """
    # Validate window indices
    assert 0 <= window_indices[0] < num_windows_h, f"window row {window_indices[0]} >= num_windows_h {num_windows_h}"
    assert 0 <= window_indices[1] < num_windows_w, f"window col {window_indices[1]} >= num_windows_w {num_windows_w}"
    
    # Select random pixels
    pixel_indices = _select_random_pixels(window_size, num_pixels, pixel_selection_seed)

    # Calculate flat window index and extract
    window_idx = window_indices[0] * num_windows_w + window_indices[1]
    window_self_attn = self_attention_weights[window_idx]
    window_cross_attn = cross_attention_weights[window_idx]

    # Add batch dimension for extraction
    window_self_attn = window_self_attn[jnp.newaxis, :, :]
    window_cross_attn = window_cross_attn[jnp.newaxis, :, :]

    # Extract and remove batch dimension
    self_attn_maps = _extract_pixel_attention_maps(window_self_attn, pixel_indices, window_size)[0]
    cross_attn_maps = _extract_pixel_attention_maps(window_cross_attn, pixel_indices, window_size)[0]

    return {
        "self_attention_maps": np.array(self_attn_maps),
        "cross_attention_maps": np.array(cross_attn_maps),
        "pixel_positions": np.array(_compute_pixel_positions(pixel_indices, window_size)),
        "seed_used": pixel_selection_seed,
    }


def log_visualizations(
    logger,
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    flow: Optional[jnp.ndarray],
    aux_data: Optional[Dict],
    step: int,
    window_size: int = 16,
    num_levels: int = 3,
):
    """Generate and log visualization figures for optical flow training.

    Orchestrates visualization from both embeddings and matching packages:
    - Embeddings: attention maps, grid overlays for each pyramid level
    - Matching: flow field colorwheel and arrows

    Args:
        logger: JaxLogger instance for TensorBoard logging
        pyramid1: List of embedding pyramids for frame 1
        pyramid2: List of embedding pyramids for frame 2
        flow: (B, H, W, 2) predicted flow field (optional)
        aux_data: Auxiliary data from training step (optional)
        step: Global step for logging
        window_size: Attention window size in pixels
        num_levels: Number of pyramid levels
    """
    import gc

    # Log flow visualization if available
    if flow is not None:
        flow_viz = np.array(flow[0]) if flow.shape[0] == 1 else np.array(flow.mean(axis=0))

        # Colorwheel visualization
        flow_rgb = flow_to_colorwheel(flow_viz, max_flow=0.3)
        logger.log_image("Flow/Predicted_Colorwheel", flow_rgb, step)

        # Arrow visualization
        arrows_rgb = flow_to_arrows(flow_viz, max_flow=0.3, scale=2.0, grid_density=8)
        logger.log_image("Flow/Predicted_Arrows", arrows_rgb, step)

    # Log visualizations for each pyramid level
    for level_idx in range(num_levels):
        level_emb = pyramid1[level_idx]
        B, H_emb, W_emb, _ = level_emb.shape

        # Calculate number of windows at this level
        num_windows_h = H_emb // window_size
        num_windows_w = W_emb // window_size

        # Skip levels that are too small
        if num_windows_h == 0 or num_windows_w == 0:
            continue

        # Select random window at this level
        rng = np.random.default_rng(seed=step + level_idx * 1000)
        window_row = int(rng.integers(0, num_windows_h))
        window_col = int(rng.integers(0, num_windows_w))
        window_indices = (window_row, window_col)

        # Extract attention data from aux if available
        if aux_data is not None and "loss" in aux_data:
            loss_aux = aux_data["loss"]
            self_attn_list = loss_aux.get("level_self_attention_weights", None)
            cross_attn_list = loss_aux.get("level_cross_attention_weights", None)

            if self_attn_list is not None and len(self_attn_list) > level_idx:
                viz_data = _extract_window_attention_data(
                    self_attention_weights=self_attn_list[level_idx],
                    cross_attention_weights=cross_attn_list[level_idx],
                    window_indices=window_indices,
                    num_windows_h=num_windows_h,
                    num_windows_w=num_windows_w,
                    window_size=window_size,
                    pixel_selection_seed=step + level_idx * 1000,
                )

                # Convert embeddings to numpy for visualization
                img1_downscaled = jax.image.resize(
                    pyramid1[level_idx][0], (H_emb, W_emb, 3), method="bilinear"
                )

                # Extract window crop
                emb_h_start = window_row * window_size
                emb_w_start = window_col * window_size
                window_crop = np.array(
                    img1_downscaled[
                        emb_h_start : emb_h_start + window_size,
                        emb_w_start : emb_w_start + window_size,
                        :,
                    ]
                )

                # Create attention maps figure
                fig_attn = create_attention_maps_figure(
                    self_attention_maps=viz_data["self_attention_maps"],
                    cross_attention_maps=viz_data["cross_attention_maps"],
                    pixel_positions=viz_data["pixel_positions"],
                    window_crop=window_crop,
                    seed_used=viz_data["seed_used"],
                )

                logger.log_figure(f"Level{level_idx}/Attention_Maps", fig_attn, step)
                plt.close(fig_attn)

    # Clean up
    gc.collect()
