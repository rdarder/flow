"""Visualization for optical flow training.

Orchestrates visualization from embeddings and matching packages.
"""

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from barevision.flow.embeddings.visualization import (
    create_attention_maps_figure,
    create_frame_with_grid_figure,
)
from barevision.flow.matching.visualization import flow_to_arrows, flow_to_colorwheel
from barevision.flow.visualization_attention import extract_window_data_for_viz


def log_visualizations(
    logger,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    flows: Optional[List[jnp.ndarray]],
    aux_data: Optional[Dict],
    metadata: dict,
    step: int,
    window_size: int = 16,
    num_levels: int = 3,
):
    """Generate and log visualization figures for optical flow training.

    Orchestrates visualization from both embeddings and matching packages:
    - Embeddings: frame grid, attention maps for each pyramid level
    - Matching: flow field colorwheel and arrows (per level)

    V1: Visualizes flow at each pyramid level independently.
    V2: May visualize priors and window shifts.

    Args:
        logger: JaxLogger instance for TensorBoard logging
        img1: Frame 1 (1, H, W, 3) - original RGB
        img2: Frame 2 (1, H, W, 3) - original RGB
        pyramid1: List of embedding pyramids for frame 1
        pyramid2: List of embedding pyramids for frame 2
        flows: List of predicted flow fields, one per level (optional)
        aux_data: Auxiliary data from training step (optional)
        metadata: dict with video_name, frame_t, frame_tk, distance
        step: Global step for logging
        window_size: Attention window size in pixels
        num_levels: Number of pyramid levels
    """
    import gc

    # Log flow visualization if available (for each level)
    if flows is not None:
        for level_idx, flow in enumerate(flows):
            flow_viz = (
                np.array(flow[0]) if flow.shape[0] == 1 else np.array(flow.mean(axis=0))
            )

            # Colorwheel visualization with adaptive scaling for better contrast
            flow_rgb = flow_to_colorwheel(flow_viz, max_flow=0.3, adaptive=True)
            logger.log_image(f"Level{level_idx}/flow_colorwheel", flow_rgb, step)

            # Arrow visualization
            arrows_rgb = flow_to_arrows(
                flow_viz,
                max_flow=0.3,
                window_size=window_size,
                grid_density=window_size,
            )
            logger.log_image(f"Level{level_idx}/flow_arrows", arrows_rgb, step)

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
            self_attn_list = loss_aux.get("level_self_attention_weights", None)
            cross_attn_list = loss_aux.get("level_cross_attention_weights", None)

            if self_attn_list is not None and len(self_attn_list) > level_idx:
                viz_data = extract_window_data_for_viz(
                    self_attention_weights=self_attn_list[level_idx],
                    cross_attention_weights=cross_attn_list[level_idx],
                    window_indices=window_indices,
                    num_windows_h=num_windows_h,
                    num_windows_w=num_windows_w,
                    window_size=window_size,
                    pixel_selection_seed=step + level_idx * 1000,
                )

                # Extract window crop from downscaled images
                emb_h_start = window_row * window_size
                emb_w_start = window_col * window_size
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

                # 2. Attention maps for selected pixels
                fig_attn = create_attention_maps_figure(
                    window_crop1=window_crop1,
                    window_crop2=window_crop2,
                    self_attn_maps=viz_data["self_attention_maps"],
                    cross_attn_maps=viz_data["cross_attention_maps"],
                    pixel_positions=viz_data["pixel_positions"],
                    window_size=window_size,
                    window_indices=window_indices,
                    frame_t=metadata.get("frame_t", 0),
                    frame_tk=metadata.get("frame_tk", 0),
                    distance=metadata.get("distance", 0),
                )
                logger.log_figure(f"Level{level_idx}/Attention_Maps", fig_attn, step)

    # Clean up
    gc.collect()
