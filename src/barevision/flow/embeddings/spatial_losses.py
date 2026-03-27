"""Spatial variance loss functions for embedding training.

Organization (top to bottom: coarse-grained to fine-grained):
1. HierarchicalSpatialVarianceLoss: Multi-level pyramid loss class
2. compute_hierarchical_spatial_variance_loss: Functional multi-level loss
3. windowed_spatial_variance_losses: Single-level window-based loss
4. cross_attention_spatial_variance: Core cross-attention variance math
5. self_attention_spatial_variance: Core self-attention variance math
6. _compute_spatial_variance: Utility function for variance computation

Loss principles:
- Self-attention: minimize spatial variance → attention concentrates near source pixel
- Cross-attention: minimize spatial variance → attention finds specific match in target frame

Spatial variance measures how concentrated vs spread out attention weights are.
Lower variance = sharper, more spatially localized attention peaks.
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp

from barevision.flow.settings import SpatialVarianceLossSettings
from barevision.utils.checks import check_value
from barevision.utils.grid import crop_to_grid_aligned, WindowGrid


def _generate_normalized_coordinates(window_size: int) -> jnp.ndarray:
    """Generate normalized coordinate grids for a window.

    Coordinates are normalized to [0, 1] range.

    Args:
        window_size: Window dimension

    Returns:
        coords: (N, 2) array of (y, x) positions where N = window_size²
    """
    # Generate 1D coordinate arrays normalized to [0, 1]
    ys = jnp.arange(window_size).astype(jnp.float32) / (window_size - 1)
    xs = jnp.arange(window_size).astype(jnp.float32) / (window_size - 1)

    # Create 2D grid and reshape to (N, 2)
    coords_2d = jnp.stack(jnp.meshgrid(ys, xs, indexing="ij"), axis=-1)
    return coords_2d.reshape(-1, 2)  # (N, 2)


def _compute_spatial_variance(
    attention_weights: jnp.ndarray,
    coords: jnp.ndarray,
) -> jnp.ndarray:
    """Compute spatial variance using weighted squared differences.

    For each query position, measures how spatially concentrated vs
    spread out its attention pattern is. This stable formulation avoids
    the E[X²] - E[X]² subtraction which can suffer from numerical instability.

    Instead computes: Σ_j (attention[j] * ||coords[j] - mean_pos||²)

    Args:
        attention_weights: (B, N, N) softmax-normalized attention matrix
        coords: (N, 2) normalized coordinate positions

    Returns:
        (B, N) variance per query position
    """
    # 1. Compute mean position (B, N, 2)
    mean_pos = jnp.einsum("bnk,kd->bnd", attention_weights, coords)

    # 2. Compute squared distance from mean for every key position
    # mean_pos: (B, N, 1, 2) to broadcast against coords: (N, 2)
    diff_sq = (
        coords[jnp.newaxis, jnp.newaxis, :, :] - mean_pos[:, :, jnp.newaxis, :]
    ) ** 2
    # sum along the (y, x) dimension: (B, N, N)
    dist_sq = jnp.sum(diff_sq, axis=-1)

    # 3. Weighted sum of squared distances
    # Sum over the 'k' dimension (the attention distribution)
    variance = jnp.einsum("bnk,bnk->bn", attention_weights, dist_sq)

    return variance


def self_attention_spatial_variance(
    windows: jnp.ndarray,
    temperature: float,
    coords: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """Compute self-attention spatial variance loss on a batch of windows.

    Self-attention should concentrate near the source pixel. Low variance
    indicates that each query position attends to a small, localized region.

    Args:
        windows: (B, H, W, D) batch of windows (already split and flattened)
        temperature: Softmax temperature for attention sharpness
        coords: (N, 2) precomputed normalized coordinates

    Returns:
        Tuple of (loss, aux_dict) where:
            - loss: (B, H, W) per-pixel spatial variance
            - aux_dict: {'attention_weights': (B, N, N), 'variance_map': (B, H, W)}
    """
    B, H, W, D = windows.shape
    N = H * W

    # Flatten spatial dimensions
    flat_windows = windows.reshape(B, N, D)

    # Compute self-attention logits: q·k for all pairs
    logits = flat_windows @ flat_windows.transpose(0, 2, 1)  # (B, N, N)

    # Softmax to get attention weights
    attn_weights = jax.nn.softmax(logits / temperature, axis=-1)

    # Compute spatial variance
    variance = _compute_spatial_variance(attn_weights, coords)

    # Reshape back to spatial grid
    variance_grid = variance.reshape(B, H, W)

    return variance_grid, {
        "attention_weights": attn_weights,
        "variance_map": variance_grid,
    }


def cross_attention_spatial_variance(
    windows1: jnp.ndarray,
    windows2: jnp.ndarray,
    temperature: float,
    coords: jnp.ndarray,
) -> Tuple[jnp.ndarray, dict]:
    """Compute cross-attention spatial variance loss on a batch of windows.

    Cross-attention should find a specific match in the target frame.
    Low variance indicates confident, localized matching.

    Args:
        windows1: (B, H, W, D) batch of windows from frame 1
        windows2: (B, H, W, D) batch of windows from frame 2
        temperature: Softmax temperature for attention sharpness
        coords: (N, 2) precomputed normalized coordinates (for window2)

    Returns:
        Tuple of (loss, aux_dict) where:
            - loss: (B, H, W) per-pixel spatial variance
            - aux_dict: {'attention_weights': (B, N, N), 'variance_map': (B, H, W)}
    """
    B, H, W, D = windows1.shape
    N = H * W

    # Flatten spatial dimensions
    flat1 = windows1.reshape(B, N, D)
    flat2 = windows2.reshape(B, N, D)

    # Compute cross-attention logits: q1·k2 for all pairs
    logits = flat1 @ flat2.transpose(0, 2, 1)  # (B, N, N)

    # Softmax to get attention weights
    attn_weights = jax.nn.softmax(logits / temperature, axis=-1)

    # Compute spatial variance
    variance = _compute_spatial_variance(attn_weights, coords)

    # Reshape back to spatial grid
    variance_grid = variance.reshape(B, H, W)

    return variance_grid, {
        "attention_weights": attn_weights,
        "variance_map": variance_grid,
    }


def windowed_spatial_variance_losses(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    window_size: int,
    lambda_self: float,
    self_temperature: float,
    cross_temperature: float,
) -> Tuple[jnp.ndarray, dict]:
    """Compute combined self and cross attention spatial variance losses.

    The losses are computed on fixed-size windows that make up the frames.

    Args:
        emb1: (B, H, W, D) embeddings from frame 1
        emb2: (B, H, W, D) embeddings from frame 2
        window_size: Size of attention windows
        lambda_self: Self-attention loss weight in [0, 1]
        self_temperature: Temperature for self-attention softmax
        cross_temperature: Temperature for cross-attention softmax

    Returns:
        Tuple of (combined_loss, aux_dict) where:
            - combined_loss: scalar combined loss value
            - aux_dict: {'self_loss': scalar, 'cross_loss': scalar,
                        'self_attention_weights': (B*N, N, N),
                        'cross_attention_weights': (B*N, N, N),
                        'self_variance_maps': (B, H, W),
                        'cross_variance_maps': (B, H, W)}
    """
    B, H, W, D = emb1.shape

    # Validate resolution
    if H % window_size != 0:
        raise ValueError(f"Height {H} not divisible by window_size {window_size}")
    if W % window_size != 0:
        raise ValueError(f"Width {W} not divisible by window_size {window_size}")

    # Validate shapes match
    check_value(
        emb2.shape == emb1.shape, f"emb2 shape {emb2.shape} != emb1 shape {emb1.shape}"
    )

    # Precompute coordinates for this window size
    coords = _generate_normalized_coordinates(window_size)

    # Split into windows
    grid = WindowGrid(window_size=window_size)
    windows1 = grid.split(emb1)
    windows2 = grid.split(emb2)

    # Flatten batch and windows
    num_windows = (H // window_size) * (W // window_size)
    flat_windows1 = windows1.reshape(B * num_windows, window_size, window_size, D)
    flat_windows2 = windows2.reshape(B * num_windows, window_size, window_size, D)

    # Compute core losses
    self_variance_flat, self_aux = self_attention_spatial_variance(
        flat_windows1,
        temperature=self_temperature,
        coords=coords,
    )
    cross_variance_flat, cross_aux = cross_attention_spatial_variance(
        flat_windows1,
        flat_windows2,
        temperature=cross_temperature,
        coords=coords,
    )

    # Reshape back to spatial grid
    def reshape_to_grid(variance_flat):
        variance = variance_flat.reshape(B, num_windows, window_size, window_size)
        variance = variance.reshape(
            B, H // window_size, W // window_size, window_size, window_size
        )
        variance = variance.transpose(0, 1, 3, 2, 4)
        return variance.reshape(B, H, W)

    # Compute mean loss per window type
    self_loss = reshape_to_grid(self_variance_flat).mean()
    cross_loss = reshape_to_grid(cross_variance_flat).mean()

    # Combine with weighting
    combined = (1 - lambda_self) * cross_loss + lambda_self * self_loss

    aux = dict(self_loss=self_loss, cross_loss=cross_loss)
    aux["self_attention_weights"] = self_aux["attention_weights"]
    aux["cross_attention_weights"] = cross_aux["attention_weights"]
    aux["self_variance_maps"] = self_aux["variance_map"]
    aux["cross_variance_maps"] = cross_aux["variance_map"]

    return combined, aux


def compute_hierarchical_spatial_variance_loss(
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    settings: SpatialVarianceLossSettings,
) -> Tuple[jnp.ndarray, dict]:
    """Compute compound spatial variance loss across all pyramid levels.

    Applies windowed spatial variance loss at each level and aggregates
    with level-weighted sum.

    Args:
        pyramid1: List of feature maps from frame 1, one per level
        pyramid2: List of feature maps from frame 2, one per level
        settings: Loss configuration

    Returns:
        Tuple of (total_loss, aux_dict) where:
            - total_loss: scalar sum of weighted per-level losses
            - aux_dict: {'self_loss': scalar, 'cross_loss': scalar,
                        'level_losses': [...], 'level_weights': [...],
                        'level_self_attention_weights': [...],
                        'level_cross_attention_weights': [...],
                        'level_self_variance_maps': [...],
                        'level_cross_variance_maps': [...]}
    """
    if len(pyramid1) != len(pyramid2):
        raise ValueError(f"Pyramid level mismatch: {len(pyramid1)} vs {len(pyramid2)}")

    num_levels = len(pyramid1)
    level_losses = []
    level_weights = []
    total_loss = jnp.array(0.0)
    total_self_loss = jnp.array(0.0)
    total_cross_loss = jnp.array(0.0)

    # Aux data storage per level
    level_self_attn = []
    level_cross_attn = []
    level_self_variance = []
    level_cross_variance = []

    for level_idx in range(num_levels):
        level_weight = settings.level_weight_decay**level_idx

        emb1 = pyramid1[level_idx]
        emb2 = pyramid2[level_idx]

        # Crop to grid-aligned dimensions
        emb1_cropped = crop_to_grid_aligned(emb1, settings.window_size)
        emb2_cropped = crop_to_grid_aligned(emb2, settings.window_size)

        B, H, W, D = emb1_cropped.shape

        # Validate we have at least one window
        num_windows_h = H // settings.window_size
        num_windows_w = W // settings.window_size
        if num_windows_h == 0 or num_windows_w == 0:
            raise ValueError(
                f"Level {level_idx}: Cropped dimensions ({H}x{W}) too small for window_size {settings.window_size}"
            )

        # Compute loss at this level
        level_loss, level_aux = windowed_spatial_variance_losses(
            emb1_cropped,
            emb2_cropped,
            window_size=settings.window_size,
            lambda_self=settings.lambda_self,
            self_temperature=settings.self_temperature,
            cross_temperature=settings.cross_temperature,
        )

        # Apply level weight
        level_loss_weighted = level_loss * level_weight

        level_losses.append(level_loss_weighted)
        level_weights.append(level_weight)
        total_self_loss += level_aux["self_loss"] * level_weight
        total_cross_loss += level_aux["cross_loss"] * level_weight
        total_loss += level_loss_weighted

        # Aggregate aux data
        level_self_attn.append(level_aux["self_attention_weights"])
        level_cross_attn.append(level_aux["cross_attention_weights"])
        level_self_variance.append(level_aux["self_variance_maps"])
        level_cross_variance.append(level_aux["cross_variance_maps"])

    # Normalize by total weight
    total_weight = sum(level_weights)
    total_loss = total_loss / total_weight
    total_self_loss = total_self_loss / total_weight
    total_cross_loss = total_cross_loss / total_weight

    aux = dict(
        self_loss=total_self_loss,
        cross_loss=total_cross_loss,
        level_losses=level_losses,
        level_weights=level_weights,
        level_self_attention_weights=level_self_attn,
        level_cross_attention_weights=level_cross_attn,
        level_self_variance_maps=level_self_variance,
        level_cross_variance_maps=level_cross_variance,
    )

    return total_loss, aux


class HierarchicalSpatialVarianceLoss:
    """Hierarchical spatial variance loss for multi-level pyramid training.

    This is the main loss class for embeddings training. It computes spatial
    variance at all pyramid levels and aggregates with configurable weighting.
    """

    def __init__(self, settings: SpatialVarianceLossSettings):
        self.settings = settings

    def __call__(
        self,
        pyramid_pair: Tuple[List[jnp.ndarray], List[jnp.ndarray]],
        need_aux: bool = True,
    ) -> Tuple[jnp.ndarray, dict]:
        """Compute hierarchical spatial variance loss.

        Args:
            pyramid_pair: Tuple of (pyramid1, pyramid2) embedding lists
            need_aux: Whether to return auxiliary data for visualization

        Returns:
            Tuple of (loss, aux_dict)
        """
        pyramid1, pyramid2 = pyramid_pair

        check_value(
            len(pyramid1) == len(pyramid2),
            f"Pyramid level mismatch: {len(pyramid1)} vs {len(pyramid2)}",
        )

        return compute_hierarchical_spatial_variance_loss(
            pyramid1, pyramid2, self.settings
        )
