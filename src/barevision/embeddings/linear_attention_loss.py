"""Linear attention flow loss for embeddings training.

Replaces softmax-based spatial variance loss with linear attention mechanism.

Core idea:
    Instead of softmax attention (O(N2)), use linear attention (O(D)):
    - Pre-compute: K_coords = K.T @ coords  # (D, 2) per window
    - Per-query: COM = Q @ K_coords  # (2,) center of mass
    - Flow: cross_com - self_com

Loss components:
    1. Warped reconstruction: frame1 ≈ warp(frame2, flow)
    2. Embedding diversity: prevent constant embedding collapse

Note: Flow concordance loss deferred to future session.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Tuple

from barevision.utils.grid import (
    crop_to_grid_aligned,
    WindowGrid,
    generate_normalized_coordinates,
)


class LinearAttentionFlowLossConfig(BaseModel):
    """Configuration for linear attention flow loss.

    Attributes:
        window_size: Size of non-overlapping windows (default 16)
        level_weight_decay: Weight decay for coarser levels (default 1.0)
        lambda_reconstruction: Weight for warped reconstruction loss (default 1.0)
        lambda_diversity: Weight for embedding diversity loss (default 0.1)
        diversity_scope: 'per_window' or 'global'. Per-window is simpler and matches attention structure.
            Note: Global scope is an alternative to explore in future ablations.
    """

    model_config = ConfigDict(frozen=True)

    window_size: int = 16
    level_weight_decay: float = 1.0
    lambda_reconstruction: float = 1.0
    lambda_diversity: float = 0.1
    diversity_scope: str = "per_window"  # 'per_window' or 'global'


def _compute_linear_attention_flow(
    windows_q: jnp.ndarray, windows_k: jnp.ndarray, coords: jnp.ndarray, window_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute flow using linear attention mechanism.

    Args:
        windows_q: Query embeddings (B*num_windows, H_w, W_w, D)
        windows_k: Key embeddings (B*num_windows, H_w, W_w, D)
        coords: Normalized coordinates (H_w*W_w, 2)
        window_size: Size of window for flow clipping

    Returns:
        Tuple of:
            - flow: (B*num_windows, H_w, W_w, 2) flow vectors
            - confidence: (B*num_windows, H_w, W_w) confidence scores (negative variance)
    """
    Bnw, H_w, W_w, D = windows_q.shape
    N = H_w * W_w

    # Flatten spatial dimensions
    q_flat = windows_q.reshape(Bnw, N, D)
    k_flat = windows_k.reshape(Bnw, N, D)

    # Pre-compute K_coords once per window: (Bnw, D, 2)
    # For each dimension, where in space does it fire?
    k_coords_self = jnp.einsum("bnd,nk->bdk", q_flat, coords)
    k_coords_cross = jnp.einsum("bnd,nk->bdk", k_flat, coords)

    # Per-query center of mass (unnormalized): (Bnw, N, 2)
    self_com_unnorm = jnp.einsum("bnd,bdk->bnk", q_flat, k_coords_self)
    cross_com_unnorm = jnp.einsum("bnd,bdk->bnk", q_flat, k_coords_cross)

    # Normalize by sum of weights to get true weighted average
    # Weight sum per position: (Bnw, N)
    weight_sum_self = q_flat.sum(axis=2)
    weight_sum_cross = k_flat.sum(axis=2)
    
    # Normalized COM: (Bnw, N, 2)
    self_com = self_com_unnorm / (weight_sum_self[..., None] + 1e-8)
    cross_com = cross_com_unnorm / (weight_sum_cross[..., None] + 1e-8)

    # Flow: (Bnw, N, 2)
    # Now in normalized coordinates: 1.0 ≈ full window span
    flow = cross_com - self_com

    # Diagnostics for monitoring
    flow_stats = {
        "self_com_min": self_com.min(),
        "self_com_max": self_com.max(),
        "cross_com_min": cross_com.min(),
        "cross_com_max": cross_com.max(),
        "flow_min": flow.min(),
        "flow_max": flow.max(),
        "weight_sum_self_mean": weight_sum_self.mean(),
        "weight_sum_cross_mean": weight_sum_cross.mean(),
    }

    # Confidence: negative variance of per-dimension flow contributions
    # Per-dimension flow: (Bnw, N, D, 2)
    self_pos_per_dim = jnp.einsum("bnd,bdk->bndk", q_flat, k_coords_self)
    cross_pos_per_dim = jnp.einsum("bnd,bdk->bndk", q_flat, k_coords_cross)
    flow_per_dim = cross_pos_per_dim - self_pos_per_dim

    # Variance across dimensions: (Bnw, N)
    flow_variance = jnp.var(flow_per_dim, axis=2).mean(axis=2)
    confidence = -flow_variance

    # Reshape to spatial dimensions
    flow = flow.reshape(Bnw, H_w, W_w, 2)
    confidence = confidence.reshape(Bnw, H_w, W_w)

    return flow, confidence, flow_stats


def _warp_embeddings(
    embeddings: jnp.ndarray, flow: jnp.ndarray, window_size: int
) -> jnp.ndarray:
    """Warp embeddings by predicted flow using bilinear interpolation.

    Uses map_coordinates for differentiable warping.
    All D channels at each position warp to the same location.

    Args:
        embeddings: Source embeddings (B*num_windows, H_w, W_w, D)
        flow: Flow vectors (B*num_windows, H_w, W_w, 2)
        window_size: Size of window for coordinate calculation

    Returns:
        Warped embeddings (B*num_windows, H_w, W_w, D)
    """
    Bnw, H_w, W_w, D = embeddings.shape

    # Create coordinate grid
    y_coords = jnp.arange(H_w, dtype=jnp.float32)
    x_coords = jnp.arange(W_w, dtype=jnp.float32)
    yy, xx = jnp.meshgrid(y_coords, x_coords, indexing="ij")

    # Warp one batch item: (H_w, W_w, D) with flow (H_w, W_w, 2)
    def warp_one(emb_one, flow_one):
        # Coordinates for map_coordinates: (2, H_w, W_w)
        coords = jnp.stack([
            yy + flow_one[:, :, 0],
            xx + flow_one[:, :, 1],
        ], axis=0)

        # Warp each channel with same coordinates
        def warp_ch(ch):
            return map_coordinates(ch, coords, order=1, mode="nearest")

        # vmap over channel dimension
        return jax.vmap(warp_ch, in_axes=-1, out_axes=-1)(emb_one)

    # vmap over batch dimension
    return jax.vmap(warp_one)(embeddings, flow)


def _compute_warped_reconstruction_loss(
    windows1: jnp.ndarray,
    windows2: jnp.ndarray,
    flow: jnp.ndarray,
    window_size: int,
) -> jnp.ndarray:
    """Compute warped reconstruction loss.

    Warp frame2 embeddings by flow, compare to frame1 embeddings.

    Args:
        windows1: Frame 1 embeddings (B*num_windows, H_w, W_w, D)
        windows2: Frame 2 embeddings (B*num_windows, H_w, W_w, D)
        flow: Predicted flow in normalized coordinates (B*num_windows, H_w, W_w, 2)
            where 1.0 ≈ full window span
        window_size: Window size

    Returns:
        Scalar MSE loss
    """
    # Convert flow from normalized to pixel coordinates
    flow_pixels = flow * window_size
    warped = _warp_embeddings(windows2, flow_pixels, window_size)
    mse = jnp.mean((windows1 - warped) ** 2)
    return mse


def _compute_embedding_variance(
    windows: jnp.ndarray, scope: str = "per_window"
) -> jnp.ndarray:
    """Compute embedding variance across spatial positions.

    Measures how much embeddings vary across space.
    Higher = more diverse embeddings.

    Args:
        windows: Embeddings (B*num_windows, H_w, W_w, D)
        scope: 'per_window' or 'global'
            - per_window: variance computed independently per window
            - global: variance computed across all positions

    Returns:
        Scalar variance (average across dimensions)
    """
    if scope == "global":
        # Flatten all spatial positions
        flat = windows.reshape(-1, windows.shape[-1])
        variance = jnp.var(flat, axis=0).mean()
    else:
        # Per-window diversity
        variance = jnp.var(windows, axis=(1, 2)).mean(axis=1)
        variance = variance.mean()

    return variance


def _compute_linear_attention_flow_loss(
    emb1: jnp.ndarray, emb2: jnp.ndarray, config: LinearAttentionFlowLossConfig
) -> Tuple[jnp.ndarray, dict]:
    """Compute linear attention flow loss for a single level.

    Args:
        emb1: Frame 1 embeddings (B, H, W, D)
        emb2: Frame 2 embeddings (B, H, W, D)
        config: Loss configuration

    Returns:
        Tuple of:
            - total_loss: scalar
            - aux: dict with per-component losses and diagnostics
    """
    B, H, W, D = emb1.shape
    window_size = config.window_size

    # Split into windows
    grid = WindowGrid(window_size=window_size)
    w1 = grid.split(emb1)
    w2 = grid.split(emb2)

    num_windows = (H // window_size) * (W // window_size)
    fw1 = w1.reshape(B * num_windows, window_size, window_size, D)
    fw2 = w2.reshape(B * num_windows, window_size, window_size, D)

    # Generate coordinates
    coords = generate_normalized_coordinates(window_size)

    # Compute flow
    flow, confidence, flow_stats = _compute_linear_attention_flow(fw1, fw2, coords, window_size)

    # Warped reconstruction loss
    loss_reconstruction = _compute_warped_reconstruction_loss(
        fw1, fw2, flow, window_size
    )

    # Embedding diversity loss (on both frames)
    # Variance is computed, then converted to normalized loss: 0 = perfect, 1 = collapse
    variance_1 = _compute_embedding_variance(fw1, config.diversity_scope)
    variance_2 = _compute_embedding_variance(fw2, config.diversity_scope)
    variance_avg = (variance_1 + variance_2) / 2
    
    # Normalized diversity loss: 1 - (variance / max_variance)
    # max_variance = 0.25 for L2-normalized embeddings in [0, 1]
    max_variance = 0.25
    loss_diversity = 1.0 - (variance_avg / max_variance)

    # Total loss (all components are now positive, 0 = perfect)
    total_loss = (
        config.lambda_reconstruction * loss_reconstruction
        + config.lambda_diversity * loss_diversity
    )

    aux = {
        "reconstruction_loss": loss_reconstruction,
        "diversity_loss": loss_diversity,
        "diversity_variance": variance_avg,
        "flow": flow,
        "confidence": confidence,
        "flow_stats": flow_stats,
    }

    return total_loss, aux


def compute_hierarchical_linear_attention_loss(
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    config: LinearAttentionFlowLossConfig,
    need_aux: bool = True,
) -> Tuple[jnp.ndarray, dict]:
    """Compute hierarchical linear attention flow loss across pyramid levels.

    Args:
        pyramid1: Frame 1 embedding pyramid
        pyramid2: Frame 2 embedding pyramid
        config: Loss configuration
        need_aux: Whether to return auxiliary data

    Returns:
        Tuple of:
            - total_loss: scalar
            - aux: dict with per-level losses and diagnostics
    """
    coords = generate_normalized_coordinates(config.window_size)

    total_loss = 0.0
    total_weight = 0.0
    total_reconstruction = 0.0
    total_diversity = 0.0

    aux = {
        "level_losses": [],
        "level_weights": [],
        "level_reconstruction_losses": [],
        "level_diversity_losses": [],
    }

    for i, (e1, e2) in enumerate(zip(pyramid1, pyramid2)):
        # Crop to grid-aligned dimensions
        e1_cropped = crop_to_grid_aligned(e1, config.window_size)
        e2_cropped = crop_to_grid_aligned(e2, config.window_size)

        weight = config.level_weight_decay**i

        level_loss, level_aux = _compute_linear_attention_flow_loss(
            e1_cropped, e2_cropped, config
        )

        total_loss += level_loss * weight
        total_reconstruction += level_aux["reconstruction_loss"] * weight
        total_diversity += level_aux["diversity_loss"] * weight
        total_weight += weight

        aux["level_losses"].append(level_loss * weight)
        aux["level_weights"].append(weight)
        aux["level_reconstruction_losses"].append(
            level_aux["reconstruction_loss"] * weight
        )
        aux["level_diversity_losses"].append(level_aux["diversity_loss"] * weight)

        if need_aux:
            # Store flow and confidence for visualization
            if "flow" not in aux:
                aux["flow"] = []
                aux["confidence"] = []
            aux["flow"].append(level_aux["flow"])
            aux["confidence"].append(level_aux["confidence"])

    aux["reconstruction_loss"] = total_reconstruction / total_weight
    aux["diversity_loss"] = total_diversity / total_weight

    return total_loss / total_weight, aux


class HierarchicalLinearAttentionFlowLoss:
    """Hierarchical linear attention flow loss wrapper."""

    def __init__(self, config: LinearAttentionFlowLossConfig):
        self.config = config

    def __call__(
        self,
        pyramid_pair: Tuple[List[jnp.ndarray], List[jnp.ndarray]],
        need_aux: bool = True,
    ) -> Tuple[jnp.ndarray, dict]:
        return compute_hierarchical_linear_attention_loss(
            pyramid_pair[0], pyramid_pair[1], self.config, need_aux
        )
