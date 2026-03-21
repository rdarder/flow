"""Latent space reconstruction loss.

Warp Frame 1 embeddings using predicted flow and compare to Frame 2 embeddings.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from barevision.flow.settings import FlowLossSettings
from barevision.utils.checks import check_value
from barevision.utils.grid import crop_to_grid_aligned


def warp_embeddings(embeddings: jnp.ndarray, flow: jnp.ndarray) -> jnp.ndarray:
    """Backward warp embeddings using flow field.

    Flow convention: (u, v) = where F1 pixel moves TO in F2

    For each pixel (y, x) in the output (F2 coordinate frame), we sample from
    the input (F1) at position (y - v, x - u). This is backward warping.

    Args:
        embeddings: (B, H, W, D) Frame 1 embeddings
        flow: (B, H, W, 2) flow field in normalized coordinates [0, 1]

    Returns:
        warped: (B, H, W, D) Frame 1 embeddings sampled at F2 coordinates
    """
    B, H, W, D = embeddings.shape

    def warp_single(emb, fl):
        # emb: (H, W, D), fl: (H, W, 2)

        # Convert normalized flow to pixel coordinates
        fl_pixels = fl * jnp.array([W - 1, H - 1], dtype=jnp.float32)

        # Create destination grid
        y_dest, x_dest = jnp.meshgrid(
            jnp.arange(H, dtype=jnp.float32),
            jnp.arange(W, dtype=jnp.float32),
            indexing="ij",
        )

        # Compute source coordinates
        x_src = x_dest - fl_pixels[..., 0]  # (H, W)
        y_src = y_dest - fl_pixels[..., 1]  # (H, W)

        # Warp each channel
        warped_ch = []
        for d in range(D):
            ch = map_coordinates(emb[..., d], (y_src, x_src), order=1, mode="nearest")
            warped_ch.append(ch)

        # Stack: (H, W, D)
        return jnp.stack(warped_ch, axis=-1)

    # Process all batches
    warped = jax.vmap(warp_single)(embeddings, flow)

    return warped


def reconstruction_loss_core(warped: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute L2 distance between warped and target embeddings.

    Args:
        warped: (B, H, W, D) warped Frame 1 embeddings
        target: (B, H, W, D) Frame 2 embeddings

    Returns:
        scalar loss value (mean L2 distance)
    """
    # Compute per-pixel L2 distance
    diff = warped - target  # (B, H, W, D)
    l2_per_pixel = jnp.sum(diff**2, axis=-1)  # (B, H, W)

    # Mean over batch and spatial dimensions
    loss = l2_per_pixel.mean()

    return loss


def hierarchical_reconstruction_loss(
    pyramid1,
    pyramid2,
    flows,
) -> tuple[jnp.ndarray, dict]:
    """Compute reconstruction loss across all pyramid levels.

    For each level:
    1. Crop embeddings to grid-aligned (divisible by window_size)
    2. Warp pyramid1[level] using flow[level]
    3. Compare warped to pyramid2[level]

    V1: Independent loss at each level, averaged across levels.
    V2: May incorporate priors and shifting.

    Args:
        pyramid1: List of embeddings from frame 1, one per level
        pyramid2: List of embeddings from frame 2, one per level
        flows: List of flow fields, one per level

    Returns:
        Tuple of (total_loss, aux_dict) where:
            - total_loss: Mean of per-level losses
            - aux_dict: {'level_losses': [...], 'level_warped': [...]}
    """
    from barevision.utils.grid import crop_to_grid_aligned

    if len(pyramid1) != len(pyramid2) != len(flows):
        raise ValueError(
            f" pyramid1 ({len(pyramid1)}), pyramid2 ({len(pyramid2)}), "
            f"and flows ({len(flows)}) must have same length"
        )

    level_losses = []
    level_warped = []

    for level_idx, (emb1, emb2, flow) in enumerate(zip(pyramid1, pyramid2, flows)):
        # Crop to grid-aligned (centered crop, same as flow estimation)
        # Use window_size=16 for all levels
        emb1_cropped = crop_to_grid_aligned(emb1, window_size=16)
        emb2_cropped = crop_to_grid_aligned(emb2, window_size=16)

        # Warp embeddings using flow
        warped = warp_embeddings(emb1_cropped, flow)
        level_warped.append(warped)

        # Compute loss at this level
        level_loss = reconstruction_loss_core(warped, emb2_cropped)
        level_losses.append(level_loss)

    # Average across levels
    total_loss = jnp.mean(jnp.stack(level_losses))

    aux = {
        "level_losses": level_losses,
        "level_warped": level_warped,
    }

    return total_loss, aux


class HierarchicalReconstructionLoss:
    def __init__(self, settings: FlowLossSettings):
        self.settings = settings

    def __call__(
        self,
        embeddings_pyramid_pairs: tuple[list[jnp.ndarray], list[jnp.ndarray]],
        flows: list[jnp.ndarray],
        need_aux: bool = True,
    ) -> tuple[jnp.ndarray, dict]:

        pyramid1, pyramid2 = embeddings_pyramid_pairs
        check_value(
            len(pyramid1) == len(pyramid2) == len(flows),
            f" pyramid1 ({len(pyramid1)}), pyramid2 ({len(pyramid2)}), "
            f"and flows ({len(flows)}) must have same length",
        )

        num_levels = len(pyramid1)
        weights = self._get_normalized_weights(num_levels)
        total_loss = jnp.array(0.0)
        levels_aux = []

        for emb1, emb2, flow, weight in zip(pyramid1, pyramid2, flows, weights):
            level_loss, level_aux = self._level_loss(emb1, emb2, flow, weight, need_aux)
            total_loss += level_loss
            if need_aux:
                levels_aux.append(level_aux)

        aux = {"loss": total_loss, "levels": levels_aux} if need_aux else {}

        return total_loss, aux

    def _get_normalized_weights(self, levels: int):
        raw_weights = jnp.arange(levels) ** self.settings.level_weight_decay
        return raw_weights / jnp.sum(raw_weights)

    def _level_loss(
        self,
        emb1: jnp.ndarray,
        emb2: jnp.ndarray,
        flow: jnp.ndarray,
        weight: jnp.ndarray,
        need_aux: bool = True,
    ) -> tuple[jnp.ndarray, dict]:
        level_weight = self.settings.level_weight_decay**weight

        # Crop to grid-aligned (centered crop, same as flow estimation)
        # Use window_size=16 for all levels
        emb1_cropped = crop_to_grid_aligned(emb1, window_size=16)
        emb2_cropped = crop_to_grid_aligned(emb2, window_size=16)

        # Warp embeddings using flow
        warped = warp_embeddings(emb1_cropped, flow)

        # Compute loss at this level
        level_loss = reconstruction_loss_core(warped, emb2_cropped)

        level_loss_weighted = level_loss * level_weight

        if need_aux:
            aux = dict(
                weighted_loss=level_loss_weighted,
                raw_loss=level_loss,
                weight=level_weight,
                warped=warped,
            )
        else:
            aux = {}

        return level_loss, aux
