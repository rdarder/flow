"""Latent space reconstruction loss.

Warp Frame 1 embeddings using predicted flow and compare to Frame 2 embeddings.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


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
