"""Flow upsampling utilities for hierarchical optical flow.

Provides 2x upsampling functions for flow and confidence fields.
Used by prior-guided hierarchical attention.
"""

import jax.numpy as jnp


def upsample_flow_2x(flow: jnp.ndarray) -> jnp.ndarray:
    """Upsample flow field by 2x using nearest neighbor interpolation.

    Args:
        flow: Flow field (B, H, W, 2) in normalized or pixel coordinates

    Returns:
        Upsampled flow (B, 2*H, 2*W, 2)
    """
    B, H, W, C = flow.shape

    # Use simple repeat (nearest neighbor) upsampling
    # (B, H, W, C) -> (B, H, 1, W, 1, C) -> (B, H, 2, W, 2, C) -> (B, 2*H, 2*W, C)
    flow_expanded = flow[:, :, None, :, None, :]  # (B, H, 1, W, 1, C)
    flow_tiled = jnp.tile(flow_expanded, (1, 1, 2, 1, 2, 1))  # (B, H, 2, W, 2, C)
    flow_upsampled = flow_tiled.reshape(B, 2 * H, 2 * W, C)

    return flow_upsampled


def upsample_confidence_2x(confidence: jnp.ndarray) -> jnp.ndarray:
    """Upsample confidence field by 2x using nearest neighbor interpolation.

    Args:
        confidence: Confidence scores (B, H, W, 1)

    Returns:
        Upsampled confidence (B, 2*H, 2*W, 1)
    """
    B, H, W, C = confidence.shape

    # Same approach as flow upsampling
    conf_expanded = confidence[:, :, None, :, None, :]
    conf_tiled = jnp.tile(conf_expanded, (1, 1, 2, 1, 2, 1))
    conf_upsampled = conf_tiled.reshape(B, 2 * H, 2 * W, C)

    return conf_upsampled
