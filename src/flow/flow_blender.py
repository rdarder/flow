"""Flow blending module for hierarchical optical flow.

Blends coarse and fine flow estimates using confidence scores as weights.
"""

import jax.numpy as jnp
from typing import Tuple, Dict, Any


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


def blend_flows(
    flow_fine: jnp.ndarray,
    conf_fine: jnp.ndarray,
    flow_coarse: jnp.ndarray,
    conf_coarse: jnp.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Blend fine and coarse flow estimates using confidence-weighted averaging.

    The blending formula:
    - When fine confidence is high: trust fine flow more
    - When fine confidence is low: trust coarse flow more

    weight_fine = conf_fine
    weight_coarse = 1 - conf_fine (or conf_coarse directly)
    flow_final = (weight_fine * flow_fine + weight_coarse * flow_coarse) / (weight_fine + weight_coarse + epsilon)

    Args:
        flow_fine: Fine flow estimate (B, H, W, 2) - already at target resolution
        conf_fine: Fine confidence (B, H, W, 1) - already at target resolution
        flow_coarse: Coarse flow estimate (B, H, W, 2) - upsampled to target resolution
        conf_coarse: Coarse confidence (B, H, W, 1) - upsampled to target resolution
        epsilon: Small value to prevent division by zero

    Returns:
        flow_final: Blended flow (B, H, W, 2)
        conf_final: Blended confidence (B, H, W, 1)
    """
    # Validate shapes match
    assert (
        flow_fine.shape == flow_coarse.shape
    ), f"Flow shapes must match: {flow_fine.shape} vs {flow_coarse.shape}"
    assert (
        conf_fine.shape == conf_coarse.shape
    ), f"Confidence shapes must match: {conf_fine.shape} vs {conf_coarse.shape}"

    # Compute weights
    # Strategy: Use fine confidence directly, coarse as complement
    # This makes intuitive sense: if fine is confident, use it; otherwise use coarse
    weight_fine = conf_fine
    weight_coarse = 1.0 - conf_fine

    # Alternative: could also use conf_coarse directly
    # weight_coarse = conf_coarse

    # Normalize weights
    weight_sum = weight_fine + weight_coarse + epsilon

    # Blend flow
    flow_final = (weight_fine * flow_fine + weight_coarse * flow_coarse) / weight_sum

    # Blend confidence (take weighted average)
    conf_final = (weight_fine * conf_fine + weight_coarse * conf_coarse) / weight_sum

    return flow_final, conf_final


class FlowBlender:
    """Blends coarse and fine flow estimates using confidence scores.

    This class handles the hierarchical blending step:
    1. Upsample coarse flow and confidence to match fine resolution
    2. Blend using confidence-weighted averaging
    3. Return final flow and confidence
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small value to prevent division by zero
        """
        self.epsilon = epsilon

    def blend_pyramid_levels(
        self,
        flow_fine: jnp.ndarray,
        conf_fine: jnp.ndarray,
        flow_coarse: jnp.ndarray,
        conf_coarse: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Blend coarse and fine pyramid levels.

        Args:
            flow_fine: Fine flow (B, H, W, 2) at target resolution
            conf_fine: Fine confidence (B, H, W, 1) at target resolution
            flow_coarse: Coarse flow (B, H/2, W/2, 2) - will be upsampled
            conf_coarse: Coarse confidence (B, H/2, W/2, 1) - will be upsampled

        Returns:
            flow_final: Blended flow (B, H, W, 2)
            conf_final: Blended confidence (B, H, W, 1)
            aux: Dictionary with intermediate outputs for debugging
        """
        B, H, W, _ = flow_fine.shape

        # Validate fine level is 2x the coarse level
        expected_coarse_H, expected_coarse_W = H // 2, W // 2
        assert flow_coarse.shape[1:3] == (expected_coarse_H, expected_coarse_W), (
            f"Coarse flow spatial dims should be ({expected_coarse_H}, {expected_coarse_W}), "
            f"got {flow_coarse.shape[1:3]}"
        )

        # Upsample coarse to match fine resolution
        flow_coarse_upsampled = upsample_flow_2x(flow_coarse)
        conf_coarse_upsampled = upsample_confidence_2x(conf_coarse)

        # Blend
        flow_final, conf_final = blend_flows(
            flow_fine,
            conf_fine,
            flow_coarse_upsampled,
            conf_coarse_upsampled,
            epsilon=self.epsilon,
        )

        # Prepare auxiliary outputs
        aux = {
            "flow_coarse_upsampled": flow_coarse_upsampled,
            "conf_coarse_upsampled": conf_coarse_upsampled,
            "weight_fine": conf_fine,
            "weight_coarse": 1.0 - conf_fine,
            "weight_sum": conf_fine + (1.0 - conf_fine) + self.epsilon,
        }

        return flow_final, conf_final, aux
