"""Flow blending and combination utilities.

This module contains strategies for combining flow estimates from different
sources, particularly for hierarchical blending of fine-level lookup results
with coarse-level prior estimates.
"""

from typing import Tuple

import jax.numpy as jnp
from flax import nnx


class PriorBlender(nnx.Module):
    """Blends flow estimates from token-level lookup with coarse-level prior.

    This implements a confidence-weighted blending strategy where:
    - High lookup confidence → trust what we found at this level
    - High prior confidence → trust the upsampled coarser estimate
    - Combined confidence reflects consensus between both sources

    This is particularly important when:
    - The prior flow points outside the lookup window (can't verify locally)
    - The current level has low confidence (occlusions, textureless regions)
    - The coarse level has high confidence (clear motion patterns)

    The blending formula is:
        flow_blended = (conf_lookup * flow_lookup + conf_prior * flow_prior)
                       / (conf_lookup + conf_prior)
        conf_blended = (conf_lookup + conf_prior) / 2  # Average consensus
    """

    def __call__(
        self,
        flow_lookup: jnp.ndarray,  # (B, N, 2) from TokenCrossAttention
        conf_lookup: jnp.ndarray,  # (B, N, 1) lookup confidence
        prior_flow: jnp.ndarray,  # (B, N, 2) upsampled from coarser level
        prior_conf: jnp.ndarray,  # (B, N, 1) prior confidence
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Blend lookup flow with prior flow using confidence-weighted combination.

        Args:
            flow_lookup: Flow estimates from token-level cross-attention lookup
            conf_lookup: Confidence scores from lookup (typically attention max)
            prior_flow: Flow estimates from coarser level (upsampled 2x)
            prior_conf: Confidence scores from coarser level

        Returns:
            flow_blended: Combined flow estimate (B, N, 2)
            conf_blended: Combined confidence score (B, N, 1)
        """
        # Confidence-weighted blend
        # High lookup confidence → trust what we found
        # High prior confidence → trust the coarse estimate
        weight_lookup = conf_lookup
        weight_prior = prior_conf
        weight_sum = weight_lookup + weight_prior + 1e-6  # epsilon for stability

        flow_blended = (
            weight_lookup * flow_lookup + weight_prior * prior_flow
        ) / weight_sum

        # Combined confidence is the average of both sources' confidence
        # This represents the consensus between what we found and what we expected
        conf_blended = (conf_lookup + prior_conf) / 2.0

        return flow_blended, conf_blended
