"""Optical flow model: combines embedding pyramid with flow matching.

End-to-end model for optical flow estimation.
"""

from typing import List, Tuple

from flax import nnx
import jax.numpy as jnp

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.matching.model import HierarchicalFlowEstimator


class JointEmbeddingFlotModel(nnx.Module):
    """End-to-end optical flow model.

    Combines hierarchical embedding pyramid with flow matching.
    """

    def __init__(
        self,
        embeddings_model: HierarchicalEmbeddingModel,
        flow_estimator: HierarchicalFlowEstimator,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding_model = embeddings_model
        self.flow_estimator = flow_estimator
        self.rngs = rngs

    def __call__(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
        """Compute optical flow between two frames.

        V1: Estimates flow independently at all pyramid levels.

        Args:
            img1: Frame 1 (B, H, W, 3)
            img2: Frame 2 (B, H, W, 3)

        Returns:
            Tuple of (flows, pyramid1, pyramid2)
            - flows: List of flow fields, one per level
            - pyramid1: List of embeddings from frame 1
            - pyramid2: List of embeddings from frame 2
        """
        pyramid1 = self.embedding_model(img1)
        pyramid2 = self.embedding_model(img2)
        flows = self.flow_estimator(pyramid1, pyramid2)

        return flows, pyramid1, pyramid2
