"""Optical flow model: combines embedding pyramid with flow matching.

End-to-end model for optical flow estimation.
"""

from flax import nnx
import jax.numpy as jnp

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.matching.model import HierarchicalFlowEstimator
from barevision.flow.settings import JointEmbeddingFlowModelSettings


class JointEmbeddingFlowModel(nnx.Module):
    """End-to-end optical flow model.

    Combines hierarchical embedding pyramid with flow matching.
    """

    def __init__(
        self,
        embeddings_model: HierarchicalEmbeddingModel,
        flow_estimator: HierarchicalFlowEstimator,
        settings: JointEmbeddingFlowModelSettings,
        *,
        rngs: nnx.Rngs,
    ):
        self.embedding_model = embeddings_model
        self.flow_estimator = flow_estimator
        self.settings = settings
        self.rngs = rngs

    def __call__(
        self,
        img_pair: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[list[jnp.ndarray], list[jnp.ndarray]], list[jnp.ndarray]]:
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
        img1, img2 = img_pair
        pyramids = (self.embedding_model(img1), self.embedding_model(img2))
        flows = self.flow_estimator(pyramids)
        return pyramids, flows
