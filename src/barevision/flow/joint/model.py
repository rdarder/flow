"""Optical flow model: combines embedding pyramid with flow matching.

End-to-end model for optical flow estimation.
"""

from typing import List, Tuple

from flax import nnx
import jax.numpy as jnp

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.matching.model import HierarchicalFlowEstimator


class Model(nnx.Module):
    """End-to-end optical flow model.

    Combines hierarchical embedding pyramid with flow matching.
    Both components are trained jointly.

    Architecture:
        img1, img2 → embedding_model → pyramid1, pyramid2
        pyramid1, pyramid2 → flow_estimator → flow_field

    The embedding_model can be used standalone for embedding extraction.
    The flow_estimator can be swapped or upgraded independently.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        num_levels: int,
        flow_hidden_dim: int,
        window_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize optical flow model.

        Args:
            hidden_dim: Hidden feature dimension for intermediate convolutions
            embed_dim: Embedding dimension per level
            num_groups: Number of groups for grouped convolutions
            num_levels: Number of pyramid levels
            flow_hidden_dim: Flow estimator hidden dimension
            window_size: Attention window size
            rngs: NNX RNG streams
        """
        # Embedding pyramid (can be used standalone)
        self.embedding_model = HierarchicalEmbeddingModel(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_groups=num_groups,
            num_levels=num_levels,
            rngs=rngs,
        )

        # Flow estimator (operates on embeddings from both frames)
        self.flow_estimator = HierarchicalFlowEstimator(
            num_levels=num_levels,
            window_size=window_size,
            hidden_dim=flow_hidden_dim,
            max_flow=0.5,  # Maximum 0.5 = half-window displacement
            rngs=rngs,
        )

        self.window_size = window_size
        self.num_levels = num_levels

    def __call__(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        temperature: float = 0.2,
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], List[jnp.ndarray]]:
        """Compute optical flow between two frames.

        V1: Estimates flow independently at all pyramid levels.

        Args:
            img1: Frame 1 (B, H, W, 3)
            img2: Frame 2 (B, H, W, 3)
            temperature: Softmax temperature for attention

        Returns:
            Tuple of (flows, pyramid1, pyramid2)
            - flows: List of flow fields, one per level
            - pyramid1: List of embeddings from frame 1
            - pyramid2: List of embeddings from frame 2
        """
        # Extract embeddings from both frames
        pyramid1 = self.embedding_model(img1)
        pyramid2 = self.embedding_model(img2)

        # Estimate flow at all levels
        flows = self.flow_estimator(pyramid1, pyramid2, temperature=temperature)

        return flows, pyramid1, pyramid2

    def extract_embeddings(
        self,
        img: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """Extract embedding pyramid from a single frame.

        Convenience method to use the embedding model standalone.

        Args:
            img: (B, H, W, 3) input frame

        Returns:
            List of embedding pyramids, one per level
        """
        return self.embedding_model(img)
