"""Optical flow model: combines embedding pyramid with flow matching.

End-to-end model for optical flow estimation.
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.matching.model import (
    FlowEstimator,
    AttentionCentroids,
    create_source_position_grid,
)


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
        self.flow_estimator = FlowEstimator(
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
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]]:
        """Compute optical flow between two frames.

        Args:
            img1: Frame 1 (B, H, W, 3)
            img2: Frame 2 (B, H, W, 3)
            temperature: Softmax temperature for attention

        Returns:
            Tuple of (flow_field, pyramid1, pyramid2)
        """
        # Extract embeddings from both frames
        pyramid1 = self.embedding_model(img1)
        pyramid2 = self.embedding_model(img2)

        # Estimate flow at coarsest level
        flow = self._estimate_flow_at_level(
            pyramid1[-1], pyramid2[-1], temperature=temperature
        )

        return flow, pyramid1, pyramid2

    def _estimate_flow_at_level(
        self,
        emb1: jnp.ndarray,
        emb2: jnp.ndarray,
        temperature: float = 0.2,
    ) -> jnp.ndarray:
        """Estimate flow from embeddings at a single pyramid level.

        Current implementation: operates on coarsest level only.

        Args:
            emb1: (B, H, W, D) embeddings from frame 1
            emb2: (B, H, W, D) embeddings from frame 2
            temperature: Softmax temperature for attention

        Returns:
            flow: (B, H, W, 2) optical flow field
        """
        B, H, W, D = emb1.shape
        N = H * W

        # Flatten spatial dimensions
        flat_emb1 = emb1.reshape(B, N, D)
        flat_emb2 = emb2.reshape(B, N, D)

        # Compute self and cross attention
        self_logits = flat_emb1 @ flat_emb1.transpose(0, 2, 1)
        cross_logits = flat_emb1 @ flat_emb2.transpose(0, 2, 1)

        self_attn = jax.nn.softmax(self_logits / temperature, axis=-1)
        cross_attn = jax.nn.softmax(cross_logits / temperature, axis=-1)

        # Compute attention centroids
        centroids_computer = AttentionCentroids(window_size=H, rngs=nnx.Rngs(0))
        centroids = centroids_computer(self_attn, cross_attn)

        # Create source position grid
        src_pos = create_source_position_grid(window_size=H)
        src_pos = jnp.broadcast_to(src_pos, (B, N, 2))

        # Predict flow
        flow = self.flow_estimator(src_pos, centroids)

        # Reshape to spatial grid
        return flow.reshape(B, H, W, 2)

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
