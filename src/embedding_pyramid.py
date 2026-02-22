"""Embedding Pyramid Module for Hierarchical Optical Flow.

Generates multi-scale embeddings using a unified approach:
- Each level: 2x2 spatial region -> flatten -> 1x1 conv -> embedding
- Level 1 (finest): 2x2 pixels -> flatten -> 1x1 conv
- Level 0 (coarser): 2x2 embeddings from below -> flatten -> 1x1 conv
"""

from typing import List, Tuple

import jax.numpy as jnp
from flax import nnx


class EmbeddingLevel(nnx.Module):
    """Single level of the embedding pyramid.
    Takes 2x2 spatial regions, flattens them, and projects to embedding_dim.
    """

    def __init__(self, in_channels: int, embed_dim: int, *, rngs: nnx.Rngs):
        """
        Args:
            in_channels: Number of input channels (4 * channels_per_pixel for 2x2 regions)
            embed_dim: Output embedding dimension
            rngs: NNX RNGs
        """
        # 1x1 convolution: projects flattened 2x2 regions to embedding_dim
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",  # 1x1 conv, no spatial change
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(num_features=embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, H, W, C) where C is already the flattened 2x2 channels

        Returns:
            Embeddings (B, H, W, embed_dim)
        """
        x = self.proj(x)  # (B, H, W, embed_dim)
        x = self.norm(x)
        x = nnx.relu(x)
        return x


class EmbeddingPyramid(nnx.Module):
    """Generates a pyramid of embeddings at multiple scales.

    Level 1 (finest): 2x2 pixel patches -> 16-dim embeddings
    Level 0 (coarser): 2x2 Level 1 embeddings -> 16-dim embeddings
    etc.
    """

    def __init__(
        self,
        num_levels: int = 2,
        embed_dim: int = 16,
        in_channels: int = 3,  # RGB or grayscale
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            num_levels: Number of pyramid levels (fine to coarse)
            embed_dim: Dimension of output embeddings at each level
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            rngs: NNX RNGs
        """
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Create embedding level for each pyramid level
        # Each level has its own learnable weights
        levels = []

        for level_idx in range(num_levels):
            if level_idx == num_levels - 1:
                # Finest level: input is raw pixels (or features from stem)
                # 2x2 patch of pixels: 4 * in_channels
                level_in_channels = 4 * in_channels
            else:
                # Coarser levels: input is embeddings from level below
                # 2x2 patch of embeddings: 4 * embed_dim
                level_in_channels = 4 * embed_dim

            level = EmbeddingLevel(
                in_channels=level_in_channels, embed_dim=embed_dim, rngs=rngs
            )
            levels.append(level)

        # Use nnx.List to store modules (required by Flax NNX)
        self.levels = nnx.List(levels)

    def _patchify_2x2(self, x: jnp.ndarray) -> jnp.ndarray:
        """Group spatial dimensions into 2x2 patches.

        Args:
            x: Input (B, H, W, C) where H, W are even

        Returns:
            Grouped tensor (B, H//2, W//2, 4*C)
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Height and width must be even, got {H}x{W}"

        # Reshape to group 2x2 spatial regions
        # (B, H, W, C) -> (B, H//2, 2, W//2, 2, C)
        x = x.reshape(B, H // 2, 2, W // 2, 2, C)

        # Move spatial dims together: (B, H//2, W//2, 2, 2, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        # Flatten 2x2xC into single dimension: (B, H//2, W//2, 4*C)
        x = x.reshape(B, H // 2, W // 2, 4 * C)

        return x

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Generate embedding pyramid.

        Args:
            x: Input image (B, H, W, C) where H, W are powers of 2 and >= 2^num_levels

        Returns:
            List of embeddings from coarsest to finest:
            [level_0, level_1, ..., level_{num_levels-1}]
            Each has shape (B, H/2^level, W/2^level, embed_dim)
        """
        embeddings = []
        current = x

        # Process from finest to coarsest
        for level_idx in range(self.num_levels - 1, -1, -1):
            # Group into 2x2 patches
            current = self._patchify_2x2(current)

            # Project to embedding dimension
            embedding = self.levels[level_idx](current)
            embeddings.append(embedding)

            # This embedding becomes input for coarser level
            current = embedding

        # Return coarsest to finest (reverse the list)
        return list(reversed(embeddings))


def compute_pyramid_shapes(
    input_size: Tuple[int, int], num_levels: int
) -> List[Tuple[int, int]]:
    """Compute the spatial shapes at each pyramid level.

    Args:
        input_size: (H, W) of input image
        num_levels: Number of pyramid levels

    Returns:
        List of (H, W) tuples for each level from coarsest to finest
    """
    H, W = input_size
    shapes = []

    for level in range(num_levels):
        # Each level halves the spatial dimensions
        level_H = H // (2 ** (num_levels - level))
        level_W = W // (2 ** (num_levels - level))
        shapes.append((level_H, level_W))

    return shapes
