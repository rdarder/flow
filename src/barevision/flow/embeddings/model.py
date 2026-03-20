"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Architecture (Decoupled Cascade, 3 levels, stride=2 downsampling):
    Input: (B, H, W, 3) RGB
      ↓
    StemBlock (Level 0):
      - Conv1: 3×3, stride=1, dense (3→32 ch) → GroupNorm → GELU
      - Conv2: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B (Downsample): 3×3, stride=2, grouped (32→32 ch)
      ↓
    StandardBlock (Level 1):
      - Conv1: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B (Downsample): 3×3, stride=2, grouped (32→32 ch)
      ↓
    StandardBlock (Level 2, last):
      - Conv1: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B: None (last level)

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Note: Uses VALID padding (no padding), so spatial dimensions shrink at each level.
      Receptive field expands through stacked stride=1 convolutions before downsampling.
      Stem Block: 25 pixel RF (two 3×3 stacked)
      Standard Block: Inherits RF from previous levels + 2 pixel expansion

Decoupled Cascade Design:
    - Feature extraction (stride=1) builds deep structural understanding
    - Downsampling (stride=2) is separate from feature extraction
    - Receptive field expands before projection to embedding space
    - GELU activation preserves more gradient flow than ReLU
    - No activation on embeddings before L2 normalization
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp
from flax import nnx

from barevision.flow.settings import EmbeddingLossSettings, EmbeddingModelSettings


class StemBlock(nnx.Module):
    """Root block of the pyramid (Level 0 only).

    Uses two stacked 3×3 convolutions to expand receptive field from 9 to 25 pixels
    before splitting into embedding and downsampling branches.

    Input: (B, H, W, 3) RGB
    Returns: (embedding, downsampled_output)
        - embedding: (B, H-4, W-4, embed_dim) L2-normalized
        - downsampled_output: (B, (H-7)//2, (W-7)//2, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize StemBlock.

        Args:
            hidden_dim: Hidden feature dimension
            embed_dim: Output embedding dimension
            num_groups: Number of groups for grouped convolutions
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        # Conv1: Dense 3×3, stride=1, RGB (3 ch) → hidden_dim ch
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=1,
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=num_groups // 2, num_features=hidden_dim, rngs=rngs
        )

        # Conv2: Grouped 3×3, stride=1, hidden_dim ch → hidden_dim ch
        self.conv2 = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=num_groups,
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=num_groups // 2, num_features=hidden_dim, rngs=rngs
        )

        # Branch A: 1×1 Conv for embedding projection
        self.embed_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

        # Branch B: 3×3 Conv for downsampling
        self.downsample_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            feature_group_count=num_groups,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through StemBlock.

        Args:
            x: Input tensor of shape (B, H, W, 3)

        Returns:
            Tuple of (embedding, downsampled_output)
        """
        # First convolution: dense 3×3, expands RF to 9 pixels
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)

        # Second convolution: grouped 3×3, expands RF to 25 pixels
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.gelu(x)
        rich_features = x

        # Branch A: Embedding projection
        embedding = self.embed_conv(rich_features)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling
        downsampled = self.downsample_conv(rich_features)

        return embedding, downsampled


class StandardBlock(nnx.Module):
    """Standard block for pyramid levels 1 to N.

    Uses a single 3×3 convolution (inherits receptive field from previous levels)
    before splitting into embedding and downsampling branches.

    Input: (B, H, W, hidden_dim)
    Returns: (embedding, downsampled_output or None)
        - embedding: (B, H-2, W-2, embed_dim) L2-normalized
        - downsampled_output: (B, (H-5)//2, (W-5)//2, hidden_dim) or None if last level
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        is_last_level: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize StandardBlock.

        Args:
            hidden_dim: Hidden feature dimension
            embed_dim: Output embedding dimension
            num_groups: Number of groups for grouped convolutions
            is_last_level: If True, skip downsampling branch
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.is_last_level = is_last_level

        # Conv1: Grouped 3×3, stride=1
        self.conv1 = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=num_groups,
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=num_groups // 2, num_features=hidden_dim, rngs=rngs
        )

        # Branch A: 1×1 Conv for embedding projection
        self.embed_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

        # Branch B: 3×3 Conv for downsampling (only if not last level)
        if not is_last_level:
            self.downsample_conv = nnx.Conv(
                in_features=hidden_dim,
                out_features=hidden_dim,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                feature_group_count=num_groups,
                rngs=rngs,
            )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass through StandardBlock.

        Args:
            x: Input tensor of shape (B, H, W, hidden_dim)

        Returns:
            Tuple of (embedding, downsampled_output or None)
        """
        # Feature extraction: grouped 3×3
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)
        rich_features = x

        # Branch A: Embedding projection
        embedding = self.embed_conv(rich_features)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling (if not last level)
        downsampled = None
        if not self.is_last_level:
            downsampled = self.downsample_conv(rich_features)

        return embedding, downsampled


class HierarchicalEmbeddingModel(nnx.Module):
    """Hierarchical embedding model with multi-scale feature pyramid.

    Uses Decoupled Cascade architecture: stacked stride=1 convolutions build
    structural understanding before splitting into embedding and downsampling
    branches. Produces embeddings at multiple scales optimized for attention-based
    matching. Uses valid convolutions only (no padding).
    """

    def __init__(
        self,
        settings: EmbeddingModelSettings,
        *,
        rngs: nnx.Rngs,
    ):
        self.settings = settings

        # Build pyramid levels
        self.blocks = nnx.List()

        # StemBlock for Level 0
        self.blocks.append(
            StemBlock(
                hidden_dim=settings.hidden_dim,
                embed_dim=settings.embed_dim,
                num_groups=settings.num_groups,
                rngs=rngs,
            )
        )

        # StandardBlocks for Levels 1 to N-1
        for i in range(1, settings.num_levels):
            is_last = i == settings.num_levels - 1
            self.blocks.append(
                StandardBlock(
                    hidden_dim=settings.hidden_dim,
                    embed_dim=settings.embed_dim,
                    num_groups=settings.num_groups,
                    is_last_level=is_last,
                    rngs=rngs,
                )
            )

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Forward pass through pyramid.

        Args:
            x: Input tensor of shape (B, H, W, 3)

        Returns:
            List of feature maps, one per level.
            Each level has shape (B, H_l, W_l, embed_dim)
        """
        feature_maps = []

        for i, block in enumerate(self.blocks):
            embedding, downsampled = block(x)
            feature_maps.append(embedding)

            if i < len(self.blocks) - 1:
                x = downsampled

        return feature_maps


def count_parameters(model: nnx.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: NNX module

    Returns:
        Total number of trainable parameters
    """
    from flax.nnx import State

    state = nnx.state(model)
    total = 0

    def count_recursive(obj):
        nonlocal total
        if isinstance(obj, State):
            for value in obj.values():
                count_recursive(value)
        elif isinstance(obj, dict):
            for value in obj.values():
                count_recursive(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                count_recursive(item)
        elif hasattr(obj, "size"):
            total += obj.size

    count_recursive(state)
    return total
