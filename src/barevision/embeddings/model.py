"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Simplified Architecture (3 identical levels):
    Input: (B, H, W, 3) RGB
      ↓
    Block0: Compact(3→4) → DW×8(stride=2) → Project(32→16, 4g) → L2
      → (B, (H-3)//2+1, (W-3)//2+1, 16)
      ↓
    Block1: Compact(16→4) → DW×8(stride=2) → Project(32→16, 4g) → L2
      → (B, (H'-3)//2+1, (W'-3)//2+1, 16)
      ↓
    Block2: Compact(16→4) → DW×8(stride=2) → Project(32→16, 4g) → L2
      → (B, (H''-3)//2+1, (W''-3)//2+1, 16)

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Note: All blocks are identical. No preprocessor, no mean subtraction, no GroupNorm.
      L2 normalization on output is REQUIRED to prevent softmax collapse.
      VALID padding throughout. Stride=2 in depthwise conv handles downsampling.

Design Rationale:
    - PW Compact (dense): Mixes input channels, reduces redundancy (3→4 or 16→4)
    - DW ×8: Eight different 3×3 spatial filters per compacted channel → 32 feature maps
    - PW Project (grouped): Compacts spatial responses back to 16D embedding
    - L2 Norm: Prevents magnitude exploitation of spatial variance loss

FLOPs per pixel (per level): ~512 (vs. ~2048 for previous architecture)
Parameters: ~1,496 (down from ~7,248)
"""

from typing import List, Tuple

import jax.numpy as jnp
from flax import nnx

from barevision.embeddings.settings import ModelSettings


class EmbeddingBlock(nnx.Module):
    """Embedding block for pyramid levels.

    Simplified architecture: Compact → Spatial (multi-filter, stride=2) → Project → L2

    Input: (B, H, W, in_channels)  # 3 for first block, 16 for others
    Output: (B, (H-3)//2+1, (W-3)//2+1, embed_dim)

    Design:
        1. PW Compact: Dense 1×1 convolution compresses channels (in_ch → 4)
        2. DW ×8: Depthwise 3×3 with 8 filters per channel, stride=2, VALID padding
        3. PW Project: Grouped 1×1 projection (32 → 16, 4 groups)
        4. L2 Norm: Required to prevent softmax collapse
    """

    def __init__(
        self,
        in_channels: int,
        settings: ModelSettings,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize EmbeddingBlock.

        Args:
            in_channels: Input channels (3 for RGB on first block, 16 for others)
            settings: Model architecture settings
            rngs: NNX RNGs for parameter initialization
        """
        self.settings = settings
        self.in_channels = in_channels

        # PW Compact: Dense 1×1, in_channels → compact_channels
        self.pw_compact = nnx.Conv(
            in_features=in_channels,
            out_features=settings.compact_channels,
            kernel_size=(1, 1),
            padding="VALID",
            feature_group_count=1,  # Dense mixing
            rngs=rngs,
        )

        # Depthwise: 3×3, stride=2, VALID padding
        # Each of the compact_channels gets depthwise_multiplier filters
        dw_channels = settings.compact_channels * settings.depthwise_multiplier
        self.dw = nnx.Conv(
            in_features=settings.compact_channels,
            out_features=dw_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            feature_group_count=settings.compact_channels,  # Depthwise
            rngs=rngs,
        )

        # PW Project: Grouped 1×1, dw_channels → embed_dim
        # Each group independently projects (dw_channels/project_groups) → (embed_dim/project_groups)
        # use_bias=False: L2 normalization makes bias redundant
        self.pw_project = nnx.Conv(
            in_features=dw_channels,
            out_features=settings.embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            feature_group_count=settings.project_groups,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through EmbeddingBlock.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            Tensor of shape (B, (H-3)//2+1, (W-3)//2+1, embed_dim)
        """
        # PW Compact: channel mixing, reduce redundancy
        x = self.pw_compact(x)
        x = nnx.gelu(x)

        # Depthwise: spatial filtering with downsampling
        x = self.dw(x)
        x = nnx.gelu(x)

        # PW Project: grouped channel mixing to embedding dim
        x = self.pw_project(x)

        # L2 normalization: REQUIRED to prevent softmax collapse
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norm + 1e-8)

        return x


class HierarchicalEmbeddingModel(nnx.Module):
    """Hierarchical embedding model with multi-scale feature pyramid.

    Three identical EmbeddingBlocks at successive resolutions.
    Each block downsamples by ~2× via stride=2 depthwise convolution.
    """

    def __init__(
        self,
        settings: ModelSettings,
        *,
        rngs: nnx.Rngs,
    ):
        self.settings = settings

        # Build pyramid levels - all blocks are identical
        self.blocks = nnx.List()
        for i in range(settings.num_levels):
            is_first = i == 0
            in_channels = 3 if is_first else settings.embed_dim
            self.blocks.append(
                EmbeddingBlock(
                    in_channels=in_channels,
                    settings=settings,
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
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)

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
