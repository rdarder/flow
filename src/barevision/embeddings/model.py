"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Architecture (Decoupled Cascade with Symmetric Mean Subtraction, 3 levels):
    Input: (B, H, W, 3) RGB
      ↓
    Preprocessor:
      - Conv: 3×3, stride=1, dense (3→hidden_dim ch) → GroupNorm → GELU
      ↓
    EmbeddingBlock (Level 0):
      - Conv1: 3×3, stride=1, grouped (num_groups, hidden_dim→hidden_dim ch)
      - GroupNorm → GELU
      - Mean Conv: 3×3, stride=1, depthwise (SAME padding) → local_mean
      - Local Contrast Normalization: rich_features - local_mean
      - Branch A (Embed): 1×1 Conv (hidden_dim→embed_dim ch) → L2 Norm
      - Branch B (Downsample): strided slice of local_mean [:, 1:-1:2, 1:-1:2, :]
      ↓
    EmbeddingBlock (Level 1):
      - Conv1: 3×3, stride=1, grouped (num_groups, hidden_dim→hidden_dim ch)
      - GroupNorm → GELU
      - Mean Conv: 3×3, stride=1, depthwise (SAME padding) → local_mean
      - Local Contrast Normalization: rich_features - local_mean
      - Branch A (Embed): 1×1 Conv (hidden_dim→embed_dim ch) → L2 Norm
      - Branch B (Downsample): strided slice of local_mean [:, 1:-1:2, 1:-1:2, :]
      ↓
    EmbeddingBlock (Level 2, last):
      - Conv1: 3×3, stride=1, grouped (num_groups, hidden_dim→hidden_dim ch)
      - GroupNorm → GELU
      - Mean Conv: 3×3, stride=1, depthwise (SAME padding) → local_mean
      - Local Contrast Normalization: rich_features - local_mean
      - Branch A (Embed): 1×1 Conv (hidden_dim→embed_dim ch) → L2 Norm
      - Branch B: None (last level)

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Note: Uses VALID padding for feature convs, SAME padding for mean_conv.
      Receptive field expands through stacked stride=1 convolutions before downsampling.
      Preprocessor + Block: 25 pixel RF (two 3×3 stacked)
      Block only: Inherits RF from previous levels + 2 pixel expansion

Symmetric Mean Subtraction Design:
    - Depthwise 3×3 convolution computes local mean per channel (learnable Gaussian init)
    - Subtracting local mean removes common background signals from hidden features
    - Local Contrast Normalization boosts uniqueness of local textures
    - 1×1 embed mixer receives "cleaned" residuals, can focus on unique signatures
    - Strided slice of mean_conv output provides pyramid downsampling (mimics VALID stride=2)
    - GELU activation preserves more gradient flow than ReLU
    - No activation on embeddings before L2 normalization
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp
from flax import nnx

from barevision.embeddings.gaussian import depthwise_gaussian_initializer
from barevision.embeddings.settings import ModelSettings


class Preprocessor(nnx.Module):
    """Preprocessor layer for hierarchical embedding model.

    Expands RGB input to hidden dimension and applies initial feature extraction.
    Separated from EmbeddingBlock to enable ablation studies.

    Input: (B, H, W, 3) RGB
    Output: (B, H-2, W-2, hidden_dim)

    Note: Drops 2 pixels due to 3×3 VALID convolution.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_groups: int,
        use_group_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Preprocessor.

        Args:
            in_channels: Input channels (typically 3 for RGB)
            hidden_dim: Output hidden dimension
            num_groups: Number of groups for GroupNorm
            use_group_norm: If True, apply GroupNorm after convolution
            rngs: NNX RNGs for parameter initialization
        """
        self.use_group_norm = use_group_norm
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=1,
            rngs=rngs,
        )
        if use_group_norm:
            self.norm = nnx.GroupNorm(
                num_groups=num_groups // 2, num_features=hidden_dim, rngs=rngs
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through preprocessor.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            Tensor of shape (B, H-2, W-2, hidden_dim)
        """
        x = self.conv(x)
        if self.use_group_norm:
            x = self.norm(x)
        return nnx.gelu(x)


class EmbeddingBlock(nnx.Module):
    """Embedding block for pyramid levels.

    Applies grouped convolution, local contrast normalization, and produces
    embedding + optional downsampling output.

    Input: (B, H, W, in_channels)
    Returns: (embedding, downsampled_output or None)
        - embedding: (B, H-2, W-2, embed_dim) L2-normalized (if use_l2_norm)
        - downsampled_output: (B, (H-5)//2, (W-5)//2, hidden_dim) or None if last level

    Symmetric Mean Subtraction:
        - depthwise 3x3 conv computes local mean per channel (SAME padding)
        - Subtracts local mean from rich_features to boost uniqueness
        - Strided slice of mean output provides downsampling for next level
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        is_last_level: bool = False,
        use_group_norm: bool = True,
        use_mean_subtraction: bool = True,
        use_l2_norm: bool = True,
        use_mean_conv_for_downsampling: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize EmbeddingBlock.

        Args:
            in_channels: Input channels (3 for RGB on first block, hidden_dim otherwise)
            hidden_dim: Hidden feature dimension
            embed_dim: Output embedding dimension
            num_groups: Number of groups for grouped convolutions
            is_last_level: If True, skip downsampling branch
            use_group_norm: If True, apply GroupNorm after conv1
            use_mean_subtraction: If True, subtract local mean from features
            use_l2_norm: If True, L2-normalize output embeddings
            use_mean_conv_for_downsampling: If True, use mean_conv for downsampling
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.is_last_level = is_last_level
        self.use_group_norm = use_group_norm
        self.use_mean_subtraction = use_mean_subtraction
        self.use_l2_norm = use_l2_norm
        self.use_mean_conv_for_downsampling = use_mean_conv_for_downsampling

        # Conv1: Grouped 3×3, stride=1
        # Use num_groups=1 (dense) for first block with 3-channel RGB input
        effective_num_groups = 1 if in_channels == 3 else num_groups
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=effective_num_groups,
            rngs=rngs,
        )
        if use_group_norm:
            # GroupNorm uses num_groups // 2 to match block's grouping
            effective_norm_groups = 1 if in_channels == 3 else num_groups // 2
            self.norm1 = nnx.GroupNorm(
                num_groups=effective_norm_groups, num_features=hidden_dim, rngs=rngs
            )

        # Mean Conv: Depthwise 3×3, stride=1, SAME padding
        # Computes local mean for contrast normalization and downsampling
        # Always created if mean subtraction is enabled OR if we need downsampling
        if use_mean_subtraction or not is_last_level:
            self.mean_conv = nnx.Conv(
                in_features=hidden_dim,
                out_features=hidden_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                feature_group_count=hidden_dim,  # Depthwise convolution
                kernel_init=depthwise_gaussian_initializer(sigma=1.0),
                rngs=rngs,
            )

        # Branch A: 1×1 Conv for embedding projection (operates on residuals)
        self.embed_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass through EmbeddingBlock.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            Tuple of (embedding, downsampled_output or None)
        """
        # Feature extraction: grouped 3×3
        x = self.conv1(x)
        if self.use_group_norm:
            x = self.norm1(x)
        x = nnx.gelu(x)
        rich_features = x

        # Local Contrast Normalization (optional)
        # Compute local_mean if needed for subtraction OR downsampling
        need_mean_conv = self.use_mean_subtraction or (not self.is_last_level and self.use_mean_conv_for_downsampling)
        local_mean = None
        if need_mean_conv and hasattr(self, 'mean_conv'):
            local_mean = self.mean_conv(rich_features)
        
        if self.use_mean_subtraction and local_mean is not None:
            x_unique = rich_features - local_mean
        else:
            x_unique = rich_features

        # Branch A: Embedding projection (operates on residuals)
        embedding = self.embed_conv(x_unique)
        if self.use_l2_norm:
            norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
            embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling (if not last level)
        downsampled = None
        if not self.is_last_level:
            if self.use_mean_conv_for_downsampling and local_mean is not None:
                downsampled = local_mean[:, 1:-1:2, 1:-1:2, :]
            else:
                downsampled = rich_features[:, 1:-1:2, 1:-1:2, :]

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
        settings: ModelSettings,
        *,
        rngs: nnx.Rngs,
    ):
        self.settings = settings

        # Preprocessor: expands RGB to hidden_dim (optional)
        if settings.use_preprocessor:
            self.preprocessor = Preprocessor(
                in_channels=3,
                hidden_dim=settings.hidden_dim,
                num_groups=settings.num_groups,
                use_group_norm=settings.use_group_norm,
                rngs=rngs,
            )
        else:
            self.preprocessor = None

        # Build pyramid levels - all blocks are identical
        # First block takes RGB (3 ch) if no preprocessor, otherwise hidden_dim
        self.blocks = nnx.List()
        for i in range(settings.num_levels):
            is_last = i == settings.num_levels - 1
            is_first = i == 0
            in_channels = 3 if (is_first and not settings.use_preprocessor) else settings.hidden_dim
            self.blocks.append(
                EmbeddingBlock(
                    in_channels=in_channels,
                    hidden_dim=settings.hidden_dim,
                    embed_dim=settings.embed_dim,
                    num_groups=settings.num_groups,
                    is_last_level=is_last,
                    use_group_norm=settings.use_group_norm,
                    use_mean_subtraction=settings.use_mean_subtraction,
                    use_l2_norm=settings.use_l2_norm,
                    use_mean_conv_for_downsampling=settings.use_mean_conv_for_downsampling,
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
        # Preprocess RGB to hidden_dim (if enabled)
        if self.preprocessor is not None:
            x = self.preprocessor(x)

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
