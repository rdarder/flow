"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Architecture (Decoupled Cascade with Symmetric Mean Subtraction, 3 levels):
    Input: (B, H, W, 3) RGB
      ↓
    StemBlock (Level 0):
      - Conv1: 3×3, stride=1, dense (3→32 ch) → GroupNorm → GELU
      - Conv2: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Mean Conv: 3×3, stride=1, depthwise (32 groups, SAME padding) → local_mean
      - Local Contrast Normalization: rich_features - local_mean
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B (Downsample): strided slice of local_mean [:, 1:-1:2, 1:-1:2, :]
      ↓
    StandardBlock (Level 1):
      - Conv1: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Mean Conv: 3×3, stride=1, depthwise (32 groups, SAME padding) → local_mean
      - Local Contrast Normalization: rich_features - local_mean
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B (Downsample): strided slice of local_mean [:, 1:-1:2, 1:-1:2, :]
      ↓
    StandardBlock (Level 2, last):
      - Conv1: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Mean Conv: 3×3, stride=1, depthwise (32 groups, SAME padding) → local_mean
      - Local Contrast Normalization: rich_features - local_mean
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B: None (last level)

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Note: Uses VALID padding for feature convs, SAME padding for mean_conv.
      Receptive field expands through stacked stride=1 convolutions before downsampling.
      Stem Block: 25 pixel RF (two 3×3 stacked)
      Standard Block: Inherits RF from previous levels + 2 pixel expansion

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

import jax
import jax.numpy as jnp
from flax import nnx

from barevision.flow.settings import (
    EmbeddingLossSettings,
    EmbeddingModelSettings,
    EmbeddingsModelSettings,
)


def gaussian_kernel_2d(sigma: float = 1.0) -> jnp.ndarray:
    """Create a 2D Gaussian kernel for initialization.

    Args:
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        3x3 kernel normalized to sum to 1.0
    """
    # Create 3x3 grid centered at 0
    ax = jnp.arange(-1, 2, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(ax, ax)

    # 2D Gaussian
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # Normalize to sum to 1
    kernel = kernel / jnp.sum(kernel)

    return kernel


def depthwise_gaussian_initializer(
    sigma: float = 1.0,
):
    """Create an initializer for depthwise convolution with Gaussian kernels.

    This creates a separate Gaussian kernel for each input channel.

    Args:
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        Initializer function compatible with nnx.Conv
    """

    def init(key, input_shape, dtype=jnp.float32):
        # input_shape: (height, width, in_features, out_features)
        # For depthwise: out_features = in_features
        _, _, in_features, out_features = input_shape

        # Create single 3x3 Gaussian kernel
        single_kernel = gaussian_kernel_2d(sigma).astype(dtype)

        # For depthwise convolution, we need shape (3, 3, in_features, out_features)
        # where each input channel connects to corresponding output channel
        # Block diagonal structure: each channel has its own kernel
        kernel = jnp.zeros((3, 3, in_features, out_features), dtype=dtype)

        # Fill diagonal blocks (each input channel → corresponding output channel)
        for i in range(min(in_features, out_features)):
            kernel = kernel.at[:, :, i, i].set(single_kernel)

        return kernel

    return init


class StemBlock(nnx.Module):
    """Root block of the pyramid (Level 0 only).

    Uses two stacked 3×3 convolutions to expand receptive field from 9 to 25 pixels
    before applying Local Contrast Normalization and splitting into embedding and
    downsampling branches.

    Input: (B, H, W, 3) RGB
    Returns: (embedding, downsampled_output)
        - embedding: (B, H-4, W-4, embed_dim) L2-normalized
        - downsampled_output: (B, (H-7)//2, (W-7)//2, hidden_dim)

    Symmetric Mean Subtraction:
        - depthwise 3x3 conv computes local mean per channel (SAME padding)
        - Subtracts local mean from rich_features to boost uniqueness
        - Strided slice of mean output provides downsampling for next level
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        use_local_contrast_normalization: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize StemBlock.

        Args:
            hidden_dim: Hidden feature dimension
            embed_dim: Output embedding dimension
            num_groups: Number of groups for grouped convolutions
            use_local_contrast_normalization: Enable LCN (default: True)
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.use_local_contrast_normalization = use_local_contrast_normalization

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

        # Mean Conv: Depthwise 3×3, stride=1, SAME padding
        # Only created if LCN is enabled
        if use_local_contrast_normalization:
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

        # Local Contrast Normalization (optional)
        if self.use_local_contrast_normalization:
            # Depthwise conv computes per-channel local mean (preserves dimensions via SAME)
            local_mean = self.mean_conv(rich_features)
            # Subtract local mean to remove common background signals
            # This boosts uniqueness of local textures before embedding projection
            x_unique = rich_features - local_mean
        else:
            # Pass through without LCN
            x_unique = rich_features
            local_mean = None

        # Branch A: Embedding projection (operates on residuals or raw features)
        embedding = self.embed_conv(x_unique)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling via strided slice of local_mean (or rich_features if no LCN)
        # Mimics 3x3 VALID stride=2 convolution behavior
        # Slice [1:-1:2] starts at index 1, ends before last, stride 2
        downsample_source = local_mean if local_mean is not None else rich_features
        downsampled = downsample_source[:, 1:-1:2, 1:-1:2, :]

        return embedding, downsampled


class StandardBlock(nnx.Module):
    """Standard block for pyramid levels 1 to N.

    Uses a single 3×3 convolution (inherits receptive field from previous levels)
    before applying Local Contrast Normalization and splitting into embedding and
    downsampling branches.

    Input: (B, H, W, hidden_dim)
    Returns: (embedding, downsampled_output or None)
        - embedding: (B, H-2, W-2, embed_dim) L2-normalized
        - downsampled_output: (B, (H-5)//2, (W-5)//2, hidden_dim) or None if last level

    Symmetric Mean Subtraction:
        - depthwise 3x3 conv computes local mean per channel (SAME padding)
        - Subtracts local mean from rich_features to boost uniqueness
        - Strided slice of mean output provides downsampling for next level
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        is_last_level: bool = False,
        use_local_contrast_normalization: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize StandardBlock.

        Args:
            hidden_dim: Hidden feature dimension
            embed_dim: Output embedding dimension
            num_groups: Number of groups for grouped convolutions
            is_last_level: If True, skip downsampling branch
            use_local_contrast_normalization: Enable LCN (default: True)
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.is_last_level = is_last_level
        self.use_local_contrast_normalization = use_local_contrast_normalization

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

        # Mean Conv: Depthwise 3×3, stride=1, SAME padding
        # Only created if LCN is enabled
        if use_local_contrast_normalization:
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

        # Local Contrast Normalization (optional)
        if self.use_local_contrast_normalization:
            # Depthwise conv computes per-channel local mean (preserves dimensions via SAME)
            local_mean = self.mean_conv(rich_features)
            # Subtract local mean to remove common background signals
            # This boosts uniqueness of local textures before embedding projection
            x_unique = rich_features - local_mean
        else:
            # Pass through without LCN
            x_unique = rich_features
            local_mean = None

        # Branch A: Embedding projection (operates on residuals or raw features)
        embedding = self.embed_conv(x_unique)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling via strided slice of local_mean (if not last level)
        # Mimics 3x3 VALID stride=2 convolution behavior
        downsample_source = local_mean if local_mean is not None else rich_features
        downsampled = None
        if not self.is_last_level:
            downsampled = downsample_source[:, 1:-1:2, 1:-1:2, :]

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
                use_local_contrast_normalization=settings.use_local_contrast_normalization,
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
                    use_local_contrast_normalization=settings.use_local_contrast_normalization,
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


def build_model_from_settings(
    settings: EmbeddingsModelSettings,
    *,
    rngs: nnx.Rngs,
) -> HierarchicalEmbeddingModel:
    """Factory function to build model from settings.

    Args:
        settings: Model settings
        rngs: NNX RNGs

    Returns:
        HierarchicalEmbeddingModel instance
    """
    return HierarchicalEmbeddingModel(
        settings=settings,
        rngs=rngs,
    )


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
