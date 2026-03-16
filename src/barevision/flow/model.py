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

Parameters: ~4,560 total
    - Stem Block: Dense 3×3 (3→32) = 3*32*9=864 + Grouped 3×3 (8g, 32→32) = 32*4*9=1152
                  + 1×1 (32→16) = 528 + Downsample 3×3 (8g, 32→32) = 1152 = 3,696
    - Standard Block: Grouped 3×3 (8g, 32→32) = 1152 + 1×1 (32→16) = 528 + Downsample = 1152 = 2,832
    - Last Standard Block: 1152 + 528 = 1,680 (no downsample)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx


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
            hidden_dim: Hidden feature dimension (default 32)
            embed_dim: Output embedding dimension (default 16)
            num_groups: Number of groups for grouped convolutions (default 8)
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        # Conv1: Dense 3×3, stride=1, RGB (3 ch) → hidden_dim ch
        # No feature groups since 3 input channels isn't divisible by num_groups
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=1,  # Dense convolution
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

        # Branch A: 1×1 Conv for embedding projection (hidden_dim → embed_dim)
        self.embed_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

        # Branch B: 3×3 Conv for downsampling (hidden_dim → hidden_dim)
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
            Tuple of (embedding, downsampled_output):
                - embedding: (B, H-4, W-4, embed_dim) L2-normalized
                - downsampled_output: (B, (H-7)//2, (W-7)//2, 32)
        """
        # First convolution: dense 3×3, expands RF to 9 pixels
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)

        # Second convolution: grouped 3×3, expands RF to 25 pixels
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.gelu(x)
        rich_features = x  # (B, H-4, W-4, 32)

        # Branch A: Embedding projection
        embedding = self.embed_conv(rich_features)  # (B, H-4, W-4, embed_dim)
        # L2 normalization (no activation before norm)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling
        downsampled = self.downsample_conv(rich_features)  # (B, (H-7)//2, (W-7)//2, 32)

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
            hidden_dim: Hidden feature dimension (default 32)
            embed_dim: Output embedding dimension (default 16)
            num_groups: Number of groups for grouped convolutions (default 8)
            is_last_level: If True, skip downsampling branch
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.is_last_level = is_last_level

        # Conv1: Grouped 3×3, stride=1, hidden_dim ch → hidden_dim ch
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

        # Branch A: 1×1 Conv for embedding projection (hidden_dim → embed_dim)
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
            x: Input tensor of shape (B, H, W, 32)

        Returns:
            Tuple of (embedding, downsampled_output or None):
                - embedding: (B, H-2, W-2, embed_dim) L2-normalized
                - downsampled_output: (B, (H-5)//2, (W-5)//2, 32) or None
        """
        # Feature extraction: grouped 3×3, expands RF by 2 pixels
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)
        rich_features = x  # (B, H-2, W-2, 32)

        # Branch A: Embedding projection
        embedding = self.embed_conv(rich_features)  # (B, H-2, W-2, embed_dim)
        # L2 normalization (no activation before norm)
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
        hidden_dim: int,
        embed_dim: int,
        num_groups: int,
        num_levels: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the hierarchical embedding model.

        Args:
            hidden_dim: Hidden feature dimension for intermediate convolutions
            embed_dim: Output embedding dimension per level
            num_groups: Number of groups for grouped convolutions
            num_levels: Number of pyramid levels
            rngs: NNX RNGs for parameter initialization
        """
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_groups = num_groups
        self.num_levels = num_levels

        # Build pyramid levels using nnx.List for proper module tracking
        # Level 0: StemBlock (RGB input, two stacked convs)
        # Levels 1 to N-1: StandardBlock (feature input, single conv)
        self.blocks = nnx.List()

        # StemBlock for Level 0
        self.blocks.append(
            StemBlock(
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                num_groups=num_groups,
                rngs=rngs,
            )
        )

        # StandardBlocks for Levels 1 to N-1
        for i in range(1, num_levels):
            is_last = i == num_levels - 1
            self.blocks.append(
                StandardBlock(
                    hidden_dim=hidden_dim,
                    embed_dim=embed_dim,
                    num_groups=num_groups,
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
            where H_l, W_l shrink at each level due to VALID padding.
        """
        feature_maps = []

        # Iterate through all blocks
        for i, block in enumerate(self.blocks):
            embedding, downsampled = block(x)
            feature_maps.append(embedding)

            # Pass downsampled output to next block (if not last)
            if i < len(self.blocks) - 1:
                x = downsampled

        return feature_maps


def _compute_entropy(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution.

    Args:
        probabilities: (..., N) array where last dimension sums to 1

    Returns:
        Entropy values with shape (...)
    """
    eps = 1e-10
    return -jnp.sum(probabilities * jnp.log(probabilities + eps), axis=-1)


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


def calculate_required_input_size(
    target_coarse_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate required input image size to achieve target coarse dimension.

    Works backwards from the target coarse-level spatial dimension through
    the pyramid to determine the exact input size needed.

    For the Decoupled Cascade architecture with VALID padding:
    - Standard Block FE reverse: Adds 2 pixels (from Conv1 3×3)
    - Stem Block FE reverse: Adds 4 pixels (from Conv1 + Conv2 3×3)
    - Downsample reverse: (size - 1) * stride + kernel_size

    Args:
        target_coarse_dim: Target spatial dimension at coarsest level
                          (e.g., 48 for 3×3 grid of 16×16 windows)
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Required input image dimension (height and width are same)

    Example:
        For 3 levels targeting 16×16 at coarsest (L2):
        L2 drops 2 → 18
        L1 downsample reversed → (18-1)*2 + 3 = 37
        L1 drops 2 → 39
        L0 downsample reversed → (39-1)*2 + 3 = 79
        L0 drops 4 → 83
        Result: 83×83 input required
    """
    size = target_coarse_dim
    # Walk backward from top level (num_levels - 1) down to 0
    for i in reversed(range(num_levels)):
        # Reverse the Feature Extraction drop (Stem drops 4, Standard drops 2)
        size += 4 if i == 0 else 2
        # Reverse the downsample that fed into this level (only if there is a level below it)
        if i > 0:
            size = (size - 1) * stride + kernel_size
    return size


def calculate_coarse_output_size(
    input_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate coarse-level output size for a given input.

    Forward calculation through the pyramid.

    For the Decoupled Cascade architecture with VALID padding:
    - Feature Extraction drop: Stem drops 4, Standard drops 2
    - Downsample: (size - kernel_size) // stride + 1

    Args:
        input_dim: Input image dimension
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Spatial dimension at coarsest level
    """
    size = input_dim
    for i in range(num_levels):
        # Apply Feature Extraction drop
        size -= 4 if i == 0 else 2
        # Apply Downsample (if not the last level)
        if i < num_levels - 1:
            size = (size - kernel_size) // stride + 1
    return size
