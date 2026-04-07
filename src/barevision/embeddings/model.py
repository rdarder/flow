"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Architecture (MobileNet V4-inspired Universal Inverted Blocks):
    Input: (B, H, W, 3) RGB
      ↓
    Level 0:
      UIB_0: DW(3→3) → PW(3→32) → DW(32→32) → PW(32→16)
      UIB_1: DW(16→16) → PW(16→32) → DW(32→32) → PW(32→16) → Downsample → L2
      → output_0: (B, (H-9)//2, (W-9)//2, 16)
      ↓
    Level 1:
      UIB_0: DW(16→16) → PW(16→32) → DW(32→32) → PW(32→16)
      UIB_1: DW(16→16) → PW(16→32) → DW(32→32) → PW(32→16) → Downsample → L2
      → output_1: (B, (H'-9)//2, (W'-9)//2, 16)
      ↓
    Level 2:
      UIB_0: DW(16→16) → PW(16→32) → DW(32→32) → PW(32→16)
      UIB_1: DW(16→16) → PW(16→32) → DW(32→32) → PW(32→16) → Downsample → L2
      → output_2: (B, (H''-9)//2, (W''-9)//2, 16)

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Universal Inverted Block (UIB) structure:
    - DW before expand (optional): 3×3 VALID, preserves channels
    - PW expand: 1×1 VALID, expands to 32 channels
    - DW after expand (optional): 3×3 VALID, preserves expanded channels
    - PW compress: 1×1 VALID, compresses to output channels (16)
    - Downsample (optional): 3×3 stride=2 VALID
    - GroupNorm + ReLU after each conv
    - L2 norm on output (configurable, default True for last UIB of each level)

Note: VALID padding throughout. Each conv followed by GroupNorm + ReLU.
      L2 normalization on level outputs prevents softmax collapse.
"""

from dataclasses import dataclass
from typing import List, Optional

import jax.numpy as jnp
from flax import nnx

from barevision.embeddings.settings import ModelSettings


@dataclass(frozen=True)
class UIBConfig:
    """Configuration for a Universal Inverted Block.

    Attributes:
        in_channels: Input channel count
        out_channels: Output channel count
        expanded_channels: Channel count after expansion (typically 32)
        use_dw_before_expand: If True, add DW conv before expansion
        use_dw_after_expand: If True, add DW conv after expansion
        downsample_after: If True, add 3×3 stride=2 DW conv at end
        use_l2_norm: If True, L2-normalize output
    """

    in_channels: int
    out_channels: int
    expanded_channels: int = 32
    use_dw_before_expand: bool = True
    use_dw_after_expand: bool = True
    downsample_after: bool = False
    use_l2_norm: bool = False


class UniversalInvertedBlock(nnx.Module):
    """Universal Inverted Block (UIB) inspired by MobileNet V4.

    Flexible block with optional DW convs before/after expansion,
    optional downsampling, and configurable L2 normalization.

    Structure:
        Input
          ↓
        [DW 3×3] → GN → ReLU         (if use_dw_before_expand)
          ↓
        PW Expand 1×1 → GN → ReLU
          ↓
        [DW 3×3] → GN → ReLU         (if use_dw_after_expand)
          ↓
        PW Compress 1×1 → GN → ReLU
          ↓
        [DW 3×3 stride=2] → GN → ReLU  (if downsample_after)
          ↓
        [L2 norm]                      (if use_l2_norm)
        Output

    All convolutions use VALID padding.
    """

    def __init__(
        self,
        config: UIBConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize UniversalInvertedBlock.

        Args:
            config: UIB configuration
            rngs: NNX RNGs for parameter initialization
        """
        self.config = config

        # DW before expand (optional)
        if config.use_dw_before_expand:
            self.dw_before = nnx.Conv(
                in_features=config.in_channels,
                out_features=config.in_channels,
                kernel_size=(3, 3),
                padding="VALID",
                feature_group_count=config.in_channels,  # Depthwise
                rngs=rngs,
            )
            self.norm_dw_before = nnx.GroupNorm(
                num_groups=max(1, config.in_channels // 4),
                num_features=config.in_channels,
                rngs=rngs,
            )

        # PW Expand: 1×1, in_channels → expanded_channels
        self.pw_expand = nnx.Conv(
            in_features=config.in_channels,
            out_features=config.expanded_channels,
            kernel_size=(1, 1),
            padding="VALID",
            feature_group_count=1,  # Dense
            rngs=rngs,
        )
        self.norm_expand = nnx.GroupNorm(
            num_groups=max(1, config.expanded_channels // 4),
            num_features=config.expanded_channels,
            rngs=rngs,
        )

        # DW after expand (optional)
        if config.use_dw_after_expand:
            self.dw_after = nnx.Conv(
                in_features=config.expanded_channels,
                out_features=config.expanded_channels,
                kernel_size=(3, 3),
                padding="VALID",
                feature_group_count=config.expanded_channels,  # Depthwise
                rngs=rngs,
            )
            self.norm_dw_after = nnx.GroupNorm(
                num_groups=max(1, config.expanded_channels // 4),
                num_features=config.expanded_channels,
                rngs=rngs,
            )

        # PW Compress: 1×1, expanded_channels → out_channels
        self.pw_compress = nnx.Conv(
            in_features=config.expanded_channels,
            out_features=config.out_channels,
            kernel_size=(1, 1),
            padding="VALID",
            feature_group_count=1,  # Dense
            rngs=rngs,
        )
        self.norm_compress = nnx.GroupNorm(
            num_groups=max(1, config.out_channels // 4),
            num_features=config.out_channels,
            rngs=rngs,
        )

        # Downsample (optional): 3×3 stride=2 DW
        if config.downsample_after:
            self.downsample = nnx.Conv(
                in_features=config.out_channels,
                out_features=config.out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                feature_group_count=config.out_channels,  # Depthwise
                rngs=rngs,
            )
            self.norm_downsample = nnx.GroupNorm(
                num_groups=max(1, config.out_channels // 4),
                num_features=config.out_channels,
                rngs=rngs,
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through UIB.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            Tensor of shape (B, H', W', out_channels)
        """
        # DW before expand
        if self.config.use_dw_before_expand:
            x = self.dw_before(x)
            x = self.norm_dw_before(x)
            x = nnx.relu(x)

        # PW Expand
        x = self.pw_expand(x)
        x = self.norm_expand(x)
        x = nnx.relu(x)

        # DW after expand
        if self.config.use_dw_after_expand:
            x = self.dw_after(x)
            x = self.norm_dw_after(x)
            x = nnx.relu(x)

        # PW Compress
        x = self.pw_compress(x)
        x = self.norm_compress(x)
        x = nnx.relu(x)

        # Downsample
        if self.config.downsample_after:
            x = self.downsample(x)
            x = self.norm_downsample(x)
            x = nnx.relu(x)

        # L2 normalization (optional)
        if self.config.use_l2_norm:
            norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
            x = x / (norm + 1e-8)

        return x


class HierarchicalEmbeddingModel(nnx.Module):
    """Hierarchical embedding model with multi-scale feature pyramid.

    Builds a 3-level pyramid with 2 UIBs per level.
    Each level downsamples once (on the second UIB).
    Each level outputs 16D L2-normalized embeddings.

    Default configuration (hardcoded for now):
        Level 0: UIB_0(3→16, no downsample) → UIB_1(16→16, downsample, L2)
        Level 1: UIB_0(16→16, no downsample) → UIB_1(16→16, downsample, L2)
        Level 2: UIB_0(16→16, no downsample) → UIB_1(16→16, downsample, L2)
    """

    def __init__(
        self,
        settings: ModelSettings,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize HierarchicalEmbeddingModel.

        Args:
            settings: Model settings (embed_dim, num_levels)
            rngs: NNX RNGs for parameter initialization
        """
        self.settings = settings
        self.expanded_channels = 32

        # Build levels: each level is a list of 2 UIBs
        self.levels = nnx.List()
        for level_idx in range(settings.num_levels):
            level_blocks = nnx.List()

            for uib_idx in range(2):
                is_first_uib = uib_idx == 0
                is_last_uib = uib_idx == 1
                is_first_level = level_idx == 0

                # Determine input channels
                if is_first_level and is_first_uib:
                    in_channels = 3
                else:
                    in_channels = settings.embed_dim

                # Determine output channels
                out_channels = settings.embed_dim

                # Downsample on second UIB of each level
                downsample_after = is_last_uib

                # L2 norm on last UIB of each level
                use_l2_norm = is_last_uib

                config = UIBConfig(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expanded_channels=self.expanded_channels,
                    use_dw_before_expand=True,
                    use_dw_after_expand=True,
                    downsample_after=downsample_after,
                    use_l2_norm=use_l2_norm,
                )

                level_blocks.append(UniversalInvertedBlock(config, rngs=rngs))

            self.levels.append(level_blocks)

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Forward pass through pyramid.

        Args:
            x: Input tensor of shape (B, H, W, 3)

        Returns:
            List of feature maps, one per level.
            Each level has shape (B, H_l, W_l, embed_dim)
        """
        feature_maps = []

        for level in self.levels:
            # Run through all UIBs in this level
            for uib in level:
                x = uib(x)

            # Collect output (last UIB of level, already L2-normalized)
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
