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
from typing import List, Tuple

import jax.numpy as jnp
from flax import nnx


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

    def output_size(self, input_size: int) -> int:
        """Calculate output spatial size given input size (forward).

        Args:
            input_size: Input spatial dimension (H or W)

        Returns:
            Output spatial dimension after this UIB

        Note: All convs use VALID padding.
            - DW 3×3: -2 pixels
            - PW 1×1: no change
            - Downsample 3×3 stride=2: (in - 1) // 2
        """
        size = input_size
        if self.use_dw_before_expand:
            size -= 2
        if self.use_dw_after_expand:
            size -= 2
        if self.downsample_after:
            size = (size - 1) // 2
        return size

    def required_input_size(self, output_size: int) -> int:
        """Calculate required input size given desired output (inverse).

        Args:
            output_size: Desired output spatial dimension (H or W)

        Returns:
            Required input spatial dimension for this UIB

        Note: Inverse of forward calculation.
            - Downsample 3×3 stride=2: out * 2 + 1
            - DW 3×3: +2 pixels
        """
        size = output_size
        if self.downsample_after:
            size = size * 2 + 1
        if self.use_dw_after_expand:
            size += 2
        if self.use_dw_before_expand:
            size += 2
        return size


@dataclass(frozen=True)
class LevelConfig:
    """Configuration for a pyramid level.

    Attributes:
        level_idx: Level index (0 = finest, increasing = coarser)
        uib_configs: Tuple of UIB configurations in order
    """

    level_idx: int
    uib_configs: Tuple[UIBConfig, ...]

    def output_size(self, input_size: int) -> int:
        """Calculate output spatial size after all UIBs in this level (forward).

        Args:
            input_size: Input spatial dimension (H or W)

        Returns:
            Output spatial dimension after this level
        """
        size = input_size
        for uib_config in self.uib_configs:
            size = uib_config.output_size(size)
        return size

    def required_input_size(self, output_size: int) -> int:
        """Calculate required input size given desired output (inverse).

        Args:
            output_size: Desired output spatial dimension (H or W)

        Returns:
            Required input spatial dimension for this level
        """
        size = output_size
        for uib_config in reversed(self.uib_configs):
            size = uib_config.required_input_size(size)
        return size


@dataclass(frozen=True)
class HierarchicalModelConfig:
    """Full model configuration for hierarchical embedding pyramid.

    This is the main configuration class for the model. It owns all
    size calculation methods and can build the model.

    Attributes:
        embed_dim: Embedding dimension per level (default 16)
        num_levels: Number of pyramid levels (default 3)
        expanded_channels: Channel count after PW expand (default 32)
        uibs_per_level: UIBs per level (default 2)
    """

    embed_dim: int = 16
    num_levels: int = 3
    expanded_channels: int = 32
    uibs_per_level: int = 2

    def _make_level_config(self, level_idx: int) -> LevelConfig:
        """Build default config for a level.

        Args:
            level_idx: Level index (0 = finest)

        Returns:
            LevelConfig with configured UIBs
        """
        uib_configs = []
        for uib_idx in range(self.uibs_per_level):
            is_first_uib = uib_idx == 0
            is_first_level = level_idx == 0

            in_channels = 3 if (is_first_level and is_first_uib) else self.embed_dim

            uib_configs.append(
                UIBConfig(
                    in_channels=in_channels,
                    out_channels=self.embed_dim,
                    expanded_channels=self.expanded_channels,
                    use_dw_before_expand=True,
                    use_dw_after_expand=True,
                    downsample_after=not is_first_uib,
                    use_l2_norm=not is_first_uib,
                )
            )

        return LevelConfig(level_idx=level_idx, uib_configs=tuple(uib_configs))

    def level_configs(self) -> Tuple[LevelConfig, ...]:
        """Return all level configs."""
        return tuple(self._make_level_config(i) for i in range(self.num_levels))

    def output_size(self, input_size: int) -> int:
        """Calculate final coarse output size given input (forward).

        Args:
            input_size: Input spatial dimension (H or W)

        Returns:
            Coarsest level output spatial dimension
        """
        size = input_size
        for level_config in self.level_configs():
            size = level_config.output_size(size)
        return size

    def required_input_size(self, output_size: int) -> int:
        """Calculate required input size given desired coarse output (inverse).

        Args:
            output_size: Desired coarsest level output spatial dimension

        Returns:
            Required input spatial dimension
        """
        size = output_size
        for level_config in reversed(self.level_configs()):
            size = level_config.required_input_size(size)
        return size

    def target_to_input(self, coarsest_grid_size: int, window_size: int) -> Tuple[int, int]:
        """Calculate required input image size for target coarse grid.

        Args:
            coarsest_grid_size: Target grid dimension at coarsest level
            window_size: Window size at coarsest level

        Returns:
            Tuple of (height, width) - assumed square
        """
        target = coarsest_grid_size * window_size
        required = self.required_input_size(target)
        return required, required

    def build_model(self, *, rngs: nnx.Rngs) -> "HierarchicalEmbeddingModel":
        """Build HierarchicalEmbeddingModel from this config.

        Args:
            rngs: NNX RNGs for parameter initialization

        Returns:
            Instantiated HierarchicalEmbeddingModel
        """
        return HierarchicalEmbeddingModel(self, rngs=rngs)


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


class Level(nnx.Module):
    """Pyramid level containing multiple UIBs.

    A level is a sequence of UIBs that processes features at one scale.
    The level owns its configuration and delegates size calculations to it.
    """

    def __init__(
        self,
        config: LevelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Level.

        Args:
            config: Level configuration
            rngs: NNX RNGs for parameter initialization
        """
        self.config = config
        self.uibs = nnx.List([
            UniversalInvertedBlock(uib_config, rngs=rngs)
            for uib_config in config.uib_configs
        ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through all UIBs in this level.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            Tensor of shape (B, H', W', out_channels)
        """
        for uib in self.uibs:
            x = uib(x)
        return x


class HierarchicalEmbeddingModel(nnx.Module):
    """Hierarchical embedding model with multi-scale feature pyramid.

    Built from HierarchicalModelConfig. Each level has 2 UIBs,
    second UIB downsamples. Each level outputs 16D L2-normalized embeddings.

    Default configuration:
        Level 0: UIB_0(3→16, no downsample) → UIB_1(16→16, downsample, L2)
        Level 1: UIB_0(16→16, no downsample) → UIB_1(16→16, downsample, L2)
        Level 2: UIB_0(16→16, no downsample) → UIB_1(16→16, downsample, L2)
    """

    def __init__(
        self,
        config: HierarchicalModelConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize HierarchicalEmbeddingModel.

        Args:
            config: Model configuration
            rngs: NNX RNGs for parameter initialization
        """
        self.config = config

        # Build levels from config
        self.levels = nnx.List([
            Level(level_config, rngs=rngs)
            for level_config in config.level_configs()
        ])

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
            x = level(x)
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
