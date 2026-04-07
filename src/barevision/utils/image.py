"""Image size and resolution calculations for UIB-based models.

All convolutions use VALID padding.
- 3×3 VALID: out = in - 2
- 3×3 stride=2 VALID: out = (in - 2) // 2 + 1 = (in - 1) // 2
- 1×1 VALID: out = in (no change)
"""

from typing import List

from barevision.embeddings.model import UIBConfig


def calculate_uib_output_size(input_size: int, config: UIBConfig) -> int:
    """Calculate output spatial size for a single UIB.

    Args:
        input_size: Input spatial dimension (H or W)
        config: UIB configuration

    Returns:
        Output spatial dimension after this UIB

    Note: All convs use VALID padding.
        - DW 3×3: -2 pixels
        - PW 1×1: no change
        - Downsample 3×3 stride=2: (in - 2) // 2 + 1 = (in - 1) // 2
    """
    size = input_size

    # DW before expand
    if config.use_dw_before_expand:
        size -= 2

    # PW expand: 1×1, no spatial change

    # DW after expand
    if config.use_dw_after_expand:
        size -= 2

    # PW compress: 1×1, no spatial change

    # Downsample
    if config.downsample_after:
        size = (size - 1) // 2

    return size


def calculate_level_output_size(
    input_size: int, uib_configs: List[UIBConfig]
) -> int:
    """Calculate output spatial size after a level (sequence of UIBs).

    Args:
        input_size: Input spatial dimension (H or W)
        uib_configs: List of UIB configs in this level (in order)

    Returns:
        Output spatial dimension after this level
    """
    size = input_size
    for config in uib_configs:
        size = calculate_uib_output_size(size, config)
    return size


def calculate_required_input_size(
    target_coarse_dim: int,
    num_levels: int,
    uib_configs_per_level: List[List[UIBConfig]],
) -> int:
    """Calculate required input image size to achieve target coarse dimension.

    Works backwards from target output size through all levels.

    Args:
        target_coarse_dim: Desired spatial dimension at coarsest level
        num_levels: Number of pyramid levels
        uib_configs_per_level: List of UIB config lists (one per level)

    Returns:
        Required input spatial dimension (H and W assumed equal)

    Note: Inverse of forward calculation. For each operation:
        - DW 3×3: in = out + 2
        - Downsample 3×3 stride=2: in = out * 2 + 1
    """
    size = target_coarse_dim

    # Work backwards through levels (coarsest to finest)
    for level_idx in reversed(range(num_levels)):
        uib_configs = uib_configs_per_level[level_idx]

        # Work backwards through UIBs in this level (last to first)
        for config in reversed(uib_configs):
            # Inverse of downsample
            if config.downsample_after:
                size = size * 2 + 1

            # Inverse of PW compress: 1×1, no change

            # Inverse of DW after expand
            if config.use_dw_after_expand:
                size += 2

            # Inverse of PW expand: 1×1, no change

            # Inverse of DW before expand
            if config.use_dw_before_expand:
                size += 2

    return size


def calculate_coarse_output_size(
    input_dim: int,
    num_levels: int,
    uib_configs_per_level: List[List[UIBConfig]],
) -> int:
    """Calculate coarse-level output size for a given input.

    Args:
        input_dim: Input spatial dimension (H or W)
        num_levels: Number of pyramid levels
        uib_configs_per_level: List of UIB config lists (one per level)

    Returns:
        Coarsest level output spatial dimension
    """
    size = input_dim
    for level_idx in range(num_levels):
        uib_configs = uib_configs_per_level[level_idx]
        size = calculate_level_output_size(size, uib_configs)
    return size


def image_size(coarsest_grid_size: int, window_size: int, levels: int) -> tuple:
    """Calculate required input image size for default UIB configuration.

    Default config: 2 UIBs per level, second UIB downsamples,
    both UIBs have 2 DW convs.

    Args:
        coarsest_grid_size: Target grid dimension at coarsest level
        window_size: Window size at coarsest level
        levels: Number of pyramid levels

    Returns:
        Tuple of (height, width) - assumed square
    """
    # Target coarse dimension
    target_coarse_dim = coarsest_grid_size * window_size

    # Build default UIB configs (2 per level, second downsamples)
    uib_configs_per_level = []
    for level_idx in range(levels):
        level_configs = []

        for uib_idx in range(2):
            is_first_uib = uib_idx == 0
            is_first_level = level_idx == 0

            if is_first_level and is_first_uib:
                in_channels = 3
            else:
                in_channels = 16

            config = UIBConfig(
                in_channels=in_channels,
                out_channels=16,
                expanded_channels=32,
                use_dw_before_expand=True,
                use_dw_after_expand=True,
                downsample_after=not is_first_uib,
                use_l2_norm=False,  # Not relevant for size calc
            )
            level_configs.append(config)

        uib_configs_per_level.append(level_configs)

    # Calculate required input size
    input_size = calculate_required_input_size(
        target_coarse_dim=target_coarse_dim,
        num_levels=levels,
        uib_configs_per_level=uib_configs_per_level,
    )

    return input_size, input_size
