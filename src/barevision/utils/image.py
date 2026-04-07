"""Image resolution utilities for barevision.

Calculates input/output dimensions for the embedding pyramid.

New Architecture (stride=2 DW, VALID 3×3):
    output = (input - 3) // 2 + 1
    reverse: input = (output - 1) * 2 + 3

For 3 levels targeting 16×16 at coarsest:
    Level 2: 16 → input = (16-1)*2+3 = 33
    Level 1: 33 → input = (33-1)*2+3 = 67
    Level 0: 67 → input = (67-1)*2+3 = 135
    Input image: 135×135
"""

from barevision.embeddings.settings import DatasetSettings


def calculate_required_input_size(
    target_coarse_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate required input image size to achieve target coarse dimension.

    For new architecture with stride=2 depthwise conv and VALID padding:
        output = (input - kernel_size) // stride + 1
        reverse: input = (output - 1) * stride + kernel_size

    Args:
        target_coarse_dim: Target spatial dimension at coarsest level
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Required input image dimension
    """
    size = target_coarse_dim
    for _ in range(num_levels):
        size = (size - 1) * stride + kernel_size
    return size


def calculate_coarse_output_size(
    input_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate coarse-level output size for a given input.

    For new architecture with stride=2 depthwise conv and VALID padding:
        output = (input - kernel_size) // stride + 1

    Args:
        input_dim: Input image dimension
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Output dimension at coarsest level
    """
    size = input_dim
    for _ in range(num_levels):
        size = (size - kernel_size) // stride + 1
    return size


def image_size(coarsest_grid_size: int, window_size: int, levels: int):
    """Calculate required input image size for given grid and window configuration.

    Args:
        coarsest_grid_size: Number of windows at coarsest level (default 1)
        window_size: Window size in pixels (default 16)
        levels: Number of pyramid levels

    Returns:
        Tuple of (height, width) for input images
    """
    # Target coarse dimension: grid_size × window_size
    target_coarse_dim = coarsest_grid_size * window_size

    # Calculate required input size
    input_size = calculate_required_input_size(
        target_coarse_dim=target_coarse_dim,
        num_levels=levels,
    )
    return input_size, input_size
