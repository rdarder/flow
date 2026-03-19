from barevision.flow.settings import DatasetSettings


def calculate_required_input_size(
    target_coarse_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate required input image size to achieve target coarse dimension."""
    size = target_coarse_dim
    for i in reversed(range(num_levels)):
        size += 4 if i == 0 else 2
        if i > 0:
            size = (size - 1) * stride + kernel_size
    return size


def calculate_coarse_output_size(
    input_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate coarse-level output size for a given input."""
    size = input_dim
    for i in range(num_levels):
        size -= 4 if i == 0 else 2
        if i < num_levels - 1:
            size = (size - kernel_size) // stride + 1
    return size

def image_size(coarsest_grid_size: int, window_size: int, levels: int):
    # Target coarse dimension: grid_size × window_size
    target_coarse_dim = coarsest_grid_size * window_size

    # Calculate required input size
    input_size = calculate_required_input_size(
        target_coarse_dim=target_coarse_dim,
        num_levels=levels,
    )
    return input_size, input_size

