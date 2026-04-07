from jax import numpy as jnp


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

    For depthwise convolutions with feature_group_count=in_features, each
    output channel receives input from exactly one input channel. The kernel
    shape is (3, 3, 1, out_features).

    Args:
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        Initializer function compatible with nnx.Conv
    """

    def init(key, input_shape, dtype=jnp.float32):
        # input_shape: (height, width, in_features, out_features)
        # For depthwise conv in Flax/JAX: actual kernel shape is (3, 3, 1, out_features)
        _, _, in_features, out_features = input_shape

        # Create single 3x3 Gaussian kernel
        single_kernel = gaussian_kernel_2d(sigma).astype(dtype)  # (3, 3)

        # For depthwise convolution, Flax uses shape (3, 3, 1, out_features)
        # where each output channel has its own 3x3 kernel applied to 1 input channel
        # Broadcast the same Gaussian to all output channels
        kernel = jnp.broadcast_to(
            single_kernel[:, :, None, None], (3, 3, 1, out_features)
        ).astype(dtype)

        return kernel

    return init
