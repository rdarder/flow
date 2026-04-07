"""Tests for Gaussian kernel initialization in downsampling convolutions."""

import jax.numpy as jnp
from flax import nnx

from barevision.embeddings.gaussian import gaussian_kernel_2d, depthwise_gaussian_initializer
from barevision.embeddings.model import UIBConfig, UniversalInvertedBlock


def test_downsample_uses_gaussian_kernel_default_sigma():
    """Verify that downsampling uses Gaussian kernel with default sigma=1.0."""
    config = UIBConfig(
        in_channels=16,
        out_channels=16,
        expanded_channels=32,
        use_dw_before_expand=True,
        use_dw_after_expand=True,
        downsample_after=True,
        # downsample_gaussian_sigma defaults to 1.0
        use_l2_norm=False,
    )

    rngs = nnx.Rngs(42)
    model = UniversalInvertedBlock(config, rngs=rngs)

    # Extract downsample kernel
    downsample_kernel = model.downsample.kernel[...]

    # Expected: Gaussian kernel (sigma=1.0) broadcast to depthwise shape
    expected_kernel = depthwise_gaussian_initializer(1.0)(
        None, (None, None, 16, 16), dtype=jnp.float32
    )

    assert downsample_kernel.shape == (3, 3, 1, 16)
    assert jnp.allclose(downsample_kernel, expected_kernel)


def test_downsample_gaussian_values():
    """Verify Gaussian kernel has expected values for sigma=1.0."""
    config = UIBConfig(
        in_channels=8,
        out_channels=8,
        expanded_channels=16,
        downsample_after=True,
        downsample_gaussian_sigma=1.0,
        use_dw_before_expand=False,
        use_dw_after_expand=False,
        use_l2_norm=False,
    )

    rngs = nnx.Rngs(42)
    model = UniversalInvertedBlock(config, rngs=rngs)

    # Get single channel kernel (should be same for all channels)
    kernel = model.downsample.kernel[..., 0, 0]

    # Compare with reference Gaussian
    expected = gaussian_kernel_2d(1.0)

    assert jnp.allclose(kernel, expected)

    # Verify kernel sums to 1 (proper averaging)
    assert jnp.allclose(jnp.sum(kernel), 1.0)

    # Verify center has highest weight
    assert kernel[1, 1] > kernel[0, 0]
    assert kernel[1, 1] > kernel[0, 1]


def test_downsample_custom_sigma():
    """Verify that custom sigma values work correctly."""
    config = UIBConfig(
        in_channels=16,
        out_channels=16,
        expanded_channels=32,
        downsample_after=True,
        downsample_gaussian_sigma=0.5,  # Sharper Gaussian
        use_dw_before_expand=True,
        use_dw_after_expand=True,
        use_l2_norm=False,
    )

    rngs = nnx.Rngs(42)
    model = UniversalInvertedBlock(config, rngs=rngs)

    kernel = model.downsample.kernel[..., 0, 0]
    expected = gaussian_kernel_2d(0.5)

    assert jnp.allclose(kernel, expected)
    # Sigma=0.5 should be more concentrated than sigma=1.0
    assert kernel[1, 1] > gaussian_kernel_2d(1.0)[1, 1]


def test_gaussian_init_ignored_when_no_downsample():
    """Verify that gaussian_sigma is ignored when downsample_after=False."""
    config = UIBConfig(
        in_channels=16,
        out_channels=16,
        expanded_channels=32,
        downsample_after=False,  # No downsampling
        downsample_gaussian_sigma=1.0,
        use_dw_before_expand=True,
        use_dw_after_expand=True,
        use_l2_norm=False,
    )

    rngs = nnx.Rngs(42)
    model = UniversalInvertedBlock(config, rngs=rngs)

    # Should not have downsample attribute
    assert not hasattr(model, 'downsample')
