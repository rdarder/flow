"""Tests for mean convolution kernel analysis."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.mean_conv_analysis import (
    analyze_mean_conv_kernels,
    gaussian_kernel_2d,
)
from barevision.flow.settings import EmbeddingModelSettings


class TestGaussianKernel:
    """Tests for reference Gaussian kernel generation."""

    def test_gaussian_kernel_shape(self):
        """Test kernel has correct shape."""
        kernel = gaussian_kernel_2d(sigma=1.0, size=3)
        assert kernel.shape == (3, 3)

    def test_gaussian_kernel_normalized(self):
        """Test kernel sums to 1.0."""
        kernel = gaussian_kernel_2d(sigma=1.0, size=3)
        assert jnp.allclose(jnp.sum(kernel), 1.0)

    def test_gaussian_kernel_center_peak(self):
        """Test center is highest value."""
        kernel = gaussian_kernel_2d(sigma=1.0, size=3)
        center = kernel[1, 1]
        assert all(
            center > kernel[i, j]
            for i in range(3)
            for j in range(3)
            if (i, j) != (1, 1)
        )

    def test_gaussian_larger_sigma(self):
        """Test larger sigma produces flatter kernel."""
        kernel_small_sigma = gaussian_kernel_2d(sigma=0.5, size=3)
        kernel_large_sigma = gaussian_kernel_2d(sigma=2.0, size=3)

        # Larger sigma should have lower center peak
        assert kernel_small_sigma[1, 1] > kernel_large_sigma[1, 1]

        # Larger sigma should have higher corners
        assert kernel_large_sigma[0, 0] > kernel_small_sigma[0, 0]


class TestAnalyzeMeanConvKernels:
    """Tests for kernel analysis functionality."""

    def test_analysis_returns_all_levels(self):
        """Test analysis includes all pyramid levels."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=3,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=3, hidden_dim=32)

        assert "level_0" in analysis
        assert "level_1" in analysis
        assert "level_2" in analysis

    def test_analysis_computes_all_scalars(self):
        """Test all expected scalar metrics are computed."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        scalars = analysis["level_0"]["scalars"]

        expected_scalars = [
            "weight_sum_mean",
            "weight_sum_std",
            "weight_sum_min",
            "weight_sum_max",
            "center_surround_ratio_mean",
            "center_surround_ratio_std",
            "drift_from_init_mean",
            "drift_from_init_std",
            "effective_sigma_mean",
            "effective_sigma_std",
            "channel_specialization",
            "positive_weight_ratio",
        ]

        for scalar in expected_scalars:
            assert scalar in scalars, f"Missing scalar: {scalar}"

    def test_weight_sums_near_one(self):
        """Test that initialized kernels have weight sums near 1.0."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        scalars = analysis["level_0"]["scalars"]

        # After Gaussian init, weight sums should be very close to 1.0
        assert jnp.isclose(scalars["weight_sum_mean"], 1.0, atol=1e-5)
        assert scalars["weight_sum_std"] < 1e-5

    def test_center_surround_ratio_highOnInit(self):
        """Test that initialized kernels have high center-surround ratio."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        scalars = analysis["level_0"]["scalars"]

        # Gaussian with sigma=1.0 should have center ~4-5× higher than corners
        assert scalars["center_surround_ratio_mean"] > 2.0

    def test_drift_from_init_zeroOnInit(self):
        """Test that drift from initialization is near zero at init."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        scalars = analysis["level_0"]["scalars"]

        # At initialization, drift should be near zero
        assert scalars["drift_from_init_mean"] < 1e-5

    def test_effective_sigma_near_oneOnInit(self):
        """Test effective sigma is near 1.0 at initialization."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        scalars = analysis["level_0"]["scalars"]

        # Should be close to sigma=1.0
        assert jnp.isclose(scalars["effective_sigma_mean"], 1.0, atol=0.1)

    def test_histograms_have_correct_shape(self):
        """Test histogram arrays have correct dimensionality."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        histograms = analysis["level_0"]["histograms"]

        # All histograms should have 32 values (one per channel)
        for name, values in histograms.items():
            assert values.shape == (32,), f"{name} has wrong shape: {values.shape}"

    def test_kernels_for_visualization(self):
        """Test kernel array is suitable for visualization."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        analysis = analyze_mean_conv_kernels(model_state, num_levels=1, hidden_dim=32)
        kernels = analysis["level_0"]["kernels"]

        # Should be (3, 3, 32) for visualization
        assert kernels.shape == (3, 3, 32)

    def test_handles_missing_levels(self):
        """Test analysis gracefully handles missing levels."""
        model = HierarchicalEmbeddingModel(
            EmbeddingModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        model_state = nnx.state(model)

        # Request analysis for 3 levels when model only has 1
        analysis = analyze_mean_conv_kernels(model_state, num_levels=3, hidden_dim=32)

        # Should only return level_0
        assert "level_0" in analysis
        assert "level_1" not in analysis
        assert "level_2" not in analysis
