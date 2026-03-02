"""Tests for hierarchical optical flow model."""

import jax.numpy as jnp
import pytest
from flax import nnx

from .hierarchical_model import HierarchicalFlowModel


class TestHierarchicalFlowModelInit:
    """Test model initialization."""

    def test_init_default(self):
        """Test default initialization."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(rngs=rngs)

        assert model.num_levels == 2
        assert model.embed_dim == 16
        assert model.in_channels == 3
        assert model.window_size == 16
        assert model.auto_crop is True

    def test_init_custom(self):
        """Test custom initialization."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(
            num_levels=3,
            embed_dim=32,
            in_channels=1,
            window_size=16,
            auto_crop=False,
            rngs=rngs,
        )

        assert model.num_levels == 3
        assert model.embed_dim == 32
        assert model.in_channels == 1
        assert model.auto_crop is False


class TestInputValidation:
    """Test input validation and cropping."""

    def test_valid_size_no_crop_needed(self):
        """Test that exact size images work without cropping."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, rngs=rngs)

        img1 = jnp.zeros((2, 64, 64, 3))
        img2 = jnp.zeros((2, 64, 64, 3))

        # Should not raise
        img1_out, img2_out = model._validate_or_crop_inputs(img1, img2)

        assert img1_out.shape == (2, 64, 64, 3)
        assert jnp.allclose(img1_out, img1)

    def test_auto_crop_larger_image(self):
        """Test that larger images are auto-cropped."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, auto_crop=True, rngs=rngs)

        # 70x70 should be cropped to 64x64
        img1 = jnp.arange(2 * 70 * 70 * 3).reshape(2, 70, 70, 3).astype(jnp.float32)
        img2 = img1.copy()

        img1_out, img2_out = model._validate_or_crop_inputs(img1, img2)

        assert img1_out.shape == (2, 64, 64, 3)

    def test_auto_crop_too_small_error(self):
        """Test that too-small images raise error even with auto_crop."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, auto_crop=True, rngs=rngs)

        # 60x60 is smaller than required 64x64
        img1 = jnp.zeros((2, 60, 60, 3))
        img2 = jnp.zeros((2, 60, 60, 3))

        with pytest.raises(ValueError, match="smaller than required"):
            model._validate_or_crop_inputs(img1, img2)

    def test_no_auto_crop_error(self):
        """Test that mismatched size raises error when auto_crop is False."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, auto_crop=False, rngs=rngs)

        # Even 70x70 should raise error without auto_crop
        img1 = jnp.zeros((2, 70, 70, 3))
        img2 = jnp.zeros((2, 70, 70, 3))

        with pytest.raises(ValueError, match="doesn't match required size"):
            model._validate_or_crop_inputs(img1, img2)


class TestForwardPass:
    """Test end-to-end forward pass."""

    def test_forward_64x64_two_levels(self):
        """Test forward pass with 2 levels on 64x64 images."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(
            num_levels=2,
            in_channels=1,
            rngs=rngs,
        )

        # 64x64 grayscale images
        img1 = jnp.zeros((2, 64, 64, 1))
        # Add motion
        img2 = jnp.roll(img1, shift=(3, 4), axis=(1, 2))

        flow, aux = model(img1, img2)

        # Output should be 32x32 (finest level)
        assert flow.shape == (2, 32, 32, 2)

        # Check no NaNs
        assert not jnp.any(jnp.isnan(flow))

        # Check aux outputs
        assert "flow_normalized" in aux
        assert "confidence" in aux
        assert aux["num_levels"] == 2
        assert aux["finest_resolution"] == (32, 32)

        # Normalized flow should be smaller than pixel flow
        assert jnp.all(jnp.abs(aux["flow_normalized"]) <= jnp.abs(flow))

    def test_forward_128x64_three_levels(self):
        """Test forward pass with 3 levels on 128x128 images."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(
            num_levels=3,
            in_channels=3,
            rngs=rngs,
        )

        # 128x128 RGB images
        img1 = jnp.zeros((1, 128, 128, 3))
        img2 = jnp.ones((1, 128, 128, 3)) * 0.1

        flow, aux = model(img1, img2)

        # Output should be 64x64 (finest level for 3 levels)
        assert flow.shape == (1, 64, 64, 2)
        assert aux["num_levels"] == 3
        assert aux["finest_resolution"] == (64, 64)

    def test_forward_return_intermediates(self):
        """Test that the model provides all level outputs."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        img1 = jnp.zeros((1, 64, 64, 1))
        img2 = jnp.roll(img1, shift=(2, 2), axis=(1, 2))

        flow, aux = model(img1, img2)

        # Check intermediate outputs exist
        assert "level_flows" in aux
        assert "level_confidences" in aux
        assert "level_aux" in aux
        assert "pyramid1" in aux

        # For 2 levels:
        # level 0 (coarse): 16x16 embeddings -> 16x16 flow (normalized)
        # level 1 (fine): 32x32 embeddings -> 32x32 flow (normalized)
        assert len(aux["level_flows"]) == 2
        assert aux["level_flows"][0].shape == (1, 16, 16, 2)
        assert aux["level_flows"][1].shape == (1, 32, 32, 2)  # Finest level is 32x32

        assert len(aux["level_confidences"]) == 2
        assert aux["level_confidences"][0].shape == (1, 16, 16, 1)

    def test_forward_identical_frames(self):
        """Test forward pass with identical frames.

        Note: For an untrained model, we don't expect exactly zero flow.
        We just verify the model runs without errors and produces output.
        """
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        img1 = jnp.zeros((2, 64, 64, 1))
        img2 = img1.copy()

        flow, aux = model(img1, img2)

        # Just verify the model runs and produces valid output
        assert flow.shape == (2, 32, 32, 2)
        assert not jnp.any(jnp.isnan(flow))

    def test_batch_processing(self):
        """Test that batch processing works."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        for batch_size in [1, 2, 4]:
            img1 = jnp.zeros((batch_size, 64, 64, 1))
            img2 = jnp.roll(img1, shift=(2, 3), axis=(1, 2))

            flow, aux = model(img1, img2)

            assert flow.shape[0] == batch_size


class TestGrayscaleAndRGB:
    """Test with different input channels."""

    def test_grayscale(self):
        """Test with grayscale images."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(
            num_levels=2,
            in_channels=1,
            rngs=rngs,
        )

        img1 = jnp.zeros((2, 64, 64, 1))
        img2 = jnp.roll(img1, shift=(2, 3), axis=(1, 2))

        flow, aux = model(img1, img2)

        assert flow.shape == (2, 32, 32, 2)

    def test_rgb(self):
        """Test with RGB images."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(
            num_levels=2,
            in_channels=3,
            rngs=rngs,
        )

        img1 = jnp.zeros((2, 64, 64, 3))
        img2 = jnp.roll(img1, shift=(2, 3), axis=(1, 2))

        flow, aux = model(img1, img2)

        assert flow.shape == (2, 32, 32, 2)


class TestNumericalStability:
    """Test numerical stability."""

    def test_no_nans(self):
        """Test that output never contains NaN."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        img1 = jnp.zeros((2, 64, 64, 1))
        img2 = jnp.ones((2, 64, 64, 1)) * 0.5

        flow, aux = model(img1, img2)

        assert not jnp.any(jnp.isnan(flow))
        assert not jnp.any(jnp.isnan(aux["confidence"]))

    def test_reproducibility(self):
        """Test that same inputs produce same outputs."""
        rngs = nnx.Rngs(42)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        img1 = jnp.zeros((2, 64, 64, 1))
        img2 = jnp.roll(img1, shift=(3, 4), axis=(1, 2))

        # Run twice
        flow1, aux1 = model(img1, img2)
        flow2, aux2 = model(img1, img2)

        assert jnp.allclose(flow1, flow2)


class TestFlowMagnitude:
    """Test flow magnitude and direction."""

    def test_detects_motion_signal(self):
        """Test that model produces non-zero flow when there's motion.

        Note: For an untrained model, we just verify the flow isn't zero
        and is in a reasonable range. We don't check for accurate motion
        detection since the model hasn't been trained yet.
        """
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        # Create a pattern
        img1 = jnp.zeros((1, 64, 64, 1))
        # Put a marker at a specific location
        img1 = img1.at[0, 30:35, 30:35, :].set(1.0)

        # Shift by (4, 5) pixels
        img2 = jnp.roll(img1, shift=(4, 5), axis=(1, 2))

        flow, aux = model(img1, img2)

        # Just verify flow is non-zero and reasonable (model untrained)
        assert not jnp.all(flow == 0), "Flow should be non-zero"

        # Flow magnitude should be reasonable
        max_flow = jnp.max(jnp.abs(flow))
        assert max_flow < 100.0, f"Flow magnitude {max_flow} unreasonably large"

    def test_flow_range(self):
        """Test that flow values are in reasonable range."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(num_levels=2, in_channels=1, rngs=rngs)

        # Create random-ish inputs
        img1 = jnp.arange(2 * 64 * 64 * 1).reshape(2, 64, 64, 1).astype(jnp.float32)
        img1 = img1 / jnp.max(img1)
        img2 = jnp.roll(img1, shift=(5, 6), axis=(1, 2))

        flow, aux = model(img1, img2)

        # Flow should be within image size
        max_flow = jnp.max(jnp.abs(flow))
        assert max_flow < 64.0, f"Flow magnitude {max_flow} exceeds image size"


class TestSingleLevel:
    """Test degenerate case: single level (no blending)."""

    def test_single_level(self):
        """Test with single pyramid level."""
        rngs = nnx.Rngs(0)
        model = HierarchicalFlowModel(
            num_levels=1,
            in_channels=1,
            rngs=rngs,
        )

        # For 1 level: 16 * 2^1 = 32x32 input -> 16x16 output
        img1 = jnp.zeros((2, 32, 32, 1))
        img2 = jnp.roll(img1, shift=(2, 3), axis=(1, 2))

        flow, aux = model(img1, img2)

        # Output should be 16x16
        assert flow.shape == (2, 16, 16, 2)
        assert aux["num_levels"] == 1
