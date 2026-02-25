"""Tests for flow blending module."""

import jax.numpy as jnp
import pytest

from flow.flow_blender import (
    upsample_flow_2x,
    upsample_confidence_2x,
    blend_flows,
    FlowBlender,
)


class TestUpsampling:
    """Test flow and confidence upsampling."""

    def test_upsample_flow_2x_shape(self):
        """Test upsampling doubles spatial dimensions."""
        # (B, H, W, 2) -> (B, 2H, 2W, 2)
        flow = jnp.zeros((2, 16, 16, 2))
        upsampled = upsample_flow_2x(flow)

        assert upsampled.shape == (2, 32, 32, 2)

    def test_upsample_flow_2x_values(self):
        """Test upsampling replicates values correctly (nearest neighbor)."""
        # Create flow with unique values
        flow = jnp.arange(2 * 16 * 16 * 2).reshape(2, 16, 16, 2).astype(jnp.float32)
        upsampled = upsample_flow_2x(flow)

        # Check that each 2x2 block in upsampled contains the same value
        # Position (i, j) in original should be at (2*i:2*i+2, 2*j:2*j+2) in upsampled
        for b in range(2):
            for i in range(16):
                for j in range(16):
                    original_value = flow[b, i, j]
                    # Should be replicated in 2x2 block
                    block = upsampled[b, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                    assert jnp.allclose(
                        block, original_value
                    ), f"Value at ({b}, {i}, {j}) not correctly replicated"

    def test_upsample_confidence_2x_shape(self):
        """Test confidence upsampling doubles spatial dimensions."""
        conf = jnp.zeros((2, 16, 16, 1))
        upsampled = upsample_confidence_2x(conf)

        assert upsampled.shape == (2, 32, 32, 1)

    def test_upsample_identity(self):
        """Test that upsampling then downsampling is identity (approximately)."""
        # Start with 16x16, upsample to 32x32, then average pool back
        original = jnp.arange(2 * 16 * 16 * 2).reshape(2, 16, 16, 2).astype(jnp.float32)
        upsampled = upsample_flow_2x(original)

        # Average pool back to 16x16
        downsampled = upsampled.reshape(2, 16, 2, 16, 2, 2).mean(axis=(2, 4))

        # Should be approximately equal (exact equality due to nearest neighbor)
        assert jnp.allclose(downsampled, original)


class TestFlowBlending:
    """Test confidence-weighted flow blending."""

    def test_blend_high_fine_confidence(self):
        """When fine confidence is high, output should be close to fine flow."""
        B, H, W = 2, 32, 32

        # Fine flow: some pattern
        flow_fine = jnp.ones((B, H, W, 2)) * 0.5
        # Coarse flow: different pattern
        flow_coarse = jnp.ones((B, H, W, 2)) * (-0.3)

        # High fine confidence (0.9), low coarse confidence (0.1)
        conf_fine = jnp.ones((B, H, W, 1)) * 0.9
        conf_coarse = jnp.ones((B, H, W, 1)) * 0.1

        flow_final, conf_final = blend_flows(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        # Output should be closer to fine flow
        # weight_fine = 0.9, weight_coarse = 0.1 (from 1 - 0.9)
        # flow_final = (0.9 * 0.5 + 0.1 * (-0.3)) / 1.0 = (0.45 - 0.03) = 0.42
        expected = 0.42
        assert jnp.allclose(
            flow_final, expected, atol=0.01
        ), f"Expected flow ≈ {expected}, got {jnp.mean(flow_final)}"

    def test_blend_low_fine_confidence(self):
        """When fine confidence is low, output should be close to coarse flow."""
        B, H, W = 2, 32, 32

        # Fine flow: some pattern
        flow_fine = jnp.ones((B, H, W, 2)) * 0.5
        # Coarse flow: different pattern
        flow_coarse = jnp.ones((B, H, W, 2)) * (-0.3)

        # Low fine confidence (0.1), high coarse confidence (0.9)
        conf_fine = jnp.ones((B, H, W, 1)) * 0.1
        conf_coarse = jnp.ones((B, H, W, 1)) * 0.9

        flow_final, conf_final = blend_flows(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        # Output should be closer to coarse flow
        # weight_fine = 0.1, weight_coarse = 0.9 (from 1 - 0.1)
        # flow_final = (0.1 * 0.5 + 0.9 * (-0.3)) / 1.0 = (0.05 - 0.27) = -0.22
        expected = -0.22
        assert jnp.allclose(
            flow_final, expected, atol=0.01
        ), f"Expected flow ≈ {expected}, got {jnp.mean(flow_final)}"

    def test_blend_equal_confidence(self):
        """When both confidences are equal (0.5), output should be average."""
        B, H, W = 2, 32, 32

        flow_fine = jnp.ones((B, H, W, 2)) * 0.4
        flow_coarse = jnp.ones((B, H, W, 2)) * 0.2

        conf_fine = jnp.ones((B, H, W, 1)) * 0.5
        conf_coarse = jnp.ones((B, H, W, 1)) * 0.5

        flow_final, conf_final = blend_flows(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        # Should be roughly average: (0.4 + 0.2) / 2 = 0.3
        expected = 0.3
        assert jnp.allclose(
            flow_final, expected, atol=0.01
        ), f"Expected flow ≈ {expected}, got {jnp.mean(flow_final)}"

    def test_blend_zero_confidence(self):
        """Test numerical stability when confidence is near zero."""
        B, H, W = 2, 32, 32

        flow_fine = jnp.ones((B, H, W, 2)) * 0.5
        flow_coarse = jnp.ones((B, H, W, 2)) * 0.3

        # Both near zero
        conf_fine = jnp.ones((B, H, W, 1)) * 1e-8
        conf_coarse = jnp.ones((B, H, W, 1)) * 1e-8

        # Should not produce NaN (epsilon prevents division by zero)
        flow_final, conf_final = blend_flows(
            flow_fine, conf_fine, flow_coarse, conf_coarse, epsilon=1e-6
        )

        assert not jnp.any(jnp.isnan(flow_final)), "Flow contains NaN"
        assert not jnp.any(jnp.isnan(conf_final)), "Confidence contains NaN"

    def test_blend_spatially_varying_confidence(self):
        """Test blending with spatially varying confidence."""
        B, H, W = 1, 16, 16

        # Create fine flow with high values
        flow_fine = jnp.ones((B, H, W, 2)) * 0.8
        # Create coarse flow with low values
        flow_coarse = jnp.ones((B, H, W, 2)) * 0.2

        # Create confidence mask: left half high, right half low
        conf_fine = jnp.zeros((B, H, W, 1))
        conf_fine = conf_fine.at[:, :, : W // 2, :].set(0.9)  # Left: trust fine
        conf_fine = conf_fine.at[:, :, W // 2 :, :].set(0.1)  # Right: trust coarse

        conf_coarse = jnp.ones((B, H, W, 1)) * 0.5  # Doesn't matter much

        flow_final, _ = blend_flows(flow_fine, conf_fine, flow_coarse, conf_coarse)

        # Left side should be closer to fine (0.8)
        left_flow = flow_final[0, :, : W // 2, :]
        assert jnp.mean(left_flow) > 0.6, "Left side should favor fine flow"

        # Right side should be closer to coarse (0.2)
        right_flow = flow_final[0, :, W // 2 :, :]
        assert jnp.mean(right_flow) < 0.4, "Right side should favor coarse flow"

    def test_blend_confidence_output(self):
        """Test that blended confidence is reasonable."""
        B, H, W = 2, 16, 16

        flow_fine = jnp.zeros((B, H, W, 2))
        flow_coarse = jnp.zeros((B, H, W, 2))

        conf_fine = jnp.ones((B, H, W, 1)) * 0.7
        conf_coarse = jnp.ones((B, H, W, 1)) * 0.3

        _, conf_final = blend_flows(flow_fine, conf_fine, flow_coarse, conf_coarse)

        # Blended confidence should be between input confidences
        assert jnp.all(conf_final >= 0.0), "Confidence should be >= 0"
        assert jnp.all(conf_final <= 1.0), "Confidence should be <= 1"

        # With weight_fine=0.7 and weight_coarse=0.3, blended conf should be:
        # (0.7*0.7 + 0.3*0.3) / 1.0 = 0.58
        expected_conf = 0.58
        assert jnp.allclose(conf_final, expected_conf, atol=0.01)


class TestFlowBlenderClass:
    """Test FlowBlender class interface."""

    def test_init(self):
        """Test FlowBlender initialization."""
        blender = FlowBlender(epsilon=1e-5)
        assert blender.epsilon == 1e-5

        blender_default = FlowBlender()
        assert blender_default.epsilon == 1e-6

    def test_blend_pyramid_levels(self):
        """Test full pyramid level blending."""
        blender = FlowBlender()

        # Fine level: 32x32
        flow_fine = jnp.ones((2, 32, 32, 2)) * 0.5
        conf_fine = jnp.ones((2, 32, 32, 1)) * 0.8

        # Coarse level: 16x16 (will be upsampled to 32x32)
        flow_coarse = jnp.ones((2, 16, 16, 2)) * 0.2
        conf_coarse = jnp.ones((2, 16, 16, 1)) * 0.4

        flow_final, conf_final, aux = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        # Check outputs
        assert flow_final.shape == (2, 32, 32, 2)
        assert conf_final.shape == (2, 32, 32, 1)

        # Check auxiliary outputs
        assert "flow_coarse_upsampled" in aux
        assert "conf_coarse_upsampled" in aux
        assert "weight_fine" in aux
        assert aux["flow_coarse_upsampled"].shape == (2, 32, 32, 2)
        assert aux["conf_coarse_upsampled"].shape == (2, 32, 32, 1)

        # Check no NaNs
        assert not jnp.any(jnp.isnan(flow_final))
        assert not jnp.any(jnp.isnan(conf_final))

    def test_blend_pyramid_levels_shape_mismatch(self):
        """Test that shape validation works."""
        blender = FlowBlender()

        flow_fine = jnp.zeros((2, 32, 32, 2))
        conf_fine = jnp.zeros((2, 32, 32, 1))

        # Coarse is 20x20, not 16x16 (should be half of fine)
        flow_coarse = jnp.zeros((2, 20, 20, 2))
        conf_coarse = jnp.zeros((2, 20, 20, 1))

        with pytest.raises(AssertionError, match="should be"):
            blender.blend_pyramid_levels(flow_fine, conf_fine, flow_coarse, conf_coarse)

    def test_blend_pyramid_levels_64x64(self):
        """Test with larger resolution."""
        blender = FlowBlender()

        # Fine: 64x64, Coarse: 32x32
        flow_fine = jnp.zeros((1, 64, 64, 2))
        conf_fine = jnp.ones((1, 64, 64, 1)) * 0.7

        flow_coarse = jnp.zeros((1, 32, 32, 2))
        conf_coarse = jnp.ones((1, 32, 32, 1)) * 0.3

        flow_final, conf_final, aux = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        assert flow_final.shape == (1, 64, 64, 2)
        assert aux["flow_coarse_upsampled"].shape == (1, 64, 64, 2)

    def test_reproducibility(self):
        """Test that blending is deterministic."""
        blender = FlowBlender()

        flow_fine = jnp.ones((2, 16, 16, 2)) * 0.5
        conf_fine = jnp.ones((2, 16, 16, 1)) * 0.7
        flow_coarse = jnp.ones((2, 8, 8, 2)) * 0.3
        conf_coarse = jnp.ones((2, 8, 8, 1)) * 0.4

        # Run twice
        flow1, conf1, _ = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )
        flow2, conf2, _ = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        assert jnp.allclose(flow1, flow2)
        assert jnp.allclose(conf1, conf2)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_blend_mismatched_shapes_error(self):
        """Test that mismatched flow shapes raise error."""
        flow_fine = jnp.zeros((2, 32, 32, 2))
        flow_coarse = jnp.zeros((2, 16, 16, 2))  # Different size
        conf_fine = jnp.zeros((2, 32, 32, 1))
        conf_coarse = jnp.zeros((2, 16, 16, 1))

        with pytest.raises(AssertionError, match="Flow shapes must match"):
            blend_flows(flow_fine, conf_fine, flow_coarse, conf_coarse)

    def test_blend_mismatched_confidence_error(self):
        """Test that mismatched confidence shapes raise error."""
        flow_fine = jnp.zeros((2, 32, 32, 2))
        flow_coarse = jnp.zeros((2, 32, 32, 2))
        conf_fine = jnp.zeros((2, 32, 32, 1))
        conf_coarse = jnp.zeros((2, 16, 16, 1))  # Different size

        with pytest.raises(AssertionError, match="Confidence shapes must match"):
            blend_flows(flow_fine, conf_fine, flow_coarse, conf_coarse)

    def test_upsample_uneven_dimensions(self):
        """Test upsampling with odd dimensions."""
        # 15x15 is odd, but should still upsample to 30x30
        flow = jnp.zeros((1, 15, 15, 2))
        upsampled = upsample_flow_2x(flow)

        assert upsampled.shape == (1, 30, 30, 2)

    def test_very_small_flow(self):
        """Test with very small flow values."""
        blender = FlowBlender()

        flow_fine = jnp.ones((1, 8, 8, 2)) * 1e-10
        conf_fine = jnp.ones((1, 8, 8, 1)) * 0.5
        flow_coarse = jnp.ones((1, 4, 4, 2)) * 2e-10
        conf_coarse = jnp.ones((1, 4, 4, 1)) * 0.5

        flow_final, conf_final, _ = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        assert not jnp.any(jnp.isnan(flow_final))
        assert flow_final.shape == (1, 8, 8, 2)


class TestRealisticScenarios:
    """Test realistic blending scenarios."""

    def test_occlusion_scenario(self):
        """Simulate occlusion: fine level has low confidence in occluded region."""
        blender = FlowBlender()

        H, W = 32, 32

        # Fine flow has good estimates except in occluded region (center)
        flow_fine = jnp.ones((1, H, W, 2)) * 0.5
        conf_fine = jnp.ones((1, H, W, 1)) * 0.9
        # Occluded region: low confidence
        conf_fine = conf_fine.at[0, 12:20, 12:20, :].set(0.1)

        # Coarse flow provides reasonable estimate everywhere
        flow_coarse = jnp.ones((1, 16, 16, 2)) * 0.4
        conf_coarse = jnp.ones((1, 16, 16, 1)) * 0.6

        flow_final, _, aux = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        # In non-occluded regions: should use fine flow (0.5)
        non_occluded = flow_final[0, :12, :12, :]
        assert jnp.allclose(non_occluded, 0.5, atol=0.05)

        # In occluded region: should use more of coarse flow (0.4)
        occluded = flow_final[0, 12:20, 12:20, :]
        # weight_fine = 0.1, weight_coarse = 0.9
        # expected = (0.1*0.5 + 0.9*0.4) / 1.0 = 0.41
        expected_occluded = 0.41
        assert jnp.allclose(occluded, expected_occluded, atol=0.05)

    def test_textureless_region_scenario(self):
        """Simulate textureless region where fine level is uncertain."""
        blender = FlowBlender()

        # Entire image has low fine confidence (textureless)
        flow_fine = jnp.ones((1, 32, 32, 2)) * 0.3
        conf_fine = jnp.ones((1, 32, 32, 1)) * 0.2  # Low confidence everywhere

        flow_coarse = jnp.ones((1, 16, 16, 2)) * 0.6  # Better coarse estimate
        conf_coarse = jnp.ones((1, 16, 16, 1)) * 0.7

        flow_final, _, _ = blender.blend_pyramid_levels(
            flow_fine, conf_fine, flow_coarse, conf_coarse
        )

        # Should mostly use coarse flow (0.6) since fine confidence is low
        # weight_fine = 0.2, weight_coarse = 0.8
        # expected = (0.2*0.3 + 0.8*0.6) / 1.0 = 0.54
        expected = 0.54
        assert jnp.allclose(flow_final, expected, atol=0.05)
