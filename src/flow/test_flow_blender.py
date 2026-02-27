"""Tests for flow upsampling module."""

import jax.numpy as jnp
import pytest

from flow.flow_blender import (
    upsample_flow_2x,
    upsample_confidence_2x,
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


class TestUpsamplingEdgeCases:
    """Test edge cases for upsampling."""

    def test_upsample_uneven_dimensions(self):
        """Test upsampling with odd dimensions."""
        # 15x15 is odd, but should still upsample to 30x30
        flow = jnp.zeros((1, 15, 15, 2))

        upsampled = upsample_flow_2x(flow)

        assert upsampled.shape == (1, 30, 30, 2)
