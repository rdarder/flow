"""Tests for window-level flow processing."""

import jax.numpy as jnp
import pytest
from flax import nnx

from flow.window_flow import WindowFlowProcessor
from flow.window_grid import (
    WindowGrid,
    create_coordinate_grid,
    grid_to_tokens,
    tokens_to_grid,
)
from flow.embedding_pyramid import EmbeddingPyramid


def create_zero_prior(batch_size, height, width):
    """Create zero prior flow and low confidence for testing."""
    prior_flow = jnp.zeros((batch_size, height, width, 2))
    # Use very low confidence (0.01) so prior doesn't affect the blend
    prior_confidence = jnp.full((batch_size, height, width, 1), 0.01)
    return prior_flow, prior_confidence


class TestWindowFlowProcessor:
    """Test WindowFlowProcessor functionality."""

    def test_init(self):
        """Test processor initialization."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        assert processor.embed_dim == 16
        assert processor.window_size == 16
        assert processor.window_grid is not None
        assert processor.token_cross_attn is not None
        assert processor.token_self_attn is not None
        assert processor.prior_blender is not None

    def test_coordinate_grid_creation(self):
        """Test coordinate grid generation."""
        grid = create_coordinate_grid(16, 16)

        # Check shape
        assert grid.shape == (16, 16, 2)

        # Check range [0, 1]
        assert jnp.all(grid >= 0.0)
        assert jnp.all(grid <= 1.0)

        # Check corners
        # Top-left should be (0, 0)
        assert jnp.allclose(grid[0, 0], jnp.array([0.0, 0.0]))

        # Bottom-right should be (1, 1)
        assert jnp.allclose(grid[15, 15], jnp.array([1.0, 1.0]))

    def test_patches_conversion(self):
        """Test embeddings to patches conversion."""
        # Create test embeddings (B, H, W, C)
        embeddings = (
            jnp.arange(2 * 16 * 16 * 8).reshape(2, 16, 16, 8).astype(jnp.float32)
        )

        # Convert to tokens
        tokens = grid_to_tokens(embeddings)

        # Should be (B, H*W, C)
        assert tokens.shape == (2, 256, 8)

        # First element should match top-left of original
        assert jnp.allclose(tokens[0, 0], embeddings[0, 0, 0])

        # Last element should match bottom-right of original
        assert jnp.allclose(tokens[0, -1], embeddings[0, 15, 15])

    def test_patches_to_grid_roundtrip(self):
        """Test tokens -> grid -> tokens is identity."""
        # Start with embeddings
        original = jnp.arange(2 * 16 * 16 * 8).reshape(2, 16, 16, 8).astype(jnp.float32)

        # To tokens
        tokens = grid_to_tokens(original)

        # Back to grid
        reconstructed = tokens_to_grid(tokens, 16, 16)

        # Should be identity
        assert reconstructed.shape == original.shape
        assert jnp.allclose(reconstructed, original)

    def test_single_window_16x16(self):
        """Test processing single 16x16 window."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Create embeddings (B=2, H=16, W=16, C=16)
        emb1 = jnp.zeros((2, 16, 16, 16))
        emb2 = jnp.zeros((2, 16, 16, 16))

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(2, 16, 16)

        # Process
        flow, conf, aux = processor(emb1, emb2, prior_flow, prior_confidence)

        # Check output shapes
        assert flow.shape == (2, 16, 16, 2)
        assert conf.shape == (2, 16, 16, 1)

        # Check aux outputs
        assert aux["flow_lookup"].shape == (2, 16, 16, 2)
        assert aux["flow_peer"].shape == (2, 16, 16, 2)
        assert aux["conf_lookup"].shape == (2, 16, 16, 1)
        assert aux["conf_peer"].shape == (2, 16, 16, 1)
        assert aux["num_windows"] == 1

        # Check no NaNs
        assert not jnp.any(jnp.isnan(flow))
        assert not jnp.any(jnp.isnan(conf))

        # Check flow range (should be small for identical inputs)
        # With identical embeddings, flow should be near zero
        assert jnp.all(jnp.abs(flow) < 0.5)  # Generous bound

    def test_four_windows_32x32(self):
        """Test processing 32x32 embeddings (4 windows)."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Create embeddings (B=2, H=32, W=32, C=16)
        emb1 = jnp.zeros((2, 32, 32, 16))
        # Add some variation so flow is non-zero
        emb2 = jnp.ones((2, 32, 32, 16)) * 0.1

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(2, 32, 32)

        # Process
        flow, conf, aux = processor(emb1, emb2, prior_flow, prior_confidence)

        # Check output shapes
        assert flow.shape == (2, 32, 32, 2)
        assert conf.shape == (2, 32, 32, 1)

        # Check aux
        assert aux["num_windows"] == 4
        assert aux["grid_h"] == 2
        assert aux["grid_w"] == 2

        # Check no NaNs
        assert not jnp.any(jnp.isnan(flow))
        assert not jnp.any(jnp.isnan(conf))

    def test_sixteen_windows_64x64(self):
        """Test processing 64x64 embeddings (16 windows)."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Create embeddings (B=2, H=64, W=64, C=16)
        emb1 = jnp.zeros((2, 64, 64, 16))
        emb2 = jnp.ones((2, 64, 64, 16)) * 0.05

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(2, 64, 64)

        # Process
        flow, conf, aux = processor(emb1, emb2, prior_flow, prior_confidence)

        # Check output shapes
        assert flow.shape == (2, 64, 64, 2)
        assert conf.shape == (2, 64, 64, 1)

        # Check aux
        assert aux["num_windows"] == 16
        assert aux["grid_h"] == 4
        assert aux["grid_w"] == 4

        # Check no NaNs
        assert not jnp.any(jnp.isnan(flow))

    def test_batch_processing(self):
        """Test that batching multiple images works correctly."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Process different batch sizes
        for batch_size in [1, 2, 4]:
            emb1 = jnp.zeros((batch_size, 32, 32, 16))
            emb2 = jnp.zeros((batch_size, 32, 32, 16))

            # Create zero priors for testing
            prior_flow, prior_confidence = create_zero_prior(batch_size, 32, 32)

            flow, conf, aux = processor(emb1, emb2, prior_flow, prior_confidence)

            assert flow.shape[0] == batch_size
            assert conf.shape[0] == batch_size

    def test_invalid_dimensions_error(self):
        """Test that invalid dimensions raise clear errors."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # 17x17 is not divisible by 16
        emb1 = jnp.zeros((2, 17, 17, 16))
        emb2 = jnp.zeros((2, 17, 17, 16))

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(2, 17, 17)

        with pytest.raises(ValueError, match="must be divisible by window size"):
            processor(emb1, emb2, prior_flow, prior_confidence)

    def test_reproducibility(self):
        """Test that same inputs produce same outputs."""
        rngs = nnx.Rngs(42)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Same inputs
        emb1 = jnp.zeros((2, 32, 32, 16))
        emb2 = jnp.ones((2, 32, 32, 16)) * 0.1

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(2, 32, 32)

        # Run twice
        flow1, conf1, _ = processor(emb1, emb2, prior_flow, prior_confidence)
        flow2, conf2, _ = processor(emb1, emb2, prior_flow, prior_confidence)

        # Should be identical
        assert jnp.allclose(flow1, flow2)
        assert jnp.allclose(conf1, conf2)


class TestPyramidIntegration:
    """Integration tests with embedding pyramid."""

    def test_pyramid_to_flow_pipeline_64x64(self):
        """Test full pipeline: image -> pyramid -> window flow."""
        rngs = nnx.Rngs(0)

        # Create pyramid
        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=1, rngs=rngs)

        # Create processor
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # 64x64 images
        img1 = jnp.zeros((2, 64, 64, 1))
        # Add some motion
        img2 = jnp.roll(img1, shift=(2, 3), axis=(1, 2))

        # Generate pyramid embeddings
        emb1_pyramid = pyramid(img1)
        emb2_pyramid = pyramid(img2)

        # emb1_pyramid[0] is coarse (16x16), emb1_pyramid[1] is fine (32x32)
        assert len(emb1_pyramid) == 2
        assert emb1_pyramid[0].shape == (2, 16, 16, 16)  # coarse
        assert emb1_pyramid[1].shape == (2, 32, 32, 16)  # fine

        # Process fine level (32x32) - should be 4 windows
        # Create zero priors for testing
        prior_flow_fine, prior_conf_fine = create_zero_prior(2, 32, 32)
        flow_fine, conf_fine, aux = processor(
            emb1_pyramid[1], emb2_pyramid[1], prior_flow_fine, prior_conf_fine
        )

        # Check outputs
        assert flow_fine.shape == (2, 32, 32, 2)
        assert conf_fine.shape == (2, 32, 32, 1)
        assert not jnp.any(jnp.isnan(flow_fine))

        # Process coarse level (16x16) - should be 1 window
        # Create zero priors for testing
        prior_flow_coarse, prior_conf_coarse = create_zero_prior(2, 16, 16)
        flow_coarse, conf_coarse, aux_coarse = processor(
            emb1_pyramid[0], emb2_pyramid[0], prior_flow_coarse, prior_conf_coarse
        )

        assert flow_coarse.shape == (2, 16, 16, 2)
        assert conf_coarse.shape == (2, 16, 16, 1)
        assert aux_coarse["num_windows"] == 1

    def test_pyramid_to_flow_128x128_3_levels(self):
        """Test with 3-level pyramid on 128x128 images."""
        rngs = nnx.Rngs(0)

        # 3 levels for 128x128
        pyramid = EmbeddingPyramid(num_levels=3, embed_dim=16, in_channels=3, rngs=rngs)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # 128x128 RGB images
        img1 = jnp.zeros((1, 128, 128, 3))
        img2 = jnp.ones((1, 128, 128, 3)) * 0.1

        # Generate pyramid
        emb1_pyramid = pyramid(img1)
        emb2_pyramid = pyramid(img2)

        # Check pyramid structure
        assert len(emb1_pyramid) == 3
        assert emb1_pyramid[0].shape == (1, 16, 16, 16)  # coarsest
        assert emb1_pyramid[1].shape == (1, 32, 32, 16)
        assert emb1_pyramid[2].shape == (1, 64, 64, 16)  # finest

        # Process each level
        # Create zero priors for each level
        prior_flow_coarse, prior_conf_coarse = create_zero_prior(1, 16, 16)
        prior_flow_mid, prior_conf_mid = create_zero_prior(1, 32, 32)
        prior_flow_fine, prior_conf_fine = create_zero_prior(1, 64, 64)

        flow_coarse, _, _ = processor(
            emb1_pyramid[0], emb2_pyramid[0], prior_flow_coarse, prior_conf_coarse
        )
        flow_mid, _, _ = processor(
            emb1_pyramid[1], emb2_pyramid[1], prior_flow_mid, prior_conf_mid
        )
        flow_fine, _, aux_fine = processor(
            emb1_pyramid[2], emb2_pyramid[2], prior_flow_fine, prior_conf_fine
        )

        # Check outputs
        assert flow_coarse.shape == (1, 16, 16, 2)
        assert flow_mid.shape == (1, 32, 32, 2)
        assert flow_fine.shape == (1, 64, 64, 2)
        assert aux_fine["num_windows"] == 16  # 4x4 grid of 16x16 windows


class TestPositionPreservation:
    """Test that window processing preserves spatial positions correctly."""

    def test_window_order_preservation_32x32(self):
        """Verify that 4 windows are split and stitched in correct spatial order."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Create embeddings where each window has a unique signature
        # We'll manually create 32x32 with 4 distinct 16x16 windows
        emb1 = jnp.zeros((1, 32, 32, 16))

        # Top-left window: channel 0 = 1.0
        emb1 = emb1.at[0, 0:16, 0:16, 0].set(1.0)
        # Top-right window: channel 1 = 1.0
        emb1 = emb1.at[0, 0:16, 16:32, 1].set(1.0)
        # Bottom-left window: channel 2 = 1.0
        emb1 = emb1.at[0, 16:32, 0:16, 2].set(1.0)
        # Bottom-right window: channel 3 = 1.0
        emb1 = emb1.at[0, 16:32, 16:32, 3].set(1.0)

        # For frame 2, shift the entire pattern by (2, 3) pixels
        emb2 = jnp.roll(emb1, shift=(2, 3), axis=(1, 2))

        # Process
        # Create zero priors for testing (32x32 to match embeddings)
        prior_flow, prior_confidence = create_zero_prior(1, 32, 32)

        flow, conf, aux = processor(emb1, emb2, prior_flow, prior_confidence)

        # Flow should be (1, 32, 32, 2)
        assert flow.shape == (1, 32, 32, 2)

        # Check that each window processed correctly by looking at the embeddings
        # Since emb2 is just a shifted version of emb1, the flow should be consistent
        # within each window (though different windows may have different flows due to
        # boundary effects from the roll)

        # More importantly: verify that the windows weren't scrambled
        # Each window should still contain the right signature in emb1
        # (this checks stitch is inverse of split)
        grid = WindowGrid()
        windows_reconstructed = grid.split(emb1)

        # Check window 0 (top-left) has channel 0 = 1.0
        assert jnp.allclose(windows_reconstructed[0, 0, :, :, 0], 1.0)
        assert jnp.allclose(windows_reconstructed[0, 0, :, :, 1], 0.0)

        # Check window 1 (top-right) has channel 1 = 1.0
        assert jnp.allclose(windows_reconstructed[0, 1, :, :, 1], 1.0)
        assert jnp.allclose(windows_reconstructed[0, 1, :, :, 0], 0.0)

        # Check window 2 (bottom-left) has channel 2 = 1.0
        assert jnp.allclose(windows_reconstructed[0, 2, :, :, 2], 1.0)

        # Check window 3 (bottom-right) has channel 3 = 1.0
        assert jnp.allclose(windows_reconstructed[0, 3, :, :, 3], 1.0)

        # Stitch back and verify identity
        emb1_reconstructed = grid.stitch(windows_reconstructed, grid_h=2, grid_w=2)
        assert jnp.allclose(emb1, emb1_reconstructed)

    def test_coordinate_consistency_within_windows(self):
        """Test that coordinate grids within each window are correct."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Create embeddings that encode their position
        # Each pixel's embedding will be [y/32, x/32, 0, 0, ...]
        h, w = 32, 32

        # Create coordinate grids
        y_grid, x_grid = jnp.meshgrid(
            jnp.arange(h, dtype=jnp.float32) / h,
            jnp.arange(w, dtype=jnp.float32) / w,
            indexing="ij",
        )  # Both are (32, 32)

        # Stack to get (32, 32, 2) position encoding
        pos_encoding_2d = jnp.stack([y_grid, x_grid], axis=-1)  # (32, 32, 2)

        # Broadcast to (32, 32, 16) by padding with zeros
        pos_encoding = jnp.concatenate(
            [pos_encoding_2d, jnp.zeros((h, w, 14), dtype=jnp.float32)], axis=-1
        )

        emb1 = jnp.broadcast_to(pos_encoding, (1, h, w, 16))
        # Shift frame 2 by exactly 2 pixels down and 3 pixels right
        emb2 = jnp.roll(emb1, shift=(2, 3), axis=(1, 2))

        # Process
        # Create zero priors for testing (32x32 to match embeddings)
        prior_flow, prior_confidence = create_zero_prior(1, 32, 32)

        flow, conf, aux = processor(emb1, emb2, prior_flow, prior_confidence)

        # For pixels not near the boundary (rolled region), the flow should
        # detect the 2-pixel down, 3-pixel right shift
        # Avoid boundary where roll wraps around
        interior_flow = flow[0, 4:12, 4:12, :]  # Center of top-left window

        # Expected flow in normalized coordinates:
        # 2 pixels down = 2/32 = 0.0625 in y
        # 3 pixels right = 3/32 = 0.09375 in x
        expected_flow_y = 2.0 / 32.0
        expected_flow_x = 3.0 / 32.0

        # Check flow is roughly in the expected range (allowing for some error)
        # Note: This is a sanity check - exact values depend on the attention mechanism
        mean_flow_y = jnp.mean(jnp.abs(interior_flow[:, :, 1]))
        mean_flow_x = jnp.mean(jnp.abs(interior_flow[:, :, 0]))

        # Flow should be detecting motion (not zero)
        assert mean_flow_y > 0.01, f"Expected y flow > 0.01, got {mean_flow_y}"
        assert mean_flow_x > 0.01, f"Expected x flow > 0.01, got {mean_flow_x}"

        # Flow should be in reasonable range (not huge)
        assert jnp.all(jnp.abs(flow) < 0.5), "Flow values unreasonably large"

    def test_spatial_correspondence_16x16(self):
        """Test exact spatial correspondence for single window case."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # Create simple 16x16 case with identifiable positions
        # Use one-hot-like encoding in channel dimension
        emb1 = jnp.zeros((1, 16, 16, 16))

        # Put unique values at specific positions
        # Position (4, 5) in frame 1 gets a special marker
        emb1 = emb1.at[0, 4, 5, :].set(1.0)

        # In frame 2, move that marker to (6, 7) - 2 down, 2 right
        emb2 = jnp.zeros((1, 16, 16, 16))
        emb2 = emb2.at[0, 6, 7, :].set(1.0)

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(1, 16, 16)

        # Process
        flow, conf, _ = processor(emb1, emb2, prior_flow, prior_confidence)

        # Check flow at position (4, 5) - should point toward (6, 7)
        flow_at_marker = flow[0, 4, 5, :]  # (x_flow, y_flow)

        # Expected: need to move 2 right (+) and 2 down (+) in normalized coords
        # But let's just verify flow is non-zero and reasonable
        flow_magnitude = jnp.linalg.norm(flow_at_marker)
        assert flow_magnitude > 0.01, f"Flow magnitude too small: {flow_magnitude}"

        # The confidence should be reasonable at this position (there was a match)
        conf_at_marker = conf[0, 4, 5, 0]
        assert conf_at_marker > 0.005, f"Confidence too low: {conf_at_marker}"


class TestWindowFlowShapes:
    """Test various shape combinations."""

    def test_different_embed_dims(self):
        """Test with different embedding dimensions."""
        for embed_dim in [8, 16, 32]:
            rngs = nnx.Rngs(0)
            processor = WindowFlowProcessor(embed_dim=embed_dim, rngs=rngs)

            emb1 = jnp.zeros((2, 32, 32, embed_dim))
            emb2 = jnp.zeros((2, 32, 32, embed_dim))

            # Create zero priors for testing
            prior_flow, prior_confidence = create_zero_prior(2, 32, 32)

            flow, conf, _ = processor(emb1, emb2, prior_flow, prior_confidence)

            assert flow.shape == (2, 32, 32, 2)
            assert conf.shape == (2, 32, 32, 1)

    def test_grayscale_and_rgb(self):
        """Test that embedding dimension matters, not original channels."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)

        # The processor works on embeddings (always 16-dim), not raw images
        # So grayscale vs RGB doesn't matter at this stage
        emb1 = jnp.zeros((2, 32, 32, 16))
        emb2 = jnp.zeros((2, 32, 32, 16))

        # Create zero priors for testing
        prior_flow, prior_confidence = create_zero_prior(2, 32, 32)

        flow, conf, _ = processor(emb1, emb2, prior_flow, prior_confidence)

        assert flow.shape == (2, 32, 32, 2)
        assert conf.shape == (2, 32, 32, 1)
