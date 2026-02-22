"""Tests for the embedding pyramid module."""

import jax
import jax.numpy as jnp
from flax import nnx

from flow.embedding_pyramid import EmbeddingPyramid, compute_pyramid_shapes


class TestEmbeddingPyramid:
    """Test suite for EmbeddingPyramid."""

    def test_two_level_64x64_grayscale(self):
        """Test 2-level pyramid with 64x64 grayscale input."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        # 64x64 grayscale image
        x = jnp.zeros((1, 64, 64, 1))

        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=1, rngs=rngs)
        embeddings = pyramid(x)

        # Should return 2 levels: coarse to fine
        assert len(embeddings) == 2

        # Level 0 (coarse): 16x16 (64/4)
        assert embeddings[0].shape == (1, 16, 16, 16)

        # Level 1 (fine): 32x32 (64/2)
        assert embeddings[1].shape == (1, 32, 32, 16)

    def test_two_level_64x64_rgb(self):
        """Test 2-level pyramid with 64x64 RGB input."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        # 64x64 RGB image
        x = jnp.zeros((2, 64, 64, 3))  # batch of 2

        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=rngs)
        embeddings = pyramid(x)

        assert len(embeddings) == 2
        assert embeddings[0].shape == (2, 16, 16, 16)
        assert embeddings[1].shape == (2, 32, 32, 16)

    def test_single_level(self):
        """Test single level pyramid (just finest level)."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        x = jnp.zeros((1, 32, 32, 1))

        pyramid = EmbeddingPyramid(num_levels=1, embed_dim=16, in_channels=1, rngs=rngs)
        embeddings = pyramid(x)

        # Single level: finest only
        assert len(embeddings) == 1
        assert embeddings[0].shape == (1, 16, 16, 16)  # 32/2 = 16

    def test_three_levels(self):
        """Test 3-level pyramid."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        x = jnp.zeros((1, 64, 64, 1))

        pyramid = EmbeddingPyramid(num_levels=3, embed_dim=16, in_channels=1, rngs=rngs)
        embeddings = pyramid(x)

        assert len(embeddings) == 3
        # Level 0 (coarsest): 64/8 = 8
        assert embeddings[0].shape == (1, 8, 8, 16)
        # Level 1: 64/4 = 16
        assert embeddings[1].shape == (1, 16, 16, 16)
        # Level 2 (finest): 64/2 = 32
        assert embeddings[2].shape == (1, 32, 32, 16)

    def test_different_embed_dims(self):
        """Test with different embedding dimensions."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        x = jnp.zeros((1, 64, 64, 1))

        for embed_dim in [8, 16, 32, 64]:
            pyramid = EmbeddingPyramid(
                num_levels=2, embed_dim=embed_dim, in_channels=1, rngs=rngs
            )
            embeddings = pyramid(x)

            assert embeddings[0].shape[-1] == embed_dim
            assert embeddings[1].shape[-1] == embed_dim

    def test_level_weights_independent(self):
        """Test that each level has its own independent weights."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=rngs)

        # Finest level: 4 * 3 = 12 input channels
        finest_proj_shape = pyramid.levels[1].proj.kernel.get_value().shape
        assert finest_proj_shape[2] == 12  # in_channels
        assert finest_proj_shape[3] == 16  # embed_dim

        # Coarse level: 4 * 16 = 64 input channels (embeddings from below)
        coarse_proj_shape = pyramid.levels[0].proj.kernel.get_value().shape
        assert coarse_proj_shape[2] == 64  # 4 * embed_dim
        assert coarse_proj_shape[3] == 16  # embed_dim

    def test_compute_pyramid_shapes(self):
        """Test shape computation utility."""
        # 64x64, 2 levels
        shapes = compute_pyramid_shapes((64, 64), 2)
        assert shapes == [(16, 16), (32, 32)]

        # 64x64, 3 levels
        shapes = compute_pyramid_shapes((64, 64), 3)
        assert shapes == [(8, 8), (16, 16), (32, 32)]

        # 128x128, 2 levels
        shapes = compute_pyramid_shapes((128, 128), 2)
        assert shapes == [(32, 32), (64, 64)]

    def test_actual_values_not_zeros(self):
        """Test that output contains actual values, not all zeros."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        # Random input (not zeros)
        x = jax.random.normal(jax.random.PRNGKey(123), (1, 64, 64, 3))

        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=rngs)
        embeddings = pyramid(x)

        # Embeddings should contain non-zero values
        assert jnp.any(embeddings[0] != 0)
        assert jnp.any(embeddings[1] != 0)

    def test_reproducibility(self):
        """Test that same input produces same output with same seed."""
        key = jax.random.PRNGKey(42)

        x = jax.random.normal(jax.random.PRNGKey(123), (1, 64, 64, 3))

        # First run
        pyramid1 = EmbeddingPyramid(
            num_levels=2, embed_dim=16, in_channels=3, rngs=nnx.Rngs(key)
        )
        embeddings1 = pyramid1(x)

        # Second run with same key
        pyramid2 = EmbeddingPyramid(
            num_levels=2, embed_dim=16, in_channels=3, rngs=nnx.Rngs(key)
        )
        embeddings2 = pyramid2(x)

        # Should be identical
        assert jnp.allclose(embeddings1[0], embeddings2[0])
        assert jnp.allclose(embeddings1[1], embeddings2[1])

    def test_jit_compilation(self):
        """Test that module can be JIT compiled."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        x = jax.random.normal(jax.random.PRNGKey(123), (1, 64, 64, 3))

        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=rngs)

        # JIT compile the __call__ method
        @jax.jit
        def run_pyramid(x):
            return pyramid(x)

        # Should run without error
        embeddings = run_pyramid(x)
        assert len(embeddings) == 2

    def test_patchify_2x2_correctness(self):
        """Test that _patchify_2x2 correctly groups spatial regions.

        Create a known pattern and verify the 2x2 patches are flattened correctly.
        """
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        pyramid = EmbeddingPyramid(num_levels=1, embed_dim=16, in_channels=1, rngs=rngs)

        # Create a 4x4 single-channel image with sequential values:
        # [[ 0  1  2  3]
        #  [ 4  5  6  7]
        #  [ 8  9 10 11]
        #  [12 13 14 15]]
        x = jnp.arange(16).reshape(1, 4, 4, 1).astype(jnp.float32)

        # Apply patchify
        patched = pyramid._patchify_2x2(x)

        # Should produce a 2x2 grid with 4 channels each
        assert patched.shape == (1, 2, 2, 4)

        # Verify each 2x2 region is flattened in the correct order
        # Top-left 2x2: [0, 1, 4, 5]
        expected_top_left = jnp.array([0, 1, 4, 5])
        assert jnp.allclose(
            patched[0, 0, 0, :], expected_top_left
        ), f"Top-left mismatch: got {patched[0, 0, 0, :]}, expected {expected_top_left}"

        # Top-right 2x2: [2, 3, 6, 7]
        expected_top_right = jnp.array([2, 3, 6, 7])
        assert jnp.allclose(
            patched[0, 0, 1, :], expected_top_right
        ), f"Top-right mismatch: got {patched[0, 0, 1, :]}, expected {expected_top_right}"

        # Bottom-left 2x2: [8, 9, 12, 13]
        expected_bottom_left = jnp.array([8, 9, 12, 13])
        assert jnp.allclose(
            patched[0, 1, 0, :], expected_bottom_left
        ), f"Bottom-left mismatch: got {patched[0, 1, 0, :]}, expected {expected_bottom_left}"

        # Bottom-right 2x2: [10, 11, 14, 15]
        expected_bottom_right = jnp.array([10, 11, 14, 15])
        assert jnp.allclose(
            patched[0, 1, 1, :], expected_bottom_right
        ), f"Bottom-right mismatch: got {patched[0, 1, 1, :]}, expected {expected_bottom_right}"

    def test_patchify_multi_channel(self):
        """Test _patchify_2x2 with multiple channels.

        Verify that channels are grouped correctly in the flattened output.
        """
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)

        pyramid = EmbeddingPyramid(num_levels=1, embed_dim=16, in_channels=3, rngs=rngs)

        # Create a 4x4 RGB image where each channel has different values
        # Channel 0: values 0-15, Channel 1: values 100-115, Channel 2: values 200-215
        c0 = jnp.arange(16).reshape(4, 4)
        c1 = jnp.arange(100, 116).reshape(4, 4)
        c2 = jnp.arange(200, 216).reshape(4, 4)
        x = jnp.stack([c0, c1, c2], axis=-1).reshape(1, 4, 4, 3).astype(jnp.float32)

        # Apply patchify
        patched = pyramid._patchify_2x2(x)

        # Should produce a 2x2 grid with 12 channels (4*3)
        assert patched.shape == (1, 2, 2, 12)

        # The order is: all channels for each spatial position, then next position
        # Top-left 2x2 positions: (0,0), (0,1), (1,0), (1,1)
        # For each position: [c0, c1, c2]
        # = [0, 100, 200, 1, 101, 201, 4, 104, 204, 5, 105, 205]
        expected = jnp.array([0, 100, 200, 1, 101, 201, 4, 104, 204, 5, 105, 205])
        assert jnp.allclose(
            patched[0, 0, 0, :], expected
        ), f"Multi-channel mismatch: got {patched[0, 0, 0, :]}, expected {expected}"


if __name__ == "__main__":
    test = TestEmbeddingPyramid()

    test.test_two_level_64x64_grayscale()
    print("✓ test_two_level_64x64_grayscale passed")

    test.test_two_level_64x64_rgb()
    print("✓ test_two_level_64x64_rgb passed")

    test.test_single_level()
    print("✓ test_single_level passed")

    test.test_three_levels()
    print("✓ test_three_levels passed")

    test.test_different_embed_dims()
    print("✓ test_different_embed_dims passed")

    test.test_level_weights_independent()
    print("✓ test_level_weights_independent passed")

    test.test_compute_pyramid_shapes()
    print("✓ test_compute_pyramid_shapes passed")

    test.test_actual_values_not_zeros()
    print("✓ test_actual_values_not_zeros passed")

    test.test_reproducibility()
    print("✓ test_reproducibility passed")

    test.test_jit_compilation()
    print("✓ test_jit_compilation passed")

    test.test_patchify_2x2_correctness()
    print("✓ test_patchify_2x2_correctness passed")

    test.test_patchify_multi_channel()
    print("✓ test_patchify_multi_channel passed")

    print("\n=== All embedding pyramid tests passed! ===")
