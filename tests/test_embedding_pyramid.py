"""Tests for the embedding pyramid module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
from flax import nnx
from embedding_pyramid import EmbeddingPyramid, compute_pyramid_shapes


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
            pyramid = EmbeddingPyramid(num_levels=2, embed_dim=embed_dim, in_channels=1, rngs=rngs)
            embeddings = pyramid(x)
            
            assert embeddings[0].shape[-1] == embed_dim
            assert embeddings[1].shape[-1] == embed_dim
    
    def test_level_weights_independent(self):
        """Test that each level has its own independent weights."""
        key = jax.random.PRNGKey(42)
        rngs = nnx.Rngs(key)
        
        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=rngs)
        
        # Finest level: 4 * 3 = 12 input channels
        finest_proj_shape = pyramid.levels[1].proj.kernel.value.shape
        assert finest_proj_shape[2] == 12  # in_channels
        assert finest_proj_shape[3] == 16  # embed_dim
        
        # Coarse level: 4 * 16 = 64 input channels (embeddings from below)
        coarse_proj_shape = pyramid.levels[0].proj.kernel.value.shape
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
        pyramid1 = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=nnx.Rngs(key))
        embeddings1 = pyramid1(x)
        
        # Second run with same key
        pyramid2 = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=3, rngs=nnx.Rngs(key))
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
    
    print("\n=== All embedding pyramid tests passed! ===")
