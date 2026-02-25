"""Tests for window-level flow processing."""

import jax.numpy as jnp
import pytest
from flax import nnx

from flow.window_flow import WindowFlowProcessor
from flow.window_grid import WindowGrid
from flow.embedding_pyramid import EmbeddingPyramid


class TestWindowFlowProcessor:
    """Test WindowFlowProcessor functionality."""
    
    def test_init(self):
        """Test processor initialization."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)
        
        assert processor.embed_dim == 16
        assert processor.window_size == 16
        assert processor.window_grid is not None
        assert processor.patch_lookup is not None
        assert processor.peer_prop is not None
    
    def test_coordinate_grid_creation(self):
        """Test coordinate grid generation."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)
        
        grid = processor._create_coordinate_grid(16, 16)
        
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
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)
        
        # Create test embeddings (B, H, W, C)
        embeddings = jnp.arange(2 * 16 * 16 * 8).reshape(2, 16, 16, 8).astype(jnp.float32)
        
        # Convert to patches
        patches = processor._embeddings_to_patches(embeddings)
        
        # Should be (B, H*W, C)
        assert patches.shape == (2, 256, 8)
        
        # First element should match top-left of original
        assert jnp.allclose(patches[0, 0], embeddings[0, 0, 0])
        
        # Last element should match bottom-right of original
        assert jnp.allclose(patches[0, -1], embeddings[0, 15, 15])
    
    def test_patches_to_grid_roundtrip(self):
        """Test patches -> grid -> patches is identity."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)
        
        # Start with embeddings
        original = jnp.arange(2 * 16 * 16 * 8).reshape(2, 16, 16, 8).astype(jnp.float32)
        
        # To patches
        patches = processor._embeddings_to_patches(original)
        
        # Back to grid
        reconstructed = processor._patches_to_grid(patches, 16, 16)
        
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
        
        # Process
        flow, conf, aux = processor(emb1, emb2)
        
        # Check output shapes
        assert flow.shape == (2, 16, 16, 2)
        assert conf.shape == (2, 16, 16, 1)
        
        # Check aux outputs
        assert aux['flow_lookup'].shape == (2, 16, 16, 2)
        assert aux['flow_peer'].shape == (2, 16, 16, 2)
        assert aux['conf_lookup'].shape == (2, 16, 16, 1)
        assert aux['conf_peer'].shape == (2, 16, 16, 1)
        assert aux['num_windows'] == 1
        
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
        
        # Process
        flow, conf, aux = processor(emb1, emb2)
        
        # Check output shapes
        assert flow.shape == (2, 32, 32, 2)
        assert conf.shape == (2, 32, 32, 1)
        
        # Check aux
        assert aux['num_windows'] == 4
        assert aux['grid_h'] == 2
        assert aux['grid_w'] == 2
        
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
        
        # Process
        flow, conf, aux = processor(emb1, emb2)
        
        # Check output shapes
        assert flow.shape == (2, 64, 64, 2)
        assert conf.shape == (2, 64, 64, 1)
        
        # Check aux
        assert aux['num_windows'] == 16
        assert aux['grid_h'] == 4
        assert aux['grid_w'] == 4
        
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
            
            flow, conf, aux = processor(emb1, emb2)
            
            assert flow.shape[0] == batch_size
            assert conf.shape[0] == batch_size
    
    def test_invalid_dimensions_error(self):
        """Test that invalid dimensions raise clear errors."""
        rngs = nnx.Rngs(0)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)
        
        # 17x17 is not divisible by 16
        emb1 = jnp.zeros((2, 17, 17, 16))
        emb2 = jnp.zeros((2, 17, 17, 16))
        
        with pytest.raises(ValueError, match="must be divisible by window size"):
            processor(emb1, emb2)
    
    def test_reproducibility(self):
        """Test that same inputs produce same outputs."""
        rngs = nnx.Rngs(42)
        processor = WindowFlowProcessor(embed_dim=16, rngs=rngs)
        
        # Same inputs
        emb1 = jnp.zeros((2, 32, 32, 16))
        emb2 = jnp.ones((2, 32, 32, 16)) * 0.1
        
        # Run twice
        flow1, conf1, _ = processor(emb1, emb2)
        flow2, conf2, _ = processor(emb1, emb2)
        
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
        flow_fine, conf_fine, aux = processor(emb1_pyramid[1], emb2_pyramid[1])
        
        # Check outputs
        assert flow_fine.shape == (2, 32, 32, 2)
        assert conf_fine.shape == (2, 32, 32, 1)
        assert not jnp.any(jnp.isnan(flow_fine))
        
        # Process coarse level (16x16) - should be 1 window
        flow_coarse, conf_coarse, aux_coarse = processor(emb1_pyramid[0], emb2_pyramid[0])
        
        assert flow_coarse.shape == (2, 16, 16, 2)
        assert conf_coarse.shape == (2, 16, 16, 1)
        assert aux_coarse['num_windows'] == 1
    
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
        assert emb1_pyramid[0].shape == (1, 16, 16, 16)   # coarsest
        assert emb1_pyramid[1].shape == (1, 32, 32, 16)
        assert emb1_pyramid[2].shape == (1, 64, 64, 16)   # finest
        
        # Process each level
        flow_coarse, _, _ = processor(emb1_pyramid[0], emb2_pyramid[0])
        flow_mid, _, _ = processor(emb1_pyramid[1], emb2_pyramid[1])
        flow_fine, _, aux_fine = processor(emb1_pyramid[2], emb2_pyramid[2])
        
        # Check outputs
        assert flow_coarse.shape == (1, 16, 16, 2)
        assert flow_mid.shape == (1, 32, 32, 2)
        assert flow_fine.shape == (1, 64, 64, 2)
        assert aux_fine['num_windows'] == 16  # 4x4 grid of 16x16 windows


class TestWindowFlowShapes:
    """Test various shape combinations."""
    
    def test_different_embed_dims(self):
        """Test with different embedding dimensions."""
        for embed_dim in [8, 16, 32]:
            rngs = nnx.Rngs(0)
            processor = WindowFlowProcessor(embed_dim=embed_dim, rngs=rngs)
            
            emb1 = jnp.zeros((2, 32, 32, embed_dim))
            emb2 = jnp.zeros((2, 32, 32, embed_dim))
            
            flow, conf, _ = processor(emb1, emb2)
            
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
        
        flow, conf, _ = processor(emb1, emb2)
        
        assert flow.shape == (2, 32, 32, 2)
        assert conf.shape == (2, 32, 32, 1)
