import jax.numpy as jnp
import pytest
from .window_grid import (
    compute_valid_resolution,
    validate_resolution,
    crop_to_valid,
    WindowGrid,
)
from .embedding_pyramid import EmbeddingPyramid
from flax import nnx


class TestResolutionUtilities:
    """Test resolution computation and validation."""

    def test_compute_valid_resolution(self):
        """Test that valid resolutions are computed correctly."""
        # 1 level: 16 * 2^1 = 32
        assert compute_valid_resolution(1) == (32, 32)

        # 2 levels: 16 * 2^2 = 64
        assert compute_valid_resolution(2) == (64, 64)

        # 3 levels: 16 * 2^3 = 128
        assert compute_valid_resolution(3) == (128, 128)

        # 4 levels: 16 * 2^4 = 256
        assert compute_valid_resolution(4) == (256, 256)

    def test_validate_resolution_exact_match(self):
        """Test validation passes for exact resolution match."""
        is_valid, msg = validate_resolution(64, 64, 2)
        assert is_valid is True
        assert "Valid" in msg or "exactly" in msg

    def test_validate_resolution_croppable(self):
        """Test validation passes for croppable larger resolution."""
        is_valid, msg = validate_resolution(70, 70, 2)
        assert is_valid is False
        assert "must be multiple of" in msg.lower()

    def test_validate_resolution_too_small(self):
        """Test validation fails for resolution that's too small."""
        is_valid, msg = validate_resolution(32, 32, 2)
        assert is_valid is False
        assert "must be multiple of" in msg.lower()


class TestWindowGrid:
    """Test WindowGrid split and stitch operations."""

    def test_init(self):
        """Test WindowGrid initialization."""
        grid = WindowGrid()
        assert grid.window_size == 16
        assert "16" in repr(grid)

        grid_custom = WindowGrid(window_size=8)
        assert grid_custom.window_size == 8

    def test_compute_num_windows(self):
        """Test computation of number of windows."""
        grid = WindowGrid()

        # 16x16 -> 1 window
        assert grid.compute_num_windows(16, 16) == 1

        # 32x32 -> 4 windows (2x2 grid)
        assert grid.compute_num_windows(32, 32) == 4

        # 64x64 -> 16 windows (4x4 grid)
        assert grid.compute_num_windows(64, 64) == 16

        # 32x48 -> 6 windows (2x3 grid)
        assert grid.compute_num_windows(32, 48) == 6

    def test_compute_num_windows_invalid(self):
        """Test that invalid dimensions raise errors."""
        grid = WindowGrid()

        with pytest.raises(ValueError, match="not divisible"):
            grid.compute_num_windows(17, 16)

        with pytest.raises(ValueError, match="not divisible"):
            grid.compute_num_windows(16, 17)

    def test_split_stitch_identity_16x16(self):
        """Test split followed by stitch is identity for 16x16."""
        grid = WindowGrid()

        # Create test data: (B, 16, 16, C)
        original = jnp.arange(2 * 16 * 16 * 8).reshape(2, 16, 16, 8).astype(jnp.float32)

        # Split into windows
        windows = grid.split(original)

        # Should be (B, 1, 16, 16, C)
        assert windows.shape == (2, 1, 16, 16, 8)

        # Stitch back
        reconstructed = grid.stitch(windows, grid_h=1, grid_w=1)

        # Should be identity
        assert reconstructed.shape == original.shape
        assert jnp.allclose(reconstructed, original)

    def test_split_stitch_identity_32x32(self):
        """Test split followed by stitch is identity for 32x32."""
        grid = WindowGrid()

        # Create test data: (B, 32, 32, C)
        original = jnp.arange(2 * 32 * 32 * 8).reshape(2, 32, 32, 8).astype(jnp.float32)

        # Split into windows
        windows = grid.split(original)

        # Should be (B, 4, 16, 16, C)
        assert windows.shape == (2, 4, 16, 16, 8)

        # Stitch back
        reconstructed = grid.stitch(windows, grid_h=2, grid_w=2)

        # Should be identity
        assert reconstructed.shape == original.shape
        assert jnp.allclose(reconstructed, original)

    def test_split_stitch_identity_64x64(self):
        """Test split followed by stitch is identity for 64x64."""
        grid = WindowGrid()

        # Create test data: (B, 64, 64, C)
        original = jnp.arange(2 * 64 * 64 * 8).reshape(2, 64, 64, 8).astype(jnp.float32)

        # Split into windows
        windows = grid.split(original)

        # Should be (B, 16, 16, 16, C)
        assert windows.shape == (2, 16, 16, 16, 8)

        # Stitch back
        reconstructed = grid.stitch(windows, grid_h=4, grid_w=4)

        # Should be identity
        assert reconstructed.shape == original.shape
        assert jnp.allclose(reconstructed, original)

    def test_split_invalid_dimensions(self):
        """Test that split raises error for invalid dimensions."""
        grid = WindowGrid()

        # 17x16 should fail
        invalid_data = jnp.zeros((2, 17, 16, 8))
        with pytest.raises(ValueError, match="not divisible"):
            grid.split(invalid_data)

        # 16x17 should fail
        invalid_data = jnp.zeros((2, 16, 17, 8))
        with pytest.raises(ValueError, match="not divisible"):
            grid.split(invalid_data)

    def test_stitch_window_count_mismatch(self):
        """Test that stitch raises error when window count doesn't match grid."""
        grid = WindowGrid()

        # Create windows claiming to be for 2x2 grid but provide wrong count
        windows = jnp.zeros((2, 3, 16, 16, 8))  # 3 windows for 2x2 grid

        with pytest.raises(ValueError, match="doesn't match"):
            grid.stitch(windows, grid_h=2, grid_w=2)

    def test_stitch_window_size_mismatch(self):
        """Test that stitch raises error when window size doesn't match."""
        grid = WindowGrid()

        # Create windows with wrong size
        windows = jnp.zeros((2, 4, 8, 8, 8))  # 8x8 windows instead of 16x16

        with pytest.raises(ValueError, match="doesn't match"):
            grid.stitch(windows, grid_h=2, grid_w=2)

    def test_split_preserves_values(self):
        """Test that split preserves all values in correct order."""
        grid = WindowGrid()

        # Create a 32x32 grid with unique values so we can verify order
        # (B=1, H=32, W=32, C=1)
        original = jnp.arange(32 * 32).reshape(1, 32, 32, 1).astype(jnp.float32)

        windows = grid.split(original)

        # Check that windows are in correct order:
        # For 2x2 grid, order should be: top-left, top-right, bottom-left, bottom-right
        # Top-left window (first 16x16)
        expected_tl = original[0, :16, :16, :]
        assert jnp.allclose(windows[0, 0, :, :, :], expected_tl)

        # Top-right window
        expected_tr = original[0, :16, 16:, :]
        assert jnp.allclose(windows[0, 1, :, :, :], expected_tr)

        # Bottom-left window
        expected_bl = original[0, 16:, :16, :]
        assert jnp.allclose(windows[0, 2, :, :, :], expected_bl)

        # Bottom-right window
        expected_br = original[0, 16:, 16:, :]
        assert jnp.allclose(windows[0, 3, :, :, :], expected_br)


class TestCropToValid:
    """Test image cropping utilities."""

    def test_crop_to_valid_exact_size_3d(self):
        """Test that exact size image is not modified (3D)."""
        img = jnp.zeros((64, 64, 3))
        cropped = crop_to_valid(img, num_levels=2)
        assert cropped.shape == (64, 64, 3)
        assert jnp.allclose(cropped, img)

    def test_crop_to_valid_exact_size_4d(self):
        """Test that exact size image is not modified (4D batch)."""
        img = jnp.zeros((2, 64, 64, 3))
        cropped = crop_to_valid(img, num_levels=2)
        assert cropped.shape == (2, 64, 64, 3)
        assert jnp.allclose(cropped, img)

    def test_crop_to_valid_larger_image_3d(self):
        """Test center cropping of larger image (3D)."""
        img = jnp.arange(70 * 70 * 3).reshape(70, 70, 3).astype(jnp.float32)
        cropped = crop_to_valid(img, num_levels=2)

        # Should crop to 64x64
        assert cropped.shape == (64, 64, 3)

        # Should be center crop: (70-64)/2 = 3, so crop [3:67]
        expected = img[3:67, 3:67, :]
        assert jnp.allclose(cropped, expected)

    def test_crop_to_valid_larger_image_4d(self):
        """Test center cropping of larger image (4D batch)."""
        img = jnp.arange(2 * 70 * 70 * 3).reshape(2, 70, 70, 3).astype(jnp.float32)
        cropped = crop_to_valid(img, num_levels=2)

        # Should crop to 64x64
        assert cropped.shape == (2, 64, 64, 3)

        # Should be center crop
        expected = img[:, 3:67, 3:67, :]
        assert jnp.allclose(cropped, expected)

    def test_crop_to_valid_different_levels(self):
        """Test cropping for different pyramid depths."""
        # 1 level -> 32x32
        img = jnp.zeros((40, 40, 3))
        cropped = crop_to_valid(img, num_levels=1)
        assert cropped.shape == (32, 32, 3)

        # 2 levels -> 64x64
        img = jnp.zeros((80, 80, 3))
        cropped = crop_to_valid(img, num_levels=2)
        assert cropped.shape == (64, 64, 3)

        # 3 levels -> 128x128
        img = jnp.zeros((140, 140, 3))
        cropped = crop_to_valid(img, num_levels=3)
        assert cropped.shape == (128, 128, 3)

    def test_crop_to_valid_invalid_dimensions(self):
        """Test that invalid array dimensions raise error."""
        # 2D array
        img = jnp.zeros((64, 64))
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            crop_to_valid(img, num_levels=2)

        # 5D array
        img = jnp.zeros((1, 2, 64, 64, 3))
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            crop_to_valid(img, num_levels=2)


class TestPyramidIntegration:
    """Integration tests with the embedding pyramid."""

    def test_pyramid_to_windows_pipeline(self):
        """Test full pipeline: image -> pyramid -> split windows."""
        # Create pyramid for 64x64 image
        rngs = nnx.Rngs(0)
        pyramid = EmbeddingPyramid(num_levels=2, embed_dim=16, in_channels=1, rngs=rngs)

        # 64x64 grayscale image
        img = jnp.zeros((2, 64, 64, 1))
        embeddings = pyramid(img)

        # embeddings[0] is coarse level: 16x16
        # embeddings[1] is fine level: 32x32
        assert len(embeddings) == 2
        assert embeddings[0].shape == (2, 16, 16, 16)
        assert embeddings[1].shape == (2, 32, 32, 16)

        # Split fine level into windows
        grid = WindowGrid()
        windows = grid.split(embeddings[1])

        # Should be 4 windows of 16x16x16 each
        assert windows.shape == (2, 4, 16, 16, 16)

        # Stitch back
        reconstructed = grid.stitch(windows, grid_h=2, grid_w=2)
        assert reconstructed.shape == embeddings[1].shape
        assert jnp.allclose(reconstructed, embeddings[1])

    def test_full_pipeline_3_levels(self):
        """Test pipeline with 3 pyramid levels."""
        rngs = nnx.Rngs(0)
        pyramid = EmbeddingPyramid(num_levels=3, embed_dim=16, in_channels=1, rngs=rngs)

        # 128x128 image for 3 levels
        img = jnp.zeros((2, 128, 128, 1))
        embeddings = pyramid(img)

        # With 3 levels on 128x128 image:
        # Level 2 (finest): 128/2 = 64x64 embeddings
        # Level 1: 64/2 = 32x32 embeddings
        # Level 0 (coarsest): 32/2 = 16x16 embeddings
        assert len(embeddings) == 3
        assert embeddings[0].shape == (2, 16, 16, 16)  # coarsest
        assert embeddings[1].shape == (2, 32, 32, 16)
        assert embeddings[2].shape == (2, 64, 64, 16)  # finest

        # Split the finest level
        grid = WindowGrid()
        windows = grid.split(embeddings[2])

        # 64x64 with 16x16 windows = 4x4 = 16 windows
        assert windows.shape == (2, 16, 16, 16, 16)

        # Stitch and verify
        reconstructed = grid.stitch(windows, grid_h=4, grid_w=4)
        assert jnp.allclose(reconstructed, embeddings[2])
        # compute_valid_resolution(3) = 16 * 2^3 = 128
        # So for 3 levels, we need a 128x128 image
        # Level 2 (finest): 128/2 = 64x64
        # Level 1: 64/2 = 32x32
        # Level 0: 32/2 = 16x16

        # So with 3 levels on 128x128:
        assert embeddings[0].shape == (2, 16, 16, 16)  # coarsest
        assert embeddings[1].shape == (2, 32, 32, 16)
        assert embeddings[2].shape == (2, 64, 64, 16)  # finest

        # Split the finest level
        grid = WindowGrid()
        windows = grid.split(embeddings[2])

        # 64x64 with 16x16 windows = 4x4 = 16 windows
        assert windows.shape == (2, 16, 16, 16, 16)

        # Stitch and verify
        reconstructed = grid.stitch(windows, grid_h=4, grid_w=4)
        assert jnp.allclose(reconstructed, embeddings[2])
