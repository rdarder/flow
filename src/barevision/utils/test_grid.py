"""Unit tests for grid utilities."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from barevision.utils.grid import (
    WindowGrid,
    compute_valid_resolution,
    create_coordinate_grid,
    crop_to_valid,
    grid_to_tokens,
    tokens_to_grid,
    validate_resolution,
)


class TestWindowGrid:
    """Tests for WindowGrid class."""

    def test_split_basic(self):
        """Test splitting embeddings into windows."""
        grid = WindowGrid(window_size=16)
        embeddings = jnp.ones((1, 32, 32, 16))  # B=1, H=32, W=32, C=16
        windows = grid.split(embeddings)

        assert windows.shape == (1, 4, 16, 16, 16)  # 4 windows (2×2)

    def test_split_batch(self):
        """Test splitting with batch dimension."""
        grid = WindowGrid(window_size=16)
        embeddings = jnp.ones((4, 64, 64, 16))
        windows = grid.split(embeddings)

        assert windows.shape == (4, 16, 16, 16, 16)  # 16 windows (4×4)

    def test_stitch_basic(self):
        """Test stitching windows back into grid."""
        grid = WindowGrid(window_size=16)
        embeddings = jnp.ones((1, 32, 32, 16))
        windows = grid.split(embeddings)

        # Stitch back
        reconstructed = grid.stitch(windows, grid_h=2, grid_w=2)
        assert reconstructed.shape == (1, 32, 32, 16)
        assert jnp.allclose(embeddings, reconstructed)

    def test_stitch_roundtrip(self):
        """Test split-stitch roundtrip preserves data."""
        grid = WindowGrid(window_size=16)
        key = jr.PRNGKey(0)
        embeddings = jr.normal(key, (2, 64, 64, 16))

        windows = grid.split(embeddings)
        reconstructed = grid.stitch(windows, grid_h=4, grid_w=4)

        assert jnp.allclose(embeddings, reconstructed)

    def test_compute_num_windows(self):
        """Test window count computation."""
        grid = WindowGrid(window_size=16)

        assert grid.compute_num_windows(32, 32) == 4
        assert grid.compute_num_windows(64, 64) == 16
        assert grid.compute_num_windows(128, 64) == 32

    def test_split_invalid_size(self):
        """Test error on invalid dimensions."""
        grid = WindowGrid(window_size=16)
        embeddings = jnp.ones((1, 30, 32, 16))  # 30 not divisible by 16

        try:
            grid.split(embeddings)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not divisible by window size" in str(e)


class TestCoordinateGrid:
    """Tests for create_coordinate_grid."""

    def test_basic_grid(self):
        """Test basic coordinate grid creation."""
        grid = create_coordinate_grid(16, 16)

        assert grid.shape == (16, 16, 2)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_grid_corners(self):
        """Test grid corners are at (0,0) and (1,1)."""
        grid = create_coordinate_grid(16, 16)

        # Top-left corner should be (0, 0)
        assert jnp.allclose(grid[0, 0], jnp.array([0.0, 0.0]))

        # Bottom-right corner should be (1, 1)
        assert jnp.allclose(grid[15, 15], jnp.array([1.0, 1.0]))

    def test_non_square_grid(self):
        """Test non-square grid."""
        grid = create_coordinate_grid(32, 16)

        assert grid.shape == (32, 16, 2)
        assert grid[0, 0, 0] == 0.0  # x at left
        assert grid[0, 15, 0] == 1.0  # x at right
        assert grid[0, 0, 1] == 0.0  # y at top
        assert grid[31, 0, 1] == 1.0  # y at bottom


class TestGridToTokens:
    """Tests for grid_to_tokens and tokens_to_grid."""

    def test_grid_to_tokens(self):
        """Test flattening grid to tokens."""
        grid = jnp.ones((2, 32, 32, 16))
        tokens = grid_to_tokens(grid)

        assert tokens.shape == (2, 32 * 32, 16)

    def test_tokens_to_grid(self):
        """Test reshaping tokens back to grid."""
        tokens = jnp.ones((2, 32 * 32, 16))
        grid = tokens_to_grid(tokens, 32, 32)

        assert grid.shape == (2, 32, 32, 16)

    def test_roundtrip(self):
        """Test grid→tokens→grid roundtrip."""
        key = jr.PRNGKey(0)
        grid = jr.normal(key, (2, 32, 32, 16))

        tokens = grid_to_tokens(grid)
        reconstructed = tokens_to_grid(tokens, 32, 32)

        assert jnp.allclose(grid, reconstructed)


class TestResolutionValidation:
    """Tests for resolution validation utilities."""

    def test_valid_resolution_minimum(self):
        """Test minimum valid resolution."""
        is_valid, msg = validate_resolution(32, 32, num_levels=1, window_size=16)
        assert is_valid
        assert "minimum size" in msg

    def test_valid_resolution_multiple(self):
        """Test resolution that's a multiple of minimum."""
        is_valid, msg = validate_resolution(64, 64, num_levels=1, window_size=16)
        assert is_valid
        assert "compatible" in msg

    def test_invalid_resolution(self):
        """Test invalid resolution."""
        is_valid, msg = validate_resolution(30, 32, num_levels=1, window_size=16)
        assert not is_valid
        assert "Invalid dimensions" in msg

    def test_compute_valid_resolution(self):
        """Test valid resolution computation."""
        h, w = compute_valid_resolution(num_levels=2, window_size=16)
        assert h == 64 and w == 64  # 16 * 2^2 = 64

        h, w = compute_valid_resolution(num_levels=3, window_size=16)
        assert h == 128 and w == 128  # 16 * 2^3 = 128


class TestCropToValid:
    """Tests for crop_to_valid."""

    def test_crop_3d(self):
        """Test cropping 3D array."""
        img = jnp.ones((100, 100, 3))
        cropped = crop_to_valid(img, num_levels=1, window_size=16)

        assert cropped.shape == (32, 32, 3)

    def test_crop_4d(self):
        """Test cropping 4D array."""
        img = jnp.ones((2, 100, 100, 3))
        cropped = crop_to_valid(img, num_levels=1, window_size=16)

        assert cropped.shape == (2, 32, 32, 3)

    def test_crop_already_valid(self):
        """Test no-op when already valid size."""
        img = jnp.ones((1, 32, 32, 3))
        cropped = crop_to_valid(img, num_levels=1, window_size=16)

        assert jnp.array_equal(img, cropped)
