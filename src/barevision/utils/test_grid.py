"""Unit tests for WindowGrid."""

import jax.random as jr
import jax.numpy as jnp
import pytest

from barevision.utils.grid import WindowGrid


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
