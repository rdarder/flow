"""Tests for hierarchical embedding visualization functions.

Run: python -m barevision.flow.test_visualization
"""

import numpy as np
import jax.random as jr
from flax import nnx

from barevision.flow.model import HierarchicalEmbeddingModel
from barevision.flow.visualization import (
    create_frame_with_grid_figure,
    create_attention_maps_figure,
)


def test_frame_with_grid_figure():
    """Test frame with grid overlay visualization."""
    # Coarse level is 48×48 for 3-level pyramid
    img1 = np.random.rand(48, 48, 3).astype(np.float32)
    img2 = np.random.rand(48, 48, 3).astype(np.float32)
    metadata = {"video_name": "test", "frame_t": 10, "frame_tk": 13, "distance": 3}

    # Without highlighted window (3×3 grid of 16×16)
    fig = create_frame_with_grid_figure(img1, img2, metadata, 16)
    assert fig.dtype == np.uint8
    assert fig.shape[2] == 3  # RGB
    assert fig.shape[0] > 100  # Reasonable height
    print("✓ test_frame_with_grid_figure (no highlight)")

    # With highlighted window
    fig = create_frame_with_grid_figure(
        img1, img2, metadata, 16, highlighted_window=(1, 1)
    )
    assert fig.dtype == np.uint8
    print("✓ test_frame_with_grid_figure (with highlight)")


def test_attention_maps_figure():
    """Test attention maps visualization with both frame crops and auto-scaling."""
    window_crop1 = np.random.rand(16, 16, 3).astype(np.float32)
    window_crop2 = np.random.rand(16, 16, 3).astype(np.float32)
    self_attn = np.random.rand(4, 16, 16).astype(np.float32)
    cross_attn = np.random.rand(4, 16, 16).astype(np.float32)
    pixel_positions = np.array([[1, 2], [5, 6], [9, 10], [13, 14]], dtype=np.int32)

    fig = create_attention_maps_figure(
        window_crop1,
        window_crop2,  # Both frame crops
        self_attn,
        cross_attn,
        pixel_positions,
        window_indices=(1, 1),
        frame_t=100,
        frame_tk=103,
        distance=3,
    )

    assert fig.dtype == np.uint8
    assert fig.shape[2] == 3
    print("✓ test_attention_maps_figure")


def test_model_compute_attention_maps():
    """Test model's compute_attention_maps method for hierarchical model."""
    model = HierarchicalEmbeddingModel(
        embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
    )
    # Input size for 3-level pyramid targeting 48×48 coarse
    img1 = jr.uniform(jr.PRNGKey(1), (1, 391, 391, 3))
    img2 = jr.uniform(jr.PRNGKey(2), (1, 391, 391, 3))

    # Test coarsest level (level_index=-1)
    attn_data = model.compute_attention_maps(
        img1, img2, window_indices=(0, 0), level_index=-1
    )

    # Coarsest level is 48×48
    assert attn_data.embeddings1.shape == (48, 48, 16)
    assert attn_data.self_attention.shape[0] == 4  # 4 random pixels
    assert attn_data.self_attention.shape[1:] == (16, 16)
    assert attn_data.pixel_positions.shape == (4, 2)
    print("✓ test_model_compute_attention_maps (coarsest level)")


def test_model_random_pixel_selection():
    """Test that different windows get different random pixels."""
    model = HierarchicalEmbeddingModel(
        embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
    )
    img1 = jr.uniform(jr.PRNGKey(1), (1, 391, 391, 3))
    img2 = jr.uniform(jr.PRNGKey(2), (1, 391, 391, 3))

    # Get attention maps for different windows at coarsest level
    attn_0_0 = model.compute_attention_maps(img1, img2, (0, 0), level_index=-1)
    attn_1_1 = model.compute_attention_maps(img1, img2, (1, 1), level_index=-1)

    # Pixel positions should be different (deterministic but window-dependent)
    assert not np.array_equal(attn_0_0.pixel_positions, attn_1_1.pixel_positions)
    print("✓ test_model_random_pixel_selection")


if __name__ == "__main__":
    print("Running visualization tests...\n")

    test_frame_with_grid_figure()
    test_attention_maps_figure()
    test_model_compute_attention_maps()
    test_model_random_pixel_selection()

    print("\n✅ All visualization tests passed!")
