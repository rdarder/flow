"""Tests for embedding visualization functions.

Run: python -m barevision.embeddings.test_visualization
"""

import numpy as np
import jax.random as jr
from flax import nnx

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.visualization import (
    create_frame_with_grid_figure,
    create_loss_heatmap_figures,
    create_attention_maps_figure,
    create_similarity_matrix_figure,
    create_entropy_maps_figure,
)
from barevision.embeddings.settings import TrainingSettings, create_smoke_test_settings


def test_frame_with_grid_figure():
    """Test frame with grid overlay visualization."""
    img1 = np.random.rand(194, 194, 3).astype(np.float32)
    img2 = np.random.rand(194, 194, 3).astype(np.float32)
    metadata = {'video_name': 'test', 'frame_t': 10, 'frame_tk': 13, 'distance': 3}

    # Without highlighted window
    fig = create_frame_with_grid_figure(img1, img2, metadata, 16)
    assert fig.dtype == np.uint8
    assert fig.shape[2] == 3  # RGB
    assert fig.shape[0] > 100  # Reasonable height
    print("✓ test_frame_with_grid_figure (no highlight)")

    # With highlighted window
    fig = create_frame_with_grid_figure(img1, img2, metadata, 16, highlighted_window=(5, 6))
    assert fig.dtype == np.uint8
    print("✓ test_frame_with_grid_figure (with highlight)")


def test_loss_heatmap_figures():
    """Test loss heatmap visualizations."""
    img = np.random.rand(194, 194, 3).astype(np.float32)
    self_loss = np.random.rand(194, 194).astype(np.float32)
    cross_loss = np.random.rand(194, 194).astype(np.float32)

    fig_self, fig_cross = create_loss_heatmap_figures(img, self_loss, cross_loss, 16)

    assert fig_self.dtype == np.uint8
    assert fig_cross.dtype == np.uint8
    assert fig_self.shape == fig_cross.shape
    print("✓ test_loss_heatmap_figures")


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
        window_indices=(5, 6),
        frame_t=100,
        frame_tk=103,
        distance=3,
    )

    assert fig.dtype == np.uint8
    assert fig.shape[2] == 3
    print("✓ test_attention_maps_figure")


def test_similarity_matrix_figure():
    """Test similarity matrix visualization."""
    # Create 256x256 similarity matrix (16x16 window = 256 pixels)
    sim_matrix = np.random.rand(256, 256).astype(np.float32)
    attn_weights = np.random.rand(256, 256).astype(np.float32)
    attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

    fig = create_similarity_matrix_figure(sim_matrix, attn_weights, window_indices=(7, 8))

    assert fig.dtype == np.uint8
    assert fig.shape[2] == 3
    print("✓ test_similarity_matrix_figure")


def test_entropy_maps_figure():
    """Test entropy maps visualization."""
    self_entropy = np.random.rand(16, 16).astype(np.float32)
    cross_entropy = np.random.rand(16, 16).astype(np.float32)
    pixel_positions = np.array([[1, 2], [5, 6]], dtype=np.int32)

    fig = create_entropy_maps_figure(self_entropy, cross_entropy, pixel_positions, window_indices=(9, 10))

    assert fig.dtype == np.uint8
    assert fig.shape[2] == 3
    print("✓ test_entropy_maps_figure")


def test_settings_include_visualization_freq():
    """Test that settings include visualization frequency parameter."""
    settings = TrainingSettings()
    assert hasattr(settings, "log_visualizations_every_steps")
    assert settings.log_visualizations_every_steps == 20  # Default
    print("✓ test_settings_include_visualization_freq")


def test_smoke_test_settings_enable_visualizations():
    """Test that smoke test settings enable frequent visualizations."""
    settings = create_smoke_test_settings()
    assert settings.training.log_visualizations_every_steps == 1
    print("✓ test_smoke_test_settings_enable_visualizations")


def test_model_compute_attention_maps():
    """Test model's compute_attention_maps method."""
    model = SimpleEmbeddingModel(rngs=nnx.Rngs(jr.PRNGKey(0)))
    img1 = jr.uniform(jr.PRNGKey(1), (1, 194, 194, 3))
    img2 = jr.uniform(jr.PRNGKey(2), (1, 194, 194, 3))

    attn_data = model.compute_attention_maps(img1, img2, window_indices=(0, 0))

    assert attn_data.embeddings1.shape == (192, 192, 16)
    assert attn_data.self_attention.shape[0] == 4  # 4 random pixels
    assert attn_data.self_attention.shape[1:] == (16, 16)
    assert attn_data.pixel_positions.shape == (4, 2)
    print("✓ test_model_compute_attention_maps")


def test_model_random_pixel_selection():
    """Test that different windows get different random pixels."""
    model = SimpleEmbeddingModel(rngs=nnx.Rngs(jr.PRNGKey(0)))
    img1 = jr.uniform(jr.PRNGKey(1), (1, 194, 194, 3))
    img2 = jr.uniform(jr.PRNGKey(2), (1, 194, 194, 3))

    # Get attention maps for different windows
    attn_0_0 = model.compute_attention_maps(img1, img2, (0, 0))
    attn_1_1 = model.compute_attention_maps(img1, img2, (1, 1))

    # Pixel positions should be different (deterministic but window-dependent)
    assert not np.array_equal(attn_0_0.pixel_positions, attn_1_1.pixel_positions)
    print("✓ test_model_random_pixel_selection")


if __name__ == "__main__":
    print("Running visualization tests...\n")

    test_frame_with_grid_figure()
    test_loss_heatmap_figures()
    test_attention_maps_figure()
    test_similarity_matrix_figure()
    test_entropy_maps_figure()
    test_settings_include_visualization_freq()
    test_smoke_test_settings_enable_visualizations()
    test_model_compute_attention_maps()
    test_model_random_pixel_selection()

    print("\n✅ All visualization tests passed!")
