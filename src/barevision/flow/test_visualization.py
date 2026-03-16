"""Tests for hierarchical embedding visualization functions.

Run: python -m barevision.flow.test_visualization
"""

import numpy as np
import jax.random as jr
from flax import nnx

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.embeddings.visualization import (
    create_frame_with_grid_figure,
    create_attention_maps_figure,
)


def test_frame_with_grid_figure():
    """Test frame with grid overlay visualization."""
    # Coarse level is 48×48 for 3-level pyramid
    img = np.random.rand(48, 48, 3).astype(np.float32)

    # Create figure
    fig = create_frame_with_grid_figure(img, window_size=16)
    assert fig is not None
    assert hasattr(fig, 'axes')
    print("✓ test_frame_with_grid_figure")


def test_attention_maps_figure():
    """Test attention maps visualization with auto-scaling."""
    window_crop = np.random.rand(16, 16, 3).astype(np.float32)
    self_attn = np.random.rand(4, 16, 16).astype(np.float32)
    cross_attn = np.random.rand(4, 16, 16).astype(np.float32)
    pixel_positions = np.array([[1, 2], [5, 6], [9, 10], [13, 14]], dtype=np.int32)

    fig = create_attention_maps_figure(
        self_attention_maps=self_attn,
        cross_attention_maps=cross_attn,
        pixel_positions=pixel_positions,
        window_crop=window_crop,
        seed_used=42,
    )

    assert fig is not None
    assert hasattr(fig, 'axes')
    print("✓ test_attention_maps_figure")


def test_loss_returns_attention_weights():
    """Test that loss functions return attention weights when requested."""
    from barevision.flow.embeddings.losses import compute_hierarchical_entropy_loss
    
    # Create simple pyramid
    key = jr.PRNGKey(0)
    pyramid1 = [jr.normal(key, (2, 48 - i * 4, 48 - i * 4, 16)) for i in range(3)]
    pyramid2 = [jr.normal(key, (2, 48 - i * 4, 48 - i * 4, 16)) for i in range(3)]
    
    # Test without attention weights (default)
    loss, aux = compute_hierarchical_entropy_loss(
        pyramid1, pyramid2, return_attention_weights=False
    )
    assert "self_loss" in aux
    assert "level_self_attention_weights" not in aux
    
    # Test with attention weights
    loss, aux = compute_hierarchical_entropy_loss(
        pyramid1, pyramid2, return_attention_weights=True
    )
    assert "self_loss" in aux
    assert "level_self_attention_weights" in aux
    assert len(aux["level_self_attention_weights"]) == 3  # 3 levels
    print("✓ test_loss_returns_attention_weights")


def test_visualization_attention_extraction():
    """Test that visualization_attention module can extract window data."""
    from barevision.flow.utils.visualization_attention import extract_window_data_for_viz
    
    # Create attention weights for a level with 3x3 windows
    # The loss function returns attention per window: (B*num_windows, window_size^2, window_size^2)
    num_windows_h, num_windows_w = 3, 3
    num_windows = num_windows_h * num_windows_w
    window_size = 16
    window_N = window_size * window_size
    
    key = jr.PRNGKey(0)
    # Shape: (num_windows, window_size^2, window_size^2) - one attention matrix per window
    self_attn = jr.uniform(key, (num_windows, window_N, window_N))
    cross_attn = jr.uniform(key, (num_windows, window_N, window_N))
    self_entropy = jr.uniform(key, (1, num_windows_h * window_size, num_windows_w * window_size))
    cross_entropy = jr.uniform(key, (1, num_windows_h * window_size, num_windows_w * window_size))
    
    # Extract window at position (1, 1)
    viz_data = extract_window_data_for_viz(
        self_attention_weights=self_attn,
        cross_attention_weights=cross_attn,
        self_entropy_map=self_entropy,
        cross_entropy_map=cross_entropy,
        window_indices=(1, 1),
        num_windows_h=num_windows_h,
        num_windows_w=num_windows_w,
        window_size=window_size,
        pixel_selection_seed=42,
    )
    
    assert "self_attention_maps" in viz_data
    assert "cross_attention_maps" in viz_data
    assert "pixel_positions" in viz_data
    assert viz_data["self_attention_maps"].shape[0] == 4  # 4 pixels
    assert viz_data["pixel_positions"].shape == (4, 2)
    print("✓ test_visualization_attention_extraction")


if __name__ == "__main__":
    print("Running visualization tests...\n")

    test_frame_with_grid_figure()
    test_attention_maps_figure()
    test_loss_returns_attention_weights()
    test_visualization_attention_extraction()

    print("\n✅ All visualization tests passed!")
