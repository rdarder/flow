"""Tests for hierarchical embedding visualization functions.

Run: python -m barevision.flow.test_visualization
"""

import numpy as np
import jax.random as jr
from flax import nnx

from barevision.flow.embeddings.model import HierarchicalEmbeddingModel
from barevision.flow.embeddings.visualization import (
    create_attention_maps_figure,
)


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
    """Test that window attention extraction works correctly."""
    from barevision.flow.training.visualization import _extract_window_attention_data
    
    # Create attention weights for a level with 3x3 windows
    # The loss function returns attention per window: (num_windows, window_size^2, window_size^2)
    num_windows_h, num_windows_w = 3, 3
    num_windows = num_windows_h * num_windows_w
    window_size = 16
    window_N = window_size * window_size
    
    key = jr.PRNGKey(0)
    # Shape: (num_windows, window_size^2, window_size^2) - one attention matrix per window
    self_attn = jr.uniform(key, (num_windows, window_N, window_N))
    cross_attn = jr.uniform(key, (num_windows, window_N, window_N))
    
    # Extract window at position (1, 1)
    viz_data = _extract_window_attention_data(
        self_attention_weights=self_attn,
        cross_attention_weights=cross_attn,
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

    test_attention_maps_figure()
    test_loss_returns_attention_weights()
    test_visualization_attention_extraction()

    print("\n✅ All visualization tests passed!")
