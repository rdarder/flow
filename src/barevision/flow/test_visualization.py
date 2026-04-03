"""Tests for hierarchical embedding visualization functions.

Run: python -m barevision.flow.test_visualization
"""

import numpy as np
import jax.random as jr

from barevision.embeddings.visualization import (
    create_attention_maps_figure,
    create_frame_with_grid_figure,
)


def test_attention_maps_figure():
    """Test attention maps visualization with auto-scaling."""
    window_crop1 = np.random.rand(16, 16, 3).astype(np.float32)
    window_crop2 = np.random.rand(16, 16, 3).astype(np.float32)
    self_attn = np.random.rand(4, 16, 16).astype(np.float32)
    cross_attn = np.random.rand(4, 16, 16).astype(np.float32)
    pixel_positions = np.array([[1, 2], [5, 6], [9, 10], [13, 14]], dtype=np.int32)

    fig_array = create_attention_maps_figure(
        window_crop1=window_crop1,
        window_crop2=window_crop2,
        self_attn_maps=self_attn,
        cross_attn_maps=cross_attn,
        pixel_positions=pixel_positions,
        window_indices=(0, 1),
        frame_t=0,
        frame_tk=1,
        distance=1,
    )

    assert fig_array is not None
    assert isinstance(fig_array, np.ndarray)
    assert fig_array.ndim == 3  # H, W, C
    assert fig_array.shape[2] == 3  # RGB
    print("✓ test_attention_maps_figure")


def test_frame_with_grid_figure():
    """Test frame with grid overlay visualization."""
    img1 = np.random.rand(64, 64, 3).astype(np.float32)
    img2 = np.random.rand(64, 64, 3).astype(np.float32)
    metadata = {"video_name": "test", "frame_t": 0, "frame_tk": 1, "distance": 1}

    fig_array = create_frame_with_grid_figure(
        img1=img1,
        img2=img2,
        metadata=metadata,
        window_size=16,
        highlighted_window=(1, 2),
    )

    assert fig_array is not None
    assert isinstance(fig_array, np.ndarray)
    assert fig_array.ndim == 3
    assert fig_array.shape[2] == 3
    print("✓ test_frame_with_grid_figure")


def test_loss_returns_attention_weights():
    """Test that loss functions return attention weights when requested.

    Uses smaller pyramid to reduce JAX compilation overhead.
    """
    from barevision.embeddings import compute_hierarchical_entropy_loss

    # Create simple 2-level pyramid (faster than 3 levels)
    key = jr.PRNGKey(0)
    pyramid1 = [
        jr.normal(key, (1, 32, 32, 16)),  # Level 0
        jr.normal(key, (1, 16, 16, 16)),  # Level 1
    ]
    pyramid2 = [
        jr.normal(key, (1, 32, 32, 16)),
        jr.normal(key, (1, 16, 16, 16)),
    ]

    # Test that attention weights are always returned
    loss, aux = compute_hierarchical_entropy_loss(
        pyramid1,
        pyramid2,
        window_size=16,
        lambda_entropy=0.5,
        level_weight_decay=1.0,
        temperature=1.0,
    )
    assert "self_loss" in aux
    assert "level_self_attention_weights" in aux
    assert len(aux["level_self_attention_weights"]) == 2  # 2 levels
    print("✓ test_loss_returns_attention_weights")


def test_visualization_attention_extraction():
    """Test that window attention extraction works correctly."""
    from barevision.flow.visualization_attention import extract_window_data_for_viz

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
    viz_data = extract_window_data_for_viz(
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
    test_frame_with_grid_figure()
    test_loss_returns_attention_weights()
    test_visualization_attention_extraction()

    print("\n✅ All visualization tests passed!")
