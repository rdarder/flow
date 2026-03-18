"""Tests for flow estimation components."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barevision.flow.matching import (
    AttentionFeatures,
    FlowEstimator,
    create_source_position_grid,
    flow_to_dense,
    warp_embeddings,
    reconstruction_loss_core,
)


def test_attention_features_shape():
    """AttentionFeatures output has correct shape."""
    B, N = 2, 256  # 16x16 window

    # Create dummy attention maps (uniform for simplicity)
    self_attn = jnp.ones((B, N, N)) / N
    cross_attn = jnp.ones((B, N, N)) / N

    # Create source position grid
    src_pos = create_source_position_grid(window_size=16)
    src_pos = jnp.broadcast_to(src_pos, (B, N, 2))

    features_computer = AttentionFeatures(window_size=16)
    features = features_computer(self_attn, cross_attn, src_pos)

    assert features.shape == (B, N, 8), f"Expected (B, N, 8), got {features.shape}"
    print(f"✓ AttentionFeatures shape: {features.shape}")


def test_attention_features_peak():
    """Features from delta-function attention are at peak location."""
    H = W = 16
    N = H * W

    # Test with peak at known position
    peak_y, peak_x = 5, 10
    peak_idx = peak_y * W + peak_x

    # Create delta-function attention (all weight on one pixel)
    self_attn = jnp.zeros((1, N, N))
    self_attn = self_attn.at[:, :, peak_idx].set(1.0)

    cross_attn = jnp.zeros((1, N, N))
    cross_attn = cross_attn.at[:, :, peak_idx].set(1.0)

    # Create source position at the same location (self-attention should center here)
    src_pos = jnp.zeros((1, N, 2))
    src_pos = src_pos.at[:, peak_idx, :].set(
        jnp.array([peak_x / (W - 1), peak_y / (H - 1)])
    )

    features_computer = AttentionFeatures(window_size=16)
    features = features_computer(self_attn, cross_attn, src_pos)

    # Expected normalized coordinates
    expected_cx = peak_x / (W - 1)
    expected_cy = peak_y / (H - 1)

    # For peak location, check features
    # self_relative should be ~0 (self-attention centered on source)
    self_rel_cx = features[0, peak_idx, 0]
    self_rel_cy = features[0, peak_idx, 1]

    np.testing.assert_allclose(self_rel_cx, 0.0, atol=1e-6)
    np.testing.assert_allclose(self_rel_cy, 0.0, atol=1e-6)

    # cross_relative should be ~0 (cross-attention at same location)
    cross_rel_cx = features[0, peak_idx, 2]
    cross_rel_cy = features[0, peak_idx, 3]

    np.testing.assert_allclose(cross_rel_cx, 0.0, atol=1e-6)
    np.testing.assert_allclose(cross_rel_cy, 0.0, atol=1e-6)

    # cross_absolute should be at peak location
    cross_abs_cx = features[0, peak_idx, 4]
    cross_abs_cy = features[0, peak_idx, 5]

    np.testing.assert_allclose(cross_abs_cx, expected_cx, rtol=1e-5)
    np.testing.assert_allclose(cross_abs_cy, expected_cy, rtol=1e-5)

    # Max peaks should be 1.0 (delta function)
    self_max = features[0, peak_idx, 6]
    cross_max = features[0, peak_idx, 7]

    np.testing.assert_allclose(self_max, 1.0, rtol=1e-5)
    np.testing.assert_allclose(cross_max, 1.0, rtol=1e-5)

    print(f"✓ AttentionFeatures peak: all features correct at peak location")


def test_attention_features_translation():
    """Features correctly encode translation between self and cross attention."""
    H = W = 16
    N = H * W

    # Self-attention peak at center
    self_peak_y, self_peak_x = 8, 8
    self_peak_idx = self_peak_y * W + self_peak_x

    # Cross-attention peak shifted by (2, 3) pixels
    cross_peak_y, cross_peak_x = 10, 11
    cross_peak_idx = cross_peak_y * W + cross_peak_x

    # Create delta-function attention
    self_attn = jnp.zeros((1, N, N))
    self_attn = self_attn.at[:, :, self_peak_idx].set(1.0)

    cross_attn = jnp.zeros((1, N, N))
    cross_attn = cross_attn.at[:, :, cross_peak_idx].set(1.0)

    # Source position at self-attention peak
    src_x = self_peak_x / (W - 1)
    src_y = self_peak_y / (H - 1)
    src_pos = jnp.broadcast_to(jnp.array([[[src_x, src_y]]]), (1, N, 2))

    features_computer = AttentionFeatures(window_size=16)
    features = features_computer(self_attn, cross_attn, src_pos)

    # Expected flow (normalized)
    expected_flow_x = (cross_peak_x - self_peak_x) / (W - 1)
    expected_flow_y = (cross_peak_y - self_peak_y) / (H - 1)

    # cross_relative should encode the flow
    cross_rel_cx = features[0, 0, 2]
    cross_rel_cy = features[0, 0, 3]

    np.testing.assert_allclose(cross_rel_cx, expected_flow_x, rtol=1e-5)
    np.testing.assert_allclose(cross_rel_cy, expected_flow_y, rtol=1e-5)

    print(
        f"✓ AttentionFeatures translation: flow ({expected_flow_x:.3f}, {expected_flow_y:.3f}) correctly encoded"
    )


def test_flow_estimator_shape():
    """Flow estimator produces (B, N, 2) output."""
    B, N = 2, 256

    # Create dummy features (8 features per pixel)
    features = jnp.zeros((B, N, 8))

    flow_estimator = FlowEstimator(
        window_size=16, hidden_dim=32, max_flow=0.5, rngs=nnx.Rngs(0)
    )
    flow = flow_estimator(features)

    assert flow.shape == (B, N, 2), f"Expected (B, N, 2), got {flow.shape}"
    print(f"✓ FlowEstimator shape: {flow.shape}")


def test_flow_estimator_parameters():
    """Flow estimator has expected parameter count."""
    flow_estimator = FlowEstimator(
        window_size=16, hidden_dim=32, max_flow=0.5, rngs=nnx.Rngs(0)
    )

    from barevision.flow.embeddings.model import count_parameters

    params = count_parameters(flow_estimator)

    # Linear(8→32) = 8*32 + 32 = 288
    # Linear(32→32) = 32*32 + 32 = 1056
    # Linear(32→32) = 32*32 + 32 = 1056
    # Linear(32→2) = 32*2 + 2 = 66
    # Total = 2466
    expected_params = (8 * 32 + 32) + (32 * 32 + 32) + (32 * 32 + 32) + (32 * 2 + 2)
    assert params == expected_params, f"Expected {expected_params} params, got {params}"
    print(f"✓ FlowEstimator parameters: {params}")


def test_flow_estimator_bounded():
    """Flow estimator output is bounded to [-max_flow, max_flow]."""
    B, N = 2, 256
    max_flow = 0.5

    # Create random features that could produce large outputs
    rng = jax.random.PRNGKey(0)
    features = jax.random.uniform(rng, (B, N, 8), minval=-10, maxval=10)

    flow_estimator = FlowEstimator(
        window_size=16, hidden_dim=32, max_flow=max_flow, rngs=nnx.Rngs(0)
    )
    flow = flow_estimator(features)

    # Check that flow is bounded
    assert (
        flow.max() <= max_flow + 1e-6
    ), f"Flow max {flow.max()} exceeds max_flow {max_flow}"
    assert (
        flow.min() >= -max_flow - 1e-6
    ), f"Flow min {flow.min()} below -max_flow {-max_flow}"

    print(
        f"✓ FlowEstimator bounded: range [{flow.min():.4f}, {flow.max():.4f}] within [-{max_flow}, {max_flow}]"
    )


def test_warp_embeddings_identity():
    """Zero flow → warped == original."""
    B, H, W, D = 1, 16, 16, 8

    # Create random embeddings
    rng = jax.random.PRNGKey(0)
    embeddings = jax.random.normal(rng, (B, H, W, D))

    # Zero flow
    flow = jnp.zeros((B, H, W, 2))

    warped = warp_embeddings(embeddings, flow)

    np.testing.assert_allclose(warped, embeddings, rtol=1e-5)
    print(f"✓ warp_embeddings identity: zero flow produces identical output")


def test_warp_embeddings_shift():
    """Known shift → correct warping."""
    B, H, W = 1, 16, 16

    # Create simple test pattern: gradient in x direction
    x = jnp.arange(W, dtype=jnp.float32) / (W - 1)
    y = jnp.arange(H, dtype=jnp.float32) / (H - 1)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    embeddings = jnp.stack([X, Y], axis=-1)  # (H, W, 2)
    embeddings = embeddings[jnp.newaxis, ...]  # (B, H, W, 2)

    # Shift right by 2 pixels (normalized: 2/(W-1))
    shift_pixels = 2
    shift_normalized = shift_pixels / (W - 1)
    flow = jnp.broadcast_to(jnp.array([[[[shift_normalized, 0.0]]]]), (B, H, W, 2))

    warped = warp_embeddings(embeddings, flow)

    # After warping, position x should have value from position x-shift
    # So warped[..., x, :] should equal embeddings[..., x-shift, :]
    # For x >= shift, check equality
    for x_idx in range(shift_pixels, W):
        expected_val = (x_idx - shift_pixels) / (W - 1)
        actual_val = warped[0, 0, x_idx, 0]
        np.testing.assert_allclose(actual_val, expected_val, rtol=0.15)

    print(f"✓ warp_embeddings shift: {shift_pixels} pixel shift verified")


def test_reconstruction_loss_zero():
    """Identical frames with identity flow → zero loss."""
    B, H, W, D = 1, 16, 16, 8

    rng = jax.random.PRNGKey(0)
    embeddings = jax.random.normal(rng, (B, H, W, D))

    # Warped == target → zero loss
    loss = reconstruction_loss_core(embeddings, embeddings)

    np.testing.assert_allclose(loss, 0.0, atol=1e-7)
    print(f"✓ reconstruction_loss_core: identical inputs → zero loss ({loss:.2e})")


def test_reconstruction_loss_nonzero():
    """Different frames → non-zero loss."""
    B, H, W, D = 1, 16, 16, 8

    rng1 = jax.random.PRNGKey(0)
    rng2 = jax.random.PRNGKey(1)
    warped = jax.random.normal(rng1, (B, H, W, D))
    target = jax.random.normal(rng2, (B, H, W, D))

    loss = reconstruction_loss_core(warped, target)

    assert loss > 0, f"Expected positive loss, got {loss}"
    print(f"✓ reconstruction_loss_core: different inputs → positive loss ({loss:.4f})")


def test_flow_to_dense():
    """Flow reshaping from token to spatial format."""
    B, H, W = 2, 16, 16
    N = H * W

    flow_tokens = jnp.ones((B, N, 2))
    flow_dense = flow_to_dense(flow_tokens, H, W)

    assert flow_dense.shape == (
        B,
        H,
        W,
        2,
    ), f"Expected (B, H, W, 2), got {flow_dense.shape}"
    print(f"✓ flow_to_dense shape: {flow_dense.shape}")


def test_create_source_position_grid():
    """Source position grid has correct shape and range."""
    H = W = 16
    N = H * W

    src_pos = create_source_position_grid(window_size=16)

    assert src_pos.shape == (N, 2), f"Expected (N, 2), got {src_pos.shape}"

    # Check range [0, 1]
    assert src_pos.min() >= 0.0, f"Min value {src_pos.min()} < 0"
    assert src_pos.max() <= 1.0, f"Max value {src_pos.max()} > 1"

    # Check corners
    # Top-left (0, 0) should be (0, 0)
    np.testing.assert_allclose(src_pos[0], [0.0, 0.0], atol=1e-6)

    # Bottom-right (15, 15) should be (1, 1)
    np.testing.assert_allclose(src_pos[-1], [1.0, 1.0], atol=1e-6)

    print(
        f"✓ create_source_position_grid: shape {src_pos.shape}, range [{src_pos.min():.2f}, {src_pos.max():.2f}]"
    )


if __name__ == "__main__":
    print("Running flow component tests...\n")

    test_attention_features_shape()
    test_attention_features_peak()
    test_attention_features_translation()
    test_flow_estimator_shape()
    test_flow_estimator_parameters()
    test_flow_estimator_bounded()
    test_warp_embeddings_identity()
    test_warp_embeddings_shift()
    test_reconstruction_loss_zero()
    test_reconstruction_loss_nonzero()
    test_flow_to_dense()
    test_create_source_position_grid()

    print("\n✓ All flow component tests passed!")
