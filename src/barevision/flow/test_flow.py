"""Tests for flow estimation components."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barevision.flow.flow_estimator import (
    AttentionCentroids,
    FlowEstimator,
    create_source_position_grid,
    flow_to_dense,
)
from barevision.flow.reconstruction_loss import (
    reconstruction_loss_core,
    warp_embeddings,
)


def test_attention_centroids_shape():
    """Centroids output has correct shape."""
    B, N = 2, 256  # 16x16 window

    # Create dummy attention maps (uniform for simplicity)
    self_attn = jnp.ones((B, N, N)) / N
    cross_attn = jnp.ones((B, N, N)) / N

    centroids_computer = AttentionCentroids(window_size=16, rngs=nnx.Rngs(0))
    centroids = centroids_computer(self_attn, cross_attn)

    assert centroids.shape == (B, N, 4), f"Expected (B, N, 4), got {centroids.shape}"
    print(f"✓ AttentionCentroids shape: {centroids.shape}")


def test_attention_centroids_peak():
    """Centroid of delta-function attention is at peak location."""
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

    centroids_computer = AttentionCentroids(window_size=16, rngs=nnx.Rngs(0))
    centroids = centroids_computer(self_attn, cross_attn)

    # Expected normalized coordinates
    expected_cx = peak_x / (W - 1)
    expected_cy = peak_y / (H - 1)

    # Check self-attention centroid
    self_cx = centroids[0, 0, 0]
    self_cy = centroids[0, 0, 1]

    np.testing.assert_allclose(self_cx, expected_cx, rtol=1e-5)
    np.testing.assert_allclose(self_cy, expected_cy, rtol=1e-5)

    # Check cross-attention centroid
    cross_cx = centroids[0, 0, 2]
    cross_cy = centroids[0, 0, 3]

    np.testing.assert_allclose(cross_cx, expected_cx, rtol=1e-5)
    np.testing.assert_allclose(cross_cy, expected_cy, rtol=1e-5)

    print(
        f"✓ AttentionCentroids peak: centroid at ({self_cx:.3f}, {self_cy:.3f}) matches peak at ({expected_cx:.3f}, {expected_cy:.3f})"
    )


def test_flow_estimator_shape():
    """Flow estimator produces (B, N, 2) output."""
    B, N = 2, 256

    # Create dummy inputs
    src_pos = jnp.zeros((B, N, 2))
    centroids = jnp.zeros((B, N, 4))

    flow_estimator = FlowEstimator(window_size=16, hidden_dim=24, rngs=nnx.Rngs(0))
    flow = flow_estimator(src_pos, centroids)

    assert flow.shape == (B, N, 2), f"Expected (B, N, 2), got {flow.shape}"
    print(f"✓ FlowEstimator shape: {flow.shape}")


def test_flow_estimator_parameters():
    """Flow estimator has expected parameter count."""
    flow_estimator = FlowEstimator(window_size=16, hidden_dim=24, rngs=nnx.Rngs(0))

    from barevision.flow.model import count_parameters

    params = count_parameters(flow_estimator)

    # Linear(6→24) = 6*24 + 24 = 168
    # Linear(24→2) = 24*2 + 2 = 50
    # Total = 218
    expected_params = 6 * 24 + 24 + 24 * 2 + 2
    assert params == expected_params, f"Expected {expected_params} params, got {params}"
    print(f"✓ FlowEstimator parameters: {params}")


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

    test_attention_centroids_shape()
    test_attention_centroids_peak()
    test_flow_estimator_shape()
    test_flow_estimator_parameters()
    test_warp_embeddings_identity()
    test_warp_embeddings_shift()
    test_reconstruction_loss_zero()
    test_reconstruction_loss_nonzero()
    test_flow_to_dense()
    test_create_source_position_grid()

    print("\n✓ All flow component tests passed!")
