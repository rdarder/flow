"""Diagnostic script to test FlowEstimator bias with identical frames.

Run: python -m barevision.flow.diagnose_flow_bias
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barevision.flow.matching.model import (
    FlowEstimator,
    AttentionCentroids,
    create_source_position_grid,
)


def test_identical_frames():
    """Test flow prediction when both frames are identical.

    Expected: Zero flow (or very close to zero).
    """
    print("=" * 60)
    print("Test: Identical Frames → Should Predict Zero Flow")
    print("=" * 60)

    # Setup
    window_size = 16
    H = W = window_size
    N = H * W
    B = 1

    # Create identical embeddings (simulating identical frames)
    rng = jax.random.PRNGKey(42)
    embeddings = jax.random.normal(rng, (B, H, W, 16))

    # Flatten for attention computation
    flat_emb = embeddings.reshape(B, N, 16)

    # Compute self and cross attention (identical → cross should equal self)
    temperature = 0.3
    self_logits = flat_emb @ flat_emb.transpose(0, 2, 1)
    cross_logits = flat_emb @ flat_emb.transpose(0, 2, 1)  # Same as self

    self_attn = jax.nn.softmax(self_logits / temperature, axis=-1)
    cross_attn = jax.nn.softmax(cross_logits / temperature, axis=-1)

    # Compute centroids
    centroids_computer = AttentionCentroids(window_size=window_size, rngs=nnx.Rngs(0))
    centroids = centroids_computer(self_attn, cross_attn)

    # Create source position grid
    src_pos = create_source_position_grid(window_size=window_size)
    src_pos = jnp.broadcast_to(src_pos, (B, N, 2))

    # Initialize flow estimator
    flow_estimator = FlowEstimator(
        window_size=window_size, hidden_dim=24, max_flow=0.5, rngs=nnx.Rngs(0)
    )

    # Predict flow
    flow = flow_estimator(src_pos, centroids)

    # Analyze results
    magnitude = jnp.linalg.norm(flow, axis=-1)

    print(f"\nFlow statistics (normalized window coordinates):")
    print(f"  Mean magnitude: {magnitude.mean():.6f}")
    print(f"  Max magnitude:  {magnitude.max():.6f}")
    print(f"  Min magnitude:  {magnitude.min():.6f}")
    print(f"  Std magnitude:  {magnitude.std():.6f}")

    print(f"\nFlow component statistics:")
    print(
        f"  U: mean={flow[..., 0].mean():.6f}, range=[{flow[..., 0].min():.6f}, {flow[..., 0].max():.6f}]"
    )
    print(
        f"  V: mean={flow[..., 1].mean():.6f}, range=[{flow[..., 1].min():.6f}, {flow[..., 1].max():.6f}]"
    )

    # Check for bias
    print(f"\nBias diagnostic:")
    if magnitude.mean() > 0.01:
        print(f"  ⚠️  WARNING: Mean flow {magnitude.mean():.4f} > 0.01")
        print(f"     This suggests the MLP has learned a systematic bias.")
        print(f"     For identical frames, expected flow ≈ 0.0")
    else:
        print(f"  ✓ Mean flow is near zero ({magnitude.mean():.6f})")

    # Check if centroids match source positions (they should for identical frames)
    # For identical frames: self_attn centroid should be at source position
    # cross_attn centroid should also be at source position
    self_cx = centroids[..., 0]
    self_cy = centroids[..., 1]
    cross_cx = centroids[..., 2]
    cross_cy = centroids[..., 3]

    src_x = src_pos[..., 0].reshape(H, W)
    src_y = src_pos[..., 1].reshape(H, W)

    cx_error = jnp.abs(self_cx.reshape(H, W) - src_x).mean()
    cy_error = jnp.abs(self_cy.reshape(H, W) - src_y).mean()

    print(f"\nCentroid accuracy (should match source for identical frames):")
    print(f"  Self-attention centroid error: {cx_error:.6f} (cx), {cy_error:.6f} (cy)")

    cross_cx_error = jnp.abs(cross_cx.reshape(H, W) - src_x).mean()
    cross_cy_error = jnp.abs(cross_cy.reshape(H, W) - src_y).mean()
    print(
        f"  Cross-attention centroid error: {cross_cx_error:.6f} (cx), {cross_cy_error:.6f} (cy)"
    )

    if cx_error > 0.01 or cy_error > 0.01:
        print(f"\n  Note: Centroids don't perfectly match source positions.")
        print(
            f"        This is expected - attention is distributed, not a delta function."
        )
        print(
            f"        The FlowEstimator should learn to output zero flow when centroids ≈ source."
        )

    print("\n" + "=" * 60)


def test_random_initialization():
    """Test flow with randomly initialized weights.

    Expected: Flow centered around zero with some variance.
    """
    print("\n" + "=" * 60)
    print("Test: Random Initialization → Flow Distribution")
    print("=" * 60)

    window_size = 16
    N = window_size * window_size
    B = 1

    # Random inputs
    rng = jax.random.PRNGKey(42)
    src_pos = jax.random.uniform(rng, (B, N, 2)) * 0.5 + 0.25  # Centered at 0.5
    centroids = jax.random.uniform(rng, (B, N, 4)) * 0.5 + 0.25  # Centered at 0.5

    # Initialize flow estimator
    flow_estimator = FlowEstimator(
        window_size=window_size,
        hidden_dim=24,
        max_flow=0.5,
        rngs=nnx.Rngs(42),  # Different seed
    )

    # Predict flow
    flow = flow_estimator(src_pos, centroids)
    magnitude = jnp.linalg.norm(flow, axis=-1)

    print(f"\nFlow statistics (random weights):")
    print(f"  Mean magnitude: {magnitude.mean():.6f}")
    print(f"  Max magnitude:  {magnitude.max():.6f}")
    print(f"  Min magnitude:  {magnitude.min():.6f}")
    print(f"  Bounded to: [-0.5, 0.5] (tanh * max_flow)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_identical_frames()
    test_random_initialization()
    print("\nDiagnostic complete!")
