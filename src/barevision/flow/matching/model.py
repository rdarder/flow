"""Flow estimation from attention centroids.

Predicts optical flow from self/cross attention centroid positions.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class AttentionCentroids(nnx.Module):
    """Computes centroids from attention weight maps.

    Input: self_attn (B, N, N), cross_attn (B, N, N) where N = H*W
    Output: (self_cx, self_cy, cross_cx, cross_cy) all normalized [0,1]

    The centroid is computed as the center-of-mass of attention weights.
    """

    def __init__(self, window_size: int):
        """Initialize centroid computer.

        Args:
            window_size: Size of attention window
        """
        self.window_size = window_size

        # Pre-compute normalized coordinate grid
        y, x = jnp.meshgrid(
            jnp.arange(window_size, dtype=jnp.float32),
            jnp.arange(window_size, dtype=jnp.float32),
            indexing="ij",
        )

        # Normalize to [0, 1]
        self.norm_coords = jnp.stack(
            [
                x / (window_size - 1),  # cx normalized
                y / (window_size - 1),  # cy normalized
            ],
            axis=-1,
        )  # (H, W, 2)

    def __call__(self, self_attn: jnp.ndarray, cross_attn: jnp.ndarray) -> jnp.ndarray:
        """Compute centroids for self and cross attention maps.

        Args:
            self_attn: (B, N, N) self-attention weights
            cross_attn: (B, N, N) cross-attention weights

        Returns:
            centroids: (B, N, 4) = [self_cx, self_cy, cross_cx, cross_cy] per query pixel
        """
        B, N, _ = self_attn.shape
        H = W = self.window_size

        # Reshape attention maps to spatial format
        self_attn_spatial = self_attn.reshape(B, N, H, W)
        cross_attn_spatial = cross_attn.reshape(B, N, H, W)

        # Compute centroid for each query pixel
        self_cx = jnp.sum(self_attn_spatial * self.norm_coords[..., 0], axis=(2, 3))
        self_cy = jnp.sum(self_attn_spatial * self.norm_coords[..., 1], axis=(2, 3))
        cross_cx = jnp.sum(cross_attn_spatial * self.norm_coords[..., 0], axis=(2, 3))
        cross_cy = jnp.sum(cross_attn_spatial * self.norm_coords[..., 1], axis=(2, 3))

        # Stack: (B, N, 4)
        centroids = jnp.stack([self_cx, self_cy, cross_cx, cross_cy], axis=-1)

        return centroids


class FlowEstimator(nnx.Module):
    """Predicts residual flow from attention centroids.

    Input: 6 floats per pixel (src_x, src_y, self_cx, self_cy, cross_cx, cross_cy)
    Output: 2 floats per pixel (residual_u, residual_v)

    All coordinates are normalized to [0, 1] range.
    """

    def __init__(
        self,
        window_size: int,
        hidden_dim: int,
        max_flow: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize flow estimator.

        Args:
            window_size: Size of attention window
            hidden_dim: Hidden layer dimension
            max_flow: Maximum flow magnitude in normalized coordinates (0.5 = half window)
            rngs: NNX RNGs for parameter initialization
        """
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.max_flow = max_flow

        # First layer with standard initialization
        self.linear1 = nnx.Linear(6, hidden_dim, rngs=rngs)

        # Second layer: initialize with small weights to start with near-zero flow predictions
        # This is a common pattern in residual networks - start conservative and learn the signal
        # Using normal(0.02) keeps initial outputs small while maintaining gradient flow
        # Zero bias ensures no systematic direction bias at initialization
        # The tanh activation further bounds output to [-1, 1], scaled by max_flow
        self.linear2 = nnx.Linear(
            hidden_dim,
            2,
            kernel_init=nnx.initializers.normal(0.02),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, src_pos: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
        """Predict flow from source positions and centroids.

        Args:
            src_pos: (B, N, 2) normalized source coordinates [x, y]
            centroids: (B, N, 4) from AttentionCentroids

        Returns:
            flow: (B, N, 2) predicted flow [u, v] in normalized coordinates, bounded to [-max_flow, max_flow]
        """
        # Concatenate inputs: (B, N, 6)
        x = jnp.concatenate([src_pos, centroids], axis=-1)

        # First layer with ReLU
        x = self.linear1(x)
        x = nnx.relu(x)

        # Second layer with tanh activation and scaling
        flow_raw = self.linear2(x)
        flow = jnp.tanh(flow_raw) * self.max_flow

        return flow


def create_source_position_grid(window_size: int) -> jnp.ndarray:
    """Create normalized source position grid.

    Args:
        window_size: Size of window

    Returns:
        src_pos: (N, 2) normalized coordinates [x, y] for each pixel
    """
    H = W = window_size
    N = H * W

    y, x = jnp.meshgrid(
        jnp.arange(H, dtype=jnp.float32),
        jnp.arange(W, dtype=jnp.float32),
        indexing="ij",
    )

    # Flatten and normalize to [0, 1]
    x_flat = x.ravel() / (W - 1)
    y_flat = y.ravel() / (H - 1)

    # Stack: (N, 2)
    src_pos = jnp.stack([x_flat, y_flat], axis=-1)

    return src_pos


def flow_to_dense(flow: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """Reshape flow from token format to spatial grid.

    Args:
        flow: (B, N, 2) flow in token format
        H: Target height
        W: Target width

    Returns:
        flow_dense: (B, H, W, 2) flow as spatial grid
    """
    B, N, _ = flow.shape
    assert N == H * W, f"Flow has {N} tokens but grid is {H}x{W}"
    return flow.reshape(B, H, W, 2)
