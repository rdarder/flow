"""Flow estimation from attention features.

Predicts optical flow from attention centroids, positions, and confidence features.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class AttentionFeatures(nnx.Module):
    """Computes spatial and confidence features from attention weight maps.

    Input: self_attn (B, N, N), cross_attn (B, N, N) where N = H*W
    Output: 8 features per pixel:
        - self_relative (2): self-centroid offset from source (should be ~0)
        - cross_relative (2): cross-centroid offset from source (flow vector)
        - cross_absolute (2): cross-centroid absolute position [0, 1]
        - self_max_peak (1): max self-attention weight (confidence)
        - cross_max_peak (1): max cross-attention weight (confidence)

    The centroid is computed as the center-of-mass of attention weights.
    """

    def __init__(self, window_size: int):
        """Initialize feature computer.

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

    def __call__(
        self, self_attn: jnp.ndarray, cross_attn: jnp.ndarray, src_pos: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute features for self and cross attention maps.

        Args:
            self_attn: (B, N, N) self-attention weights
            cross_attn: (B, N, N) cross-attention weights
            src_pos: (B, N, 2) normalized source coordinates [x, y]

        Returns:
            features: (B, N, 8) feature vector per query pixel
        """
        B, N, _ = self_attn.shape
        H = W = self.window_size

        # Reshape attention maps to spatial format
        self_attn_spatial = self_attn.reshape(B, N, H, W)
        cross_attn_spatial = cross_attn.reshape(B, N, H, W)

        # Compute absolute centroids for self and cross attention
        self_cx_abs = jnp.sum(self_attn_spatial * self.norm_coords[..., 0], axis=(2, 3))
        self_cy_abs = jnp.sum(self_attn_spatial * self.norm_coords[..., 1], axis=(2, 3))
        cross_cx_abs = jnp.sum(
            cross_attn_spatial * self.norm_coords[..., 0], axis=(2, 3)
        )
        cross_cy_abs = jnp.sum(
            cross_attn_spatial * self.norm_coords[..., 1], axis=(2, 3)
        )

        # Compute relative centroids (offset from source position)
        self_cx_rel = self_cx_abs - src_pos[..., 0]
        self_cy_rel = self_cy_abs - src_pos[..., 1]
        cross_cx_rel = cross_cx_abs - src_pos[..., 0]
        cross_cy_rel = cross_cy_abs - src_pos[..., 1]

        # Compute max peak values (confidence features)
        self_max = jnp.max(self_attn, axis=-1)
        cross_max = jnp.max(cross_attn, axis=-1)

        # Stack all features: (B, N, 8)
        features = jnp.stack(
            [
                self_cx_rel,
                self_cy_rel,  # self_relative (2)
                cross_cx_rel,
                cross_cy_rel,  # cross_relative (2)
                cross_cx_abs,
                cross_cy_abs,  # cross_absolute (2)
                self_max,
                cross_max,  # confidence (2)
            ],
            axis=-1,
        )

        return features


class FlowEstimator(nnx.Module):
    """Predicts residual flow from attention features.

    Input: 8 floats per pixel:
        - self_relative (2): self-centroid offset from source
        - cross_relative (2): cross-centroid offset from source (flow vector)
        - cross_absolute (2): cross-centroid absolute position [0, 1]
        - self_max_peak (1): max self-attention weight
        - cross_max_peak (1): max cross-attention weight

    Output: 2 floats per pixel (residual_u, residual_v)

    All coordinates are normalized to [0, 1] range, relative features are 0-mean.
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

        # Three hidden layers for better capacity with richer features
        self.linear1 = nnx.Linear(8, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        # Output layer: initialize with small weights to start with near-zero flow predictions
        # This is a common pattern in residual networks - start conservative and learn the signal
        # Using normal(0.02) keeps initial outputs small while maintaining gradient flow
        # Zero bias ensures no systematic direction bias at initialization
        # The tanh activation further bounds output to [-1, 1], scaled by max_flow
        self.linear_out = nnx.Linear(
            hidden_dim,
            2,
            kernel_init=nnx.initializers.normal(0.02),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Predict flow from attention features.

        Args:
            features: (B, N, 8) from AttentionFeatures

        Returns:
            flow: (B, N, 2) predicted flow [u, v] in normalized coordinates, bounded to [-max_flow, max_flow]
        """
        # Three hidden layers with ReLU activation
        x = self.linear1(features)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.linear3(x)
        x = nnx.relu(x)

        # Output layer with tanh activation and scaling
        flow_raw = self.linear_out(x)
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
