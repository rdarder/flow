"""Simple Embedding Model v0 for Barevision.

Baseline architecture: depthwise separable convolutions for efficient
local feature extraction optimized for patch matching.

Architecture:
    Input: (B, H, W, 3) RGB
      ↓
    3×3 depthwise conv: 3 in → 12 out (4 filters per channel)
      ↓
    ReLU
      ↓
    1×1 conv: 12 in → 16 out
      ↓
    Output: (B, H-2, W-2, 16) embeddings

Note: Uses valid convolutions (no padding), so output is 2 pixels smaller
than input on each dimension.

Parameters: ~326 total
    - Depthwise: 3 channels × 4 filters × 9 weights = 108
    - 1×1 conv: 12 in × 16 out + 16 bias = 208
    - Total: 316 + 10 (approx, depends on implementation)
"""

from flax import nnx
import jax.numpy as jnp


class SimpleEmbeddingModel(nnx.Module):
    """Baseline embedding model using depthwise separable convolutions.

    Produces 16-dimensional embeddings optimized for attention-based matching.
    Uses valid convolutions only (no padding) to avoid border artifacts.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        in_channels: int = 3,
        depthwise_out_channels: int = 12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the embedding model.

        Args:
            embed_dim: Output embedding dimension (default 16)
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            depthwise_out_channels: Number of output channels from depthwise conv
                (default 12 = 4 filters per input channel for RGB)
            rngs: NNX RNGs for parameter initialization
        """
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.depthwise_out_channels = depthwise_out_channels

        # 3×3 depthwise convolution
        # Each input channel gets depthwise_out_channels / in_channels filters
        self.depthwise_conv = nnx.Conv(
            in_features=in_channels,
            out_features=depthwise_out_channels,
            kernel_size=(3, 3),
            padding="VALID",  # No padding - valid convolution only
            feature_group_count=in_channels,  # Depthwise: one filter set per input channel
            rngs=rngs,
        )

        # 1×1 pointwise convolution to mix channels
        self.pointwise_conv = nnx.Conv(
            in_features=depthwise_out_channels,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            Embeddings of shape (B, H-2, W-2, embed_dim)
        """
        # Depthwise convolution + ReLU
        x = self.depthwise_conv(x)
        x = nnx.relu(x)

        # Pointwise convolution
        x = self.pointwise_conv(x)

        return x


def count_parameters(model: nnx.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: NNX module

    Returns:
        Total number of trainable parameters
    """
    state = nnx.state(model)
    total = 0
    for module_state in state.values():
        for param_value in module_state.values():
            if hasattr(param_value, "size"):
                total += param_value.size
    return total
