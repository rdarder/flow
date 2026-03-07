"""Simple Embedding Model v0 for Barevision.

Baseline architecture: depthwise separable convolutions for efficient
local feature extraction optimized for patch matching.

Architecture:
    Input: (B, H, W, 3) RGB
      ↓
    5×5 depthwise conv: 3 in → 48 out (16 filters per channel)
      ↓
    LeakyReLU (negative_slope=0.1) - prevents dying ReLU
      ↓
    1×1 conv: 48 in → 16 out
      ↓
    Output: (B, H-4, W-4, 16) embeddings, L2-normalized

Note: Uses valid convolutions (no padding), so output is 4 pixels smaller
than input on each dimension (vs 2 pixels with 3×3 kernel).

Parameters: ~2,032 total
    - Depthwise: 3 channels × 16 filters × 25 weights = 1,200
    - 1×1 conv: 48 in × 16 out + 16 bias = 784
    - Total: 1,984 + bias terms
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from barevision.utils.grid import WindowGrid


class SimpleEmbeddingModel(nnx.Module):
    """Baseline embedding model using depthwise separable convolutions.

    Produces 16-dimensional embeddings optimized for attention-based matching.
    Uses valid convolutions only (no padding) to avoid border artifacts.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        in_channels: int = 3,
        depthwise_out_channels: int = 48,
        kernel_size: int = 5,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the embedding model.

        Args:
            embed_dim: Output embedding dimension (default 16)
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            depthwise_out_channels: Number of output channels from depthwise conv
                (default 48 = 16 filters per input channel for RGB)
            kernel_size: Size of depthwise convolution kernel (default 5 for larger receptive field)
            rngs: NNX RNGs for parameter initialization
        """
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.depthwise_out_channels = depthwise_out_channels
        self.kernel_size = kernel_size

        # 5×5 depthwise convolution (larger receptive field)
        # Each input channel gets depthwise_out_channels / in_channels filters
        self.depthwise_conv = nnx.Conv(
            in_features=in_channels,
            out_features=depthwise_out_channels,
            kernel_size=(kernel_size, kernel_size),
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
            Embeddings of shape (B, H-4, W-4, embed_dim), L2-normalized to unit norm
        """
        # Depthwise convolution + LeakyReLU (prevents dying ReLU problem)
        x = self.depthwise_conv(x)
        x = nnx.leaky_relu(x, negative_slope=0.1)

        # Pointwise convolution
        x = self.pointwise_conv(x)
        
        # L2 normalize embeddings to unit norm (per-pixel)
        # This ensures self-attention peak is at self (q·q = ||q||² = 1 for all pixels)
        # Without this, high-norm embeddings dominate attention regardless of location
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norm + 1e-8)

        return x

    def compute_attention_maps(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        window_indices: Tuple[int, int],
        window_size: int = 16,
        pixel_indices: Optional[jnp.ndarray] = None,
    ) -> AttentionMaps:
        """Compute attention maps for visualization (not used in training loss).

        This method is called separately from training for diagnostic visualization.
        It returns detailed attention information that would be wasteful to compute
        during training.

        Args:
            img1: Frame 1 (1, H, W, 3) - batch size must be 1
            img2: Frame 2 (1, H, W, 3) - batch size must be 1
            window_indices: (row, col) of window to analyze within the grid
            window_size: Attention window size (default 16)
            pixel_indices: List of pixel indices within window to show attention maps for.
                          If None, picks 4 random pixels using a deterministic seed.

        Returns:
            AttentionMaps dataclass containing all visualization data

        Raises:
            ValueError: If batch size != 1 or window indices out of bounds
        """
        # Validate batch size
        if img1.shape[0] != 1 or img2.shape[0] != 1:
            raise ValueError(
                f"Batch size must be 1 for visualization, got {img1.shape[0]}"
            )

        # Compute embeddings
        emb1 = self(img1)[0]  # Remove batch dimension: (H-2, W-2, 16)
        emb2 = self(img2)[0]

        # Validate window indices
        H_emb, W_emb, _ = emb1.shape
        num_windows_h = H_emb // window_size
        num_windows_w = W_emb // window_size
        row, col = window_indices

        if row < 0 or row >= num_windows_h or col < 0 or col >= num_windows_w:
            raise ValueError(
                f"Window indices ({row}, {col}) out of bounds. "
                f"Valid range: row [0, {num_windows_h}), col [0, {num_windows_w})"
            )

        # Extract window from embeddings
        emb_h_start = row * window_size
        emb_h_end = emb_h_start + window_size
        emb_w_start = col * window_size
        emb_w_end = emb_w_start + window_size

        window_emb1 = emb1[
            emb_h_start:emb_h_end, emb_w_start:emb_w_end, :
        ]  # (16, 16, 16)
        window_emb2 = emb2[emb_h_start:emb_h_end, emb_w_start:emb_w_end, :]

        # Extract image crop for visualization
        # Account for 2-pixel border from valid convolutions
        img_h_start = emb_h_start
        img_h_end = img_h_start + window_size
        img_w_start = emb_w_start
        img_w_end = img_w_start + window_size
        window_crop = img1[
            0, img_h_start:img_h_end, img_w_start:img_w_end, :
        ]  # (16, 16, 3)

        # Flatten windows for attention computation
        flat_emb1 = window_emb1.reshape(window_size * window_size, -1)  # (256, 16)
        flat_emb2 = window_emb2.reshape(window_size * window_size, -1)

        # Select pixel indices
        N = window_size * window_size
        if pixel_indices is None:
            # Use deterministic random selection based on window position
            seed = row * 1000 + col
            key = jax.random.PRNGKey(seed)
            pixel_indices = jax.random.choice(key, N, shape=(4,), replace=False)

        # Ensure pixel_indices is a jnp array
        pixel_indices = jnp.asarray(pixel_indices)

        # Compute self-attention logits (no masking, no penalty - embeddings are normalized)
        self_logits = flat_emb1 @ flat_emb1.T  # (256, 256)

        # Compute self-attention weights
        self_attn_weights = jax.nn.softmax(self_logits, axis=-1)  # (256, 256)

        # Compute cross-attention logits
        cross_logits = flat_emb1 @ flat_emb2.T  # (256, 256)
        cross_attn_weights = jax.nn.softmax(cross_logits, axis=-1)  # (256, 256)

        # Extract attention maps for selected pixels
        self_attn_maps = self_attn_weights[pixel_indices].reshape(
            -1, window_size, window_size
        )  # (N_sel, 16, 16)
        cross_attn_maps = cross_attn_weights[pixel_indices].reshape(
            -1, window_size, window_size
        )

        # Compute per-pixel entropy maps
        self_entropy = _compute_entropy(self_attn_weights).reshape(
            window_size, window_size
        )  # (16, 16)
        cross_entropy = _compute_entropy(cross_attn_weights).reshape(
            window_size, window_size
        )

        # Compute pixel positions (y, x) for each selected index
        pixel_y = pixel_indices // window_size
        pixel_x = pixel_indices % window_size
        pixel_positions = jnp.stack([pixel_y, pixel_x], axis=-1)  # (N_sel, 2)

        return AttentionMaps(
            embeddings1=emb1,
            embeddings2=emb2,
            self_attention=self_attn_maps,
            cross_attention=cross_attn_maps,
            self_entropy=self_entropy,
            cross_entropy=cross_entropy,
            window_crop=window_crop,
            pixel_positions=pixel_positions,
        )

        # Compute embeddings
        emb1 = self(img1)[0]  # Remove batch dimension: (H-2, W-2, 16)
        emb2 = self(img2)[0]

        # Validate window indices
        H_emb, W_emb, _ = emb1.shape
        num_windows_h = H_emb // window_size
        num_windows_w = W_emb // window_size
        row, col = window_indices

        if row < 0 or row >= num_windows_h or col < 0 or col >= num_windows_w:
            raise ValueError(
                f"Window indices ({row}, {col}) out of bounds. "
                f"Valid range: row [0, {num_windows_h}), col [0, {num_windows_w})"
            )

        # Extract window from embeddings
        emb_h_start = row * window_size
        emb_h_end = emb_h_start + window_size
        emb_w_start = col * window_size
        emb_w_end = emb_w_start + window_size

        window_emb1 = emb1[
            emb_h_start:emb_h_end, emb_w_start:emb_w_end, :
        ]  # (16, 16, 16)
        window_emb2 = emb2[emb_h_start:emb_h_end, emb_w_start:emb_w_end, :]

        # Extract image crop for visualization
        # Account for 2-pixel border from valid convolutions
        img_h_start = emb_h_start
        img_h_end = img_h_start + window_size
        img_w_start = emb_w_start
        img_w_end = img_w_start + window_size
        window_crop = img1[
            0, img_h_start:img_h_end, img_w_start:img_w_end, :
        ]  # (16, 16, 3)

        # Flatten windows for attention computation
        flat_emb1 = window_emb1.reshape(window_size * window_size, -1)  # (256, 16)
        flat_emb2 = window_emb2.reshape(window_size * window_size, -1)

        # Select pixel indices
        N = window_size * window_size
        if pixel_indices is None:
            # Use deterministic random selection based on window position
            seed = row * 1000 + col
            key = jax.random.PRNGKey(seed)
            pixel_indices = jax.random.choice(key, N, shape=(4,), replace=False)

        # Ensure pixel_indices is a jnp array
        pixel_indices = jnp.asarray(pixel_indices)

        # Compute self-attention logits
        self_logits = flat_emb1 @ flat_emb1.T  # (256, 256)
        spatial_matrix = _spatial_logits_matrix(window_size, spatial_scale)
        self_logits = self_logits + spatial_matrix

        # Compute self-attention weights
        self_attn_weights = jax.nn.softmax(self_logits, axis=-1)  # (256, 256)

        # Compute cross-attention logits
        cross_logits = flat_emb1 @ flat_emb2.T  # (256, 256)
        cross_attn_weights = jax.nn.softmax(cross_logits, axis=-1)  # (256, 256)

        # Extract attention maps for selected pixels
        self_attn_maps = self_attn_weights[pixel_indices].reshape(
            -1, window_size, window_size
        )  # (N_sel, 16, 16)
        cross_attn_maps = cross_attn_weights[pixel_indices].reshape(
            -1, window_size, window_size
        )

        # Compute per-pixel entropy maps
        self_entropy = _compute_entropy(self_attn_weights).reshape(
            window_size, window_size
        )  # (16, 16)
        cross_entropy = _compute_entropy(cross_attn_weights).reshape(
            window_size, window_size
        )

        # Compute pixel positions (y, x) for each selected index
        pixel_y = pixel_indices // window_size
        pixel_x = pixel_indices % window_size
        pixel_positions = jnp.stack([pixel_y, pixel_x], axis=-1)  # (N_sel, 2)

        return AttentionMaps(
            embeddings1=emb1,
            embeddings2=emb2,
            self_attention=self_attn_maps,
            cross_attention=cross_attn_maps,
            self_entropy=self_entropy,
            cross_entropy=cross_entropy,
            window_crop=window_crop,
            pixel_positions=pixel_positions,
        )


@dataclass
class AttentionMaps:
    """Container for attention map data used in visualization.

    Returned by SimpleEmbeddingModel.compute_attention_maps().
    Not used in training loss - only for diagnostic visualization.

    Attributes:
        embeddings1: (H-2, W-2, 16) embeddings for frame 1
        embeddings2: (H-2, W-2, 16) embeddings for frame 2
        self_attention: (N, 16, 16) self-attention weights for N query pixels
        cross_attention: (N, 16, 16) cross-attention weights for N query pixels
        self_entropy: (16, 16) per-pixel self-attention entropy within window
        cross_entropy: (16, 16) per-pixel cross-attention entropy within window
        window_crop: (16, 16, 3) image crop for the analyzed window
        pixel_positions: (N, 2) (y, x) positions of queried pixels within window
    """

    embeddings1: jnp.ndarray
    embeddings2: jnp.ndarray
    self_attention: jnp.ndarray
    cross_attention: jnp.ndarray
    self_entropy: jnp.ndarray
    cross_entropy: jnp.ndarray
    window_crop: jnp.ndarray
    pixel_positions: jnp.ndarray


def _compute_entropy(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution.

    Args:
        probabilities: (..., N) array where last dimension sums to 1

    Returns:
        Entropy values with shape (...)
    """
    eps = 1e-10
    return -jnp.sum(probabilities * jnp.log(probabilities + eps), axis=-1)


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
