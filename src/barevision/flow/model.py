"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Architecture (3 levels, stride=2 downsampling):
    Input: (B, H, W, 3) RGB
      ↓
    Level 0: GroupedConv(3×3, 3 groups) → 36 ch → Conv(1×1) → 16 channels
      ↓
    Level 1: GroupedConv(3×3, 8 groups) → 32 ch → Conv(1×1) → 16 channels
      ↓
    Level 2: GroupedConv(3×3, 8 groups) → 32 ch → Conv(1×1) → 16 channels

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Note: Uses VALID padding (no padding), so spatial dimensions shrink at each level.
      For 3 levels targeting 16×16 at coarsest level, input must be 135×135.

Grouped convolution design:
    - Level 0 (RGB→features): 3 groups, 3→36 channels (12 ch/group), then 36→16
    - Levels 1+ (features→features): 8 groups, 16→32 channels (4 ch/group), then 32→16
    - Groups increase channel mixing expressiveness while keeping params efficient

Parameters: ~2,648 total
    - Level 0: 3×3 grouped (3 groups, 12 out/group) = 3*12*9=324 + 1×1 (36*16+16=592) = 916
    - Levels 1+: 3×3 grouped (8 groups, 4 out/group) = 16*4*9=576 + 1×1 (32*16+16=528) = 1,104 each
    - Total: 916 + 1,104 + 1,104 = 3,124 (but actual may vary slightly)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from barevision.utils.grid import WindowGrid
from barevision.flow.loss import TEMPERATURE


class HierarchicalEmbeddingModel(nnx.Module):
    """Hierarchical embedding model with multi-scale feature pyramid.

    Produces 16-dimensional embeddings at multiple scales optimized for
    attention-based matching. Uses valid convolutions only (no padding).
    """

    def __init__(
        self,
        embed_dim: int = 16,
        in_channels: int = 3,
        num_levels: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the hierarchical embedding model.

        Args:
            embed_dim: Output embedding dimension per level (default 16)
            in_channels: Number of input channels (3 for RGB)
            num_levels: Number of pyramid levels (default 3)
            rngs: NNX RNGs for parameter initialization
        """
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_levels = num_levels

        # Build pyramid levels using nnx.List for proper module tracking
        self.spatial_convs = nnx.List()
        self.pointwise_convs = nnx.List()
        self.norms = nnx.List()

        for i in range(num_levels):
            level_in_ch = in_channels if i == 0 else embed_dim

            # Level 0 (RGB input): 3 groups, 3→36 channels
            # Levels 1+ (feature input): 8 groups, 16→32 channels
            if i == 0:
                # First level: RGB to features with 3 groups
                num_groups = 3
                intermediate_channels = 36  # 12 channels per group
            else:
                # Deeper levels: features to features with 8 groups
                num_groups = 8
                intermediate_channels = (
                    32  # 4 channels per group (16/8=2 in, 32/8=4 out)
                )

            # 3×3 spatial convolution with stride=2 (downsampling) with groups
            spatial_conv = nnx.Conv(
                in_features=level_in_ch,
                out_features=intermediate_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                feature_group_count=num_groups,
                rngs=rngs,
            )
            norm_layer = nnx.GroupNorm(
                num_groups=4, num_features=intermediate_channels, rngs=rngs
            )

            # 1×1 convolution for feature mixing to embed_dim
            pointwise_conv = nnx.Conv(
                in_features=intermediate_channels,
                out_features=embed_dim,
                kernel_size=(1, 1),
                padding="VALID",
                rngs=rngs,
            )

            self.spatial_convs.append(spatial_conv)
            self.pointwise_convs.append(pointwise_conv)
            self.norms.append(norm_layer)  # Add this list to __init__

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Forward pass through pyramid.

        Args:
            x: Input tensor of shape (B, H, W, in_channels)

        Returns:
            List of feature maps, one per level.
            Each level has shape (B, H_l, W_l, embed_dim)
            where H_l, W_l shrink by ~2x at each level.
        """
        feature_maps = []

        for i in range(self.num_levels):
            # Spatial downsampling convolution
            x = self.spatial_convs[i](x)
            x = self.norms[i](x)
            x = nnx.relu(x)

            # 1×1 projection
            x = self.pointwise_convs[i](x)

            # L2 normalize embeddings to unit norm (per-pixel)
            norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
            x = x / (norm + 1e-8)

            feature_maps.append(x)

        return feature_maps

    def compute_attention_maps(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        window_indices: Tuple[int, int],
        window_size: int = 16,
        level_index: int = -1,
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
            level_index: Which pyramid level to visualize (-1 for coarsest)
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

        # Compute pyramid and select level
        pyramid1 = self(img1)
        pyramid2 = self(img2)

        level_idx = level_index if level_index >= 0 else len(pyramid1) + level_index
        emb1 = pyramid1[level_idx][0]  # Remove batch dimension
        emb2 = pyramid2[level_idx][0]

        # Validate window indices
        H_emb, W_emb, _ = emb1.shape
        num_windows_h = H_emb // window_size
        num_windows_w = W_emb // window_size
        row, col = window_indices

        if row < 0 or row >= num_windows_h or col < 0 or col >= num_windows_w:
            raise ValueError(
                f"Window indices ({row}, {col}) out of bounds for level {level_idx}. "
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
        # Need to map from coarse level coordinates back to input image
        # Each level halves the spatial dimensions, so coarsest level is 1/8 of input
        scale_factor = 2 ** (level_idx + 1)
        img_h_start = emb_h_start * scale_factor
        img_h_end = img_h_start + window_size * scale_factor
        img_w_start = emb_w_start * scale_factor
        img_w_end = img_w_start + window_size * scale_factor

        window_crop = img1[
            0, img_h_start:img_h_end, img_w_start:img_w_end, :
        ]  # (16*scale, 16*scale, 3)

        # Resize crop to window_size for visualization consistency
        window_crop = jax.image.resize(
            window_crop, (window_size, window_size, 3), method="bilinear"
        )

        # Flatten windows for attention computation
        flat_emb1 = window_emb1.reshape(window_size * window_size, -1)  # (256, 16)
        flat_emb2 = window_emb2.reshape(window_size * window_size, -1)

        # Select pixel indices
        N = window_size * window_size
        if pixel_indices is None:
            # Use deterministic random selection based on window position
            seed = row * 1000 + col + level_idx * 10000
            key = jax.random.PRNGKey(seed)
            pixel_indices = jax.random.choice(key, N, shape=(4,), replace=False)

        # Ensure pixel_indices is a jnp array
        pixel_indices = jnp.asarray(pixel_indices)

        # Compute self-attention logits (no masking, no penalty - embeddings are normalized)
        self_logits = flat_emb1 @ flat_emb1.T  # (256, 256)

        # Compute self-attention weights with temperature scaling
        self_attn_weights = jax.nn.softmax(
            self_logits / TEMPERATURE, axis=-1
        )  # (256, 256)

        # Compute cross-attention logits
        cross_logits = flat_emb1 @ flat_emb2.T  # (256, 256)
        cross_attn_weights = jax.nn.softmax(
            cross_logits / TEMPERATURE, axis=-1
        )  # (256, 256)

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

    Returned by HierarchicalEmbeddingModel.compute_attention_maps().
    Not used in training loss - only for diagnostic visualization.

    Attributes:
        embeddings1: (H, W, 16) embeddings for frame 1 at selected level
        embeddings2: (H, W, 16) embeddings for frame 2 at selected level
        self_attention: (N, 16, 16) self-attention weights for N query pixels
        cross_attention: (N, 16, 16) cross-attention weights for N query pixels
        self_entropy: (16, 16) per-pixel self-attention entropy within window
        cross_entropy: (16, 16) per-pixel cross-attention entropy within window
        window_crop: (16, 16, 3) image crop for the analyzed window (resized)
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
    from flax.nnx import State

    state = nnx.state(model)
    total = 0

    def count_recursive(obj):
        nonlocal total
        if isinstance(obj, State):
            for value in obj.values():
                count_recursive(value)
        elif isinstance(obj, dict):
            for value in obj.values():
                count_recursive(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                count_recursive(item)
        elif hasattr(obj, "size"):
            total += obj.size

    count_recursive(state)
    return total


def calculate_required_input_size(
    target_coarse_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate required input image size to achieve target coarse dimension.

    Works backwards from the target coarse-level spatial dimension through
    the pyramid to determine the exact input size needed.

    For VALID padding with stride=2, each level transforms:
        output_size = (input_size - kernel_size) // stride + 1

    Reversed:
        input_size = (output_size - 1) * stride + kernel_size

    Args:
        target_coarse_dim: Target spatial dimension at coarsest level
                          (e.g., 48 for 3×3 grid of 16×16 windows)
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Required input image dimension (height and width are same)

    Example:
        For 3 levels targeting 48×48 at coarsest:
        Level 2 output: 48
        Level 1 output → Level 2 input: (48-1)*2 + 3 = 97
        Level 0 output → Level 1 input: (97-1)*2 + 3 = 195
        Raw input → Level 0 input: (195-1)*2 + 3 = 391
        Result: 391×391 input required
    """
    size = target_coarse_dim
    for _ in range(num_levels):
        size = (size - 1) * stride + kernel_size
    return size


def calculate_coarse_output_size(
    input_dim: int,
    num_levels: int,
    kernel_size: int = 3,
    stride: int = 2,
) -> int:
    """Calculate coarse-level output size for a given input.

    Forward calculation through the pyramid.

    For VALID padding:
        output_size = (input_size - kernel_size) // stride + 1

    Args:
        input_dim: Input image dimension
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Spatial dimension at coarsest level
    """
    size = input_dim
    for _ in range(num_levels):
        size = (size - kernel_size) // stride + 1
    return size
