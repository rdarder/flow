"""Hierarchical Embedding Model for Barevision.

Multi-scale feature pyramid for coarse-to-fine patch matching.

Architecture (Decoupled Cascade, 3 levels, stride=2 downsampling):
    Input: (B, H, W, 3) RGB
      ↓
    StemBlock (Level 0):
      - Conv1: 3×3, stride=1, dense (3→32 ch) → GroupNorm → GELU
      - Conv2: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B (Downsample): 3×3, stride=2, grouped (32→32 ch)
      ↓
    StandardBlock (Level 1):
      - Conv1: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B (Downsample): 3×3, stride=2, grouped (32→32 ch)
      ↓
    StandardBlock (Level 2, last):
      - Conv1: 3×3, stride=1, grouped (8 groups, 32→32 ch) → GroupNorm → GELU
      - Branch A (Embed): 1×1 Conv (32→16 ch) → L2 Norm
      - Branch B: None (last level)

Output: List of feature maps [Level_0, Level_1, Level_2]
        Each level has 16 channels, spatial dimensions halve at each level.

Note: Uses VALID padding (no padding), so spatial dimensions shrink at each level.
      Receptive field expands through stacked stride=1 convolutions before downsampling.
      Stem Block: 25 pixel RF (two 3×3 stacked)
      Standard Block: Inherits RF from previous levels + 2 pixel expansion

Decoupled Cascade Design:
    - Feature extraction (stride=1) builds deep structural understanding
    - Downsampling (stride=2) is separate from feature extraction
    - Receptive field expands before projection to embedding space
    - GELU activation preserves more gradient flow than ReLU
    - No activation on embeddings before L2 normalization

Parameters: ~4,560 total
    - Stem Block: Dense 3×3 (3→32) = 3*32*9=864 + Grouped 3×3 (8g, 32→32) = 32*4*9=1152
                  + 1×1 (32→16) = 528 + Downsample 3×3 (8g, 32→32) = 1152 = 3,696
    - Standard Block: Grouped 3×3 (8g, 32→32) = 1152 + 1×1 (32→16) = 528 + Downsample = 1152 = 2,832
    - Last Standard Block: 1152 + 528 = 1,680 (no downsample)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

# Global constants for blocks
HIDDEN_CHANNELS = 32
EMBED_DIM = 16
NUM_GROUPS = 8  # For grouped convolutions (32/8 = 4 ch/group)


class StemBlock(nnx.Module):
    """Root block of the pyramid (Level 0 only).

    Uses two stacked 3×3 convolutions to expand receptive field from 9 to 25 pixels
    before splitting into embedding and downsampling branches.

    Input: (B, H, W, 3) RGB
    Returns: (embedding, downsampled_output)
        - embedding: (B, H-4, W-4, 16) L2-normalized
        - downsampled_output: (B, (H-7)//2, (W-7)//2, 32)
    """

    def __init__(self, embed_dim: int = EMBED_DIM, *, rngs: nnx.Rngs):
        """Initialize StemBlock.

        Args:
            embed_dim: Output embedding dimension (default 16)
            rngs: NNX RNGs for parameter initialization
        """
        self.embed_dim = embed_dim

        # Conv1: Dense 3×3, stride=1, RGB (3 ch) → 32 ch
        # No feature groups since 3 input channels isn't divisible by 8
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=HIDDEN_CHANNELS,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=1,  # Dense convolution
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=4, num_features=HIDDEN_CHANNELS, rngs=rngs
        )

        # Conv2: Grouped 3×3, stride=1, 32 ch → 32 ch
        # 8 groups, 4 ch/group input, 4 ch/group output
        self.conv2 = nnx.Conv(
            in_features=HIDDEN_CHANNELS,
            out_features=HIDDEN_CHANNELS,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=NUM_GROUPS,
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=4, num_features=HIDDEN_CHANNELS, rngs=rngs
        )

        # Branch A: 1×1 Conv for embedding projection (32 → embed_dim)
        self.embed_conv = nnx.Conv(
            in_features=HIDDEN_CHANNELS,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

        # Branch B: 3×3 Conv for downsampling (32 → 32)
        self.downsample_conv = nnx.Conv(
            in_features=HIDDEN_CHANNELS,
            out_features=HIDDEN_CHANNELS,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            feature_group_count=NUM_GROUPS,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through StemBlock.

        Args:
            x: Input tensor of shape (B, H, W, 3)

        Returns:
            Tuple of (embedding, downsampled_output):
                - embedding: (B, H-4, W-4, embed_dim) L2-normalized
                - downsampled_output: (B, (H-7)//2, (W-7)//2, 32)
        """
        # First convolution: dense 3×3, expands RF to 9 pixels
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)

        # Second convolution: grouped 3×3, expands RF to 25 pixels
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.gelu(x)
        rich_features = x  # (B, H-4, W-4, 32)

        # Branch A: Embedding projection
        embedding = self.embed_conv(rich_features)  # (B, H-4, W-4, embed_dim)
        # L2 normalization (no activation before norm)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling
        downsampled = self.downsample_conv(rich_features)  # (B, (H-7)//2, (W-7)//2, 32)

        return embedding, downsampled


class StandardBlock(nnx.Module):
    """Standard block for pyramid levels 1 to N.

    Uses a single 3×3 convolution (inherits receptive field from previous levels)
    before splitting into embedding and downsampling branches.

    Input: (B, H, W, 32)
    Returns: (embedding, downsampled_output or None)
        - embedding: (B, H-2, W-2, 16) L2-normalized
        - downsampled_output: (B, (H-5)//2, (W-5)//2, 32) or None if last level
    """

    def __init__(
        self, embed_dim: int = EMBED_DIM, is_last_level: bool = False, *, rngs: nnx.Rngs
    ):
        """Initialize StandardBlock.

        Args:
            embed_dim: Output embedding dimension (default 16)
            is_last_level: If True, skip downsampling branch
            rngs: NNX RNGs for parameter initialization
        """
        self.embed_dim = embed_dim
        self.is_last_level = is_last_level

        # Conv1: Grouped 3×3, stride=1, 32 ch → 32 ch
        self.conv1 = nnx.Conv(
            in_features=HIDDEN_CHANNELS,
            out_features=HIDDEN_CHANNELS,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            feature_group_count=NUM_GROUPS,
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=4, num_features=HIDDEN_CHANNELS, rngs=rngs
        )

        # Branch A: 1×1 Conv for embedding projection (32 → embed_dim)
        self.embed_conv = nnx.Conv(
            in_features=HIDDEN_CHANNELS,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",
            rngs=rngs,
        )

        # Branch B: 3×3 Conv for downsampling (only if not last level)
        if not is_last_level:
            self.downsample_conv = nnx.Conv(
                in_features=HIDDEN_CHANNELS,
                out_features=HIDDEN_CHANNELS,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="VALID",
                feature_group_count=NUM_GROUPS,
                rngs=rngs,
            )

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass through StandardBlock.

        Args:
            x: Input tensor of shape (B, H, W, 32)

        Returns:
            Tuple of (embedding, downsampled_output or None):
                - embedding: (B, H-2, W-2, embed_dim) L2-normalized
                - downsampled_output: (B, (H-5)//2, (W-5)//2, 32) or None
        """
        # Feature extraction: grouped 3×3, expands RF by 2 pixels
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)
        rich_features = x  # (B, H-2, W-2, 32)

        # Branch A: Embedding projection
        embedding = self.embed_conv(rich_features)  # (B, H-2, W-2, embed_dim)
        # L2 normalization (no activation before norm)
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        embedding = embedding / (norm + 1e-8)

        # Branch B: Downsampling (if not last level)
        downsampled = None
        if not self.is_last_level:
            downsampled = self.downsample_conv(rich_features)

        return embedding, downsampled


class HierarchicalEmbeddingModel(nnx.Module):
    """Hierarchical embedding model with multi-scale feature pyramid.

    Uses Decoupled Cascade architecture: stacked stride=1 convolutions build
    structural understanding before splitting into embedding and downsampling
    branches. Produces 16-dimensional embeddings at multiple scales optimized
    for attention-based matching. Uses valid convolutions only (no padding).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
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
        # Level 0: StemBlock (RGB input, two stacked convs)
        # Levels 1 to N-1: StandardBlock (feature input, single conv)
        self.blocks = nnx.List()

        # StemBlock for Level 0
        self.blocks.append(StemBlock(embed_dim=embed_dim, rngs=rngs))

        # StandardBlocks for Levels 1 to N-1
        for i in range(1, num_levels):
            is_last = i == num_levels - 1
            self.blocks.append(
                StandardBlock(
                    embed_dim=embed_dim,
                    is_last_level=is_last,
                    rngs=rngs,
                )
            )

    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        """Forward pass through pyramid.

        Args:
            x: Input tensor of shape (B, H, W, 3)

        Returns:
            List of feature maps, one per level.
            Each level has shape (B, H_l, W_l, embed_dim)
            where H_l, W_l shrink at each level due to VALID padding.
        """
        feature_maps = []

        # Iterate through all blocks
        for i, block in enumerate(self.blocks):
            embedding, downsampled = block(x)
            feature_maps.append(embedding)

            # Pass downsampled output to next block (if not last)
            if i < len(self.blocks) - 1:
                x = downsampled

        return feature_maps

    def compute_attention_maps(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        window_indices: Tuple[int, int],
        window_size: int = 16,
        level_index: int = -1,
        pixel_indices: Optional[jnp.ndarray] = None,
        temperature: float = 0.2,
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
            temperature: Softmax temperature (default 0.2)

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
        # For the new architecture, we need to map embedding coordinates back to input image
        # The downscaled image already matches the embedding dimensions, so we can extract directly
        # Note: The visualization code downscales the image to match embedding dimensions
        # So we extract the same window from the downscaled image that was passed in
        # The caller is responsible for providing the correctly downscaled image
        # We use the embedding coordinates directly on the input image with proper scaling
        #
        # Since img1 is the original input (not downscaled), we need to calculate the mapping
        # Each level has different spatial dimensions, so we compute the ratio
        H_img, W_img = img1.shape[1], img1.shape[2]
        scale_h = H_img / H_emb
        scale_w = W_img / W_emb

        img_h_start = int(emb_h_start * scale_h)
        img_h_end = int(emb_h_end * scale_h)
        img_w_start = int(emb_w_start * scale_w)
        img_w_end = int(emb_w_end * scale_w)

        # Ensure we don't go out of bounds
        img_h_end = min(img_h_end, H_img)
        img_w_end = min(img_w_end, W_img)

        window_crop = img1[
            0, img_h_start:img_h_end, img_w_start:img_w_end, :
        ]  # (window_size*scale, window_size*scale, 3)

        # Resize crop to window_size for visualization consistency
        if window_crop.shape[0] > 0 and window_crop.shape[1] > 0:
            window_crop = jax.image.resize(
                window_crop, (window_size, window_size, 3), method="bilinear"
            )
        else:
            # Fallback: create a blank crop if dimensions are invalid
            window_crop = jnp.zeros((window_size, window_size, 3))

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
            self_logits / temperature, axis=-1
        )  # (256, 256)

        # Compute cross-attention logits
        cross_logits = flat_emb1 @ flat_emb2.T  # (256, 256)
        cross_attn_weights = jax.nn.softmax(
            cross_logits / temperature, axis=-1
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

    def compute_flow(
        self,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        flow_estimator,
        temperature: float = 0.15,
    ) -> jnp.ndarray:
        """Compute flow field at coarsest level.

        Flow convention: (u, v) = where F1 pixel moves TO in F2

        Args:
            img1: Frame 1 (B, H, W, 3)
            img2: Frame 2 (B, H, W, 3)
            flow_estimator: FlowEstimator module
            temperature: Softmax temperature (default 0.15)

        Returns:
            flow: (B, H, W, 2) normalized flow field at coarsest level
        """
        from barevision.flow.flow_estimator import (
            AttentionCentroids,
            create_source_position_grid,
        )

        # Get embeddings at coarsest level (last level)
        pyramid1 = self(img1)
        pyramid2 = self(img2)

        emb1 = pyramid1[-1]  # (B, H, W, D)
        emb2 = pyramid2[-1]  # (B, H, W, D)
        B, H, W, D = emb1.shape

        # Flatten spatial dimensions: (B, N, D) where N = H*W
        N = H * W
        flat_emb1 = emb1.reshape(B, N, D)
        flat_emb2 = emb2.reshape(B, N, D)

        # Compute self and cross attention logits
        self_logits = flat_emb1 @ flat_emb1.transpose(0, 2, 1)  # (B, N, N)
        cross_logits = flat_emb1 @ flat_emb2.transpose(0, 2, 1)  # (B, N, N)

        # Apply temperature and softmax
        self_attn = jax.nn.softmax(self_logits / temperature, axis=-1)  # (B, N, N)
        cross_attn = jax.nn.softmax(cross_logits / temperature, axis=-1)  # (B, N, N)

        # Compute centroids
        centroids_computer = AttentionCentroids(window_size=H, rngs=nnx.Rngs(0))
        centroids = centroids_computer(self_attn, cross_attn)  # (B, N, 4)

        # Create source position grid
        src_pos = create_source_position_grid(window_size=H)  # (N, 2)
        src_pos = jnp.broadcast_to(src_pos, (B, N, 2))  # (B, N, 2)

        # Predict flow through estimator
        flow = flow_estimator(src_pos, centroids)  # (B, N, 2)

        # Reshape to spatial grid: (B, H, W, 2)
        flow_dense = flow.reshape(B, H, W, 2)

        return flow_dense


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

    For the Decoupled Cascade architecture with VALID padding:
    - Standard Block FE reverse: Adds 2 pixels (from Conv1 3×3)
    - Stem Block FE reverse: Adds 4 pixels (from Conv1 + Conv2 3×3)
    - Downsample reverse: (size - 1) * stride + kernel_size

    Args:
        target_coarse_dim: Target spatial dimension at coarsest level
                          (e.g., 48 for 3×3 grid of 16×16 windows)
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Required input image dimension (height and width are same)

    Example:
        For 3 levels targeting 16×16 at coarsest (L2):
        L2 drops 2 → 18
        L1 downsample reversed → (18-1)*2 + 3 = 37
        L1 drops 2 → 39
        L0 downsample reversed → (39-1)*2 + 3 = 79
        L0 drops 4 → 83
        Result: 83×83 input required
    """
    size = target_coarse_dim
    # Walk backward from top level (num_levels - 1) down to 0
    for i in reversed(range(num_levels)):
        # Reverse the Feature Extraction drop (Stem drops 4, Standard drops 2)
        size += 4 if i == 0 else 2
        # Reverse the downsample that fed into this level (only if there is a level below it)
        if i > 0:
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

    For the Decoupled Cascade architecture with VALID padding:
    - Feature Extraction drop: Stem drops 4, Standard drops 2
    - Downsample: (size - kernel_size) // stride + 1

    Args:
        input_dim: Input image dimension
        num_levels: Number of pyramid levels
        kernel_size: Convolution kernel size (default 3)
        stride: Downsampling stride (default 2)

    Returns:
        Spatial dimension at coarsest level
    """
    size = input_dim
    for i in range(num_levels):
        # Apply Feature Extraction drop
        size -= 4 if i == 0 else 2
        # Apply Downsample (if not the last level)
        if i < num_levels - 1:
            size = (size - kernel_size) // stride + 1
    return size
