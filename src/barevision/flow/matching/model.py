"""Flow estimation from attention features.

Predicts optical flow from attention centroids, positions, and confidence features.
"""

from typing import List

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


class LevelFlowEstimator(nnx.Module):
    """Predicts residual flow from attention features for a single pyramid level.

    Input: 8 floats per pixel:
        - self_relative (2): self-centroid offset from source
        - cross_relative (2): cross-centroid offset from source (flow vector)
        - cross_absolute (2): cross-centroid absolute position [0, 1]
        - self_max_peak (1): max self-attention weight
        - cross_max_peak (1): max cross-attention weight

    Output: 2 floats per pixel (residual_u, residual_v)

    All coordinates are normalized to [0, 1] range, relative features are 0-mean.

    V1: Operates on a single 16×16 window at a time.
    V2: Will receive prior flow from coarser level and window shift offset.
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

        # Two hidden layers: sufficient capacity for 8→2 mapping
        # Simpler architecture reduces risk of overfitting per-window
        self.linear1 = nnx.Linear(8, hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)

        # Output layer: no bias since tanh output is centered at zero
        # We want to start with no systematic direction preference
        # Small kernel init keeps initial predictions near zero
        self.linear_out = nnx.Linear(
            hidden_dim,
            2,
            use_bias=False,  # No bias: output centered at zero flow
            kernel_init=nnx.initializers.normal(0.02),
            rngs=rngs,
        )

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Predict flow from attention features.

        Args:
            features: (B, N, 8) from AttentionFeatures

        Returns:
            flow: (B, N, 2) predicted flow [u, v] in normalized coordinates, bounded to [-max_flow, max_flow]
        """
        # Two hidden layers with ReLU activation
        x = self.linear1(features)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.relu(x)

        # Output layer with tanh activation and scaling
        # No bias means output is centered at zero (no preferred direction)
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


class HierarchicalFlowEstimator(nnx.Module):
    """Orchestrates coarse-to-fine flow estimation across pyramid levels.

    V1 (current): Runs LevelFlowEstimator independently at each level.
        - Crops each level to grid-aligned (divisible by window_size)
        - Splits into 16×16 windows
        - Runs LevelFlowEstimator on each window
        - Returns list of flow fields, one per level

    V2 (planned): Will cascade with upscaled priors and window shifting.
        - Start at coarsest level (single window)
        - Upscale flow 2× and pass as prior to next finer level
        - Extract centered region at finer level based on prior
        - Shift windows to cancel ego-motion
        - Refine flow prediction with prior context

    Architecture:
        pyramid1, pyramid2 → [Level 0: 16 windows] → flow_L0
                           → [Level 1: 4 windows]  → flow_L1
                           → [Level 2: 1 window]   → flow_L2
        Returns: [flow_L0, flow_L1, flow_L2]

    Each flow field is at its level's native resolution (after cropping).
    """

    def __init__(
        self,
        num_levels: int,
        window_size: int,
        hidden_dim: int,
        max_flow: float,
        temperature: float,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize hierarchical flow estimator.

        Args:
            num_levels: Number of pyramid levels
            window_size: Size of attention windows (default 16)
            hidden_dim: Hidden dimension for LevelFlowEstimator
            temperature: attention softmax temperature
            max_flow: Maximum flow magnitude (default 0.5 = half-window)
            rngs: NNX RNGs for parameter initialization
        """
        self.num_levels = num_levels
        self.window_size = window_size
        self.temperature = temperature

        # Create independent LevelFlowEstimator per level
        # V2: Could share weights across levels, but V1 uses independent estimators
        self.level_estimators = nnx.List(
            [
                LevelFlowEstimator(
                    window_size=window_size,
                    hidden_dim=hidden_dim,
                    max_flow=max_flow,
                    rngs=rngs,
                )
                for _ in range(num_levels)
            ]
        )

    def __call__(
        self,
        pyramid1: List[jnp.ndarray],
        pyramid2: List[jnp.ndarray],
    ) -> List[jnp.ndarray]:
        """Estimate flow at all pyramid levels.

        V1: Independent estimation at each level (no priors, no shifting).

        Args:
            pyramid1: List of embeddings from frame 1, one per level
            pyramid2: List of embeddings from frame 2, one per level
            temperature: Softmax temperature for attention

        Returns:
            List of flow fields, one per level.
            Each flow field has shape (B, H_l, W_l, 2) where H_l, W_l are
            the cropped dimensions at that level.
        """
        from barevision.utils.grid import WindowGrid, crop_to_grid_aligned

        if len(pyramid1) != len(pyramid2) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} pyramid levels, "
                f"got {len(pyramid1)} and {len(pyramid2)}"
            )

        flows = []
        grid = WindowGrid(window_size=self.window_size)

        for level_idx in range(self.num_levels):
            emb1 = pyramid1[level_idx]
            emb2 = pyramid2[level_idx]
            estimator = self.level_estimators[level_idx]

            # Crop to grid-aligned (centered crop for symmetric buffer)
            emb1_cropped = crop_to_grid_aligned(emb1, self.window_size)
            emb2_cropped = crop_to_grid_aligned(emb2, self.window_size)

            B, H, W, D = emb1_cropped.shape

            # Split into windows
            windows1 = grid.split(emb1_cropped)  # (B, num_windows, 16, 16, D)
            windows2 = grid.split(emb2_cropped)

            num_windows_h = H // self.window_size
            num_windows_w = W // self.window_size
            num_windows = num_windows_h * num_windows_w

            # Flatten batch and windows for processing
            flat_windows1 = windows1.reshape(
                B * num_windows, self.window_size, self.window_size, D
            )
            flat_windows2 = windows2.reshape(
                B * num_windows, self.window_size, self.window_size, D
            )

            # Flatten spatial dimensions for attention
            N = self.window_size * self.window_size
            flat_emb1 = flat_windows1.reshape(B * num_windows, N, D)
            flat_emb2 = flat_windows2.reshape(B * num_windows, N, D)

            # Compute self and cross attention
            self_logits = flat_emb1 @ flat_emb1.transpose(0, 2, 1)
            cross_logits = flat_emb1 @ flat_emb2.transpose(0, 2, 1)

            self_attn = jax.nn.softmax(self_logits / self.temperature, axis=-1)
            cross_attn = jax.nn.softmax(cross_logits / self.temperature, axis=-1)

            # Create source position grid
            src_pos = create_source_position_grid(window_size=self.window_size)
            src_pos = jnp.broadcast_to(src_pos, (B * num_windows, N, 2))

            # Compute attention features (8 floats per pixel)
            features_computer = AttentionFeatures(window_size=self.window_size)
            features = features_computer(self_attn, cross_attn, src_pos)

            # Predict flow
            flow = estimator(features)  # (B * num_windows, N, 2)

            # Reshape back to spatial grid per window
            flow_per_window = flow.reshape(
                B * num_windows, self.window_size, self.window_size, 2
            )

            # Unflatten batch and windows
            flow_unflat = flow_per_window.reshape(
                B, num_windows, self.window_size, self.window_size, 2
            )

            # Rearrange windows into grid layout
            # (B, num_windows, WH, WW, 2) → (B, num_h, num_w, WH, WW, 2)
            flow_grid = flow_unflat.reshape(
                B, num_windows_h, num_windows_w, self.window_size, self.window_size, 2
            )

            # Transpose to interleave: (B, num_h, WH, num_w, WW, 2)
            flow_grid = flow_grid.transpose(0, 1, 3, 2, 4, 5)

            # Reshape to spatial grid: (B, H, W, 2)
            flow_grid = flow_grid.reshape(B, H, W, 2)

            flows.append(flow_grid)

        return flows
