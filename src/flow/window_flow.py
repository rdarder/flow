"""Window-level flow processing module.

Processes pyramid embeddings through windowed attention to estimate flow.
Batches multiple windows together for efficient processing.
"""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from flax import nnx

from flow.token_attention import TokenCrossAttention, TokenSelfAttention
from flow.window_grid import WindowGrid


class WindowFlowProcessor(nnx.Module):
    """Processes embeddings through windows to estimate flow.

    Takes pyramid embeddings at a given level, splits into windows,
    applies TokenCrossAttention + TokenSelfAttention to each window, and stitches
    results back together.
    """

    def __init__(
        self,
        embed_dim: int,
        window_size: int = 16,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            embed_dim: Dimension of embeddings (e.g., 16)
            window_size: Size of attention windows (default 16)
            rngs: NNX RNGs
        """
        self.embed_dim = embed_dim
        self.window_size = window_size

        # Window grid for split/stitch operations
        self.window_grid = WindowGrid(window_size=window_size)

        # Attention modules
        self.token_cross_attn = TokenCrossAttention(embed_dim=embed_dim, rngs=rngs)
        self.token_self_attn = TokenSelfAttention(embed_dim=embed_dim, rngs=rngs)

    def _create_coordinate_grid(self, h: int, w: int) -> jnp.ndarray:
        """Create normalized coordinate grid [0, 1] for a window.

        Args:
            h: Height of grid
            w: Width of grid

        Returns:
            Grid of shape (h, w, 2) with (x, y) coordinates in [0, 1]
        """
        # Create coordinate grid
        y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")

        # Normalize to [0, 1]
        x_norm = (
            x.astype(jnp.float32) / float(w - 1)
            if w > 1
            else jnp.zeros_like(x, dtype=jnp.float32)
        )
        y_norm = (
            y.astype(jnp.float32) / float(h - 1)
            if h > 1
            else jnp.zeros_like(y, dtype=jnp.float32)
        )

        # Stack to get (h, w, 2)
        grid = jnp.stack([x_norm, y_norm], axis=-1)
        return grid

    def _embeddings_to_patches(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Convert embeddings from (B, H, W, C) to (B, H*W, C) patches.

        Args:
            embeddings: (B, H, W, C) tensor

        Returns:
            (B, H*W, C) patches
        """
        B, H, W, C = embeddings.shape
        return embeddings.reshape(B, H * W, C)

    def _patches_to_grid(self, patches: jnp.ndarray, h: int, w: int) -> jnp.ndarray:
        """Convert patches from (B, H*W, C) back to (B, H, W, C) grid.

        Args:
            patches: (B, H*W, C) tensor
            h: Target height
            w: Target width

        Returns:
            (B, H, W, C) tensor
        """
        B, _, C = patches.shape
        return patches.reshape(B, h, w, C)

    def __call__(
        self,
        emb1: jnp.ndarray,  # (B, H, W, C)
        emb2: jnp.ndarray,  # (B, H, W, C)
        prior_flow: jnp.ndarray,  # (B, H, W, 2) - Flow from coarser level
        prior_confidence: jnp.ndarray,  # (B, H, W, 1) - Confidence in prior
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Process embeddings through windows to estimate flow.

        Args:
            emb1: Embeddings from frame 1 (B, H, W, C)
            emb2: Embeddings from frame 2 (B, H, W, C)
            prior_flow: Flow estimate from coarser level (B, H, W, 2)
            prior_confidence: Confidence in prior flow (B, H, W, 1)

        Returns:
            flow: Estimated flow (B, H, W, 2) in normalized coordinates [0, 1]
            confidence: Confidence scores (B, H, W, 1)
            aux: Dictionary with intermediate outputs for debugging
        """
        B, H, W, C = emb1.shape

        # Validate dimensions are compatible with window size
        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                f"Embedding dimensions ({H}, {W}) must be divisible by window size "
                f"({self.window_size}). Consider cropping to compatible size."
            )

        # Split embeddings into windows
        # (B, H, W, C) -> (B, num_windows, window_size, window_size, C)
        windows1 = self.window_grid.split(emb1)
        windows2 = self.window_grid.split(emb2)

        # Split priors into windows (same pattern as embeddings)
        prior_flow_windows = self.window_grid.split(prior_flow)
        prior_conf_windows = self.window_grid.split(prior_confidence)

        num_windows = windows1.shape[1]

        # Reshape for batching: (B, num_windows, W, W, C) -> (B*num_windows, W, W, C)
        windows1_batched = windows1.reshape(
            B * num_windows, self.window_size, self.window_size, C
        )
        windows2_batched = windows2.reshape(
            B * num_windows, self.window_size, self.window_size, C
        )

        # Reshape priors: (B, num_windows, W, W, 2) -> (B*num_windows, W, W, 2)
        prior_flow_batched = prior_flow_windows.reshape(
            B * num_windows, self.window_size, self.window_size, 2
        )
        prior_conf_batched = prior_conf_windows.reshape(
            B * num_windows, self.window_size, self.window_size, 1
        )

        # Convert to patches: (B*num_windows, W, W, C) -> (B*num_windows, W*W, C)
        patches1 = self._embeddings_to_patches(windows1_batched)
        patches2 = self._embeddings_to_patches(windows2_batched)

        # Convert priors to patches: (B*num_windows, W*W, 2/1)
        prior_flow_patches = self._embeddings_to_patches(prior_flow_batched)
        prior_conf_patches = self._embeddings_to_patches(prior_conf_batched)

        # Create coordinate grid for a single window
        window_coords = self._create_coordinate_grid(
            self.window_size, self.window_size
        )  # (W, W, 2)
        window_coords_flat = window_coords.reshape(-1, 2)  # (W*W, 2)

        # Broadcast to batch dimension: (B*num_windows, W*W, 2)
        q_pos = jnp.broadcast_to(
            window_coords_flat,
            (B * num_windows, self.window_size * self.window_size, 2),
        )
        k_pos = q_pos  # Same positions for key

        # Run TokenCrossAttention (cross-attention between frames with prior guidance)
        flow_lookup, conf_lookup, attn_weights_lookup = self.token_cross_attn(
            patches1, patches2, q_pos, k_pos, prior_flow_patches, prior_conf_patches
        )

        # Blend lookup result with prior flow (allows outside-window flow to enter)
        # High prior confidence -> more weight on prior flow outside window
        weight_lookup = conf_lookup
        weight_prior = prior_conf_patches
        weight_sum = weight_lookup + weight_prior + 1e-6

        flow_mixed = (
            weight_lookup * flow_lookup + weight_prior * prior_flow_patches
        ) / weight_sum
        # Combined confidence is the average of both sources' confidence
        # (consensus between what we found and what we expected)
        conf_mixed = (conf_lookup + prior_conf_patches) / 2

        # Run TokenSelfAttention (self-attention within frame 1) with blended flow
        flow_peer, attn_weights_peer, conf_peer = self.token_self_attn(
            patches1, q_pos, flow_mixed, conf_mixed
        )

        # Convert back to grid format: (B*num_windows, W*W, 2) -> (B*num_windows, W, W, 2)
        flow_grid_batched = self._patches_to_grid(
            flow_peer, self.window_size, self.window_size
        )
        conf_grid_batched = self._patches_to_grid(
            conf_peer, self.window_size, self.window_size
        )

        # Reshape back to window format: (B*num_windows, W, W, C) -> (B, num_windows, W, W, C)
        flow_windows = flow_grid_batched.reshape(
            B, num_windows, self.window_size, self.window_size, 2
        )
        conf_windows = conf_grid_batched.reshape(
            B, num_windows, self.window_size, self.window_size, 1
        )

        # Stitch windows back to full resolution
        num_h = H // self.window_size
        num_w = W // self.window_size

        flow = self.window_grid.stitch(flow_windows, grid_h=num_h, grid_w=num_w)
        confidence = self.window_grid.stitch(conf_windows, grid_h=num_h, grid_w=num_w)

        # Prepare auxiliary outputs for debugging
        aux = {
            "flow_lookup": self.window_grid.stitch(
                self._patches_to_grid(
                    flow_lookup, self.window_size, self.window_size
                ).reshape(B, num_windows, self.window_size, self.window_size, 2),
                grid_h=num_h,
                grid_w=num_w,
            ),
            "flow_mixed": self.window_grid.stitch(
                self._patches_to_grid(
                    flow_mixed, self.window_size, self.window_size
                ).reshape(B, num_windows, self.window_size, self.window_size, 2),
                grid_h=num_h,
                grid_w=num_w,
            ),
            "flow_peer": self.window_grid.stitch(
                self._patches_to_grid(
                    flow_peer, self.window_size, self.window_size
                ).reshape(B, num_windows, self.window_size, self.window_size, 2),
                grid_h=num_h,
                grid_w=num_w,
            ),
            "conf_lookup": self.window_grid.stitch(
                self._patches_to_grid(
                    conf_lookup, self.window_size, self.window_size
                ).reshape(B, num_windows, self.window_size, self.window_size, 1),
                grid_h=num_h,
                grid_w=num_w,
            ),
            "conf_mixed": self.window_grid.stitch(
                self._patches_to_grid(
                    conf_mixed, self.window_size, self.window_size
                ).reshape(B, num_windows, self.window_size, self.window_size, 1),
                grid_h=num_h,
                grid_w=num_w,
            ),
            "conf_peer": self.window_grid.stitch(
                self._patches_to_grid(
                    conf_peer, self.window_size, self.window_size
                ).reshape(B, num_windows, self.window_size, self.window_size, 1),
                grid_h=num_h,
                grid_w=num_w,
            ),
            "prior_flow": prior_flow,  # Prior flow passed to this level
            "prior_confidence": prior_confidence,  # Prior confidence passed to this level
            "num_windows": num_windows,
            "grid_h": num_h,
            "grid_w": num_w,
        }

        return flow, confidence, aux
