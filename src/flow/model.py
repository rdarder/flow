import jax
import jax.numpy as jnp
from flax import nnx


class Stem(nnx.Module):
    """
    Pixel-level feature extractor.
    Input: (B, H+2, W+2, 3) - 18x18 image patch
    Output: (B, H, W, C) - 16x16 feature grid
    """

    def __init__(self, embed_dim: int, dw_patterns: int = 12, *, rngs: nnx.Rngs):
        # 1. Depthwise Conv (3x3, VALID)
        # Reduces spatial dim by 2 (1px border lost)
        self.dw1 = nnx.Conv(
            in_features=3,
            out_features=dw_patterns * 3,
            kernel_size=(3, 3),
            padding="VALID",  # No padding
            feature_group_count=3,
            use_bias=False,
            rngs=rngs,
        )
        # 2. Pointwise Conv (1x1)
        self.pw1 = nnx.Conv(
            in_features=dw_patterns * 3,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding="VALID",  # No padding needed for 1x1
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(num_features=embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input: (B, 18, 18, 3)
        x = self.dw1(x)  # -> (B, 16, 16, 36)
        x = self.pw1(x)  # -> (B, 16, 16, C)
        x = self.norm1(x)
        x = nnx.relu(x)
        return x


class SpatialScore(nnx.Module):
    """
    Computes the Gaussian Kernel spatial score between two sets of positions.
    Score = -scale * ||pos1 - pos2||^2
    Uses the expanded square trick for efficiency: ||A-B||^2 = A^2 + B^2 - 2AB
    """

    def __init__(self, initial_scale: float = 10.0, *, rngs: nnx.Rngs):
        # Initialized to 10.0 because normalized distances (0-1) are small.
        self.scale = nnx.Param(initial_scale)

    def __call__(self, pos1: jnp.ndarray, pos2: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            pos1: (B, N, 2)
            pos2: (B, N, 2)
        Returns:
            scores: (B, N, N) pairwise spatial scores
        """
        # Squared Norms
        p1_sq = jnp.sum(jnp.square(pos1), axis=-1, keepdims=True)  # (B, N, 1)
        p2_sq = jnp.sum(jnp.square(pos2), axis=-1, keepdims=True)  # (B, N, 1)
        p2_sq_T = jnp.swapaxes(p2_sq, -2, -1)  # (B, 1, N)

        # Cross Term (2 * A . B)
        # (B, N, 2) @ (B, 2, N) -> (B, N, N)
        cross = 2.0 * (pos1 @ jnp.swapaxes(pos2, -2, -1))

        # Combine
        dist_sq = p1_sq + p2_sq_T - cross

        # Clip negative values (numerical noise from float32 subtraction)
        dist_sq = jnp.maximum(dist_sq, 0.0)

        # Convert distance to log-probability (Gaussian Kernel logit)
        return -jnp.abs(self.scale.value) * dist_sq


class PatchLookup(nnx.Module):
    """
    Core 'Line 4' Attention Module.
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))
        self.spatial_score = SpatialScore(initial_scale=10.0, rngs=rngs)

    def __call__(
        self,
        q_features: jnp.ndarray,  # (B, N, C)
        k_features: jnp.ndarray,  # (B, N, C)
        q_pos: jnp.ndarray,  # (B, N, 2) - Normalized [0, 1]
        k_pos: jnp.ndarray,  # (B, N, 2) - Normalized [0, 1]
    ):
        B, N, C = q_features.shape

        # --- 1. Visual Similarity ---
        visual_logits = q_features @ jnp.swapaxes(k_features, -2, -1)
        visual_score = visual_logits * self.visual_scale.value

        # --- 2. Spatial Proximity ---
        spatial_score = self.spatial_score(q_pos, k_pos)

        # --- 3. Combine & Softmax ---
        logits = visual_score + spatial_score
        attn_weights = nnx.softmax(logits, axis=-1)

        # --- 4. Value Aggregation ---
        target_pos_est = attn_weights @ k_pos

        # --- 5. Output Calculation ---
        flow = target_pos_est - q_pos
        consensus = jnp.max(attn_weights, axis=-1, keepdims=True)

        return flow, consensus, attn_weights


class PeerPropagation(nnx.Module):
    """
    Module 2: Peer Propagation (V2) - Normalized Coords
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))
        self.spatial_score = SpatialScore(initial_scale=10.0, rngs=rngs)
        self.consensus_bias_scale = nnx.Param(5.0)

    def __call__(
        self,
        features: jnp.ndarray,  # (B, N, C)
        pos: jnp.ndarray,  # (B, N, 2) - Normalized
        flow_v1: jnp.ndarray,  # (B, N, 2) - Normalized Flow
        consensus_v1: jnp.ndarray,  # (B, N, 1)
    ):
        B, N, C = features.shape

        # --- 1. Visual Similarity ---
        visual_logits = features @ jnp.swapaxes(features, -2, -1)
        visual_score = visual_logits * self.visual_scale.value

        # --- 2. Spatial Proximity ---
        spatial_score = self.spatial_score(pos, pos)

        # --- 3. Consensus Bias ---
        consensus_key = jnp.swapaxes(consensus_v1, -2, -1)
        consensus_score = consensus_key * self.consensus_bias_scale.value

        # --- 4. Combine & Mask ---
        logits = visual_score + spatial_score + consensus_score

        mask = jnp.eye(N, dtype=bool)
        logits = logits + (mask * -1e9)

        attn_weights = nnx.softmax(logits, axis=-1)

        # --- 5. Value Aggregation ---
        flow_peer = attn_weights @ flow_v1

        # --- 6. Peer Consensus ---
        peer_consensus = attn_weights @ consensus_v1

        return flow_peer, attn_weights, peer_consensus


class BarebonesFlowModel(nnx.Module):
    """
    The main "Orchestrator" model (Line 4).
    Uses Normalized Cartesian Coordinates.
    """

    def __init__(self, img_size_hw: tuple[int, int], embed_dim: int, *, rngs):
        super().__init__()
        self.embed_dim = embed_dim
        self.raw_hw = img_size_hw  # Expected 18x18

        # 1. Stem
        self.stem = Stem(embed_dim=embed_dim, dw_patterns=12, rngs=rngs)

        # 2. Coordinate Grid (Normalized 0-1)
        h, w = (img_size_hw[0] - 2, img_size_hw[1] - 2)  # 16x16
        self.grid_hw = (h, w)
        self.P = h * w

        self.norm_scale = float(max(img_size_hw))  # e.g. 18.0

        # Generate grid 0..15, offset to align with valid region
        y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        y = y + 1.0
        x = x + 1.0

        y = y / self.norm_scale
        x = x / self.norm_scale

        xy_grid = jnp.stack([x, y], axis=-1).reshape(self.P, 2).astype(jnp.float32)
        self.xy_grid = nnx.Variable(xy_grid)

        # 3. Attention Layers
        self.patch_lookup = PatchLookup(embed_dim=embed_dim, rngs=rngs)
        self.peer_prop = PeerPropagation(embed_dim=embed_dim, rngs=rngs)

        # 4. Blend Parameter
        self.lookup_blend = nnx.Param(4.0)

    def _img_to_patches(self, img):
        # img: (B, 18, 18, 3) -> patches: (B, 256, C)
        B = img.shape[0]
        stem = self.stem(img)  # (B, 16, 16, C)
        patches = stem.reshape(B, self.P, self.embed_dim)
        return patches

    def _pad_output(self, flow_16x16, batch_size):
        return jnp.pad(flow_16x16, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")

    def __call__(self, frame1, frame2):
        batch_size = frame1.shape[0]

        # 1. Get Features
        f1_patches = self._img_to_patches(frame1)
        f2_patches = self._img_to_patches(frame2)

        # 2. Get Coordinates (Normalized)
        xy_grid_val = self.xy_grid.value
        q_pos = jnp.broadcast_to(xy_grid_val, (batch_size, self.P, 2))
        k_pos = jnp.broadcast_to(xy_grid_val, (batch_size, self.P, 2))

        # 3. Run V1: Patch Lookup
        F_cross, C_cross, A_cross = self.patch_lookup(
            f1_patches, f2_patches, q_pos, k_pos
        )

        # 4. Run V2: Peer Propagation
        F_peer, A_peer, C_peer = self.peer_prop(f1_patches, q_pos, F_cross, C_cross)

        # 5. Blend Flows
        w1 = jnp.power(C_cross, self.lookup_blend.value)
        w2 = 1.0 - w1

        F_final_flat = (w1 * F_cross) + (w2 * F_peer)

        # 6. Reshape
        h, w = self.grid_hw
        F_final_grid = F_final_flat.reshape(batch_size, h, w, 2)

        # 7. Denormalize Flow (Back to Pixels)
        F_final_pixels = F_final_grid * self.norm_scale

        # 8. Pad to 18x18
        F_final_padded = self._pad_output(F_final_pixels, batch_size)

        # Aux logging
        F_cross_grid = F_cross.reshape(batch_size, h, w, 2) * self.norm_scale
        F_peer_grid = F_peer.reshape(batch_size, h, w, 2) * self.norm_scale
        C_cross_grid = C_cross.reshape(batch_size, h, w, 1)
        C_peer_grid = C_peer.reshape(batch_size, h, w, 1)

        return F_final_padded, dict(
            F_cross=self._pad_output(F_cross_grid, batch_size),
            F_peer=self._pad_output(F_peer_grid, batch_size),
            C_cross=self._pad_output(C_cross_grid, batch_size),
            C_peer=self._pad_output(C_peer_grid, batch_size),
            A_cross=A_cross,
            A_peer=A_peer,
        )
