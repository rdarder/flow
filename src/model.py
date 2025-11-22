import jax
import jax.numpy as jnp
from flax import nnx


class Stem(nnx.Module):
    """
    A simple 2-layer stem to extract features from a 16x16 patch.
    Input: (B, 16, 16, 3) image patch
    Output: (B, 8, 8, C) feature grid
    """

    def __init__(self, embed_dim: int, dw_patterns: int = 12, *, rngs: nnx.Rngs):
        # 1. Depthwise Conv (3x3)
        self.dw1 = nnx.Conv(
            in_features=3,
            out_features=dw_patterns * 3,
            kernel_size=(3, 3),
            padding='SAME',
            feature_group_count=3,
            use_bias=False,
            rngs=rngs
        )
        # 2. Pointwise Conv (1x1)
        self.pw1 = nnx.Conv(
            in_features=dw_patterns * 3,
            out_features=embed_dim,
            kernel_size=(1, 1),
            padding='SAME',
            rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(num_features=embed_dim, use_bias=True)
        self.pool1 = nnx.AvgPool(
            window_shape=(2, 2),
            strides=(2, 2)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Input: (B, 16, 16, 3)
        x = self.dw1(x)
        x = self.pw1(x)
        # -> (B, 16, 16, C)
        x = self.pool1(x)
        # -> (B, 8, 8, C)
        x = self.norm1(x)
        x = nnx.relu(x)
        return x

    def geometry(self, input_shape_hw):
        h_in, w_in = input_shape_hw
        return (h_in // 2, w_in // 2), 2, (h_in, w_in)


class PatchLookup(nnx.Module):
    """
    Core 'Line 4' Attention Module.
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))
        self.spatial_scale = nnx.Param(0.1)

    def __call__(
            self,
            q_features: jnp.ndarray,  # (B, N, C)
            k_features: jnp.ndarray,  # (B, N, C)
            q_pos: jnp.ndarray,  # (B, N, 2)
            k_pos: jnp.ndarray  # (B, N, 2)
    ):
        B, N, C = q_features.shape

        # --- 1. Visual Similarity ---
        visual_logits = q_features @ jnp.swapaxes(k_features, -2, -1)
        visual_score = visual_logits * self.visual_scale.value

        # --- 2. Spatial Proximity (Gaussian Kernel) ---
        q_pos_sq = jnp.sum(jnp.square(q_pos), axis=-1, keepdims=True)  # (B, N, 1)
        k_pos_sq = jnp.sum(jnp.square(k_pos), axis=-1, keepdims=True)  # (B, N, 1)
        k_pos_sq_T = jnp.swapaxes(k_pos_sq, -2, -1)  # (B, 1, N)

        pos_cross = 2.0 * (q_pos @ jnp.swapaxes(k_pos, -2, -1))
        dist_sq = q_pos_sq + k_pos_sq_T - pos_cross

        spatial_score = -jnp.abs(self.spatial_scale.value) * dist_sq

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
    Module 2: Peer Propagation (V2)
    Uses consensus as a bias for attention, and calculates C_peer
    as the weighted average of neighbor consensus.
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))
        self.spatial_scale = nnx.Param(0.1)
        self.consensus_bias_scale = nnx.Param(5.0)

    def __call__(
            self,
            features: jnp.ndarray,  # (B, N, C)
            pos: jnp.ndarray,  # (B, N, 2)
            flow_v1: jnp.ndarray,  # (B, N, 2)
            consensus_v1: jnp.ndarray  # (B, N, 1)
    ):
        B, N, C = features.shape

        # --- 1. Visual Similarity ---
        visual_logits = features @ jnp.swapaxes(features, -2, -1)
        visual_score = visual_logits * self.visual_scale.value

        # --- 2. Spatial Proximity ---
        pos_sq = jnp.sum(jnp.square(pos), axis=-1, keepdims=True)
        pos_sq_T = jnp.swapaxes(pos_sq, -2, -1)
        pos_cross = 2.0 * (pos @ jnp.swapaxes(pos, -2, -1))
        dist_sq = pos_sq + pos_sq_T - pos_cross

        spatial_score = -jnp.abs(self.spatial_scale.value) * dist_sq

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

        # --- 6. Peer Consensus Metric (The Simple One) ---
        # Calculate weighted average of neighbors' consensus.
        # (B, N, N) @ (B, N, 1) -> (B, N, 1)
        # "How confident are the neighbors I am listening to?"
        peer_consensus = attn_weights @ consensus_v1

        return flow_peer, attn_weights, peer_consensus


class BarebonesFlowModel(nnx.Module):
    """
    The main "Orchestrator" model (m2).
    """

    def __init__(self, img_size_hw: tuple[int, int], embed_dim: int, *, rngs):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Stem
        self.stem = Stem(embed_dim=embed_dim, dw_patterns=12, rngs=rngs)
        self.grid_size_hw, self.stride, self.inner_hw = self.stem.geometry(img_size_hw)
        self.P = self.grid_size_hw[0] * self.grid_size_hw[1]

        # 2. Global Coordinate Grid
        h, w = self.grid_size_hw
        x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing='xy')
        xy_grid = jnp.stack([x, y], axis=-1).reshape(self.P, 2).astype(jnp.float32)
        self.xy_grid = nnx.Variable(xy_grid)

        # 3. Attention Layers
        self.patch_lookup = PatchLookup(embed_dim=embed_dim, rngs=rngs)
        self.peer_prop = PeerPropagation(embed_dim=embed_dim, rngs=rngs)

        # 4. Blend Parameter
        self.blend_sharpness = nnx.Param(4.0)

    def _img_to_patches(self, img):
        B = img.shape[0]
        stem = self.stem(img)
        patches = stem.reshape(B, self.P, self.embed_dim)
        return patches

    def _upsample_flow_to_dense(self, flow_pred_grid_units, batch_size):
        h, w = self.grid_size_hw
        H, W = self.img_size_hw
        flow_dense = jax.image.resize(
            flow_pred_grid_units,
            shape=(batch_size, H, W, 2),
            method='nearest'
        )
        return flow_dense

    def __call__(self, frame1, frame2):
        batch_size = frame1.shape[0]

        # 1. Get Features
        f1_patches = self._img_to_patches(frame1)
        f2_patches = self._img_to_patches(frame2)

        # 2. Get Coordinates
        xy_grid_val = self.xy_grid.value
        q_pos = jnp.broadcast_to(xy_grid_val, (batch_size, self.P, 2))
        k_pos = jnp.broadcast_to(xy_grid_val, (batch_size, self.P, 2))

        # 3. Run V1: Patch Lookup
        F_cross, C_cross, A_cross = self.patch_lookup(
            f1_patches, f2_patches, q_pos, k_pos
        )

        # 4. Run V2: Peer Propagation
        F_peer, A_peer, C_peer = self.peer_prop(
            f1_patches, q_pos, F_cross, C_cross
        )

        # 5. Blend Flows
        w1 = jnp.power(C_cross, self.blend_sharpness.value)
        w2 = 1.0 - w1

        F_final_flat = (w1 * F_cross) + (w2 * F_peer)

        # 6. Reshape & Upsample
        h_grid, w_grid = self.grid_size_hw
        F_final_grid = F_final_flat.reshape(batch_size, h_grid, w_grid, 2)

        # Aux logging
        F_cross_grid = F_cross.reshape(batch_size, h_grid, w_grid, 2)
        F_peer_grid = F_peer.reshape(batch_size, h_grid, w_grid, 2)
        C_cross_grid = C_cross.reshape(batch_size, h_grid, w_grid, 1)
        C_peer_grid = C_peer.reshape(batch_size, h_grid, w_grid, 1)

        return F_final_grid, dict(
            F_cross=self._upsample_flow_to_dense(F_cross_grid, batch_size),
            F_peer=self._upsample_flow_to_dense(F_peer_grid, batch_size),
            C_cross=jax.image.resize(C_cross_grid, (batch_size, 18, 18, 1), method='nearest'),
            C_peer=jax.image.resize(C_peer_grid, (batch_size, 18, 18, 1), method='nearest'),
            A_cross=A_cross,
            A_peer=A_peer
        )