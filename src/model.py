import jax
import jax.numpy as jnp
from jax.nn import softmax
from flax import nnx
from flax.linen import avg_pool


class Stem(nnx.Module):
    def __init__(self, dw_patterns: int, embed_dim: int, *, rngs):
        init = nnx.initializers.he_normal()
        self.dw1 = nnx.Conv(
            in_features=3,
            out_features=3 * dw_patterns,
            kernel_size=(3, 3),
            strides=(1, 1),
            feature_group_count=3,
            padding='VALID',
            use_bias=False,
            kernel_init=init,
            rngs=rngs
        )
        self.pw1 = nnx.Conv(
            in_features=3 * dw_patterns,
            out_features=embed_dim,
            kernel_size=(1, 1),
            use_bias=True,
            kernel_init=init,
            rngs=rngs
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=embed_dim,
            num_features=embed_dim,
            use_bias=True,
            use_scale=False,
            rngs=rngs
        )

    def geometry(self, img_size_hw: tuple[int, int]):
        h, w = img_size_hw
        h_inner, w_inner = h - 2, w - 2  # 18 -> 16
        h_out, w_out = h_inner // 2, w_inner // 2  # 16 -> 8
        stride = 2.0
        return (int(h_out), int(w_out)), stride, (int(h_inner), int(w_inner))

    def __call__(self, x):
        x = self.dw1(x)
        x = self.pw1(x)
        x = jax.nn.gelu(x)
        x = avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = self.norm1(x)
        return x


class SinusoidalPosEncoding(nnx.Module):
    """
    Creates a static 2D sinusoidal positional encoding (PE) matrix.
    Shape: (H*W, C) e.g., (64, 16)
    """

    def __init__(self, grid_size_hw: tuple[int, int], embed_dim: int):
        h, w = grid_size_hw
        num_patches = h * w


        # Ensure embed_dim is even
        if embed_dim % 4 != 0:
            raise ValueError(f"Embedding dim {embed_dim} must be divisible by 4 for 2D PE.")

        half_dim = embed_dim // 2

        max_period = jnp.float32(w * 2)
        div_term = jnp.exp(jnp.arange(0, half_dim, 2) * -(jnp.log(max_period) / (half_dim // 2)))

        # Create 1D PEs
        pos_x = jnp.arange(w, dtype=jnp.float32).reshape(-1, 1)
        pos_y = jnp.arange(h, dtype=jnp.float32).reshape(-1, 1)

        pe_x = jnp.zeros((w, half_dim))
        pe_x = pe_x.at[:, 0::2].set(jnp.sin(pos_x * div_term))
        pe_x = pe_x.at[:, 1::2].set(jnp.cos(pos_x * div_term))

        pe_y = jnp.zeros((h, half_dim))
        pe_y = pe_y.at[:, 0::2].set(jnp.sin(pos_y * div_term))
        pe_y = pe_y.at[:, 1::2].set(jnp.cos(pos_y * div_term))

        # Create 2D PE
        pe_x_broadcast = jnp.tile(pe_x, (h, 1))
        pe_y_broadcast = jnp.repeat(pe_y, w, axis=0)

        pe = jnp.concatenate([pe_y_broadcast, pe_x_broadcast], axis=1)

        # Store as a non-learnable Variable
        self.pe = nnx.Variable(pe.reshape(num_patches, embed_dim))

    def __call__(self):
        return self.pe.value


class PatchLookupAttention(nnx.Module):
    """
    V1 "Patch Lookup" Block.
    """

    def __init__(self, *, rngs):
        self.attn_temperature = nnx.Param(1.0)
        self.pos_scale = nnx.Param(2.0)

    def __call__(self, f1_patches, f2_patches, pe, location_grid_L):
        """
        Args:
          f1_patches: (B, 64, C) - *Raw* features
          f2_patches: (B, 64, C) - *Raw* features
          pe: (64, C) - *Raw* positional encoding
          location_grid_L: (B, 64, 2) - Broadcasted V matrix
        """
        # 0. Fuse in positional encoding with our own scale.
        pe_scaled = pe * self.pos_scale
        f1_fused = f1_patches + pe_scaled
        f2_fused = f2_patches + pe_scaled

        # 1. Get Attention
        logits = (f1_fused @ f2_fused.transpose(0, 2, 1)) * self.attn_temperature
        A_cross = softmax(logits, axis=-1)  # (B, 64, 64)

        # 2. Get Flow (L_target - L_source)
        # (attn @ L) - L
        F_cross = (A_cross @ location_grid_L) - location_grid_L  # (B, 64, 2)

        # 3. Get Confidence (using jnp.max as discussed)
        max_val = jnp.max(A_cross, axis=-1, keepdims=True)
        num_choices = A_cross.shape[-1]

        # Rescale [1/N, 1.0] -> [0.0, 1.0]
        C_cross = (max_val - (1.0 / num_choices)) / (1.0 - (1.0 / num_choices))
        C_cross = jnp.clip(C_cross, 0.0, 1.0)  # (B, 64, 1)

        return F_cross, C_cross, A_cross


class PeerPropagationAttention(nnx.Module):
    """
    V2.1 "Peer Propagation" Block.
    """

    def __init__(self, *, rngs):
        self.attn_temperature = nnx.Param(1.0)
        self.pos_scale = nnx.Param(4.0)
        self.self_mask = nnx.Variable(jnp.eye(64) * -1e9)  # (64, 64)

    def __call__(self, f1_patches, pe, F_cross, C_cross):
        """
        Args:
          f1_patches: (B, 64, C) - *Raw* features
          pe: (64, C) - *Raw* positional encoding
          F_cross: (B, 64, 2) - V1 flow (serves as V)
          C_cross: (B, 64, 1) - V1 confidence (serves as gate)
        """
        # 0. Fuse in positional encoding with our own scale.
        pe_scaled = pe * self.pos_scale
        f1_fused = f1_patches + pe_scaled

        # 1. Define Q, K, V (as per our V2.1 design)
        Q = f1_fused  # (B, 64, C)
        K = f1_fused * C_cross  # (B, 64, C) - Gated Key
        V = F_cross  # (B, 64, 2)

        # 2. Get Peer Attention
        logits_peer = (Q @ K.transpose(0, 2, 1)) * self.attn_temperature

        # 3. Mask self-attention
        logits_peer = logits_peer + self.self_mask.value  # (B, 64, 64)

        A_peer = softmax(logits_peer, axis=-1)

        # 4. Get Peer Flow
        F_peer = A_peer @ V  # (B, 64, 2)

        return F_peer, A_peer


class BarebonesFlowModel(nnx.Module):
    """
    The main "Orchestrator" model.
    """

    def __init__(self, img_size_hw: tuple[int, int], embed_dim: int, *, rngs):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size_hw = img_size_hw

        # 1. Stem
        self.stem = Stem(embed_dim=embed_dim, dw_patterns=12, rngs=rngs)
        self.grid_size_hw, self.stride, self.inner_hw = self.stem.geometry(img_size_hw)

        # 2. Positional Mechanisms
        # The new, unified PE for Q/K fusion
        self.pos_encoding = SinusoidalPosEncoding(self.grid_size_hw, embed_dim)
        # The V matrix (L) for flow calculation (unchanged)
        self.location_grid = nnx.Variable(self._create_location_tensor(self.grid_size_hw))

        # 3. Attention Layers
        self.patch_lookup = PatchLookupAttention(rngs=rngs)
        self.peer_prop = PeerPropagationAttention(rngs=rngs)

    def _create_location_tensor(self, grid_size_hw: tuple[int, int]) -> jnp.ndarray:
        """(Unchanged) Creates the (H*W, 2) location grid L (for V matrix)."""
        h, w = grid_size_hw
        x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing='xy')
        L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
        return L.astype(jnp.float32)

    def _img_to_patches(self, img):
        """Converts (B, H, W, C) image to (B, P, C) patches."""
        stem = self.stem(img)
        B, H, W, C = stem.shape
        patches = stem.reshape(B, H * W, C)
        return patches

    def _upsample_flow_to_dense(self, flow_pred_grid_units, batch_size):
        """Upsamples (B, 64, 2) flat flow grid to (B, 18, 18, 2) dense flow."""

        # --- FIX ---
        # Input `flow_pred_grid_units` is (B, 64, 2). Reshape to (B, 8, 8, 2).
        h_grid, w_grid = self.grid_size_hw
        flow_pred_map = flow_pred_grid_units.reshape(batch_size, h_grid, w_grid, 2)
        # --- END FIX ---

        flow_pred_pixels = flow_pred_map * self.stride # Use the reshaped map
        h_inner, w_inner = self.inner_hw
        flow_16x16 = jax.image.resize(
            flow_pred_pixels, # This is now (B, 8, 8, 2)
            shape=(batch_size, h_inner, w_inner, 2),
            method='bilinear'
        )
        paddings = ((0, 0), (1, 1), (1, 1), (0, 0))
        flow_18x18 = jnp.pad(flow_16x16, paddings, mode='edge')
        return flow_18x18

    def __call__(self, frame1, frame2, dense_flow_ground_truth):
        batch_size = frame1.shape[0]

        # --- 1. Get Patch Features ---
        f1_patches = self._img_to_patches(frame1)  # (B, 64, C)
        f2_patches = self._img_to_patches(frame2)  # (B, 64, C)

        # --- 2. Get Raw PE ---
        pe = self.pos_encoding()  # (64, C)

        # --- 3. Run V1: Patch Lookup ---
        num_patches = self.grid_size_hw[0] * self.grid_size_hw[1]
        L_broadcast = jnp.broadcast_to(
            self.location_grid.value.reshape(1, num_patches, 2),
            (batch_size, num_patches, 2)
        )

        F_cross, C_cross, A_cross = self.patch_lookup(f1_patches, f2_patches, pe, L_broadcast)

        # --- 4. Run V2.1: Peer Propagation ---
        F_peer, A_peer = self.peer_prop(f1_patches, pe, F_cross, C_cross)

        # --- 5. Blend Flows ---
        w1 = C_cross
        w2 = 1.0 - C_cross
        flow_pred_flat = (w1 * F_cross) + (w2 * F_peer)  # (B, 64, 2)

        # --- 6. Upsample to Dense ---
        h_grid, w_grid = self.grid_size_hw
        flow_pred_map = flow_pred_flat.reshape(batch_size, h_grid, w_grid, 2)
        dense_flow_pred = self._upsample_flow_to_dense(flow_pred_map, batch_size)

        # --- 7. Loss (Dense vs Dense) ---
        flow_loss = jnp.mean(jnp.abs(dense_flow_pred - dense_flow_ground_truth))

        loss_dict = dict(
            flow=flow_loss,
        )

        # --- 8. Create Rich Trace for Logging ---
        # Upsample intermediate flows for visualization
        f_cross_dense = self._upsample_flow_to_dense(F_cross, batch_size)
        f_peer_dense = self._upsample_flow_to_dense(F_peer, batch_size)

        # Upsample confidence map for visualization
        c_cross_map = C_cross.reshape(batch_size, h_grid, w_grid, 1)
        c_cross_dense = jax.image.resize(
            c_cross_map,
            shape=(batch_size, self.inner_hw[0], self.inner_hw[1], 1),
            method='nearest' # Use nearest to see the blocks
        )
        c_cross_dense = jnp.pad(c_cross_dense, ((0,0), (1,1), (1,1), (0,0)), mode='edge')


        trace_dict = dict(
            A_cross=A_cross,    # (B, 64, 64)
            A_peer=A_peer,      # (B, 64, 64)
            C_cross_grid=C_cross, # (B, 64, 1)
            F_cross_dense=f_cross_dense, # (B, 18, 18, 2)
            F_peer_dense=f_peer_dense,   # (B, 18, 18, 2)
            C_cross_dense=c_cross_dense  # (B, 18, 18, 1)
        )

        return dense_flow_pred, dict(
            loss=loss_dict,
            trace=trace_dict
        )