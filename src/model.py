import jax
import jax.numpy as jnp
from jax.nn import softmax
from flax import nnx
from flax.linen import max_pool, avg_pool


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
        """
        Calculates geometry for the new single-block stem.
        Returns:
          - (H_out, W_out): The output grid size.
          - stride: The total stride (pixels per grid step).
          - inner_hw: The H,W of the map *before* padding (for upsampling)
        """
        h, w = img_size_hw

        # Conv (k=3, s=1, valid): Eats 1 pixel border
        h_inner, w_inner = h - 2, w - 2  # e.g., 18 -> 16

        # Pool (k=2, s=2)
        h_out, w_out = h_inner // 2, w_inner // 2  # e.g., 16 -> 8

        # Total stride from input to output patch
        stride = float(img_size_hw[0] / h_out)  # e.g., 18 / 8 = 2.25 (This is complex)
        # Let's use the simpler stride:
        stride = 2.0  # The stride of the MaxPool is the dominant factor

        return (int(h_out), int(w_out)), stride, (int(h_inner), int(w_inner))

    def __call__(self, x):
        """Applies the single-block stem logic."""
        x = self.dw1(x)
        # x = jax.nn.gelu(x)
        x = self.pw1(x)
        x = jax.nn.gelu(x)
        # (18,18) -> Conv(valid) -> (16,16)
        # (16,16) -> MaxPool(2,2) -> (8,8)
        x = avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = self.norm1(x)
        return x


class BarebonesFlowModel(nnx.Module):
    def __init__(self, img_size_hw: tuple[int, int], embed_dim: int, *, rngs):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size_hw = img_size_hw

        self.stem = Stem(embed_dim=embed_dim, dw_patterns=12, rngs=rngs)
        self.zero_boost_radius = nnx.Param(10.0)
        self.attn_temperature = nnx.Param(1.0)

        self.grid_size_hw, self.stride, self.inner_hw = self.stem.geometry(img_size_hw)
        self.location_grid = nnx.Variable(self._create_location_tensor(self.grid_size_hw))

    def _img_to_patches(self, img):
        """Converts (B, H, W, C) image to (B, P, C) patches."""
        stem = self.stem(img)
        B, H, W, C = stem.shape
        patches = stem.reshape(B, H * W, C)
        return patches

    def _create_location_tensor(self, grid_size_hw: tuple[int, int]) -> jnp.ndarray:
        """Creates the (H*W, 2) location grid L (indices)."""
        h, w = grid_size_hw
        x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h), indexing='xy')
        L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
        return L.astype(jnp.float32)

    def _zero_flow_bias(self):
        """Calculates the zero-flow bias based on grid distance."""
        L = self.location_grid.value
        L_q = L[None, :, :]
        L_k = L[:, None, :]
        dist_sq = jnp.sum((L_q - L_k) ** 2, axis=-1)
        zero = jnp.exp(-dist_sq / self.zero_boost_radius)[None, ...]
        center = jnp.mean(zero, axis=-1)
        return (zero - center) + 1.0

    def compute_decorrelation_loss(self, features):
        """Calculates Barlow Twins/VICReg-style loss."""
        B, P, C = features.shape
        features_mean = features.mean(axis=(0, 1))
        features_centered = features - features_mean
        features_flat = features_centered.reshape(-1, C)
        num_features = features_flat.shape[0]
        cov = (features_flat.T @ features_flat) / (num_features - 1)
        loss_var = jnp.mean(jax.nn.relu(1.0 - jnp.diag(cov)))
        loss_cov = jnp.mean(cov.at[jnp.diag_indices(C)].set(0) ** 2)
        return loss_var, loss_cov

    def _upsample_flow_to_dense(self, flow_pred_grid_units, batch_size):
        """
        Upsamples the (B, 8, 8, 2) flow map to (B, 18, 18, 2).
        1. Converts units: Grid Units -> Pixels (multiply by stride).
        2. Interpolates: 8x8 -> 16x16
        3. Pads: 16x16 -> 18x18
        """
        # 1. Convert Grid Units -> Pixel Units
        # A flow of "1" in the 8x8 grid means 2 pixels in the 16x16 map
        flow_pred_pixels = flow_pred_grid_units * self.stride

        # 2. Interpolate 8x8 -> 16x16
        h_inner, w_inner = self.inner_hw
        flow_16x16 = jax.image.resize(
            flow_pred_pixels,
            shape=(batch_size, h_inner, w_inner, 2),
            method='bilinear'
        )

        # 3. Pad 16x16 -> 18x18
        # We need to pad 1 pixel on all 4 sides (axis 1 and 2)
        paddings = ((0, 0), (1, 1), (1, 1), (0, 0))
        flow_18x18 = jnp.pad(flow_16x16, paddings, mode='edge')

        return flow_18x18

    def __call__(self, frame1, frame2, dense_flow_ground_truth):
        """
        Expects (B, 18, 18, 3) inputs and (B, 18, 18, 2) dense GT.
        Returns dense (B, 18, 18, 2) flow prediction.
        """
        batch_size = frame1.shape[0]
        f1_patches = self._img_to_patches(frame1)
        f2_patches = self._img_to_patches(frame2)

        # --- Calculate Low-Res Flow ---
        patch_similarities = f1_patches @ f2_patches.transpose(0, 2, 1)
        # scaled_similarities = patch_similarities / jnp.sqrt(self.embed_dim)

        zero_bias = self._zero_flow_bias()
        biased_sim = patch_similarities * zero_bias

        attn = softmax(biased_sim * self.attn_temperature, axis=-1)

        h_grid, w_grid = self.grid_size_hw
        num_patches = h_grid * w_grid

        L_broadcast = jnp.broadcast_to(
            self.location_grid.value.reshape(1, num_patches, 2),
            (batch_size, num_patches, 2)
        )

        # (B, 64, 2) - Flow in "Grid Units"
        flow_pred_flat = (attn @ L_broadcast) - L_broadcast

        # Reshape to map: (B, 8, 8, 2)
        flow_pred_map = flow_pred_flat.reshape(batch_size, h_grid, w_grid, 2)

        # --- Upsample to Dense (B, 18, 18, 2) ---
        dense_flow_pred = self._upsample_flow_to_dense(flow_pred_map, batch_size)

        # --- Loss (Dense vs Dense) ---
        flow_loss = jnp.mean(jnp.abs(dense_flow_pred - dense_flow_ground_truth))

        loss_var_1, loss_cov_1 = self.compute_decorrelation_loss(f1_patches)
        loss_var_2, loss_cov_2 = self.compute_decorrelation_loss(f2_patches)

        loss_dict = dict(
            flow=flow_loss,
            variance=(loss_var_1 + loss_var_2) / 2,
            covariance=(loss_cov_1 + loss_cov_2) / 2
        )

        # Return dense flow
        return dense_flow_pred, dict(
            loss=loss_dict,
            trace=dict(attn=attn, f1_patches=f1_patches, f2_patches=f2_patches)
        )
