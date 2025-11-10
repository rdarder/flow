import jax
import jax.numpy as jnp
from jax.nn import softmax
from flax import nnx
from flax.linen import max_pool


# --- 1. The "GroupNorm" Stem (V0.12) ---
class Stem(nnx.Module):
    def __init__(self, *, rngs):
        """Initializes our V0.12 'MaxPool + Norm' stem."""
        init = nnx.initializers.lecun_normal()
        self.dw1 = nnx.Conv(
            in_features=3, out_features=24, kernel_size=(3, 3),
            strides=(1, 1),  # Find features
            feature_group_count=3, padding='SAME', use_bias=True,
            kernel_init=init, rngs=rngs
        )
        self.pw1 = nnx.Conv(
            in_features=24, out_features=32, kernel_size=(1, 1),
            use_bias=True, kernel_init=init, rngs=rngs
        )
        self.norm1 = nnx.GroupNorm(
            num_groups=1, num_features=32,  # num_groups=1 == LayerNorm
            use_bias=True, use_scale=True, rngs=rngs
        )

        self.dw2 = nnx.Conv(
            in_features=32, out_features=32, kernel_size=(3, 3),
            strides=(1, 1),  # Find features
            feature_group_count=32, padding='SAME', use_bias=True,
            kernel_init=init, rngs=rngs
        )
        self.pw2 = nnx.Conv(
            in_features=32, out_features=64, kernel_size=(1, 1),
            use_bias=True, kernel_init=init, rngs=rngs
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=1, num_features=64,
            use_bias=True, use_scale=True, rngs=rngs
        )

    def __call__(self, x):
        """Applies the stem logic."""
        x = jax.nn.gelu(self.dw1(x))
        x = self.pw1(x)  # Conv
        x = self.norm1(x)  # <-- NORMALIZE
        x = jax.nn.gelu(x)  # <-- Activate
        x = max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')

        x = jax.nn.gelu(self.dw2(x))
        x = self.pw2(x)  # Conv
        x = self.norm2(x)  # <-- NORMALIZE
        x = jax.nn.gelu(x)  # <-- Activate
        x = max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        return x


class BarebonesFlowModel(nnx.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=64, *, rngs):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size

        # --- 1. The Stem Module ---
        self.stem = Stem(rngs=rngs)

        # --- 2. Our "Barebones" Parameters ---
        # nnx.Param makes them learnable
        self.log_temp = nnx.Param(jnp.log(10.0))
        self.log_w_zero_boost = nnx.Param(jnp.log(0.1))

        # --- 3. Fixed "Data" ---
        # nnx.Variable makes it non-learnable (like a buffer)
        self.L = nnx.Variable(create_location_tensor(self.grid_size))

    def _safe_l2_norm(self, F):
        """(Our gradient-safe norm, now a class method)"""
        norm_sq = jnp.sum(F ** 2, axis=-1, keepdims=True)
        safe_norm = jnp.sqrt(norm_sq + 1e-6)
        return F / safe_norm

    def _get_zero_hint_bias(self):
        """(Our zero-boost logic, now a class method)"""
        # .value gets the array from the nnx.Variable/Param
        L = self.L.value
        L_q = L[None, :, :]
        L_k = L[:, None, :]
        dist_sq = jnp.sum((L_q - L_k) ** 2, axis=-1)
        sigma_sq = 1.0
        B_base = jnp.exp(-dist_sq / (2 * sigma_sq))
        w = jnp.exp(self.log_w_zero_boost.value)
        return 1.0 + (w * B_base)

    def _apply_patch_embed(self, img_batch):
        """(Our patch embed helper, now uses self.stem)"""
        # img_batch is (B, C, H, W)
        x_batched = self.stem(img_batch)  # (B, 64, 8, 8)

        B, H, W, C = x_batched.shape
        F_batch = x_batched.reshape(B, H * W, C)
        return F_batch

    def __call__(self, img1_batch, img2_batch):
        """
        Applies the full model logic for a BATCH.
        (We can build vmap right into the class!)
        """
        # --- 1. Get Feature Vectors ---
        F1 = self._apply_patch_embed(img1_batch)  # (B, 64, 64)
        F2 = self._apply_patch_embed(img2_batch)  # (B, 64, 64)

        # --- 2. Calculate Raw Similarity ---
        F1_norm = self._safe_l2_norm(F1)
        F2_norm = self._safe_l2_norm(F2)
        C_raw = F1_norm @ F2_norm.transpose(0, 2, 1)  # (B, 64, 64)

        # --- 3. Apply Temperature ---
        temp = jnp.exp(self.log_temp.value)
        C_scaled = C_raw * temp

        # --- 4. Apply "Zero-Flow" Gated Boost ---
        B_gated = self._get_zero_hint_bias()  # (64, 64)
        C_biased = C_scaled * B_gated[None, ...]  # Add batch dim

        # --- 5. Final Calculation ---
        C_norm = softmax(C_biased, axis=-1)

        # .value gets the array
        L_batch = jnp.broadcast_to(self.L.value[None, ...], C_norm.shape[:-1] + (2,))

        A = C_norm @ L_batch
        Flow_pred = A - L_batch

        return Flow_pred


def create_location_tensor(grid_size):
    x, y = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
    L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
    return L.astype(jnp.float32)
