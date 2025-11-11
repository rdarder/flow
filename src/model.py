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
        self.zero_boost = nnx.Param(0.1)

        # --- 3. Fixed "Data" ---
        # nnx.Variable makes it non-learnable (like a buffer)
        self.location_grid = nnx.Variable(self._create_location_tensor(self.grid_size))

    def _get_zero_hint_bias(self):
        """(Our zero-boost logic, now a class method)"""
        # .value gets the array from the nnx.Variable/Param
        L = self.location_grid.value
        L_q = L[None, :, :]
        L_k = L[:, None, :]
        dist_sq = jnp.sum((L_q - L_k) ** 2, axis=-1)
        sigma_sq = 1.0
        return jnp.exp(-dist_sq / (2 * sigma_sq))[None, ...]

    def _img_to_patches(self, img):
        """(Our patch embed helper, now uses self.stem)"""
        # img is (B, C, H, W)
        stem = self.stem(img)  # (B, 64, 8, 8)
        B, H, W, C = stem.shape
        patches = stem.reshape(B, H * W, C)
        return patches

    def _create_location_tensor(self, grid_size):
        x, y = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
        L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
        return L.astype(jnp.float32)

    def compute_decorrelation_loss(self, features):
        """
        Calculates the Barlow Twins/VICReg-style loss for a
        batch of features (B, P, C).
        """
        B, P, C = features.shape

        features_mean = features.mean(axis=(0, 1))  # (C,)
        features_centered = features - features_mean

        features_flat = features_centered.reshape(-1, C)  # (B*P, C)
        num_features = features_flat.shape[0]
        covariance_matrix = (features_flat.T @ features_flat) / (num_features - 1)  # (C, C)
        loss_variance = jnp.mean(jax.nn.relu(1.0 - jnp.diag(covariance_matrix)))

        covariance_no_diag = covariance_matrix.at[jnp.diag_indices(C)].set(0)
        loss_covariance = jnp.mean(covariance_no_diag ** 2)

        return loss_variance, loss_covariance

    def __call__(self, frame1, frame2, flow_ground_truth):
        """
        Runs the full forward pass and computes all losses
        and metrics, as you requested.
        """

        # --- 1. Get Features ---
        f1_patches = self._img_to_patches(frame1)
        f2_patches = self._img_to_patches(frame2)

        # --- 2. Calculate Flow ---
        # (This is the "no-norm" V1.0 logic)
        patch_similarities = f1_patches @ f2_patches.transpose(0, 2, 1)

        scaled_similarities = patch_similarities / (jnp.sqrt(self.embed_dim))

        zero_flow_bias = self._get_zero_hint_bias()
        alpha = self.zero_boost.value
        zero_flow_biased_similarities = alpha * zero_flow_bias + (1 - alpha) * scaled_similarities

        sharpened_patch_similarities = softmax(zero_flow_biased_similarities, axis=-1)

        location_grid = jnp.broadcast_to(self.location_grid.value[None, ...],
                                         sharpened_patch_similarities.shape[:-1] + (2,))
        target_patch_location = sharpened_patch_similarities @ location_grid
        flow_prediction = target_patch_location - location_grid

        flow_loss = jnp.mean(jnp.abs(flow_prediction - flow_ground_truth))

        loss_var_F1, loss_cov_F1 = self.compute_decorrelation_loss(f1_patches)
        loss_var_F2, loss_cov_F2 = self.compute_decorrelation_loss(f2_patches)
        total_var_loss = (loss_var_F1 + loss_var_F2) / 2.0
        total_cov_loss = (loss_cov_F1 + loss_cov_F2) / 2.0

        loss_dict = dict(
            flow=flow_loss,
            variance=total_var_loss,
            covariance=total_cov_loss,
        )
        debug = dict(
            f1_magnitude=jnp.mean(jnp.abs(f1_patches)),
            c_magnitude=jnp.mean(jnp.abs(scaled_similarities)),
            softmax_confidence=jnp.mean(jnp.max(sharpened_patch_similarities, axis=-1)),
        )
        return flow_prediction, dict(loss=loss_dict, debug=debug)

    def _safe_l2_norm(self, features):
        """Our gradient-safe L2 norm."""
        norm_sq = jnp.sum(features ** 2, axis=-1, keepdims=True)
        safe_norm = jnp.sqrt(norm_sq + 1e-6)
        return features / safe_norm
