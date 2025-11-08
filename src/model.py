import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softmax
import numpy as np


def create_location_tensor(grid_size):
    x, y = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
    L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
    return L.astype(jnp.float32)


def init_params(key, img_size=32, patch_size=4, embed_dim=16):  # <-- embed_dim is 16
    """
    Initializes your 6-layer 'V0.7' stem.
    - Bias-free
    - Kaiming init
    - Wide dw1 (24 channels)
    """
    keys = jr.split(key, 8)  # 6 conv W, log_temp, log_w

    # --- Helper: Kaiming/He stddev calculation ---
    def kaiming_std(key, shape):
        # (O, I, H, W)
        fan_in = np.prod(shape[1:])  # I * H * W
        stddev = jnp.sqrt(2.0 / fan_in)
        return jr.truncated_normal(key, -2 * stddev, 2 * stddev, shape)

    # --- Block 1: (32x32) in=3, mid=24, out=16 ---
    dw1_out = 24  # Your 8x multiplier
    # dw1: (O=24, I=1, H=3, W=3). fan_in = 1*3*3 = 9
    dw1_W = kaiming_std(keys[0], (dw1_out, 1, 3, 3))

    # pw1: (O=16, I=24, H=1, W=1). fan_in = 24
    pw1_W = kaiming_std(keys[1], (16, dw1_out, 1, 1))

    # --- Block 2: (32x32 -> 16x16) in=16, out=16 ---
    # dw2: (O=16, I=1, H=3, W=3). fan_in = 9
    dw2_W = kaiming_std(keys[2], (16, 1, 3, 3))

    # pw2: (O=16, I=16, H=1, W=1). fan_in = 16
    pw2_W = kaiming_std(keys[3], (16, 16, 1, 1))

    # --- Block 3: (16x16 -> 8x8) in=16, out=16 ---
    # dw3: (O=16, I=1, H=3, W=3). fan_in = 9
    dw3_W = kaiming_std(keys[4], (16, 1, 3, 3))

    # pw3: (O=16, I=16, H=1, W=1). fan_in = 16
    pw3_W = kaiming_std(keys[5], (16, 16, 1, 1))

    # --- Other Params ---
    log_temp = jnp.log(100.0)
    log_w_zero_boost = jnp.log(0.1)

    params = {
        'stem': {
            'dw1_W': dw1_W,
            'pw1_W': pw1_W,
            'dw2_W': dw2_W,
            'pw2_W': pw2_W,
            'dw3_W': dw3_W,
            'pw3_W': pw3_W,
        },
        'log_temp': log_temp,
        'log_w_zero_boost': log_w_zero_boost
    }

    final_patch_size = 4  # 32 -> 16 -> 8
    final_embed_dim = 16  # Final output dim

    return params, (final_patch_size, final_embed_dim)


# --- (Keep _get_zero_hint_bias) ---
# --- (Keep safe_l2_norm) ---

# --- (Keep _get_zero_hint_bias as it is) ---

# --- Helper: V0.7 "Deep & Wide" Embedder ---
def _apply_patch_embed(p_stem, img):
    # img is (C, H, W), e.g., (3, 32, 32)
    x = img[None, ...]  # Add batch dim -> (1, 3, 32, 32)
    dn = ('NCHW', 'OIHW', 'NCHW')  # Dimension conventions

    # --- Block 1 (stride 1, in=3, mid=24, out=16) ---
    # dw1: (1, 3, 32, 32) -> (1, 24, 32, 32)
    x = jax.lax.conv_general_dilated(
        x, p_stem['dw1_W'], (1, 1), 'SAME',  # Stride 1
        feature_group_count=3,  # C_in=3, groups=3
        dimension_numbers=dn
    )
    x = jax.nn.gelu(x)

    # pw1: (1, 24, 32, 32) -> (1, 16, 32, 32)
    x = jax.lax.conv_general_dilated(
        x, p_stem['pw1_W'], (1, 1), 'SAME',  # Stride 1
        dimension_numbers=dn
    )
    x = jax.nn.gelu(x)

    # --- Block 2 (stride 2, in=16, out=16) ---
    # dw2: (1, 16, 32, 32) -> (1, 16, 16, 16)
    x = jax.lax.conv_general_dilated(
        x, p_stem['dw2_W'], (2, 2), 'SAME',  # Stride 2
        feature_group_count=16,  # C_in=16, groups=16
        dimension_numbers=dn
    )
    x = jax.nn.gelu(x)

    # pw2: (1, 16, 16, 16) -> (1, 16, 16, 16)
    x = jax.lax.conv_general_dilated(
        x, p_stem['pw2_W'], (1, 1), 'SAME',  # Stride 1
        dimension_numbers=dn
    )
    x = jax.nn.gelu(x)

    # --- Block 3 (stride 2, in=16, out=16) ---
    # dw3: (1, 16, 16, 16) -> (1, 16, 8, 8)
    x = jax.lax.conv_general_dilated(
        x, p_stem['dw3_W'], (2, 2), 'SAME',  # Stride 2
        feature_group_count=16,  # C_in=16, groups=16
        dimension_numbers=dn
    )
    x = jax.nn.gelu(x)

    # pw3: (1, 16, 8, 8) -> (1, 16, 8, 8)
    x = jax.lax.conv_general_dilated(
        x, p_stem['pw3_W'], (1, 1), 'SAME',  # Stride 1
        dimension_numbers=dn
    )
    x = jax.nn.gelu(x)  # Final (1, 16, 8, 8)

    # --- Final Reshape ---
    x = x.squeeze(0)  # (16, 8, 8)
    # (C, H, W) -> (H, W, C) -> (P, C)
    x_permuted = x.transpose(1, 2, 0)  # (8, 8, 16)
    P, C = x_permuted.shape[0] * x_permuted.shape[1], x_permuted.shape[2]
    return x_permuted.reshape(P, C)

def safe_l2_norm(F):
    norm_sq = jnp.sum(F ** 2, axis=-1, keepdims=True)
    safe_norm = jnp.sqrt(norm_sq + 1e-6)
    return F / safe_norm

def apply_model(params, img1, img2, L, patch_size, embed_dim):
    """
    JAX 'apply_model' for a SINGLE sample (C, H, W),
    using the bias-free 6-layer (V0.7) stem.
    """

    # --- 1. Get Feature Vectors ---
    F1 = _apply_patch_embed(params['stem'], img1)  # (64, 16)
    F2 = _apply_patch_embed(params['stem'], img2)  # (64, 16)

    # --- 2. Calculate Raw Similarity ---
    F1_norm = safe_l2_norm(F1)
    F2_norm = safe_l2_norm(F2)
    C_raw = F1_norm @ F2_norm.T  # (64, 64)

    # --- 3. Apply Temperature ---
    temp = jnp.exp(params['log_temp'])
    C_scaled = C_raw * temp

    # --- 4. Apply "Zero-Flow" Gated Boost ---
    # temporarily removed as log_temp makes this unnecessary.
    # B_gated = _get_zero_hint_bias(L, params)
    # C_biased = C_scaled * B_gated

    # --- 5. Final Calculation (as before) ---
    C_norm = softmax(C_scaled, axis=-1)
    A = C_norm @ L
    Flow_pred = A - L

    return Flow_pred


# TODO: rather than a zero hint bias, this is a "closer patch" bias
# doesn't necessarily need to couple with the hierarchical hint.
def _get_zero_hint_bias(L, params):
    L_q = L[None, :, :]
    L_k = L[:, None, :]
    dist_sq = jnp.sum((L_q - L_k) ** 2, axis=-1)
    sigma_sq = 1.0
    B_base = jnp.exp(-dist_sq / (2 * sigma_sq))
    w = jnp.exp(params['log_w_zero_boost'])
    B_gated = 1.0 + (w * B_base)
    return B_gated
