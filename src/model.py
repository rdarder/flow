import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softmax
import numpy as np


def create_location_tensor(grid_size):
    x, y = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
    L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
    return L.astype(jnp.float32)


def init_params(key, img_size=32, patch_size=4, embed_dim=64):
    """
    Initializes the 4-layer "Gentle Ramp" stem,
    using your *wide* (24-channel) first layer.
    """
    keys = jr.split(key, 10)

    # --- YOUR NEW CHANNEL SIZES ---
    dw1_out_channels = 24  # 3 groups * 8 channels/group
    intermediate_dim = 32

    # --- END ---

    # --- Helper: Kaiming/He stddev calculation ---
    def kaiming_std(key, shape):
        fan_in = np.prod(shape[1:])  # I * H * W
        stddev = jnp.sqrt(2.0 / fan_in)
        return jr.truncated_normal(key, -2 * stddev, 2 * stddev, shape)

    # --- Block 1: 32x32 -> 16x16 (in=3, mid=24, out=32) ---
    # dw1: (O=24, I=1, H=3, W=3). fan_in = 9
    dw1_W = kaiming_std(keys[0], (dw1_out_channels, 1, 3, 3))
    dw1_b = jnp.zeros(dw1_out_channels)

    # pw1: (O=32, I=24, H=1, W=1). fan_in = 24
    pw1_W = kaiming_std(keys[1], (intermediate_dim, dw1_out_channels, 1, 1))
    pw1_b = jnp.zeros(intermediate_dim)

    # --- Block 2: 16x16 -> 8x8 (in=32, mid=32, out=64) ---
    # (This block is unchanged)
    # dw2: (O=32, I=1, H=3, W=3). fan_in = 9
    dw2_W = kaiming_std(keys[2], (intermediate_dim, 1, 3, 3))
    dw2_b = jnp.zeros(intermediate_dim)

    # pw2: (O=64, I=32, H=1, W=1). fan_in = 32
    pw2_W = kaiming_std(keys[3], (embed_dim, intermediate_dim, 1, 1))
    pw2_b = jnp.zeros(embed_dim)

    # --- Other Params (Unchanged) ---
    log_temp = jnp.log(100.0)
    log_w_zero_boost = jnp.log(0.1)

    params = {
        'stem': {
            'dw1_W': dw1_W, 'dw1_b': dw1_b,
            'pw1_W': pw1_W, 'pw1_b': pw1_b,
            'dw2_W': dw2_W, 'dw2_b': dw2_b,
            'pw2_W': pw2_W, 'pw2_b': pw2_b,
        },
        'log_temp': log_temp,
        'log_w_zero_boost': log_w_zero_boost
    }

    final_patch_size = 4
    final_embed_dim = 64

    return params, (final_patch_size, final_embed_dim)


def safe_l2_norm(F):
    norm_sq = jnp.sum(F ** 2, axis=-1, keepdims=True)
    safe_norm = jnp.sqrt(norm_sq + 1e-6)
    return F / safe_norm


# (This is your standalone function, as requested)
def apply_patch_embed(p_stem, img):
    """
    Applies the 4-layer "Gentle Ramp" stem (V0.10)
    with your *wide* (24-channel) first layer.
    """
    # img is (C, H, W), e.g., (3, 32, 32)
    x = img[None, ...]  # Add batch dim -> (1, 3, 32, 32)

    dn = ('NCHW', 'OIHW', 'NCHW')

    # --- Block 1 (Stride 2) ---
    # dw1: (1, 3, 32, 32) -> (1, 24, 16, 16)
    x = jax.lax.conv_general_dilated(
        x, p_stem['dw1_W'], (2, 2), 'SAME',  # Stride 2
        feature_group_count=3,  # <-- C_in=3, O=24 -> 8 channels/group
        dimension_numbers=dn
    ) + p_stem['dw1_b'].reshape(1, -1, 1, 1)
    x = jax.nn.gelu(x)

    # pw1: (1, 24, 16, 16) -> (1, 32, 16, 16)
    x = jax.lax.conv_general_dilated(
        x, p_stem['pw1_W'], (1, 1), 'SAME',  # Stride 1
        dimension_numbers=dn
    ) + p_stem['pw1_b'].reshape(1, -1, 1, 1)
    x = jax.nn.gelu(x)

    # --- Block 2 (Stride 2) ---
    # (This block is unchanged)
    # dw2: (1, 32, 16, 16) -> (1, 32, 8, 8)
    x = jax.lax.conv_general_dilated(
        x, p_stem['dw2_W'], (2, 2), 'SAME',  # Stride 2
        feature_group_count=32,  # C_in=32, groups=32
        dimension_numbers=dn
    ) + p_stem['dw2_b'].reshape(1, -1, 1, 1)
    x = jax.nn.gelu(x)

    # pw2: (1, 32, 8, 8) -> (1, 64, 8, 8)
    x = jax.lax.conv_general_dilated(
        x, p_stem['pw2_W'], (1, 1), 'SAME',  # Stride 1
        dimension_numbers=dn
    ) + p_stem['pw2_b'].reshape(1, -1, 1, 1)
    x = jax.nn.gelu(x)

    # --- Final Reshape ---
    x = x.squeeze(0)  # (64, 8, 8)
    x_permuted = x.transpose(1, 2, 0)  # (8, 8, 64)
    P, C = x_permuted.shape[0] * x_permuted.shape[1], x_permuted.shape[2]
    return x_permuted.reshape(P, C)


def safe_l2_norm(F):
    norm_sq = jnp.sum(F ** 2, axis=-1, keepdims=True)
    safe_norm = jnp.sqrt(norm_sq + 1e-6)
    return F / safe_norm


def apply_model(params, img1, img2, L, patch_size, embed_dim):
    """
    JAX 'apply_model' for a SINGLE sample (C, H, W),
    using the V0.4 "Gentle Ramp" stem *with biases*.
    """

    # --- 1. Get Feature Vectors ---
    F1 = apply_patch_embed(params['stem'], img1)  # (64, 64)
    F2 = apply_patch_embed(params['stem'], img2)  # (64, 64)

    # --- 2. Calculate Raw Similarity ---
    F1_norm = safe_l2_norm(F1)
    F2_norm = safe_l2_norm(F2)
    C_raw = F1_norm @ F2_norm.T  # (64, 64)

    # --- 3. Apply Temperature ---
    temp = jnp.exp(params['log_temp'])
    C_scaled = C_raw * temp

    # --- 4. Apply "Zero-Flow" Gated Boost ---
    B_gated = _get_zero_hint_bias(L, params)
    C_biased = C_scaled * B_gated

    # --- 5. Final Calculation (as before) ---
    C_norm = softmax(C_biased, axis=-1)
    A = C_norm @ L
    Flow_pred = A - L

    # --- 6. Return both values ---
    # (This is for our V0.8 variance loss)
    return Flow_pred, F1


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
