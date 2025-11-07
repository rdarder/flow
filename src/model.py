import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softmax # We'll need this

# --- 1. The Location Tensor (Fixed Data) ---

def create_location_tensor(grid_size):
    """
    Creates a (P, 2) tensor of (x, y) coordinates.
    e.g., for grid_size=2, returns:
    [[0, 0],
     [1, 0],
     [0, 1],
     [1, 1]]
    """
    x, y = jnp.meshgrid(
        jnp.arange(grid_size), 
        jnp.arange(grid_size)
    )
    # Stack them and reshape to (P, 2)
    # Note: JAX's meshgrid is 'xy' indexing by default
    L = jnp.stack([x, y], axis=-1).reshape(-1, 2)
    return L

# --- 2. The "Model" (Parameters & Apply Function) ---

def init_params(key, img_size=32, patch_size=4, embed_dim=64):
    """
    Initializes our model's weights in a nested dict.
    We are using a "V0" single-conv patch embedder.
    JAX conv kernels are (H, W, In, Out).
    """
    grid_size = img_size // patch_size
    
    key, subkey = jr.split(key)
    
    # We use kaiming/lecun normal initialization
    stddev = 1.0 / jnp.sqrt(patch_size * patch_size * 3)
    
    patch_embed_W = jr.truncated_normal(
        subkey, 
        lower=-2*stddev, 
        upper=2*stddev, 
        shape=(patch_size, patch_size, 3, embed_dim)
    )
    patch_embed_b = jnp.zeros(embed_dim)
    
    params = {
        'patch_embed': {
            'W': patch_embed_W,
            'b': patch_embed_b
        }
        # Note: 'sigma' and 'w' from our V1 doc aren't
        # needed for this V0, since we cut all hints.
    }
    return params

def apply_model(params, img1, img2, L):
    """
    The "barebones" V0 model apply function.
    This function is for a SINGLE sample.
    We will use jax.vmap to batch it later.
    
    Args:
        params: The dict of weights from init_params.
        img1, img2: (H, W, C) jnp.array, e.g., (32, 32, 3).
        L: The (P, 2) location tensor, e.g., (64, 2).
    """
    
    # --- 1. Feature Extraction (The "Stem") ---
    # We'll write a simple "apply_patch_embed" helper
    
    def _apply_patch_embed(p_embed, img):
        # JAX's simple conv primitive. Stride is a tuple.
        # 'VALID' means no padding, which is what we want for
        # a strided patch embedder.
        x = jax.lax.conv(
            img, 
            p_embed['W'], 
            (patch_size, patch_size), # strides
            'VALID' # padding
        ) + p_embed['b']
        
        # Output shape is (grid_size, grid_size, embed_dim)
        # e.g., (8, 8, 64)
        
        # Flatten to (P, C) for our matrix math
        P, C = x.shape[0] * x.shape[1], x.shape[2]
        return x.reshape(P, C)

    F1 = _apply_patch_embed(params['patch_embed'], img1)
    F2 = _apply_patch_embed(params['patch_embed'], img2)
    # F1 and F2 are shape (64, 64)
    
    # --- 2. "Barebones" Attention (Your Design) ---
    
    embed_dim = F1.shape[-1]
    
    # L2-normalize for stability (as we discussed)
    F1_norm = F1 / (jnp.linalg.norm(F1, axis=-1, keepdims=True) + 1e-6)
    F2_norm = F2 / (jnp.linalg.norm(F2, axis=-1, keepdims=True) + 1e-6)
    
    # a. Correlation (C = F1 @ F2.T)
    C_raw = F1_norm @ F2_norm.T # (64, 64) @ (64, 64) -> (64, 64)
    
    # b. Scale (Transformer-style)
    C_scaled = C_raw / jnp.sqrt(embed_dim)
    
    # c. Softmax (The "soft argmax" selector)
    C_norm = softmax(C_scaled, axis=-1)
    
    # d. Weighted Position (A = C_norm @ L)
    A = C_norm @ L # (64, 64) @ (64, 2) -> (64, 2)
    
    # e. Calculate Flow (Flow = A - L)
    Flow_pred = A - L # (64, 2)
    
    return Flow_pred
