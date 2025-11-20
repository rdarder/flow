import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Tuple

# --- 1. Architectures ---

class LearnedPE_Fourier(nnx.Module):
    # ... (Same as before) ...
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        scale = 1.0
        self.B = nnx.Variable(
            jax.random.normal(rngs.params(), (input_dim, hidden_dim // 2)) * scale,
            metadata={'is_state': True}
        )
        
        self.net = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), 
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), 
            nnx.gelu,
            nnx.Linear(hidden_dim, output_dim, rngs=rngs),
        )
        
        self.raw_sigma = nnx.Param(0.0)
        self.raw_scale = nnx.Param(0.0)
        self.raw_bias = nnx.Param(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        projection = x @ self.B.value
        features = jnp.concatenate([jnp.cos(projection), jnp.sin(projection)], axis=-1)
        return self.net(features)

    def get_metric_params(self):
        # Using the tighter sigma range we found successful
        sigma = 1.0 + (5.0 * jax.nn.sigmoid(self.raw_sigma.value))
        scale = 0.5 + (4.5 * jax.nn.sigmoid(self.raw_scale.value))
        bias = -1.0 + (2.0 * jax.nn.sigmoid(self.raw_bias.value))
        return sigma, scale, bias

class SirenLayer(nnx.Module):
    # ... (Same as before) ...
    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False, *, rngs: nnx.Rngs):
        self.w0 = w0
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        limit = np.sqrt(6 / in_features) / w0 if is_first else np.sqrt(6 / in_features) / 30.0
        key = rngs.params()
        self.linear.kernel.value = jax.random.uniform(key, (in_features, out_features), minval=-limit, maxval=limit)

    def __call__(self, x):
        return jnp.sin(self.w0 * self.linear(x))

class LearnedPE_Siren(nnx.Module):
    # ... (Same as before) ...
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            SirenLayer(input_dim, hidden_dim, w0=30.0, is_first=True, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        )
        self.raw_sigma = nnx.Param(0.0)
        self.raw_scale = nnx.Param(0.0)
        self.raw_bias = nnx.Param(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net(x)

    def get_metric_params(self):
        sigma = 1.0 + (5.0 * jax.nn.sigmoid(self.raw_sigma.value))
        scale = 0.5 + (4.5 * jax.nn.sigmoid(self.raw_scale.value))
        bias = -1.0 + (2.0 * jax.nn.sigmoid(self.raw_bias.value))
        return sigma, scale, bias

# --- 2. The Decoder ---

class Decoder(nnx.Module):
    """ Standard MLP Decoder (GELU) """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net(x)

class Decoder_Siren(nnx.Module):
    """
    SIREN Decoder.
    Symmetric with the encoder. Uses sine activations to unwrap the manifold.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            # First layer w0=1.0 is standard for decoders unless we expect crazy high freqs input
            SirenLayer(input_dim, hidden_dim, w0=1.0, is_first=True, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net(x)

# --- 3. NEW: The Zoom Operator ---
class ZoomNetwork(nnx.Module):
    # ... (Same as before) ...
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        # Try a single Linear layer first (Matrix Multiplication)
        # If this works, it means the "Shift" property is perfectly learned.
        self.net = nnx.Linear(dim, dim, rngs=rngs)
        
        # Alternative: Tiny MLP
        # self.net = nnx.Sequential(
        #    nnx.Linear(dim, dim*2, rngs=rngs),
        #    nnx.gelu,
        #    nnx.Linear(dim*2, dim, rngs=rngs)
        # )

    def __call__(self, pe: jnp.ndarray) -> jnp.ndarray:
        return self.net(pe)


# --- 4. Main Container ---

class ModelContainer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # Using SIREN as it performed best on metric correlation
        self.encoder = LearnedPE_Siren(input_dim=2, hidden_dim=64, output_dim=16, rngs=rngs)
        
        # --- SWITCHED TO SIREN DECODER ---
        self.decoder = Decoder(input_dim=16, hidden_dim=64, output_dim=2, rngs=rngs)
        
        self.zoom = ZoomNetwork(dim=16, rngs=rngs)

# --- 5. Training Logic ---

@nnx.jit
def loss_fn(model: ModelContainer, x1, x2, alpha, x_zoom_src):
    enc = model.encoder
    dec = model.decoder
    zoom = model.zoom
    
    # --- Forward Pass ---
    p1 = enc(x1)
    p2 = enc(x2)
    
    # --- Constraint 1: Reconstruction ---
    x1_rec = dec(p1)
    loss_recon = jnp.mean(jnp.square(x1_rec - x1))
    
    # --- Constraint 2: Linearity (Interpolation) ---
    x_mix_target = alpha * x1 + (1 - alpha) * x2
    p_mix = alpha * p1 + (1 - alpha) * p2
    x_mix_pred = dec(p_mix)
    
    loss_linear = jnp.mean(jnp.square(x_mix_pred - x_mix_target))
    
    # --- Constraint 3: Metric (Dot Product ~ Gaussian Kernel) ---
    dist = jnp.linalg.norm(x1 - x2, axis=-1)
    sigma, scale, bias = enc.get_metric_params()
    target_sim = scale * jnp.exp(-(dist**2) / (2 * sigma**2)) + bias
    actual_dot = jnp.sum(p1 * p2, axis=-1)
    loss_metric = jnp.mean(jnp.square(actual_dot - target_sim))
    
    # --- Constraint 4: Zoom (x -> 2x in Latent Space) ---
    # We generate a random x, encode it, zoom it, and compare to encoding of 2x
    p_src = enc(x_zoom_src)
    p_target = enc(x_zoom_src * 2.0) # The target is the encoding of 2x
    p_zoomed = zoom(p_src)
    
    loss_zoom = jnp.mean(jnp.square(p_zoomed - p_target))
    
    # Weighted Sum (Give Zoom good weight to force structure)
    total_loss = loss_recon + loss_linear + (5.0 * loss_metric) + (2.0 * loss_zoom)
    
    return total_loss, (loss_recon, loss_linear, loss_metric, loss_zoom, sigma, scale, bias)

@nnx.jit
def train_step(
    model: ModelContainer,
    optimizer: nnx.Optimizer,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    alpha: jnp.ndarray,
    x_zoom_src: jnp.ndarray
):
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x1, x2, alpha, x_zoom_src)
    optimizer.update(model, grads)
    return loss, aux

# --- 6. Main Execution ---

def main():
    config = {
        'batch_size': 1024,
        'min_val': -20.0, 'max_val': 20.0,
        'lr': 1e-4, # SIREN prefers lower LR
        'steps': 40000
    }
    
    key = jax.random.PRNGKey(0)
    model = ModelContainer(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(config['lr']), wrt=nnx.Param)
    
    print("Training Learned PE (SIREN Encoder + SIREN Decoder)...")
    for i in range(config['steps']):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        x1 = jax.random.uniform(k1, (config['batch_size'], 2), minval=config['min_val'], maxval=config['max_val'])
        x2 = jax.random.uniform(k2, (config['batch_size'], 2), minval=config['min_val'], maxval=config['max_val'])
        alpha = jax.random.uniform(k3, (config['batch_size'], 1))
        
        # Zoom samples
        x_zoom_src = jax.random.uniform(k4, (config['batch_size'], 2), minval=config['min_val']/2, maxval=config['max_val']/2)
        
        loss, (l_r, l_l, l_m, l_z, sig, sc, b) = train_step(model, optimizer, x1, x2, alpha, x_zoom_src)
        
        if i % 1000 == 0:
            print(f"Step {i}: Total={loss:.4f} | Recon={l_r:.4f} | Linear={l_l:.4f} | Metric={l_m:.4f} | Zoom={l_z:.4f}")
            print(f"          Params: Sigma={sig:.2f}, Scale={sc:.2f}, Bias={b:.2f}")
            
    print("Done.")

if __name__ == "__main__":
    main()