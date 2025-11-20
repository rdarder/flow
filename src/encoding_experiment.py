import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Tuple

# --- 1. Architectures ---

class LearnedPE_Fourier(nnx.Module):
    """ The standard 'Fourier Features + ReLU MLP' architecture. """
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
        sigma = 1.0 + (5.0 * jax.nn.sigmoid(self.raw_sigma.value))
        scale = 0.5 + (4.5 * jax.nn.sigmoid(self.raw_scale.value))
        bias = -1.0 + (2.0 * jax.nn.sigmoid(self.raw_bias.value))
        return sigma, scale, bias

class SirenLayer(nnx.Module):
    """ A single layer of a SIREN network (Linear + Sin). """
    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False, *, rngs: nnx.Rngs):
        self.w0 = w0
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        
        # SIREN initialization is specific
        limit = np.sqrt(6 / in_features) / w0 if is_first else np.sqrt(6 / in_features) / 30.0
        # We perform the init manually on the kernel
        key = rngs.params()
        self.linear.kernel.value = jax.random.uniform(key, (in_features, out_features), minval=-limit, maxval=limit)

    def __call__(self, x):
        return jnp.sin(self.w0 * self.linear(x))

class LearnedPE_Siren(nnx.Module):
    """
    SIREN Architecture: The weights *learn* the frequencies.
    No fixed Fourier basis.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        # w0=30 is standard for SIREN to encourage high-frequency learning
        self.net = nnx.Sequential(
            SirenLayer(input_dim, hidden_dim, w0=30.0, is_first=True, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            nnx.Linear(hidden_dim, output_dim, rngs=rngs) # Last layer is usually linear
        )
        
        # Metric params (Same as Fourier model)
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

# --- 2. The Decoder (Standard MLP) ---

class Decoder(nnx.Module):
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

# --- 3. Main Container ---

class ModelContainer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # SWITCH HERE: Choose Fourier or Siren
        # self.encoder = LearnedPE_Fourier(input_dim=2, hidden_dim=128, output_dim=16, rngs=rngs)
        self.encoder = LearnedPE_Siren(input_dim=2, hidden_dim=64, output_dim=8, rngs=rngs)
        
        self.decoder = Decoder(input_dim=8, hidden_dim=32, output_dim=2, rngs=rngs)

# --- 4. Training Logic (Unchanged) ---
# ... (Same loss_fn and train_step as before) ...
@nnx.jit
def loss_fn(model: ModelContainer, x1, x2, alpha):
    enc = model.encoder
    dec = model.decoder
    
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
    
    # --- Constraint 3: Bounded Adaptive Metric ---
    dist = jnp.linalg.norm(x1 - x2, axis=-1)
    
    sigma, scale, bias = enc.get_metric_params()
    
    target_sim = scale * jnp.exp(-(dist**2) / (2 * sigma**2)) + bias
    
    actual_dot = jnp.sum(p1 * p2, axis=-1)
    
    loss_metric = jnp.mean(jnp.square(actual_dot - target_sim))
    
    total_loss = loss_recon + loss_linear + (5.0 * loss_metric) 
    
    return total_loss, (loss_recon, loss_linear, loss_metric, sigma, scale, bias)

@nnx.jit
def train_step(
    model: ModelContainer,
    optimizer: nnx.Optimizer,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    alpha: jnp.ndarray
):
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x1, x2, alpha)
    optimizer.update(model, grads)
    return loss, aux

# --- 5. Main Execution ---
# ... (Same main loop as before) ...
def main():
    config = {
        'batch_size': 1024,
        'min_val': -20.0, 'max_val': 20.0,
        'lr': 1e-4,
        'steps': 40000
    }
    
    key = jax.random.PRNGKey(0)
    model = ModelContainer(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(config['lr']), wrt=nnx.Param)
    
    print("Training Learned PE (SIREN)...")
    for i in range(config['steps']):
        key, k1, k2, k3 = jax.random.split(key, 4)
        
        x1 = jax.random.uniform(k1, (config['batch_size'], 2), minval=config['min_val'], maxval=config['max_val'])
        x2 = jax.random.uniform(k2, (config['batch_size'], 2), minval=config['min_val'], maxval=config['max_val'])
        alpha = jax.random.uniform(k3, (config['batch_size'], 1))
        
        loss, (l_r, l_l, l_m, sig, sc, b) = train_step(model, optimizer, x1, x2, alpha)
        
        if i % 1000 == 0:
            print(f"Step {i}: Total={loss:.4f} | Recon={l_r:.4f} | Linear={l_l:.4f} | Metric={l_m:.4f}")
            print(f"          Params: Sigma={sig:.2f}, Scale={sc:.2f}, Bias={b:.2f}")
            
    print("Done.")
    
    # --- Validation ---
    key, k1, k2 = jax.random.split(key, 3)
    x1 = jax.random.uniform(k1, (1000, 2), minval=config['min_val'], maxval=config['max_val'])
    x2 = jax.random.uniform(k2, (1000, 2), minval=config['min_val'], maxval=config['max_val'])
    
    enc = model.encoder
    p1 = enc(x1)
    p2 = enc(x2)
    
    dists = jnp.linalg.norm(x1 - x2, axis=-1)
    dots = jnp.sum(p1 * p2, axis=-1)
    
    print("\nCorrelation Check:")
    print(f"Correlation: {jnp.corrcoef(dists, dots)[0,1]:.4f}")

if __name__ == "__main__":
    main()