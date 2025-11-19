import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Tuple

# --- 1. The Learned PE Module ---

class LearnedPE(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        # 1. Fixed Fourier Basis
        scale = 1.0
        self.B = nnx.Variable(
            jax.random.normal(rngs.params(), (input_dim, hidden_dim // 2)) * scale,
            metadata={'is_state': True}
        )
        
        # 2. Deeper, Smoother MLP
        self.net = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), 
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), 
            nnx.gelu,
            nnx.Linear(hidden_dim, output_dim, rngs=rngs),
        )
        
        # --- Bounded Learnable Metric Parameters ---
        # We store "raw" unconstrained logits
        self.raw_sigma = nnx.Param(0.0) # Initialized to center of range
        self.raw_scale = nnx.Param(0.0)
        self.raw_bias = nnx.Param(0.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        projection = x @ self.B.value
        features = jnp.concatenate([jnp.cos(projection), jnp.sin(projection)], axis=-1)
        return self.net(features)
    
    def get_metric_params(self):
        # Sigmoid squash to force params into a safe, sensible range
        
        # Sigma Range: [5.0, 20.0] (Target roughly 10.0)
        # Prevents "flat" kernel (sigma -> inf) or "dirac" kernel (sigma -> 0)
        sigma = 1.0 + (5.0 * jax.nn.sigmoid(self.raw_sigma.value))
        
        # Scale Range: [0.5, 5.0]
        # Prevents vanishing kernel (scale -> 0)
        scale = 0.5 + (4.5 * jax.nn.sigmoid(self.raw_scale.value))
        
        # Bias Range: [-1.0, 1.0]
        bias = -1.0 + (2.0 * jax.nn.sigmoid(self.raw_bias.value))
        
        return sigma, scale, bias

# --- 2. The Decoder (Matched Capacity) ---

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
        self.encoder = LearnedPE(input_dim=2, hidden_dim=128, output_dim=16, rngs=rngs)
        self.decoder = Decoder(input_dim=16, hidden_dim=128, output_dim=2, rngs=rngs)

# --- 4. Training Logic ---

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
    
    # Get the safe, bounded parameters
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

def main():
    config = {
        'batch_size': 1024,
        'min_val': -20.0, 'max_val': 20.0,
        'lr': 5e-3,
        'steps': 20000
    }
    
    key = jax.random.PRNGKey(0)
    model = ModelContainer(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(config['lr']), wrt=nnx.Param)
    
    print("Training Bounded Adaptive Learned PE...")
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
    key, subkey = jax.random.split(key)
    x1 = jax.random.uniform(subkey, (1000, 2), minval=-20, maxval=20)
    x2 = jax.random.uniform(subkey, (1000, 2), minval=-20, maxval=20)
    
    enc = model.encoder
    p1 = enc(x1)
    p2 = enc(x2)
    
    dists = jnp.linalg.norm(x1 - x2, axis=-1)
    dots = jnp.sum(p1 * p2, axis=-1)
    
    print("\nCorrelation Check:")
    print(f"Correlation: {jnp.corrcoef(dists, dots)[0,1]:.4f}")

if __name__ == "__main__":
    main()