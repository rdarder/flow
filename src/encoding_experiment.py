import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np

# --- 1. Architectures ---

class SirenLayer(nnx.Module):
    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False, *, rngs: nnx.Rngs):
        self.w0 = w0
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        limit = np.sqrt(6 / in_features) / w0 if is_first else np.sqrt(6 / in_features) / 30.0
        key = rngs.params()
        self.linear.kernel.value = jax.random.uniform(key, (in_features, out_features), minval=-limit, maxval=limit)

    def __call__(self, x):
        return jnp.sin(self.w0 * self.linear(x))

class LearnedPE_Siren(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            SirenLayer(input_dim, hidden_dim, w0=30.0, is_first=True, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            SirenLayer(hidden_dim, hidden_dim, w0=1.0, rngs=rngs),
            nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        )
        # Fixed Metric Params
        self.sigma = 10.0
        self.scale = 1.0
        self.bias = 0.0

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.net(x)

    def get_metric_params(self):
        return self.sigma, self.scale, self.bias

# --- 2. The Decoder (GELU) ---

class Decoder(nnx.Module):
    """ Standard MLP Decoder (GELU) - Proven to be robust """
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

class ZoomNetwork(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.net = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, pe: jnp.ndarray) -> jnp.ndarray:
        return self.net(pe)

class ModelContainer(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # SIREN Encoder + GELU Decoder
        self.encoder = LearnedPE_Siren(input_dim=2, hidden_dim=128, output_dim=16, rngs=rngs)
        self.decoder = Decoder(input_dim=16, hidden_dim=128, output_dim=2, rngs=rngs)
        self.zoom = ZoomNetwork(dim=16, rngs=rngs)

# --- 3. Foveated Loss Logic ---

@nnx.jit
def loss_fn(model: ModelContainer, x1, x2, alpha, x_zoom_src):
    enc = model.encoder
    dec = model.decoder
    zoom = model.zoom
    
    p1 = enc(x1)
    p2 = enc(x2)
    
    # --- 1. Foveated Reconstruction ---
    x1_rec = dec(p1)
    
    # Weighting based on distance from origin
    dist_from_origin = jnp.linalg.norm(x1, axis=-1, keepdims=True)
    recon_weight = 1.0 / (1.0 + dist_from_origin * 0.5) 
    
    loss_recon = jnp.mean(recon_weight * jnp.square(x1_rec - x1))
    
    # --- 2. Foveated Linearity (UPDATED) ---
    # Interpolation Target
    x_mix_target = alpha * x1 + (1 - alpha) * x2
    
    # Latent Interpolation
    p_mix = alpha * p1 + (1 - alpha) * p2
    x_mix_pred = dec(p_mix)
    
    # We must apply the SAME foveation logic here.
    # If the target mixed point is far away, we allow less precision.
    dist_mix = jnp.linalg.norm(x_mix_target, axis=-1, keepdims=True)
    linear_weight = 1.0 / (1.0 + dist_mix * 0.5)
    
    loss_linear = jnp.mean(linear_weight * jnp.square(x_mix_pred - x_mix_target))
    
    # --- 3. Metric (Unchanged) ---
    dist = jnp.linalg.norm(x1 - x2, axis=-1)
    sigma, scale, bias = enc.get_metric_params()
    target_sim = scale * jnp.exp(-(dist**2) / (2 * sigma**2)) + bias
    actual_dot = jnp.sum(p1 * p2, axis=-1)
    loss_metric = jnp.mean(jnp.square(actual_dot - target_sim))
    
    # --- 4. Zoom (Unchanged) ---
    p_src = enc(x_zoom_src)
    p_target = enc(x_zoom_src * 2.0)
    p_zoomed = zoom(p_src)
    loss_zoom = jnp.mean(jnp.square(p_zoomed - p_target))
    
    # Weighted Sum
    total_loss = (10.0 * loss_recon) + loss_linear + (5.0 * loss_metric) + (2.0 * loss_zoom)
    
    return total_loss, (loss_recon, loss_linear, loss_metric, loss_zoom)

@nnx.jit
def train_step(model, optimizer, x1, x2, alpha, x_zoom_src):
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x1, x2, alpha, x_zoom_src)
    optimizer.update(model, grads)
    return loss, aux

# --- 4. Evaluation ---

def evaluate_precision_at_range(model, rng, center, noise_scale, n_samples=1000):
    """ Check reconstruction error around a specific center point """
    k1 = rng
    # Generate points around 'center'
    noise = jax.random.uniform(k1, (n_samples, 2), minval=-noise_scale, maxval=noise_scale)
    x = center + noise
    
    enc = model.encoder
    dec = model.decoder
    
    p = enc(x)
    x_rec = dec(p)
    
    # RMSE
    mse = jnp.mean(jnp.square(x_rec - x))
    rmse = jnp.sqrt(mse)
    return float(rmse)

def main():
    config = {
        'batch_size': 256,
        'min_val': -200.0, 'max_val': 200.0, # HUGE RANGE
        'lr': 5e-5, 
        'steps': 20000
    }
    
    key = jax.random.PRNGKey(0)
    model = ModelContainer(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(config['lr']), wrt=nnx.Param)
    
    print(f"Training Foveated PE on range [{config['min_val']}, {config['max_val']}]...")
    
    for i in range(config['steps']):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        x1 = jax.random.uniform(k1, (config['batch_size'], 2), minval=config['min_val'], maxval=config['max_val'])
        x2 = jax.random.uniform(k2, (config['batch_size'], 2), minval=config['min_val'], maxval=config['max_val'])
        alpha = jax.random.uniform(k3, (config['batch_size'], 1))
        x_zoom_src = jax.random.uniform(k4, (config['batch_size'], 2), minval=config['min_val']/2, maxval=config['max_val']/2)
        
        loss, (l_r, l_l, l_m, l_z) = train_step(model, optimizer, x1, x2, alpha, x_zoom_src)
        
        if i % 2000 == 0:
            print(f"Step {i}: Total={loss:.4f} | Recon={l_r:.6f} | Linear={l_l:.4f} | Metric={l_m:.4f} | Zoom={l_z:.4f}")

    print("Done.")
    
    # --- Check Foveation ---
    key, k1, k2 = jax.random.split(key, 3)
    
    # Precision at Center (0,0)
    err_center = evaluate_precision_at_range(model, k1, jnp.array([0.0, 0.0]), 1.0)
    
    # Precision at Periphery (150, 150)
    err_far = evaluate_precision_at_range(model, k2, jnp.array([150.0, 150.0]), 1.0)
    
    print(f"\nFoveation Test:")
    print(f"  RMSE at (0,0) [Center]: {err_center:.6f}")
    print(f"  RMSE at (150,150) [Far]: {err_far:.6f}")
    print(f"  Ratio (Far/Center): {err_far/err_center:.2f}x worse")

if __name__ == "__main__":
    main()