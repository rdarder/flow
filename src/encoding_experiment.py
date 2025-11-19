import jax
import jax.numpy as jnp
from flax import nnx
import optax


class LearnedPE(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        # 1. Fixed Fourier Basis (Random Gaussian) to overcome spectral bias
        # We map 2D -> hidden_dim (e.g. 32 or 64)
        # This provides the "raw material" of frequencies.
        scale = 1.0  # Scale of the random frequencies
        self.B = nnx.Variable(
            jax.random.normal(rngs.params(), (input_dim, hidden_dim // 2)) * scale,
            metadata={'is_state': True}  # Static, not trained
        )
        # 2. Learnable MLP to mix these frequencies into the final PE
        # This allows the model to "shape" the geometry.
        self.net = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, output_dim, rngs=rngs),
            # No activation at the end, we want the raw vector space
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., 2)
        # Fourier Feature Mapping: [cos(Bx), sin(Bx)]
        # (..., 2) @ (2, H/2) -> (..., H/2)
        projection = x @ self.B.value
        features = jnp.concatenate([jnp.cos(projection), jnp.sin(projection)], axis=-1)
        # MLP Mixing
        return self.net(features)


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


@nnx.jit
def loss_fn(model, x1, x2, alpha):
    # Unpack state (nnx weirdness handling handled by optimizer update normally,
    # but here we have two models. We'll treat them as a tuple for optimization if needed,
    # or just update them jointly.)
    # Actually, nnx.Optimizer can wrap a container. Let's assume 'model' is a container.
    # --- Forward Pass ---
    p1 = model.encoder(x1)
    p2 = model.encoder(x2)
    # --- Constraint 1: Reconstruction ---
    x1_rec = model.decoder(p1)
    loss_recon = jnp.mean(jnp.square(x1_rec - x1))
    # --- Constraint 2: Linearity (Interpolation) ---
    # Target: Linear mix of coordinates
    x_mix_target = alpha * x1 + (1 - alpha) * x2
    # Prediction: Decode the linear mix of embeddings
    p_mix = alpha * p1 + (1 - alpha) * p2
    x_mix_pred = model.decoder(p_mix)
    loss_linear = jnp.mean(jnp.square(x_mix_pred - x_mix_target))
    # --- Constraint 3: Metric (Dot Product ~ Distance) ---
    # Real Distance
    dist = jnp.linalg.norm(x1 - x2, axis=-1)
    # Target Similarity Kernel (Gaussian-like)
    # We want Dot=1 when dist=0, Dot=0 when dist is large.
    # Let's pick a sigma that makes sense for the range.
    # If range is -20..20, max dist ~56.
    # Let's say sigma=10.0
    target_sim = jnp.exp(-(dist**2) / (2 * 10.0**2))
    # Actual Dot Product (Cosine Similarity? or Raw Dot?)
    # Let's try Raw Dot first, but normalized?
    # Let's enforce bounded magnitude if we want dot product consistency.
    # Actually, let's just match the dot product to the kernel.
    actual_dot = jnp.sum(p1 * p2, axis=-1)
    loss_metric = jnp.mean(jnp.square(actual_dot - target_sim))
    # --- Constraint 4: Unit Norm? (Optional) ---
    # To keep dot products stable, we might want vectors to be roughly unit length.
    # Or we let the 'loss_metric' enforce the scale implicitly (since target max is 1.0).

    total_loss = 10 * loss_recon + 10 * loss_linear + loss_metric

    return total_loss, (loss_recon, loss_linear, loss_metric)


def train_step(model, optimizer, x1, x2, alpha):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (l_r, l_l, l_m)), grads = grad_fn(model, x1, x2, alpha)
    optimizer.update(model, grads)
    return loss, l_r, l_l, l_m


class ModelContainer(nnx.Module):
    def __init__(self, rngs):
        self.encoder = LearnedPE(input_dim=2, hidden_dim=32, output_dim=16, rngs=rngs)
        self.decoder = Decoder(input_dim=16, hidden_dim=32, output_dim=2, rngs=rngs)


def main():
    config = {
        'batch_size': 32,
        'min_val': -20.0, 'max_val': 20.0,
        'lr': 1e-3,
        'steps': 100_000
    }
    key = jax.random.PRNGKey(0)
    model = ModelContainer(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(config['lr']), wrt=nnx.Param)
    print("Training Learned PE...")
    batch_size = config['batch_size']
    for i in range(config['steps']):

        # 1. Generate Random Points (x1, x2) for Metric & Reconstruction
        key, k1, k2, k3 = jax.random.split(key, 4)
        x1 = jax.random.uniform(
                k1, (batch_size, 2), 
                minval=config['min_val'], maxval=config['max_val']
                                )
        x2 = jax.random.uniform(k2, (batch_size, 2), minval=config['min_val'], maxval=config['max_val'])
        
        # 2. Generate Weights for Linearity (alpha)
        alpha = jax.random.uniform(k3, (batch_size, 1)) # [0, 1]


        loss, l_r, l_l, l_m = train_step(model, optimizer, x1, x2, alpha) # optimizer holds model
        
        if i % 1000 == 0:
            print(f"Step {i}: Total={loss:.4f} | Recon={l_r:.4f} | Linear={l_l:.4f} | Metric={l_m:.4f}")
            
    print("Done.")
    
    # --- Validation: Plot Distance vs Dot Product ---
    key, k1, k2 = jax.random.split(key, 3)
    x1 = jax.random.uniform(k1, (1000, 2), minval=-20, maxval=20)
    x2 = jax.random.uniform(k2, (1000, 2), minval=-20, maxval=20) # Random pairs
    
    enc = optimizer.model.encoder
    p1 = enc(x1)
    p2 = enc(x2)
    
    dists = jnp.linalg.norm(x1 - x2, axis=-1)
    dots = jnp.sum(p1 * p2, axis=-1)
    
    print("\nCorrelation Check:")
    print(f"Correlation: {jnp.corrcoef(dists, dots)[0,1]:.4f}")
    
    # Note: In a real script we'd use matplotlib, but here we just print stats
    # to see if it worked.

if __name__ == "__main__":
    main()
