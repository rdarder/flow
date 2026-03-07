"""Debug script to analyze training dynamics and magnitude scales.

Run for a few epochs to see if scales are stable or exploding/vanishing.

Usage:
    python -m barevision.embeddings.debug_scales
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import (
    self_attention_entropy_loss_core,
    cross_attention_entropy_loss_core,
)
from barevision.utils.grid import WindowGrid


def analyze_scales(model, img1, img2, step: int):
    """Analyze parameter, gradient, and embedding scales."""
    state = nnx.state(model)
    
    # Parameter scales
    param_norms = {}
    for module_path, module_state in state.items():
        for param_name, param_value in module_state.items():
            arr = jnp.array(param_value)
            key = f"{str(module_path).replace('/', '.')}.{param_name}"
            param_norms[key] = float(jnp.linalg.norm(arr))
    
    # Forward pass
    emb1 = model(img1)
    emb2 = model(img2)
    
    # Embedding scales
    emb_mean = float(jnp.mean(emb1))
    emb_std = float(jnp.std(emb1))
    emb_max = float(jnp.max(jnp.abs(emb1)))
    
    # Loss components
    window_size = 16
    grid = WindowGrid(window_size=window_size)
    windows1 = grid.split(emb1)
    windows2 = grid.split(emb2)
    B, num_windows, wh, ww, D = windows1.shape
    flat_windows1 = windows1.reshape(B * num_windows, wh, ww, D)
    flat_windows2 = windows2.reshape(B * num_windows, wh, ww, D)
    
    self_loss = self_attention_entropy_loss_core(flat_windows1)
    cross_loss = cross_attention_entropy_loss_core(flat_windows1, flat_windows2)
    combined_loss = self_loss + cross_loss
    
    # Gradient scales
    def loss_fn(m):
        e1 = m(img1)
        e2 = m(img2)
        w1 = grid.split(e1).reshape(B * num_windows, wh, ww, D)
        w2 = grid.split(e2).reshape(B * num_windows, wh, ww, D)
        sl = self_attention_entropy_loss_core(w1)
        cl = cross_attention_entropy_loss_core(w1, w2)
        return (sl + cl).mean()
    
    grads = nnx.grad(loss_fn)(model)
    grad_norms = {}
    for module_path, module_state in nnx.state(grads).items():
        for param_name, grad_value in module_state.items():
            arr = jnp.array(grad_value)
            key = f"{str(module_path).replace('/', '.')}.{param_name}"
            grad_norms[key] = float(jnp.linalg.norm(arr))
    
    total_grad_norm = float(jnp.sqrt(sum(g**2 for g in grad_norms.values())))
    
    # Print summary
    print(f"\nStep {step:4d}:")
    print(f"  Loss: combined={float(jnp.mean(combined_loss)):.4f} ± {float(jnp.std(combined_loss)):.4f}")
    print(f"        self={float(jnp.mean(self_loss)):.4f} ± {float(jnp.std(self_loss)):.4f}")
    print(f"        cross={float(jnp.mean(cross_loss)):.4f} ± {float(jnp.std(cross_loss)):.4f}")
    print(f"  Embeddings: mean={emb_mean:.4f}, std={emb_std:.4f}, max_abs={emb_max:.4f}")
    print(f"  Total grad norm: {total_grad_norm:.6f}")
    for key in sorted(grad_norms.keys()):
        print(f"    {key:30s}: {grad_norms[key]:.6f}")
    
    return {
        'step': step,
        'combined_loss': float(jnp.mean(combined_loss)),
        'self_loss': float(jnp.mean(self_loss)),
        'cross_loss': float(jnp.mean(cross_loss)),
        'emb_std': emb_std,
        'emb_max': emb_max,
        'total_grad_norm': total_grad_norm,
        'grad_norms': grad_norms,
        'param_norms': param_norms,
    }


def main():
    print("="*80)
    print("TRAINING DYNAMICS DEBUG ANALYSIS")
    print("="*80)
    
    # Settings
    learning_rate = 1e-4
    num_steps = 50
    img_size = (194, 194)
    
    print(f"\nConfiguration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps: {num_steps}")
    print(f"  Image size: {img_size}")
    
    # Initialize
    model = SimpleEmbeddingModel(rngs=nnx.Rngs(jax.random.PRNGKey(0)))
    
    # Create training state properly
    graphdef, state = nnx.split(model)
    tx = optax.adam(learning_rate)
    opt_state = tx.init(state)
    
    # Create dummy data (fixed seed for consistency)
    data_key = jax.random.PRNGKey(42)
    
    # Track metrics
    history = []
    
    print("\n" + "="*80)
    print("INITIAL STATE (Step 0):")
    print("="*80)
    
    img1 = jax.random.uniform(data_key, (1, *img_size, 3))
    img2 = jax.random.uniform(jax.random.fold_in(data_key, 1), (1, *img_size, 3))
    
    metrics = analyze_scales(model, img1, img2, 0)
    history.append(metrics)
    
    initial_grad_norm = metrics['total_grad_norm']
    
    print("\n" + "="*80)
    print("TRAINING STEPS:")
    print("="*80)
    
    for step in range(1, num_steps + 1):
        # New data each step (simulate training)
        img1 = jax.random.uniform(jax.random.fold_in(data_key, step * 2), (1, *img_size, 3))
        img2 = jax.random.uniform(jax.random.fold_in(data_key, step * 2 + 1), (1, *img_size, 3))
        
        # Training step
        grid = WindowGrid(window_size=16)
        
        def loss_fn(state):
            m = nnx.merge(graphdef, state)
            e1 = m(img1)
            e2 = m(img2)
            w1 = grid.split(e1).reshape(1, -1, 16, 16, 16).reshape(-1, 16, 16, 16)
            w2 = grid.split(e2).reshape(1, -1, 16, 16, 16).reshape(-1, 16, 16, 16)
            sl = self_attention_entropy_loss_core(w1)
            cl = cross_attention_entropy_loss_core(w1, w2)
            return (sl + cl).mean()
        
        grads = nnx.grad(loss_fn)(state)
        updates, opt_state = tx.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        
        # Update model state
        model = nnx.merge(graphdef, state)
        
        # Analyze
        metrics = analyze_scales(model, img1, img2, step)
        history.append(metrics)
        
        # Check for issues
        if metrics['total_grad_norm'] > initial_grad_norm * 10:
            print(f"\n⚠️  WARNING: Gradient norm exploded at step {step}!")
        if metrics['emb_max'] > 100:
            print(f"\n⚠️  WARNING: Embeddings exploding at step {step}!")
        if jnp.isnan(metrics['combined_loss']):
            print(f"\n⚠️  WARNING: NaN loss at step {step}!")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    
    losses = [h['combined_loss'] for h in history]
    grad_norms = [h['total_grad_norm'] for h in history]
    emb_stds = [h['emb_std'] for h in history]
    
    print(f"\nCombined Loss:")
    print(f"  Mean: {jnp.mean(losses):.4f}, Std: {jnp.std(losses):.4f}")
    print(f"  Range: [{min(losses):.4f}, {max(losses):.4f}]")
    print(f"  CV (std/mean): {jnp.std(losses) / (abs(jnp.mean(losses)) + 1e-8):.2f}")
    
    print(f"\nGradient Norm:")
    print(f"  Mean: {jnp.mean(grad_norms):.6f}, Std: {jnp.std(grad_norms):.6f}")
    print(f"  Range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    print(f"  CV (std/mean): {jnp.std(grad_norms) / (jnp.mean(grad_norms) + 1e-8):.2f}")
    
    print(f"\nEmbedding Std:")
    print(f"  Mean: {jnp.mean(emb_stds):.4f}, Std: {jnp.std(emb_stds):.4f}")
    print(f"  Trend: {emb_stds[0]:.4f} → {emb_stds[-1]:.4f} ({'increasing' if emb_stds[-1] > emb_stds[0] else 'decreasing'})")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    loss_cv = jnp.std(losses) / (abs(jnp.mean(losses)) + 1e-8)
    grad_cv = jnp.std(grad_norms) / (jnp.mean(grad_norms) + 1e-8)
    
    if loss_cv > 0.5:
        print("⚠️  HIGH LOSS VARIABILITY (CV > 0.5)")
        print("   → Consider reducing learning rate (try 1e-5 or 3e-5)")
        print("   → Consider increasing batch size (aggregate more samples)")
    
    if grad_cv > 1.0:
        print("⚠️  HIGH GRADIENT VARIABILITY (CV > 1.0)")
        print("   → Gradients are very noisy - try gradient clipping")
        print("   → Consider learning rate warmup")
    
    if emb_stds[-1] > emb_stds[0] * 2:
        print("⚠️  EMBEDDINGS GROWING")
        print("   → Add weight decay to prevent unbounded growth")
    
    if emb_stds[-1] < emb_stds[0] * 0.5:
        print("⚠️  EMBEDDINGS SHRINKING")
        print("   → Possible collapse - check loss balance")
    
    # Check loss balance
    self_losses = [h['self_loss'] for h in history]
    cross_losses = [h['cross_loss'] for h in history]
    balance_ratio = abs(jnp.mean(self_losses)) / (jnp.mean(cross_losses) + 1e-8)
    
    print(f"\nLoss Balance:")
    print(f"  |Self| / |Cross| = {balance_ratio:.2f}")
    if balance_ratio < 0.5 or balance_ratio > 2.0:
        print("   → Loss terms are imbalanced!")
        print("   → Consider adjusting alpha/beta weights in combined loss")
    
    if grad_norms[-1] < initial_grad_norm * 0.1:
        print("⚠️  GRADIENTS VANISHING")
        print("   → Gradients decreased 10x - check for saturation")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
