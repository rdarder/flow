"""Debug script to verify gradient flow in embedding model.

Run this to check if gradients are flowing through the network.

Usage:
    python -m barevision.embeddings.debug_gradients
"""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import combined_loss


def test_gradient_flow():
    """Test if gradients flow through the entire model."""
    print("="*70)
    print("GRADIENT FLOW DEBUG TEST")
    print("="*70)
    
    # Create model
    print("\n1. Creating model...")
    model = SimpleEmbeddingModel(rngs=nnx.Rngs(jr.PRNGKey(0)))
    print("   ✓ Model created")
    
    # Count parameters
    state = nnx.state(model)
    param_count = 0
    for module_state in state.values():
        for param_value in module_state.values():
            if hasattr(param_value, "size"):
                param_count += param_value.size
    print(f"   Parameters: {param_count}")
    
    # Create dummy data
    print("\n2. Creating dummy data...")
    img1 = jr.uniform(jr.PRNGKey(1), (2, 194, 194, 3))  # Batch of 2
    img2 = jr.uniform(jr.PRNGKey(2), (2, 194, 194, 3))
    print(f"   Image shape: {img1.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    emb1 = model(img1)
    emb2 = model(img2)
    print(f"   Embedding shape: {emb1.shape}")
    print(f"   Embedding stats: mean={float(emb1.mean()):.4f}, std={float(emb1.std()):.4f}")
    
    # Compute loss
    print("\n4. Computing loss...")
    loss_fn = lambda m: combined_loss(m(img1), m(img2)).mean()
    loss = loss_fn(model)
    print(f"   Loss: {float(loss):.4f}")
    
    # Compute gradients
    print("\n5. Computing gradients...")
    grads = nnx.grad(loss_fn)(model)
    
    # Check gradient statistics
    print("\n6. Gradient statistics:")
    grad_state = nnx.state(grads)
    
    total_grad_norm = 0.0
    has_zero_grads = False
    
    for module_path, module_state in grad_state.items():
        print(f"\n   Module: {module_path}")
        for param_name, grad_value in module_state.items():
            grad_array = jnp.array(grad_value)
            grad_norm = float(jnp.linalg.norm(grad_array))
            grad_max = float(jnp.max(jnp.abs(grad_array)))
            grad_mean = float(jnp.mean(jnp.abs(grad_array)))
            
            total_grad_norm += grad_norm ** 2
            
            is_zero = grad_max < 1e-10
            if is_zero:
                has_zero_grads = True
            
            status = "⚠️  ZERO GRADIENT!" if is_zero else "✓"
            print(f"      {param_name:10s}: norm={grad_norm:.6f}, max={grad_max:.6f}, mean={grad_mean:.6f} {status}")
    
    total_grad_norm = float(jnp.sqrt(total_grad_norm))
    print(f"\n   Total gradient norm: {total_grad_norm:.6f}")
    
    if has_zero_grads:
        print("\n⚠️  WARNING: Some parameters have zero gradients!")
        print("   This could indicate:")
        print("   - Dead ReLU neurons (all activations negative)")
        print("   - Loss not connected to those parameters")
        print("   - Numerical issues (NaN/Inf)")
    elif total_grad_norm < 1e-6:
        print("\n⚠️  WARNING: Total gradient norm is very small!")
        print("   This could indicate:")
        print("   - Learning rate too small")
        print("   - Loss surface is flat")
        print("   - Model is at a local minimum")
    else:
        print("\n✓ Gradients appear to be flowing correctly!")
    
    # Check for NaN/Inf
    print("\n7. Checking for NaN/Inf...")
    has_nan = False
    has_inf = False
    
    for module_path, module_state in grad_state.items():
        for param_name, grad_value in module_state.items():
            grad_array = jnp.array(grad_value)
            if jnp.any(jnp.isnan(grad_array)):
                print(f"   ⚠️  NaN in {module_path}.{param_name}")
                has_nan = True
            if jnp.any(jnp.isinf(grad_array)):
                print(f"   ⚠️  Inf in {module_path}.{param_name}")
                has_inf = True
    
    if not has_nan and not has_inf:
        print("   ✓ No NaN or Inf values detected")
    
    # Test parameter update
    print("\n8. Testing parameter update...")
    import optax
    optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
    
    # Get initial parameter values
    initial_state = nnx.state(model)
    initial_kernel = jnp.array(initial_state['depthwise_conv']['kernel'])
    
    # Apply gradients
    nnx.update(optimizer, grads)
    
    # Get updated parameter values
    updated_state = nnx.state(model)
    updated_kernel = jnp.array(updated_state['depthwise_conv']['kernel'])
    
    # Check if parameters changed
    param_change = float(jnp.max(jnp.abs(updated_kernel - initial_kernel)))
    print(f"   Max parameter change: {param_change:.8f}")
    
    if param_change < 1e-10:
        print("   ⚠️  WARNING: Parameters did not change after update!")
        print("   This could indicate:")
        print("   - Optimizer not configured correctly")
        print("   - Gradients not being applied")
        print("   - Learning rate is zero")
    else:
        print("   ✓ Parameters are updating")
    
    print("\n" + "="*70)
    print("DEBUG TEST COMPLETE")
    print("="*70)
    
    return {
        'loss': float(loss),
        'total_grad_norm': total_grad_norm,
        'has_zero_grads': has_zero_grads,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'param_update': param_change,
    }


if __name__ == "__main__":
    results = test_gradient_flow()
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   Loss: {results['loss']:.4f}")
    print(f"   Gradient norm: {results['total_grad_norm']:.6f}")
    print(f"   Zero gradients: {results['has_zero_grads']}")
    print(f"   NaN values: {results['has_nan']}")
    print(f"   Inf values: {results['has_inf']}")
    print(f"   Parameter update: {results['param_update']:.8f}")
