"""Smoke test and diagnostic analysis for embedding loss functions.

Demonstrates loss behavior and verifies correctness.

Run:
    python -m barevision.embeddings.verify_losses
"""

from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import (
    self_attention_entropy_loss,
    cross_attention_entropy_loss,
    combined_loss,
    _spatial_logits_matrix,
)
from flax import nnx
import jax
import jax.numpy as jnp
import jax.random as jr


def test_spatial_scoring():
    """Verify spatial scoring matches SpatialScore approach from flow."""
    print("=" * 60)
    print("1. Spatial Logits Matrix (Gaussian kernel)")
    print("=" * 60)

    spatial = _spatial_logits_matrix(4, scale=10.0)

    print(f"Shape: {spatial.shape}")
    print(f"Self-attention (diagonal): {spatial[0, 0]:.2f} (expected: 0)")
    print(f"Adjacent position: {spatial[0, 1]:.2f} (small penalty)")
    print(f"Far position: {spatial[0, -1]:.2f} (large penalty)")
    print()

    # Verify spatial decay pattern
    assert spatial[0, 0] == 0.0, "Self should have zero penalty"
    assert spatial[0, 1] > spatial[0, -1], "Far positions should be more penalized"
    print("✓ Spatial scoring verified\n")


def test_loss_values():
    """Verify loss values are in expected ranges."""
    print("=" * 60)
    print("2. Loss Function Values")
    print("=" * 60)

    # Create test embeddings
    emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
    emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

    self_loss = self_attention_entropy_loss(emb1, spatial_scale=10.0)
    cross_loss = cross_attention_entropy_loss(emb1, emb2)

    print(f"Self-attention loss:")
    print(f"  Shape: {self_loss.shape}")
    print(f"  Range: [{self_loss.min():.3f}, {self_loss.max():.3f}]")
    print(f"  Mean: {self_loss.mean():.3f}")
    print(f"  → Negative = maximizing entropy (after spatial weighting)")
    print()

    print(f"Cross-attention loss:")
    print(f"  Shape: {cross_loss.shape}")
    print(f"  Range: [{cross_loss.min():.3f}, {cross_loss.max():.3f}]")
    print(f"  Mean: {cross_loss.mean():.3f}")
    print(f"  → Positive = minimizing entropy (sharp matches)")
    print()

    # Verify expected properties
    assert (self_loss < 0).all(), "Self-loss should be negative"
    assert (cross_loss >= 0).all(), "Cross-loss should be non-negative"
    print("✓ Loss values verified\n")


def test_gradient_flow():
    """Verify gradients flow through both losses."""
    print("=" * 60)
    print("3. Gradient Flow")
    print("=" * 60)

    emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
    emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

    def total_loss(e1, e2):
        self_l = self_attention_entropy_loss(e1).mean()
        cross_l = cross_attention_entropy_loss(e1, e2).mean()
        return self_l + cross_l

    grad1, grad2 = jax.grad(total_loss, argnums=(0, 1))(emb1, emb2)

    print(f"∂loss/∂emb1: max={jnp.abs(grad1).max():.6f}")
    print(f"∂loss/∂emb2: max={jnp.abs(grad2).max():.6f}")

    assert jnp.abs(grad1).max() > 0, "Gradient for emb1 should be non-zero"
    assert jnp.abs(grad2).max() > 0, "Gradient for emb2 should be non-zero"
    assert jnp.isfinite(grad1).all(), "Gradient for emb1 should be finite"
    assert jnp.isfinite(grad2).all(), "Gradient for emb2 should be finite"
    print("✓ Gradients flow correctly\n")


def test_full_pipeline():
    """Verify complete pipeline: model → embeddings → losses."""
    print("=" * 60)
    print("4. Full Pipeline (Model + Losses)")
    print("=" * 60)

    model = SimpleEmbeddingModel(
        embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
    )

    img1 = jr.uniform(jr.PRNGKey(1), (2, 64, 64, 3))
    img2 = jr.uniform(jr.PRNGKey(2), (2, 64, 64, 3))

    emb1 = model(img1)
    emb2 = model(img2)

    print(f"Model: {img1.shape} → {emb1.shape}")

    self_loss = self_attention_entropy_loss(emb1)
    cross_loss = cross_attention_entropy_loss(emb1, emb2)
    total = combined_loss(self_loss, cross_loss)

    print(f"Self-loss mean: {self_loss.mean():.4f}")
    print(f"Cross-loss mean: {cross_loss.mean():.4f}")
    print(f"Combined mean: {total.mean():.4f}")

    assert jnp.isfinite(total).all(), "Combined loss should be finite"
    print("✓ Full pipeline verified\n")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("EMBEDDING LOSS FUNCTIONS - VERIFICATION")
    print("=" * 60 + "\n")

    test_spatial_scoring()
    test_loss_values()
    test_gradient_flow()
    test_full_pipeline()

    print("=" * 60)
    print("ALL VERIFICATIONS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
