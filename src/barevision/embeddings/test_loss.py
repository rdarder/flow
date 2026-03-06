"""Unit tests for loss functions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from barevision.embeddings.loss import (
    self_attention_entropy_loss_core,
    cross_attention_entropy_loss_core,
    combined_loss,
)


class TestSelfAttentionEntropyLossCore:
    """Tests for self-attention entropy core function (pure math)."""

    def test_output_shape(self):
        """Test that output shape matches input spatial dimensions."""
        windows = jr.normal(jr.PRNGKey(0), (2, 16, 16, 16))  # (B, H, W, D)
        loss = self_attention_entropy_loss_core(windows)

        assert loss.shape == (2, 16, 16), f"Expected (2, 16, 16), got {loss.shape}"

    def test_finite_values(self):
        """Test that all values are finite."""
        windows = jr.normal(jr.PRNGKey(0), (4, 16, 16, 16))
        loss = self_attention_entropy_loss_core(windows)

        assert jnp.isfinite(loss).all(), "Loss contains NaN/Inf"

    def test_gradient_flow(self):
        """Test that gradients flow."""

        def loss_fn(w):
            return self_attention_entropy_loss_core(w).mean()

        windows = jr.normal(jr.PRNGKey(0), (2, 16, 16, 16))
        grad = jax.grad(loss_fn)(windows)

        assert grad.shape == windows.shape
        assert jnp.isfinite(grad).all()
        assert jnp.abs(grad).max() > 0


class TestCrossAttentionEntropyLossCore:
    """Tests for cross-attention entropy core function (pure math)."""

    def test_output_shape(self):
        """Test that output shape matches input spatial dimensions."""
        windows1 = jr.normal(jr.PRNGKey(0), (2, 16, 16, 16))
        windows2 = jr.normal(jr.PRNGKey(1), (2, 16, 16, 16))
        loss = cross_attention_entropy_loss_core(windows1, windows2)

        assert loss.shape == (2, 16, 16)

    def test_finite_values(self):
        """Test that all values are finite."""
        windows1 = jr.normal(jr.PRNGKey(0), (4, 16, 16, 16))
        windows2 = jr.normal(jr.PRNGKey(1), (4, 16, 16, 16))
        loss = cross_attention_entropy_loss_core(windows1, windows2)

        assert jnp.isfinite(loss).all()

    def test_positive_entropy(self):
        """Cross-attention entropy is always non-negative."""
        windows1 = jr.normal(jr.PRNGKey(0), (2, 16, 16, 16))
        windows2 = jr.normal(jr.PRNGKey(1), (2, 16, 16, 16))
        loss = cross_attention_entropy_loss_core(windows1, windows2)

        assert (loss >= 0).all()

    def test_gradient_flow(self):
        """Test that gradients flow through both inputs."""

        def loss_fn(w1, w2):
            return cross_attention_entropy_loss_core(w1, w2).mean()

        windows1 = jr.normal(jr.PRNGKey(0), (2, 16, 16, 16))
        windows2 = jr.normal(jr.PRNGKey(1), (2, 16, 16, 16))
        grad1, grad2 = jax.grad(loss_fn, argnums=(0, 1))(windows1, windows2)

        assert grad1.shape == windows1.shape
        assert grad2.shape == windows2.shape
        assert jnp.isfinite(grad1).all()
        assert jnp.isfinite(grad2).all()


class TestCombinedLoss:
    """Tests for combined loss wrapper (handles splitting)."""

    def test_output_shape(self):
        """Test that combined loss returns (B, H, W)."""
        emb1 = jr.normal(jr.PRNGKey(0), (2, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (2, 32, 32, 16))
        loss = combined_loss(emb1, emb2)

        assert loss.shape == (2, 32, 32)

    def test_finite_values(self):
        """Test that combined loss is finite."""
        emb1 = jr.normal(jr.PRNGKey(0), (2, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (2, 32, 32, 16))
        loss = combined_loss(emb1, emb2)

        assert jnp.isfinite(loss).all()

    def test_misaligned_height_fails(self):
        """Test that misaligned height raises ValueError."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 31, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 31, 32, 16))

        with pytest.raises(ValueError, match="Height.*not divisible"):
            combined_loss(emb1, emb2)

    def test_misaligned_width_fails(self):
        """Test that misaligned width raises ValueError."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 33, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 33, 16))

        with pytest.raises(ValueError, match="Width.*not divisible"):
            combined_loss(emb1, emb2)

    def test_shape_mismatch_fails(self):
        """Test that mismatched shapes raise assertion."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 8))  # Different D

        with pytest.raises(AssertionError):
            combined_loss(emb1, emb2)

    def test_gradient_flow(self):
        """Test gradients flow through combined loss."""

        def loss_fn(e1, e2):
            return combined_loss(e1, e2).mean()

        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        grad1, grad2 = jax.grad(loss_fn, argnums=(0, 1))(emb1, emb2)

        assert grad1.shape == emb1.shape
        assert grad2.shape == emb2.shape
        assert jnp.isfinite(grad1).all()
        assert jnp.isfinite(grad2).all()

    def test_weights_applied(self):
        """Test that alpha/beta weights affect the loss."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

        loss1 = combined_loss(emb1, emb2, alpha=1.0, beta=0.0)  # Self only
        loss2 = combined_loss(emb1, emb2, alpha=0.0, beta=1.0)  # Cross only
        loss3 = combined_loss(emb1, emb2, alpha=1.0, beta=1.0)  # Both

        # All should be finite
        assert jnp.isfinite(loss1).all()
        assert jnp.isfinite(loss2).all()
        assert jnp.isfinite(loss3).all()

        # Combined should equal sum of parts (approximately)
        combined = loss1 + loss2
        assert jnp.allclose(loss3, combined)


class TestLossIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_training_step(self):
        """Test a single training step with combined loss."""
        B, H, W, D = 2, 64, 64, 16

        emb1 = jr.normal(jr.PRNGKey(0), (B, H, W, D))
        emb2 = jr.normal(jr.PRNGKey(1), (B, H, W, D))

        def train_loss(e1, e2):
            return combined_loss(e1, e2).mean()

        loss = train_loss(emb1, emb2)
        assert jnp.isfinite(loss)

        grad1, grad2 = jax.grad(train_loss, argnums=(0, 1))(emb1, emb2)
        assert jnp.isfinite(grad1).all()
        assert jnp.isfinite(grad2).all()

    def test_different_resolutions(self):
        """Test with different aligned resolutions."""
        for h, w in [(16, 16), (32, 32), (64, 48), (48, 64)]:
            emb1 = jr.normal(jr.PRNGKey(0), (1, h, w, 16))
            emb2 = jr.normal(jr.PRNGKey(1), (1, h, w, 16))
            loss = combined_loss(emb1, emb2)
            assert loss.shape == (1, h, w)
            assert jnp.isfinite(loss).all()
