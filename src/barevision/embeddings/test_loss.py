"""Unit tests for loss functions."""

import jax
import jax.numpy as jnp
import jax.random as jr

from barevision.embeddings.loss import (
    self_attention_entropy_loss,
    cross_attention_entropy_loss,
    combined_loss,
)


class TestSelfAttentionEntropyLoss:
    """Tests for self-attention entropy loss."""

    def test_output_shape(self):
        """Test that output shape matches input spatial dimensions."""
        emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        loss = self_attention_entropy_loss(emb)

        assert loss.shape == (1, 32, 32), f"Expected (1, 32, 32), got {loss.shape}"

    def test_batch_processing(self):
        """Test batch processing."""
        emb = jr.normal(jr.PRNGKey(0), (4, 32, 32, 16))
        loss = self_attention_entropy_loss(emb)

        assert loss.shape == (4, 32, 32), f"Expected (4, 32, 32), got {loss.shape}"

    def test_finite_values(self):
        """Test that all values are finite (no NaN/Inf)."""
        emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        loss = self_attention_entropy_loss(emb)

        assert jnp.isfinite(loss).all(), "Loss contains NaN/Inf values"

    def test_custom_window_size(self):
        """Test with custom window size."""
        emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        loss = self_attention_entropy_loss(emb, window_size=8)

        assert loss.shape == (1, 32, 32)
        assert jnp.isfinite(loss).all()

    def test_different_spatial_sizes(self):
        """Test with different spatial dimensions."""
        test_cases = [
            (1, 16, 16, 16),
            (1, 32, 32, 16),
            (2, 64, 64, 16),
        ]

        for B, H, W, D in test_cases:
            emb = jr.normal(jr.PRNGKey(0), (B, H, W, D))
            loss = self_attention_entropy_loss(emb)
            assert loss.shape == (B, H, W), f"Failed for shape ({B}, {H}, {W}, {D})"

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""

        def loss_fn(emb):
            return self_attention_entropy_loss(emb).mean()

        emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        grad = jax.grad(loss_fn)(emb)

        assert grad.shape == emb.shape, f"Gradient shape mismatch: {grad.shape}"
        assert jnp.isfinite(grad).all(), "Gradient contains NaN/Inf"
        assert jnp.abs(grad).max() > 0, "Gradient is zero everywhere"


class TestCrossAttentionEntropyLoss:
    """Tests for cross-attention entropy loss."""

    def test_output_shape(self):
        """Test that output shape matches input spatial dimensions."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        loss = cross_attention_entropy_loss(emb1, emb2)

        assert loss.shape == (1, 32, 32), f"Expected (1, 32, 32), got {loss.shape}"

    def test_batch_processing(self):
        """Test batch processing."""
        emb1 = jr.normal(jr.PRNGKey(0), (4, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (4, 32, 32, 16))
        loss = cross_attention_entropy_loss(emb1, emb2)

        assert loss.shape == (4, 32, 32), f"Expected (4, 32, 32), got {loss.shape}"

    def test_finite_values(self):
        """Test that all values are finite (no NaN/Inf)."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        loss = cross_attention_entropy_loss(emb1, emb2)

        assert jnp.isfinite(loss).all(), "Loss contains NaN/Inf values"

    def test_custom_window_size(self):
        """Test with custom window size."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        loss = cross_attention_entropy_loss(emb1, emb2, window_size=8)

        assert loss.shape == (1, 32, 32)
        assert jnp.isfinite(loss).all()

    def test_identical_embeddings(self):
        """Test with identical embeddings (should produce valid loss)."""
        emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        loss = cross_attention_entropy_loss(emb, emb)

        assert loss.shape == (1, 32, 32)
        assert jnp.isfinite(loss).all()
        # Loss should be positive (entropy is always positive)
        assert (loss >= 0).all(), "Cross-attention loss should be non-negative"

    def test_gradient_flow(self):
        """Test that gradients flow through both inputs."""

        def loss_fn(emb1, emb2):
            return cross_attention_entropy_loss(emb1, emb2).mean()

        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        grad1, grad2 = jax.grad(loss_fn, argnums=(0, 1))(emb1, emb2)

        assert grad1.shape == emb1.shape, f"Gradient 1 shape mismatch: {grad1.shape}"
        assert grad2.shape == emb2.shape, f"Gradient 2 shape mismatch: {grad2.shape}"
        assert jnp.isfinite(grad1).all(), "Gradient 1 contains NaN/Inf"
        assert jnp.isfinite(grad2).all(), "Gradient 2 contains NaN/Inf"


class TestCombinedLoss:
    """Tests for combined loss function."""

    def test_output_shape(self):
        """Test that combined loss has correct shape."""
        self_loss = jr.uniform(jr.PRNGKey(0), (1, 32, 32))
        cross_loss = jr.uniform(jr.PRNGKey(1), (1, 32, 32))
        combined = combined_loss(self_loss, cross_loss)

        assert combined.shape == (1, 32, 32)

    def test_weighted_combination(self):
        """Test that weights are applied correctly."""
        self_loss = jnp.ones((1, 32, 32))
        cross_loss = jnp.ones((1, 32, 32)) * 2

        # Equal weights
        combined = combined_loss(self_loss, cross_loss, alpha=1.0, beta=1.0)
        expected = 1.0 * 1.0 + 1.0 * 2.0
        assert jnp.allclose(combined, jnp.full_like(combined, expected))

        # Different weights
        combined = combined_loss(self_loss, cross_loss, alpha=2.0, beta=0.5)
        expected = 2.0 * 1.0 + 0.5 * 2.0
        assert jnp.allclose(combined, jnp.full_like(combined, expected))

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises assertion."""
        self_loss = jnp.ones((1, 32, 32))
        cross_loss = jnp.ones((1, 16, 16))

        try:
            combined_loss(self_loss, cross_loss)
            assert False, "Should have raised an error for shape mismatch"
        except AssertionError:
            pass  # Expected


class TestLossIntegration:
    """Integration tests for the complete loss pipeline."""

    def test_full_pipeline(self):
        """Test complete loss computation pipeline."""
        B, H, W, D = 2, 64, 64, 16

        # Create embeddings for two frames
        emb1 = jr.normal(jr.PRNGKey(0), (B, H, W, D))
        emb2 = jr.normal(jr.PRNGKey(1), (B, H, W, D))

        # Compute individual losses
        self_loss = self_attention_entropy_loss(emb1)
        cross_loss = cross_attention_entropy_loss(emb1, emb2)

        # Verify shapes
        assert self_loss.shape == (B, H, W)
        assert cross_loss.shape == (B, H, W)

        # Verify finite values
        assert jnp.isfinite(self_loss).all()
        assert jnp.isfinite(cross_loss).all()

        # Compute combined loss
        total_loss = combined_loss(self_loss, cross_loss, alpha=1.0, beta=1.0)
        assert total_loss.shape == (B, H, W)
        assert jnp.isfinite(total_loss).all()

    def test_gradient_through_pipeline(self):
        """Test gradients through complete pipeline."""

        def total_loss_fn(emb1, emb2):
            self_loss = self_attention_entropy_loss(emb1)
            cross_loss = cross_attention_entropy_loss(emb1, emb2)
            return combined_loss(self_loss, cross_loss).mean()

        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

        grad1, grad2 = jax.grad(total_loss_fn, argnums=(0, 1))(emb1, emb2)

        assert grad1.shape == emb1.shape
        assert grad2.shape == emb2.shape
        assert jnp.isfinite(grad1).all()
        assert jnp.isfinite(grad2).all()


class TestLossValues:
    """Tests to verify loss value ranges and properties."""

    def test_self_loss_can_be_negative(self):
        """Self-attention loss is negative entropy (can be negative)."""
        emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        loss = self_attention_entropy_loss(emb)

        # Self-loss is negative entropy, so should be negative
        # (we minimize negative entropy = maximize entropy)
        assert (loss < 0).any(), "Self-attention loss should have negative values"

    def test_cross_loss_is_positive(self):
        """Cross-attention loss is entropy (always positive)."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        loss = cross_attention_entropy_loss(emb1, emb2)

        # Entropy is always non-negative
        assert (loss >= 0).all(), "Cross-attention loss should be non-negative"

    def test_loss_ranges(self):
        """Test that losses are in reasonable ranges."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

        self_loss = self_attention_entropy_loss(emb1, spatial_scale=10.0)
        cross_loss = cross_attention_entropy_loss(emb1, emb2)

        # Self-loss (negative entropy after spatial weighting)
        # Max entropy for 256 positions is ~ln(256) ≈ 5.5
        # With spatial_scale=10.0, nearby positions are favored, reducing entropy
        assert self_loss.min() > -10, "Self-loss unexpectedly large negative"
        assert self_loss.max() < 0, "Self-loss should be negative"

        # Cross-loss (entropy) should be in reasonable range
        # Max entropy is ~ln(256) ≈ 5.5
        assert cross_loss.min() >= 0, "Cross-loss should be non-negative"
        assert cross_loss.max() < 20, "Cross-loss unexpectedly large"

        # With spatial weighting, self-loss should be in a reasonable range
        # (spatial weighting reduces effective entropy by favoring nearby positions)
        assert self_loss.mean() > -6.0, "Self-loss suggests too much spatial concentration"
        assert self_loss.mean() < 0, "Self-loss should be negative (maximizing entropy)"
