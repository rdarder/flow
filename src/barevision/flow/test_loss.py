"""Unit tests for loss functions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from barevision.flow.loss import (
    self_attention_entropy_loss_core,
    cross_attention_entropy_loss_core,
    compute_window_attention_losses,
    compute_hierarchical_entropy_loss,
    crop_to_grid_aligned,
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
        """Test that combined loss returns scalar values."""
        emb1 = jr.normal(jr.PRNGKey(0), (2, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (2, 32, 32, 16))
        loss, aux = compute_window_attention_losses(emb1, emb2)

        assert jnp.isscalar(loss) or loss.shape == ()
        assert jnp.isscalar(aux["self_loss"]) or aux["self_loss"].shape == ()
        assert jnp.isscalar(aux["cross_loss"]) or aux["cross_loss"].shape == ()

    def test_finite_values(self):
        """Test that combined loss is finite."""
        emb1 = jr.normal(jr.PRNGKey(0), (2, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (2, 32, 32, 16))
        loss, aux = compute_window_attention_losses(emb1, emb2)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(aux["self_loss"])
        assert jnp.isfinite(aux["cross_loss"])

    def test_misaligned_height_fails(self):
        """Test that misaligned height raises ValueError."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 31, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 31, 32, 16))

        with pytest.raises(ValueError, match="Height.*not divisible"):
            compute_window_attention_losses(emb1, emb2)

    def test_misaligned_width_fails(self):
        """Test that misaligned width raises ValueError."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 33, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 33, 16))

        with pytest.raises(ValueError, match="Width.*not divisible"):
            compute_window_attention_losses(emb1, emb2)

    def test_shape_mismatch_fails(self):
        """Test that mismatched shapes raise assertion."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 8))  # Different D

        with pytest.raises(AssertionError):
            compute_window_attention_losses(emb1, emb2)

    def test_gradient_flow(self):
        """Test gradients flow through combined loss."""

        def loss_fn(e1, e2):
            loss, _ = compute_window_attention_losses(e1, e2)
            return loss

        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))
        grad1, grad2 = jax.grad(loss_fn, argnums=(0, 1))(emb1, emb2)

        assert grad1.shape == emb1.shape
        assert grad2.shape == emb2.shape
        assert jnp.isfinite(grad1).all()
        assert jnp.isfinite(grad2).all()

    def test_weights_applied(self):
        """Test that lambda_entropy weights affect the loss."""
        emb1 = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
        emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

        loss1, aux1 = compute_window_attention_losses(
            emb1, emb2, lambda_entropy=0.0
        )  # Self only
        loss2, aux2 = compute_window_attention_losses(
            emb1, emb2, lambda_entropy=1.0
        )  # Cross only
        loss3, aux3 = compute_window_attention_losses(
            emb1, emb2, lambda_entropy=0.5
        )  # Equal mix

        # All should be finite
        assert jnp.isfinite(loss1)
        assert jnp.isfinite(loss2)
        assert jnp.isfinite(loss3)

        # Equal mix should equal average of self and cross
        expected = 0.5 * loss1 + 0.5 * loss2
        assert jnp.allclose(loss3, expected)


class TestLossIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_training_step(self):
        """Test a single training step with combined loss."""
        B, H, W, D = 2, 64, 64, 16

        emb1 = jr.normal(jr.PRNGKey(0), (B, H, W, D))
        emb2 = jr.normal(jr.PRNGKey(1), (B, H, W, D))

        def train_loss(e1, e2):
            loss, _ = compute_window_attention_losses(e1, e2)
            return loss

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
            loss, aux = compute_window_attention_losses(emb1, emb2)
            assert jnp.isscalar(loss) or loss.shape == ()
            assert jnp.isfinite(loss)


class TestCropToGridAligned:
    """Tests for grid alignment cropping utility."""

    def test_no_crop_needed(self):
        """Test that aligned dimensions are not cropped."""
        feature_map = jr.normal(jr.PRNGKey(0), (1, 64, 64, 16))
        cropped = crop_to_grid_aligned(feature_map, window_size=16)

        assert cropped.shape == (1, 64, 64, 16)

    def test_crop_removes_extra_pixels(self):
        """Test that extra pixels are cropped."""
        feature_map = jr.normal(jr.PRNGKey(0), (1, 67, 67, 16))
        cropped = crop_to_grid_aligned(feature_map, window_size=16)

        assert cropped.shape == (1, 64, 64, 16)

    def test_crop_preserves_batch(self):
        """Test that batch dimension is preserved."""
        feature_map = jr.normal(jr.PRNGKey(0), (4, 33, 33, 16))
        cropped = crop_to_grid_aligned(feature_map, window_size=16)

        assert cropped.shape == (4, 32, 32, 16)

    def test_different_window_sizes(self):
        """Test with different window sizes."""
        for ws in [8, 16, 32]:
            feature_map = jr.normal(jr.PRNGKey(0), (1, 50, 50, 16))
            cropped = crop_to_grid_aligned(feature_map, window_size=ws)
            assert cropped.shape[1] % ws == 0
            assert cropped.shape[2] % ws == 0


class TestHierarchicalEmbeddingLosses:
    """Tests for multi-level hierarchical loss computation (Phase 2)."""

    def test_single_level_pyramid(self):
        """Test with single-level pyramid."""
        pyramid1 = [jr.normal(jr.PRNGKey(0), (1, 64, 64, 16))]
        pyramid2 = [jr.normal(jr.PRNGKey(1), (1, 64, 64, 16))]

        loss, aux = compute_hierarchical_entropy_loss(pyramid1, pyramid2)

        assert jnp.isfinite(loss)
        assert jnp.isfinite(aux["self_loss"])
        assert jnp.isfinite(aux["cross_loss"])
        assert len(aux["level_losses"]) == 1

    def test_three_level_pyramid(self):
        """Test with three-level pyramid (Phase 2 default)."""
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),  # Level 0: 4×4 grid
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),  # Level 1: 2×2 grid
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),  # Level 2: 1×1 grid
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(4), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        loss, aux = compute_hierarchical_entropy_loss(pyramid1, pyramid2)

        assert jnp.isfinite(loss)
        assert len(aux["level_losses"]) == 3
        # All level losses should be finite
        for level_loss in aux["level_losses"]:
            assert jnp.isfinite(level_loss)

    def test_crops_misaligned_inputs(self):
        """Test that misaligned inputs are cropped correctly."""
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 67, 67, 16)),  # Needs crop to 64×64
            jr.normal(jr.PRNGKey(1), (1, 33, 33, 16)),  # Needs crop to 32×32
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),  # No crop needed
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 67, 67, 16)),
            jr.normal(jr.PRNGKey(4), (1, 33, 33, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        loss, aux = compute_hierarchical_entropy_loss(pyramid1, pyramid2)

        assert jnp.isfinite(loss)
        assert len(aux["level_losses"]) == 3

    def test_gradient_flow_all_levels(self):
        """Test that gradients flow through all pyramid levels."""

        def loss_fn(p1, p2):
            loss, _ = compute_hierarchical_entropy_loss(p1, p2)
            return loss

        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(4), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        grads1, grads2 = jax.grad(loss_fn, argnums=(0, 1))(pyramid1, pyramid2)

        # All levels should have gradients
        assert len(grads1) == 3
        assert len(grads2) == 3

        # All gradients should be finite
        for g1, g2 in zip(grads1, grads2):
            assert jnp.isfinite(g1).all()
            assert jnp.isfinite(g2).all()

    def test_per_level_averaging(self):
        """Test that per-level averaging prevents fine levels from dominating."""
        # Create pyramids where Level 0 has many more windows than Level 2
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),  # 16 windows
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),  # 4 windows
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),  # 1 window
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(4), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        loss, aux = compute_hierarchical_entropy_loss(pyramid1, pyramid2)

        # Each level should contribute to the total loss
        level_losses = aux["level_losses"]
        level_weights = aux["level_weights"]
        assert len(level_losses) == 3

        # All level losses should be positive (entropy is non-negative)
        for level_loss in level_losses:
            assert level_loss >= 0

        # Total loss should equal weighted sum divided by total weight
        total_weight = sum(level_weights)
        weighted_sum = sum(level_losses)
        expected = weighted_sum / total_weight
        assert jnp.allclose(loss, expected)

    def test_level_mismatch_fails(self):
        """Test that mismatched pyramid levels raise ValueError."""
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
        ]

        with pytest.raises(ValueError, match="Pyramid level mismatch"):
            compute_hierarchical_entropy_loss(pyramid1, pyramid2)

    def test_level_weight_decay(self):
        """Test that level weight decay correctly weights coarser levels higher."""
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),  # Level 0: weight = 1
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),  # Level 1: weight = 2
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),  # Level 2: weight = 4
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(4), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        # Default decay=2.0
        loss, aux = compute_hierarchical_entropy_loss(pyramid1, pyramid2)

        # Check weights are correct
        assert aux["level_weights"] == [1.0, 2.0, 4.0]

        # Verify weighted sum normalized by total weight
        weighted_sum = sum(aux["level_losses"])
        total_weight = sum(aux["level_weights"])
        expected = weighted_sum / total_weight
        assert jnp.allclose(loss, expected)

    def test_level_weight_decay_custom(self):
        """Test custom level weight decay factor."""
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(4), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        # Custom decay=3.0: weights should be [1, 3, 9]
        loss, aux = compute_hierarchical_entropy_loss(
            pyramid1, pyramid2, level_weight_decay=3.0
        )

        assert aux["level_weights"] == [1.0, 3.0, 9.0]

    def test_level_weight_decay_disabled(self):
        """Test that decay=1.0 gives equal weight to all levels."""
        pyramid1 = [
            jr.normal(jr.PRNGKey(0), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(1), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(2), (1, 16, 16, 16)),
        ]
        pyramid2 = [
            jr.normal(jr.PRNGKey(3), (1, 64, 64, 16)),
            jr.normal(jr.PRNGKey(4), (1, 32, 32, 16)),
            jr.normal(jr.PRNGKey(5), (1, 16, 16, 16)),
        ]

        # decay=1.0: all levels get equal weight
        loss, aux = compute_hierarchical_entropy_loss(
            pyramid1, pyramid2, level_weight_decay=1.0
        )

        assert aux["level_weights"] == [1.0, 1.0, 1.0]


class TestHierarchicalLossIntegration:
    """Integration tests for hierarchical loss with model forward pass."""

    def test_full_training_step_with_pyramid(self):
        """Test a complete training step using hierarchical loss."""
        from barevision.flow.model import HierarchicalEmbeddingModel
        from flax import nnx

        model = HierarchicalEmbeddingModel(
            hidden_dim=32, embed_dim=16, num_groups=8, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )

        # Input size for 3 levels targeting 16×16 at coarsest
        img1 = jr.normal(jr.PRNGKey(1), (1, 135, 135, 3))
        img2 = jr.normal(jr.PRNGKey(2), (1, 135, 135, 3))

        def train_loss(m, x1, x2):
            pyramid1 = m(x1)
            pyramid2 = m(x2)
            loss, _ = compute_hierarchical_entropy_loss(pyramid1, pyramid2)
            return loss

        loss = train_loss(model, img1, img2)
        assert jnp.isfinite(loss)

        # Test gradient flow
        grad = jax.grad(train_loss, argnums=(0, 1, 2))(model, img1, img2)
        assert grad[0] is not None  # Model has gradients
        assert jnp.isfinite(grad[1]).all()  # img1 gradients
        assert jnp.isfinite(grad[2]).all()  # img2 gradients
