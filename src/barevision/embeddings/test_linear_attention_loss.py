"""Tests for linear attention flow loss.

Tests verify:
1. Linear attention flow computation is correct
2. Warped reconstruction loss works with bilinear interpolation
3. Embedding diversity loss prevents collapse
4. Hierarchical loss aggregates correctly across levels
5. Gradients flow properly (no NaN/Inf)
"""

import jax
import jax.numpy as jnp
import pytest

from barevision.embeddings.linear_attention_loss import (
    _compute_linear_attention_flow,
    _compute_warped_reconstruction_loss,
    _compute_embedding_diversity_loss,
    _compute_linear_attention_flow_loss,
    compute_hierarchical_linear_attention_loss,
    HierarchicalLinearAttentionFlowLoss,
    LinearAttentionFlowLossConfig,
)
from barevision.utils.grid import generate_normalized_coordinates


class TestCoordinateGeneration:
    """Test coordinate grid generation."""

    def test_coordinates_shape(self):
        """Coordinates should have shape (N, 2) where N = window_size^2."""
        window_size = 4
        coords = generate_normalized_coordinates(window_size)

        N = window_size * window_size
        assert coords.shape == (N, 2)

    def test_coordinates_normalized_to_unit_interval(self):
        """Coordinates should be in [0, 1] range."""
        window_size = 16
        coords = generate_normalized_coordinates(window_size)

        assert jnp.all(coords >= 0.0)
        assert jnp.all(coords <= 1.0)


class TestLinearAttentionFlow:
    """Test linear attention flow computation."""

    def test_flow_shape(self):
        """Flow should have shape (B*num_windows, H_w, W_w, 2)."""
        Bnw, H_w, W_w, D = 2, 4, 4, 16
        q = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        k = jax.random.normal(jax.random.PRNGKey(1), (Bnw, H_w, W_w, D))
        coords = generate_normalized_coordinates(H_w)

        flow, confidence = _compute_linear_attention_flow(q, k, coords, H_w)

        assert flow.shape == (Bnw, H_w, W_w, 2)
        assert confidence.shape == (Bnw, H_w, W_w)

    def test_flow_with_identical_embeddings(self):
        """Identical embeddings should produce near-zero flow."""
        Bnw, H_w, W_w, D = 1, 8, 8, 16
        embeddings = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        coords = generate_normalized_coordinates(H_w)

        flow, _ = _compute_linear_attention_flow(embeddings, embeddings, coords, H_w)

        # Flow should be close to zero (self-matching)
        assert jnp.all(jnp.abs(flow) < 1.0)

    def test_confidence_is_negative_variance(self):
        """Confidence should be negative (variance is positive)."""
        Bnw, H_w, W_w, D = 2, 4, 4, 16
        q = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        k = jax.random.normal(jax.random.PRNGKey(1), (Bnw, H_w, W_w, D))
        coords = generate_normalized_coordinates(H_w)

        _, confidence = _compute_linear_attention_flow(q, k, coords, H_w)

        # Confidence is negative variance
        assert jnp.all(confidence <= 0.0)


class TestWarpedReconstruction:
    """Test warped reconstruction loss."""

    def test_warp_with_zero_flow(self):
        """Zero flow should return original embeddings."""
        Bnw, H_w, W_w, D = 1, 8, 8, 16
        embeddings = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        flow = jnp.zeros((Bnw, H_w, W_w, 2))
        window_size = 8

        from barevision.embeddings.linear_attention_loss import _warp_embeddings

        warped = _warp_embeddings(embeddings, flow, window_size)

        # With zero flow, warped should be very close to original
        assert jnp.allclose(warped, embeddings, rtol=1e-5)

    def test_warp_with_constant_flow(self):
        """Constant flow should shift all embeddings uniformly."""
        Bnw, H_w, W_w, D = 1, 8, 8, 16
        embeddings = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        flow = jnp.ones((Bnw, H_w, W_w, 2)) * 0.5  # Half-pixel shift
        window_size = 8

        from barevision.embeddings.linear_attention_loss import _warp_embeddings

        warped = _warp_embeddings(embeddings, flow, window_size)

        # Warped should be different from original
        assert not jnp.allclose(warped, embeddings, rtol=1e-5)

    def test_reconstruction_loss_with_perfect_alignment(self):
        """Perfect alignment (zero flow) should give zero reconstruction loss."""
        Bnw, H_w, W_w, D = 2, 8, 8, 16
        embeddings = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        flow = jnp.zeros((Bnw, H_w, W_w, 2))
        window_size = 8

        loss = _compute_warped_reconstruction_loss(
            embeddings, embeddings, flow, window_size
        )

        # Reconstructing self with zero flow should be perfect
        assert loss < 1e-5


class TestEmbeddingDiversity:
    """Test embedding diversity loss."""

    def test_diversity_loss_with_constant_embeddings(self):
        """Constant embeddings should have high diversity loss (low variance)."""
        Bnw, H_w, W_w, D = 1, 8, 8, 16
        constant = jnp.ones((Bnw, H_w, W_w, D))

        loss = _compute_embedding_diversity_loss(constant, scope="per_window")

        # Variance of constant is zero, so loss is -0 = 0
        assert jnp.isclose(loss, 0.0, atol=1e-6)

    def test_diversity_loss_with_varied_embeddings(self):
        """Varied embeddings should have low diversity loss (high variance)."""
        Bnw, H_w, W_w, D = 1, 8, 8, 16
        varied = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))

        loss = _compute_embedding_diversity_loss(varied, scope="per_window")

        # Variance should be positive, so loss is negative
        assert loss < 0.0

    def test_diversity_scope_per_window_vs_global(self):
        """Per-window and global scope should give different results."""
        Bnw, H_w, W_w, D = 4, 8, 8, 16
        embeddings = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))

        loss_per_window = _compute_embedding_diversity_loss(
            embeddings, scope="per_window"
        )
        loss_global = _compute_embedding_diversity_loss(embeddings, scope="global")

        # They should be different (unless all windows have identical statistics)
        assert not jnp.isclose(loss_per_window, loss_global, rtol=1e-5)


class TestSingleLevelLoss:
    """Test single-level linear attention flow loss."""

    def test_loss_components(self):
        """Loss should return both reconstruction and diversity components."""
        B, H, W, D = 2, 16, 16, 16
        emb1 = jax.random.normal(jax.random.PRNGKey(0), (B, H, W, D))
        emb2 = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, D))
        config = LinearAttentionFlowLossConfig()

        loss, aux = _compute_linear_attention_flow_loss(emb1, emb2, config)

        assert jnp.isscalar(loss) or loss.shape == ()
        assert "reconstruction_loss" in aux
        assert "diversity_loss" in aux
        assert "flow" in aux
        assert "confidence" in aux

    def test_loss_with_identical_frames(self):
        """Identical frames should have low reconstruction loss."""
        B, H, W, D = 2, 16, 16, 16
        emb = jax.random.normal(jax.random.PRNGKey(0), (B, H, W, D))
        config = LinearAttentionFlowLossConfig(
            lambda_reconstruction=1.0, lambda_diversity=0.0
        )

        loss, aux = _compute_linear_attention_flow_loss(emb, emb, config)

        # Reconstruction should be low (matching self)
        assert aux["reconstruction_loss"] < 1.0


class TestHierarchicalLoss:
    """Test hierarchical linear attention flow loss."""

    def test_hierarchical_loss_with_two_levels(self):
        """Hierarchical loss should aggregate across levels."""
        B, D = 2, 16

        pyramid1 = [
            jax.random.normal(jax.random.PRNGKey(0), (B, 32, 32, D)),
            jax.random.normal(jax.random.PRNGKey(1), (B, 16, 16, D)),
        ]
        pyramid2 = [
            jax.random.normal(jax.random.PRNGKey(2), (B, 32, 32, D)),
            jax.random.normal(jax.random.PRNGKey(3), (B, 16, 16, D)),
        ]

        config = LinearAttentionFlowLossConfig(
            window_size=16,
            level_weight_decay=1.0,
        )

        loss, aux = compute_hierarchical_linear_attention_loss(
            pyramid1, pyramid2, config
        )

        assert jnp.isscalar(loss) or loss.shape == ()
        assert len(aux["level_losses"]) == 2
        assert len(aux["level_weights"]) == 2

    def test_hierarchical_loss_level_weighting(self):
        """Level weight decay should affect contribution."""
        B, D = 2, 16

        pyramid1 = [
            jax.random.normal(jax.random.PRNGKey(0), (B, 32, 32, D)),
            jax.random.normal(jax.random.PRNGKey(1), (B, 16, 16, D)),
        ]
        pyramid2 = [
            jax.random.normal(jax.random.PRNGKey(2), (B, 32, 32, D)),
            jax.random.normal(jax.random.PRNGKey(3), (B, 16, 16, D)),
        ]

        config_uniform = LinearAttentionFlowLossConfig(
            window_size=16,
            level_weight_decay=1.0,
        )
        loss_uniform, aux_uniform = compute_hierarchical_linear_attention_loss(
            pyramid1, pyramid2, config_uniform
        )

        config_weighted = LinearAttentionFlowLossConfig(
            window_size=16,
            level_weight_decay=2.0,
        )
        loss_weighted, aux_weighted = compute_hierarchical_linear_attention_loss(
            pyramid1, pyramid2, config_weighted
        )

        assert aux_uniform["level_weights"] == [1.0, 1.0]
        assert aux_weighted["level_weights"] == [1.0, 2.0]

    def test_hierarchical_loss_class_interface(self):
        """HierarchicalLinearAttentionFlowLoss class should work correctly."""
        config = LinearAttentionFlowLossConfig()
        loss_fn = HierarchicalLinearAttentionFlowLoss(config)

        B, D = 2, 16
        pyramid1 = [jax.random.normal(jax.random.PRNGKey(0), (B, 32, 32, D))]
        pyramid2 = [jax.random.normal(jax.random.PRNGKey(1), (B, 32, 32, D))]

        loss, aux = loss_fn((pyramid1, pyramid2), need_aux=True)

        assert jnp.isscalar(loss) or loss.shape == ()
        assert "reconstruction_loss" in aux
        assert "diversity_loss" in aux


class TestGradientFlow:
    """Test that gradients flow properly through the loss."""

    def test_gradients_through_flow_computation(self):
        """Gradients should flow through linear attention flow."""
        Bnw, H_w, W_w, D = 2, 8, 8, 16
        q = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        k = jax.random.normal(jax.random.PRNGKey(1), (Bnw, H_w, W_w, D))
        coords = generate_normalized_coordinates(H_w)

        def loss_fn(q_in, k_in):
            flow, _ = _compute_linear_attention_flow(q_in, k_in, coords, H_w)
            return jnp.mean(flow**2)

        grad_q, grad_k = jax.grad(loss_fn, argnums=(0, 1))(q, k)

        assert grad_q.shape == q.shape
        assert grad_k.shape == k.shape
        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isinf(grad_q))

    def test_gradients_through_warped_reconstruction(self):
        """Gradients should flow through warped reconstruction."""
        Bnw, H_w, W_w, D = 1, 8, 8, 16
        emb1 = jax.random.normal(jax.random.PRNGKey(0), (Bnw, H_w, W_w, D))
        emb2 = jax.random.normal(jax.random.PRNGKey(1), (Bnw, H_w, W_w, D))
        flow = jax.random.normal(jax.random.PRNGKey(2), (Bnw, H_w, W_w, 2))
        window_size = 8

        def loss_fn(e1, e2, f):
            return _compute_warped_reconstruction_loss(e1, e2, f, window_size)

        grad1, grad2, grad_flow = jax.grad(loss_fn, argnums=(0, 1, 2))(
            emb1, emb2, flow
        )

        assert grad1.shape == emb1.shape
        assert grad2.shape == emb2.shape
        assert grad_flow.shape == flow.shape
        assert not jnp.any(jnp.isnan(grad1))
        assert not jnp.any(jnp.isnan(grad2))

    def test_gradients_through_hierarchical_loss(self):
        """Gradients should flow through hierarchical loss."""
        config = LinearAttentionFlowLossConfig()
        loss_fn_obj = HierarchicalLinearAttentionFlowLoss(config)

        B, D = 2, 16
        pyramid1 = [
            jax.random.normal(jax.random.PRNGKey(0), (B, 32, 32, D)),
            jax.random.normal(jax.random.PRNGKey(1), (B, 16, 16, D)),
        ]
        pyramid2 = [
            jax.random.normal(jax.random.PRNGKey(2), (B, 32, 32, D)),
            jax.random.normal(jax.random.PRNGKey(3), (B, 16, 16, D)),
        ]

        def loss_fn(p1, p2):
            loss, _ = loss_fn_obj((p1, p2), need_aux=False)
            return loss

        grads = jax.grad(loss_fn, argnums=0)(pyramid1, pyramid2)

        assert len(grads) == len(pyramid1)
        for grad, emb in zip(grads, pyramid1):
            assert grad.shape == emb.shape
            assert not jnp.any(jnp.isnan(grad))
            assert not jnp.any(jnp.isinf(grad))


class TestLossWeighting:
    """Test loss component weighting."""

    def test_lambda_reconstruction_weighting(self):
        """Lambda_reconstruction should scale reconstruction loss contribution."""
        B, H, W, D = 2, 16, 16, 16
        emb1 = jax.random.normal(jax.random.PRNGKey(0), (B, H, W, D))
        emb2 = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, D))

        config_no_div = LinearAttentionFlowLossConfig(
            lambda_reconstruction=1.0,
            lambda_diversity=0.0,
        )

        loss, aux = _compute_linear_attention_flow_loss(emb1, emb2, config_no_div)

        # With lambda_diversity=0, total should equal reconstruction
        assert jnp.isclose(loss, aux["reconstruction_loss"], rtol=1e-5)

    def test_lambda_diversity_weighting(self):
        """Lambda_diversity should scale diversity loss contribution."""
        B, H, W, D = 2, 16, 16, 16
        emb1 = jax.random.normal(jax.random.PRNGKey(0), (B, H, W, D))
        emb2 = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, D))

        config_no_rec = LinearAttentionFlowLossConfig(
            lambda_reconstruction=0.0,
            lambda_diversity=1.0,
        )

        loss, aux = _compute_linear_attention_flow_loss(emb1, emb2, config_no_rec)

        # With lambda_reconstruction=0, total should equal diversity
        assert jnp.isclose(loss, aux["diversity_loss"], rtol=1e-5)
