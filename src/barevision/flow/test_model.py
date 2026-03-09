"""Unit tests for embedding model architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.flow.model import SimpleEmbeddingModel, count_parameters


class TestSimpleEmbeddingModel:
    """Tests for SimpleEmbeddingModel forward pass and parameter counting."""

    def test_rgb_forward_pass(self):
        """Test forward pass with RGB input."""
        model = SimpleEmbeddingModel(
            embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 32, 32, 3))
        y = model(x)

        assert y.shape == (1, 28, 28, 16), f"Expected (1, 28, 28, 16), got {y.shape}"

    def test_grayscale_forward_pass(self):
        """Test forward pass with grayscale input."""
        model = SimpleEmbeddingModel(
            embed_dim=16, in_channels=1, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 32, 32, 1))
        y = model(x)

        assert y.shape == (1, 28, 28, 16), f"Expected (1, 28, 28, 16), got {y.shape}"

    def test_batch_processing(self):
        """Test batch processing."""
        model = SimpleEmbeddingModel(
            embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((4, 64, 64, 3))
        y = model(x)

        assert y.shape == (4, 60, 60, 16), f"Expected (4, 60, 60, 16), got {y.shape}"

    def test_parameter_count(self):
        """Test parameter counting."""
        model = SimpleEmbeddingModel(
            embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        param_count = count_parameters(model)

        # depthwise: 3 * 16 * 25 = 1200 weights + 48 bias = 1248
        # pointwise: 48 * 16 = 768 weights + 16 bias = 784
        # total: 2032
        assert param_count == 2032, f"Expected 2032 parameters, got {param_count}"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from jax import grad

        model = SimpleEmbeddingModel(
            embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 32, 32, 3))

        def loss_fn(m, inp):
            return m(inp).sum()

        # Compute gradients
        grads = grad(loss_fn, argnums=0)(model, x)

        # Check that model has gradients (non-zero)
        grad_state = nnx.state(grads)
        depthwise_kernel = grad_state["depthwise_conv"]["kernel"]
        pointwise_kernel = grad_state["pointwise_conv"]["kernel"]

        # Verify gradients exist and are non-zero
        assert depthwise_kernel.shape == (5, 5, 1, 48)
        assert pointwise_kernel.shape == (1, 1, 48, 16)

        # Extract array values from Param objects
        dw_value = (
            depthwise_kernel.get_value()
            if hasattr(depthwise_kernel, "get_value")
            else depthwise_kernel[...]
        )
        pw_value = (
            pointwise_kernel.get_value()
            if hasattr(pointwise_kernel, "get_value")
            else pointwise_kernel[...]
        )

        is_nonzero_dw = jnp.any(dw_value != 0)
        is_nonzero_pw = jnp.any(pw_value != 0)
        assert bool(is_nonzero_dw), "Zero gradients in depthwise_conv.kernel"
        assert bool(is_nonzero_pw), "Zero gradients in pointwise_conv.kernel"
