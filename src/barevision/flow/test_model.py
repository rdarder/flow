"""Unit tests for hierarchical embedding model architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.flow.model import HierarchicalEmbeddingModel, count_parameters


class TestHierarchicalEmbeddingModel:
    """Tests for HierarchicalEmbeddingModel forward pass and parameter counting."""

    def test_pyramid_output(self):
        """Test that model returns list of feature maps."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 391, 391, 3))  # Input size for 3 levels, 48×48 coarse
        pyramid = model(x)

        assert isinstance(pyramid, list), "Output should be a list"
        assert len(pyramid) == 3, f"Expected 3 levels, got {len(pyramid)}"

        # Each level should have 16 channels
        for i, level in enumerate(pyramid):
            assert level.shape[-1] == 16, f"Level {i} should have 16 channels"

        # Spatial dimensions should decrease at each level
        for i in range(len(pyramid) - 1):
            assert (
                pyramid[i].shape[1] > pyramid[i + 1].shape[1]
            ), f"Level {i} should be larger than level {i+1}"

    def test_single_level(self):
        """Test model with single level."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=1, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 50, 50, 3))
        pyramid = model(x)

        assert len(pyramid) == 1
        assert pyramid[0].shape[-1] == 16

    def test_batch_processing(self):
        """Test batch processing."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((4, 391, 391, 3))
        pyramid = model(x)

        assert len(pyramid) == 3
        for level in pyramid:
            assert level.shape[0] == 4, "Batch size should be preserved"

    def test_parameter_count(self):
        """Test parameter counting."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        param_count = count_parameters(model)

        # Each level: 3×3 conv + 1×1 conv
        # Level 0: 3→16 channels: 3*16*9 + 16 + 16*16 + 16 = 432 + 16 + 256 + 16 = 720
        # Level 1,2: 16→16 channels: 16*16*9 + 16 + 16*16 + 16 = 2304 + 16 + 256 + 16 = 2592 each
        # Total: 720 + 2592 + 2592 = 5904
        assert param_count == 5904, f"Expected 5904 parameters, got {param_count}"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from jax import grad

        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 391, 391, 3))

        def loss_fn(m, inp):
            pyramid = m(inp)
            return pyramid[-1].sum()  # Use coarsest level

        # Compute gradients
        grads = grad(loss_fn, argnums=0)(model, x)

        # Check that model has gradients (non-zero)
        grad_state = nnx.state(grads)

        # Check first level has gradients (now stored in lists)
        assert "spatial_convs" in grad_state
        assert "pointwise_convs" in grad_state

        # Verify gradients exist for level 0
        level0_spatial = grad_state["spatial_convs"][0]["kernel"]
        assert level0_spatial is not None
