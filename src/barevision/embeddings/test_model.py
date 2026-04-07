"""Unit tests for hierarchical embedding model architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.embeddings.model import (
    HierarchicalEmbeddingModel,
    EmbeddingBlock,
    count_parameters,
)
from barevision.embeddings.settings import ModelSettings
from barevision.utils.image import (
    calculate_required_input_size,
    calculate_coarse_output_size,
)


class TestEmbeddingBlock:
    """Tests for unified EmbeddingBlock."""

    def test_embedding_block_output_shape_first_level(self):
        """Test EmbeddingBlock returns correct shape for first level (RGB input)."""
        settings = ModelSettings(
            compact_channels=4,
            depthwise_multiplier=8,
            project_groups=4,
            embed_dim=16,
        )
        block = EmbeddingBlock(
            in_channels=3,
            settings=settings,
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        # For 135×135 input: (135-3)//2 + 1 = 67
        x = jnp.ones((1, 135, 135, 3))
        output = block(x)

        assert output.shape == (1, 67, 67, 16), f"Got {output.shape}"

    def test_embedding_block_output_shape_subsequent_level(self):
        """Test EmbeddingBlock returns correct shape for subsequent levels."""
        settings = ModelSettings(
            compact_channels=4,
            depthwise_multiplier=8,
            project_groups=4,
            embed_dim=16,
        )
        block = EmbeddingBlock(
            in_channels=16,
            settings=settings,
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        # For 67×67 input: (67-3)//2 + 1 = 33
        x = jnp.ones((1, 67, 67, 16))
        output = block(x)

        assert output.shape == (1, 33, 33, 16), f"Got {output.shape}"

    def test_embedding_block_gradient_flow(self):
        """Test gradients flow through EmbeddingBlock."""
        from jax import grad

        settings = ModelSettings()
        block = EmbeddingBlock(
            in_channels=16,
            settings=settings,
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        x = jnp.ones((1, 67, 67, 16))

        def loss_fn(m, inp):
            out = m(inp)
            return out.sum()

        grads = grad(loss_fn, argnums=0)(block, x)
        grad_state = nnx.state(grads)

        # Check gradients exist in all layers
        assert "pw_compact" in grad_state
        assert "dw" in grad_state
        assert "pw_project" in grad_state

        # Check gradients are non-zero
        assert not jnp.allclose(grad_state["pw_compact"]["kernel"], 0.0)
        assert not jnp.allclose(grad_state["dw"]["kernel"], 0.0)
        assert not jnp.allclose(grad_state["pw_project"]["kernel"], 0.0)


class TestHierarchicalEmbeddingModel:
    """Tests for HierarchicalEmbeddingModel forward pass and parameter counting."""

    def test_pyramid_output_shapes(self):
        """Test that model returns list of feature maps with correct shapes."""
        settings = ModelSettings(
            compact_channels=4,
            depthwise_multiplier=8,
            project_groups=4,
            embed_dim=16,
            num_levels=3,
        )
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # For 3 levels targeting 16×16 at coarsest: input must be 135×135
        x = jnp.ones((1, 135, 135, 3))
        pyramid = model(x)

        assert isinstance(pyramid, list), "Output should be a list"
        assert len(pyramid) == 3, f"Expected 3 levels, got {len(pyramid)}"

        # Each level should have 16 channels
        for i, level in enumerate(pyramid):
            assert level.shape[-1] == 16, f"Level {i} should have 16 channels"

        # Spatial dimensions should decrease at each level
        # Level 0: (135-3)//2 + 1 = 67
        # Level 1: (67-3)//2 + 1 = 33
        # Level 2: (33-3)//2 + 1 = 16
        assert pyramid[0].shape[1:] == (
            67,
            67,
            16,
        ), f"Level 0 shape: {pyramid[0].shape}"
        assert pyramid[1].shape[1:] == (
            33,
            33,
            16,
        ), f"Level 1 shape: {pyramid[1].shape}"
        assert pyramid[2].shape[1:] == (
            16,
            16,
            16,
        ), f"Level 2 shape: {pyramid[2].shape}"

    def test_pyramid_output_single_level(self):
        """Test model with single level."""
        settings = ModelSettings(
            compact_channels=4,
            depthwise_multiplier=8,
            project_groups=4,
            embed_dim=16,
            num_levels=1,
        )
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # For 1 level targeting 16×16: input = (16-1)*2 + 3 = 33
        x = jnp.ones((1, 33, 33, 3))
        pyramid = model(x)

        assert len(pyramid) == 1
        assert pyramid[0].shape[-1] == 16
        # (33-3)//2 + 1 = 16
        assert pyramid[0].shape[1] == 16

    def test_batch_processing(self):
        """Test batch processing."""
        settings = ModelSettings(num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((4, 135, 135, 3))
        pyramid = model(x)

        assert len(pyramid) == 3
        for level in pyramid:
            assert level.shape[0] == 4, "Batch size should be preserved"

    def test_parameter_count(self):
        """Test parameter counting with simplified architecture."""
        settings = ModelSettings(
            compact_channels=4,
            depthwise_multiplier=8,
            project_groups=4,
            embed_dim=16,
            num_levels=3,
        )
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))
        param_count = count_parameters(model)

        # Block (×3, identical):
        #   PW Compact (16→4 dense, first block 3→4):
        #     First block: 3*4 + 4 = 16
        #     Other blocks: 16*4 + 4 = 68
        #   DW (4 ch × 8 = 32 filters, depthwise 3×3):
        #     32 * 9 + 32 = 320
        #   PW Project (32→16, 4 groups = 4× (8→4)):
        #     4 * (8*4) + 16 = 144
        #   Per block (first): 16 + 320 + 144 = 480
        #   Per block (other): 68 + 320 + 144 = 532
        #
        # Total: 480 + 532 + 532 = 1544

        assert param_count == 1544, f"Expected 1544 parameters, got {param_count}"

    def test_gradient_flow_full_model(self):
        """Test that gradients flow through the full model."""
        from jax import grad

        settings = ModelSettings(num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))
        x = jnp.ones((1, 135, 135, 3))

        def loss_fn(m, inp):
            pyramid = m(inp)
            return pyramid[-1].sum()  # Use coarsest level

        # Compute gradients
        grads = grad(loss_fn, argnums=0)(model, x)

        # Check that model has gradients
        grad_state = nnx.state(grads)

        # Check blocks have gradients
        assert "blocks" in grad_state
        block0_grads = grad_state["blocks"][0]  # type: ignore
        assert "pw_compact" in block0_grads
        assert "dw" in block0_grads
        assert "pw_project" in block0_grads

        # Verify gradients exist and are non-zero
        level0_compact = block0_grads["pw_compact"]["kernel"]  # type: ignore
        assert level0_compact is not None
        assert not jnp.allclose(level0_compact, 0.0)


class TestDimensionalMath:
    """Tests for input/output size calculation functions."""

    def test_calculate_required_input_size_3_levels(self):
        """Test input size calculation for 3 levels targeting 16×16."""
        # For 3 levels targeting 16×16 at coarsest:
        # L2: (16-1)*2 + 3 = 33
        # L1: (33-1)*2 + 3 = 67
        # L0: (67-1)*2 + 3 = 135
        result = calculate_required_input_size(16, num_levels=3)
        assert result == 135, f"Expected 135, got {result}"

    def test_calculate_required_input_size_3_levels_48x48(self):
        """Test input size calculation for 3 levels targeting 48×48."""
        # For 3 levels targeting 48×48 at coarsest:
        # L2: (48-1)*2 + 3 = 97
        # L1: (97-1)*2 + 3 = 195
        # L0: (195-1)*2 + 3 = 391
        result = calculate_required_input_size(48, num_levels=3)
        assert result == 391, f"Expected 391, got {result}"

    def test_calculate_coarse_output_size_3_levels(self):
        """Test output size calculation for 3 levels with 135×135 input."""
        result = calculate_coarse_output_size(135, num_levels=3)
        # L0: (135-3)//2 + 1 = 67
        # L1: (67-3)//2 + 1 = 33
        # L2: (33-3)//2 + 1 = 16
        assert result == 16, f"Expected 16, got {result}"

    def test_roundtrip_consistency(self):
        """Test that input/output calculations are consistent."""
        target = 16
        num_levels = 3

        # Calculate required input for target
        required_input = calculate_required_input_size(target, num_levels)

        # Calculate output from that input
        output = calculate_coarse_output_size(required_input, num_levels)

        assert (
            output == target
        ), f"Roundtrip failed: {required_input} -> {output} (expected {target})"

    def test_single_level_math(self):
        """Test dimensional math for single level."""
        # Single level: input = (16-1)*2 + 3 = 33
        input_size = 33
        output = calculate_coarse_output_size(input_size, num_levels=1)
        assert output == 16, f"Expected 16, got {output}"

        # Reverse calculation
        required = calculate_required_input_size(16, num_levels=1)
        assert required == 33, f"Expected 33, got {required}"
