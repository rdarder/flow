"""Unit tests for hierarchical embedding model architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.flow.model import (
    HierarchicalEmbeddingModel,
    StemBlock,
    StandardBlock,
    count_parameters,
    calculate_required_input_size,
    calculate_coarse_output_size,
)


class TestStemBlock:
    """Tests for StemBlock (Level 0)."""

    def test_stem_output_shapes(self):
        """Test StemBlock returns correct shapes."""
        model = StemBlock(embed_dim=16, rngs=nnx.Rngs(jr.PRNGKey(0)))
        x = jnp.ones((1, 83, 83, 3))  # Example input
        embedding, downsampled = model(x)

        # Embedding: drops 4 pixels (two 3×3 convs), 16 channels
        assert embedding.shape == (1, 79, 79, 16), f"Got {embedding.shape}"
        # Downsampled: (79 - 3) // 2 + 1 = 39, 32 channels
        assert downsampled.shape == (1, 39, 39, 32), f"Got {downsampled.shape}"

    def test_stem_gradient_flow(self):
        """Test gradients flow through StemBlock."""
        from jax import grad

        model = StemBlock(embed_dim=16, rngs=nnx.Rngs(jr.PRNGKey(0)))
        x = jnp.ones((1, 83, 83, 3))

        def loss_fn(m, inp):
            emb, _ = m(inp)
            return emb.sum()

        grads = grad(loss_fn, argnums=0)(model, x)
        grad_state = nnx.state(grads)

        # Check gradients exist
        assert "conv1" in grad_state
        assert "conv2" in grad_state
        assert "embed_conv" in grad_state
        assert "downsample_conv" in grad_state


class TestStandardBlock:
    """Tests for StandardBlock (Levels 1+)."""

    def test_standard_block_output_shapes(self):
        """Test StandardBlock returns correct shapes."""
        model = StandardBlock(
            embed_dim=16, is_last_level=False, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 39, 39, 32))  # From Stem downsample
        embedding, downsampled = model(x)

        # Embedding: drops 2 pixels (one 3×3 conv), 16 channels
        assert embedding.shape == (1, 37, 37, 16), f"Got {embedding.shape}"
        # Downsampled: (37 - 3) // 2 + 1 = 18, 32 channels
        assert downsampled is not None
        assert downsampled.shape == (1, 18, 18, 32), f"Got {downsampled.shape}"

    def test_standard_block_last_level(self):
        """Test StandardBlock at last level returns None for downsample."""
        model = StandardBlock(
            embed_dim=16, is_last_level=True, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 18, 18, 32))
        embedding, downsampled = model(x)

        # Embedding: drops 2 pixels, 16 channels
        assert embedding.shape == (1, 16, 16, 16), f"Got {embedding.shape}"
        # No downsampling at last level
        assert downsampled is None  # type: ignore


class TestHierarchicalEmbeddingModel:
    """Tests for HierarchicalEmbeddingModel forward pass and parameter counting."""

    def test_pyramid_output(self):
        """Test that model returns list of feature maps."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        # For 3 levels targeting 16×16 at coarsest: input must be 83×83
        x = jnp.ones((1, 83, 83, 3))
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

        # Verify exact dimensions
        # Level 0: 83 - 4 = 79
        assert pyramid[0].shape[1:] == (
            79,
            79,
            16,
        ), f"Level 0 shape: {pyramid[0].shape}"
        # Level 1: (79 - 3) // 2 + 1 = 39, then 39 - 2 = 37
        assert pyramid[1].shape[1:] == (
            37,
            37,
            16,
        ), f"Level 1 shape: {pyramid[1].shape}"
        # Level 2: (37 - 3) // 2 + 1 = 18, then 18 - 2 = 16
        assert pyramid[2].shape[1:] == (
            16,
            16,
            16,
        ), f"Level 2 shape: {pyramid[2].shape}"

    def test_single_level(self):
        """Test model with single level."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=1, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        # For 1 level (Stem only) targeting any size: input drops 4
        x = jnp.ones((1, 20, 20, 3))
        pyramid = model(x)

        assert len(pyramid) == 1
        assert pyramid[0].shape[-1] == 16
        # Single StemBlock: 20 - 4 = 16
        assert pyramid[0].shape[1] == 16

    def test_batch_processing(self):
        """Test batch processing."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((4, 83, 83, 3))
        pyramid = model(x)

        assert len(pyramid) == 3
        for level in pyramid:
            assert level.shape[0] == 4, "Batch size should be preserved"

    def test_parameter_count(self):
        """Test parameter counting with new Decoupled Cascade architecture."""
        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        param_count = count_parameters(model)

        # Stem Block:
        #   Conv1 (dense 3×3, 3→32): 3*32*9 + 32 = 896
        #   Norm1 (GroupNorm, 32 ch): 32 + 32 = 64
        #   Conv2 (grouped 3×3, 8g, 32→32): 8*4*4*9 + 32 = 1184
        #   Norm2 (GroupNorm, 32 ch): 64
        #   Embed conv (1×1, 32→16): 32*16 + 16 = 528
        #   Downsample (grouped 3×3, 8g, 32→32): 1184
        #   Stem total: 896 + 64 + 1184 + 64 + 528 + 1184 = 3,920
        #
        # Standard Block 1 (non-last):
        #   Conv1 (grouped 3×3, 8g, 32→32): 1184
        #   Norm1 (GroupNorm, 32 ch): 64
        #   Embed conv (1×1, 32→16): 528
        #   Downsample (grouped 3×3, 8g, 32→32): 1184
        #   Standard 1 total: 1184 + 64 + 528 + 1184 = 2,960
        #
        # Standard Block 2 (last):
        #   Conv1: 1184
        #   Norm1: 64
        #   Embed conv: 528
        #   Standard 2 total: 1184 + 64 + 528 = 1,776
        #
        # Total: 3,920 + 2,960 + 1,776 = 8,656
        assert param_count == 8656, f"Expected 8656 parameters, got {param_count}"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from jax import grad

        model = HierarchicalEmbeddingModel(
            embed_dim=16, in_channels=3, num_levels=3, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 83, 83, 3))

        def loss_fn(m, inp):
            pyramid = m(inp)
            return pyramid[-1].sum()  # Use coarsest level

        # Compute gradients
        grads = grad(loss_fn, argnums=0)(model, x)

        # Check that model has gradients (non-zero)
        grad_state = nnx.state(grads)

        # Check first block (StemBlock) has gradients
        assert "blocks" in grad_state
        stem_grads = grad_state["blocks"][0]  # type: ignore
        assert "conv1" in stem_grads
        assert "conv2" in stem_grads

        # Verify gradients exist for level 0
        level0_conv1 = stem_grads["conv1"]["kernel"]  # type: ignore
        assert level0_conv1 is not None
        assert level0_conv1.shape == (3, 3, 3, 32)


class TestDimensionalMath:
    """Tests for input/output size calculation functions."""

    def test_calculate_required_input_size_3_levels(self):
        """Test input size calculation for 3 levels targeting 16×16."""
        # For 3 levels targeting 16×16 at coarsest:
        # L2: 16 + 2 = 18
        # L1 downsample reverse: (18-1)*2 + 3 = 37
        # L1: 37 + 2 = 39
        # L0 downsample reverse: (39-1)*2 + 3 = 79
        # L0: 79 + 4 = 83
        result = calculate_required_input_size(16, num_levels=3)
        assert result == 83, f"Expected 83, got {result}"

    def test_calculate_required_input_size_3_levels_48x48(self):
        """Test input size calculation for 3 levels targeting 48×48."""
        # For 3 levels targeting 48×48 at coarsest:
        # L2: 48 + 2 = 50
        # L1 downsample reverse: (50-1)*2 + 3 = 101
        # L1: 101 + 2 = 103
        # L0 downsample reverse: (103-1)*2 + 3 = 207
        # L0: 207 + 4 = 211
        result = calculate_required_input_size(48, num_levels=3)
        assert result == 211, f"Expected 211, got {result}"

    def test_calculate_coarse_output_size_3_levels(self):
        """Test output size calculation for 3 levels with 83×83 input."""
        result = calculate_coarse_output_size(83, num_levels=3)
        # L0: 83 - 4 = 79, downsample: (79-3)//2 + 1 = 39
        # L1: 39 - 2 = 37, downsample: (37-3)//2 + 1 = 18
        # L2: 18 - 2 = 16
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
        # Single level (Stem only): input drops 4
        input_size = 20
        output = calculate_coarse_output_size(input_size, num_levels=1)
        assert output == 16, f"Expected 16, got {output}"

        # Reverse calculation
        required = calculate_required_input_size(16, num_levels=1)
        assert required == 20, f"Expected 20, got {required}"
