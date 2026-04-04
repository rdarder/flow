"""Unit tests for hierarchical embedding model architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.embeddings.model import (
    HierarchicalEmbeddingModel,
    StemBlock,
    StandardBlock,
    count_parameters,
)
from barevision.embeddings.gaussian import (
    gaussian_kernel_2d,
    depthwise_gaussian_initializer,
)
from barevision.embeddings.settings import ModelSettings
from barevision.utils.image import (
    calculate_required_input_size,
    calculate_coarse_output_size,
)


class TestStemBlock:
    """Tests for StemBlock (Level 0)."""

    def test_stem_output_shapes(self):
        """Test StemBlock returns correct shapes."""
        model = StemBlock(
            hidden_dim=32, embed_dim=16, num_groups=8, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 83, 83, 3))  # Example input
        embedding, downsampled = model(x)

        # Embedding: drops 4 pixels (two 3×3 convs), 16 channels
        assert embedding.shape == (1, 79, 79, 16), f"Got {embedding.shape}"
        # Downsampled: (79 - 3) // 2 + 1 = 39, 32 channels
        assert downsampled.shape == (1, 39, 39, 32), f"Got {downsampled.shape}"

    def test_stem_gradient_flow(self):
        """Test gradients flow through StemBlock."""
        from jax import grad

        model = StemBlock(
            hidden_dim=32, embed_dim=16, num_groups=8, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
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
        assert "mean_conv" in grad_state  # mean_conv is learnable


class TestStandardBlock:
    """Tests for StandardBlock (Levels 1+)."""

    def test_standard_block_output_shapes(self):
        """Test StandardBlock returns correct shapes."""
        model = StandardBlock(
            hidden_dim=32,
            embed_dim=16,
            num_groups=8,
            is_last_level=False,
            rngs=nnx.Rngs(jr.PRNGKey(0)),
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
            hidden_dim=32,
            embed_dim=16,
            num_groups=8,
            is_last_level=True,
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        x = jnp.ones((1, 18, 18, 32))
        embedding, downsampled = model(x)

        # Embedding: drops 2 pixels, 16 channels
        assert embedding.shape == (1, 16, 16, 16), f"Got {embedding.shape}"
        # No downsampling at last level
        assert downsampled is None  # type: ignore


class TestSymmetricMeanSubtraction:
    """Tests for Local Contrast Normalization and mean subtraction."""

    def test_gaussian_kernel_properties(self):
        """Test Gaussian kernel initialization."""
        kernel = gaussian_kernel_2d(sigma=1.0)

        # Should be 3x3
        assert kernel.shape == (3, 3)

        # Should sum to 1.0
        assert jnp.allclose(jnp.sum(kernel), 1.0)

        # Center should be highest value
        center = kernel[1, 1]
        assert all(
            center > kernel[i, j]
            for i in range(3)
            for j in range(3)
            if (i, j) != (1, 1)
        )

        # Corners should be lowest values
        corner = kernel[0, 0]
        assert all(
            corner <= kernel[i, j]
            for i in range(3)
            for j in range(3)
            if (i, j) not in [(0, 0), (0, 2), (2, 0), (2, 2)]
        )

    def test_depthwise_gaussian_initializer(self):
        """Test depthwise Gaussian initializer creates proper depthwise kernels."""
        init_fn = depthwise_gaussian_initializer(sigma=1.0)

        # Create kernel for 32 input/output channels
        key = jr.PRNGKey(0)
        # Shape: (height, width, in_features, out_features)
        input_shape = (3, 3, 32, 32)
        kernel = init_fn(key, input_shape)

        # For depthwise convolution, Flax uses (3, 3, 1, out_features)
        # Each output channel has its own 3x3 kernel
        assert kernel.shape == (3, 3, 1, 32)

        # All channels should have the same Gaussian kernel (broadcasted)
        # Each channel's kernel should sum to 1.0
        for i in range(32):
            channel_sum = jnp.sum(kernel[:, :, 0, i])
            assert jnp.allclose(channel_sum, 1.0), f"Channel {i} should sum to 1.0"

        # Center should be highest value in all channels
        for i in range(32):
            center = kernel[1, 1, 0, i]
            assert all(
                center >= kernel[h, w, 0, i] for h in range(3) for w in range(3)
            ), f"Center should be highest in channel {i}"

    def test_lcn_subtraction_preserves_dimensions(self):
        """Test that LCN subtraction preserves spatial dimensions."""
        model = StemBlock(
            hidden_dim=32, embed_dim=16, num_groups=8, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 83, 83, 3))

        # Get intermediate features by running forward pass manually
        h = model.conv1(x)
        h = model.norm1(h)
        h = nnx.gelu(h)
        h = model.conv2(h)
        h = model.norm2(h)
        rich_features = nnx.gelu(h)

        # mean_conv with SAME padding should preserve dimensions
        local_mean = model.mean_conv(rich_features)
        assert (
            local_mean.shape == rich_features.shape
        ), "mean_conv should preserve dimensions with SAME padding"

        # Subtraction should preserve dimensions
        x_unique = rich_features - local_mean
        assert x_unique.shape == rich_features.shape

    def test_strided_slice_downsampling(self):
        """Test that strided slice produces correct downsampling dimensions."""
        model = StemBlock(
            hidden_dim=32, embed_dim=16, num_groups=8, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 83, 83, 3))

        embedding, downsampled = model(x)

        # After mean_conv (SAME): still 79x79
        # After strided slice [1:-1:2]: (79-2)//2 = 38... wait let me recalculate
        # For H=79: indices [1, 3, 5, ..., 77] → 39 values
        # This should match (79-3)//2 + 1 = 39 (original VALID stride=2 conv)
        assert downsampled.shape == (1, 39, 39, 32), f"Got {downsampled.shape}"

    def test_lcn_removes_low_frequency_content(self):
        """Test that LCN subtraction removes common background signals."""
        model = StemBlock(
            hidden_dim=32, embed_dim=16, num_groups=8, rngs=nnx.Rngs(jr.PRNGKey(0))
        )

        # Create input with uniform signal + small unique variation
        uniform = jnp.ones((1, 20, 20, 32)) * 5.0  # Strong uniform signal
        variation = jr.normal(jr.PRNGKey(1), (1, 20, 20, 32)) * 0.1  # Small variation
        rich_features = uniform + variation

        # Apply mean convolution (should extract the uniform component)
        local_mean = model.mean_conv(rich_features)

        # After subtraction, mean should be close to zero
        x_unique = rich_features - local_mean

        # The uniform component should be mostly removed
        # Mean of residuals should be much smaller than original mean
        original_mean = jnp.mean(jnp.abs(rich_features))
        residual_mean = jnp.mean(jnp.abs(x_unique))

        assert residual_mean < original_mean, "LCN should reduce mean magnitude"

    def test_mean_conv_is_learnable(self):
        """Test that mean_conv parameters are trainable."""
        from jax import grad

        model = StemBlock(
            hidden_dim=32, embed_dim=16, num_groups=8, rngs=nnx.Rngs(jr.PRNGKey(0))
        )
        x = jnp.ones((1, 83, 83, 3))

        def loss_fn(m, inp):
            emb, _ = m(inp)
            return emb.sum()

        grads = grad(loss_fn, argnums=0)(model, x)
        grad_state = nnx.state(grads)

        # mean_conv should have gradients
        assert "mean_conv" in grad_state
        assert "kernel" in grad_state["mean_conv"]

        # Gradients should be non-zero
        mean_grad = grad_state["mean_conv"]["kernel"]
        assert not jnp.allclose(
            mean_grad, 0.0
        ), "mean_conv should have non-zero gradients"


class TestHierarchicalEmbeddingModel:
    """Tests for HierarchicalEmbeddingModel forward pass and parameter counting."""

    def test_pyramid_output(self):
        """Test that model returns list of feature maps."""
        model = HierarchicalEmbeddingModel(
            ModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=3,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
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
            ), f"Level {i} should be larger than level {i + 1}"

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
            ModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=1,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
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
            ModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=3,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        x = jnp.ones((4, 83, 83, 3))
        pyramid = model(x)

        assert len(pyramid) == 3
        for level in pyramid:
            assert level.shape[0] == 4, "Batch size should be preserved"

    def test_parameter_count(self):
        """Test parameter counting with Symmetric Mean Subtraction architecture."""
        model = HierarchicalEmbeddingModel(
            ModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=3,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
        )
        param_count = count_parameters(model)

        # Stem Block:
        #   Conv1 (dense 3×3, 3→32): 3*32*9 + 32 = 896
        #   Norm1 (GroupNorm, 32 ch): 32 + 32 = 64
        #   Conv2 (grouped 3×3, 8g, 32→32): 8*4*4*9 + 32 = 1184
        #   Norm2 (GroupNorm, 32 ch): 64
        #   Mean Conv (depthwise 3×3, 32g, 32→32): 32*9 + 32 = 320
        #   Embed conv (1×1, 32→16): 32*16 + 16 = 528
        #   Stem total: 896 + 64 + 1184 + 64 + 320 + 528 = 3,056
        #
        # Standard Block 1 (non-last):
        #   Conv1 (grouped 3×3, 8g, 32→32): 1184
        #   Norm1 (GroupNorm, 32 ch): 64
        #   Mean Conv (depthwise 3×3, 32g, 32→32): 320
        #   Embed conv (1×1, 32→16): 528
        #   Standard 1 total: 1184 + 64 + 320 + 528 = 2,096
        #
        # Standard Block 2 (last):
        #   Conv1: 1184
        #   Norm1: 64
        #   Mean Conv: 320
        #   Embed conv: 528
        #   Standard 2 total: 1184 + 64 + 320 + 528 = 2,096
        #
        # Total: 3,056 + 2,096 + 2,096 = 7,248
        assert param_count == 7248, f"Expected 7248 parameters, got {param_count}"

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from jax import grad

        model = HierarchicalEmbeddingModel(
            ModelSettings(
                hidden_dim=32,
                embed_dim=16,
                num_groups=8,
                num_levels=3,
            ),
            rngs=nnx.Rngs(jr.PRNGKey(0)),
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
        assert "mean_conv" in stem_grads  # New: mean_conv should have gradients

        # Verify gradients exist for level 0
        level0_conv1 = stem_grads["conv1"]["kernel"]  # type: ignore
        assert level0_conv1 is not None
        assert level0_conv1.shape == (3, 3, 3, 32)

        # Verify mean_conv gradients exist
        level0_mean = stem_grads["mean_conv"]["kernel"]  # type: ignore
        assert level0_mean is not None
        # Depthwise convolution: feature_group_count=32 creates (3, 3, 1, 32) per group
        # But total shape is (3, 3, 1, 32) since each group processes 1 input channel
        assert level0_mean.shape == (3, 3, 1, 32)


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
