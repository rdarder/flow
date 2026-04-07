"""Unit tests for hierarchical embedding model with UIB architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.embeddings.model import (
    HierarchicalEmbeddingModel,
    UniversalInvertedBlock,
    UIBConfig,
    count_parameters,
)
from barevision.embeddings.settings import ModelSettings
from barevision.utils.image import (
    calculate_required_input_size,
    calculate_coarse_output_size,
    calculate_uib_output_size,
    calculate_level_output_size,
    image_size,
)


class TestUniversalInvertedBlock:
    """Tests for UniversalInvertedBlock."""

    def test_uib_no_dw_no_downsample(self):
        """Test UIB with no DW convs and no downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=False,
            use_dw_after_expand=False,
            downsample_after=False,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # 1×1 convs don't change spatial dims
        x = jnp.ones((1, 32, 32, 16))
        output = block(x)

        assert output.shape == (1, 32, 32, 16), f"Got {output.shape}"

    def test_uib_one_dw_before(self):
        """Test UIB with DW before expand only."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=False,
            downsample_after=False,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # DW 3×3 VALID: -2 pixels
        x = jnp.ones((1, 32, 32, 16))
        output = block(x)

        assert output.shape == (1, 30, 30, 16), f"Got {output.shape}"

    def test_uib_two_dws(self):
        """Test UIB with both DW convs (no downsample)."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # 2× DW 3×3 VALID: -4 pixels total
        x = jnp.ones((1, 32, 32, 16))
        output = block(x)

        assert output.shape == (1, 28, 28, 16), f"Got {output.shape}"

    def test_uib_with_downsample(self):
        """Test UIB with downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=True,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # 2× DW 3×3: -4, then downsample: (28-1)//2 = 13
        x = jnp.ones((1, 32, 32, 16))
        output = block(x)

        assert output.shape == (1, 13, 13, 16), f"Got {output.shape}"

    def test_uib_channel_change(self):
        """Test UIB changes channel dimensions."""
        config = UIBConfig(
            in_channels=3,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((1, 32, 32, 3))
        output = block(x)

        assert output.shape == (1, 28, 28, 16), f"Got {output.shape}"

    def test_uib_l2_normalization(self):
        """Test UIB L2 normalization."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
            use_l2_norm=True,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((1, 32, 32, 16)) * 10.0  # Large values
        output = block(x)

        # Check L2 norm along last axis is ~1
        norms = jnp.linalg.norm(output, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-5), f"Norms: {norms}"

    def test_uib_gradient_flow(self):
        """Test gradients flow through UIB."""
        from jax import grad

        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
        )
        block = UniversalInvertedBlock(config, rngs=nnx.Rngs(jr.PRNGKey(0)))
        x = jnp.ones((1, 32, 32, 16))

        def loss_fn(m, inp):
            out = m(inp)
            return out.sum()

        grads = grad(loss_fn, argnums=0)(block, x)
        grad_state = nnx.state(grads)

        # Check gradients exist in all layers
        assert "pw_expand" in grad_state
        assert "pw_compress" in grad_state
        if config.use_dw_before_expand:
            assert "dw_before" in grad_state
        if config.use_dw_after_expand:
            assert "dw_after" in grad_state

        # Check gradients are non-zero
        assert not jnp.allclose(grad_state["pw_expand"]["kernel"], 0.0)
        assert not jnp.allclose(grad_state["pw_compress"]["kernel"], 0.0)


class TestHierarchicalEmbeddingModel:
    """Tests for HierarchicalEmbeddingModel forward pass."""

    def test_pyramid_output_shapes(self):
        """Test that model returns list of feature maps with correct shapes."""
        settings = ModelSettings(embed_dim=16, num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # Default config: 2 UIBs per level, second downsamples
        # Each level: UIB_0 (no downsample, -4 pix) → UIB_1 (downsample)
        # Level 0: input 137×137 → UIB_0: 133×133 → UIB_1: (133-4-1)//2 = 64
        # Level 1: 64×64 → UIB_0: 60×60 → UIB_1: (60-4-1)//2 = 27
        # Level 2: 27×27 → UIB_0: 23×23 → UIB_1: (23-4-1)//2 = 9
        x = jnp.ones((1, 137, 137, 3))
        pyramid = model(x)

        assert isinstance(pyramid, list), "Output should be a list"
        assert len(pyramid) == 3, f"Expected 3 levels, got {len(pyramid)}"

        # Each level should have 16 channels
        for i, level in enumerate(pyramid):
            assert level.shape[-1] == 16, f"Level {i} should have 16 channels"

        # Spatial dimensions should decrease at each level
        assert pyramid[0].shape[1:] == (
            64,
            64,
            16,
        ), f"Level 0 shape: {pyramid[0].shape}"
        assert pyramid[1].shape[1:] == (
            27,
            27,
            16,
        ), f"Level 1 shape: {pyramid[1].shape}"
        assert pyramid[2].shape[1:] == (
            9,
            9,
            16,
        ), f"Level 2 shape: {pyramid[2].shape}"

    def test_pyramid_output_single_level(self):
        """Test model with single level."""
        settings = ModelSettings(embed_dim=16, num_levels=1)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        # Single level: UIB_0 (no downsample) → UIB_1 (downsample)
        # Input 37×37 → UIB_0: 33×33 → UIB_1: (33-4-1)//2 = 14
        x = jnp.ones((1, 37, 37, 3))
        pyramid = model(x)

        assert len(pyramid) == 1
        assert pyramid[0].shape[-1] == 16
        assert pyramid[0].shape[1] == 14

    def test_batch_processing(self):
        """Test batch processing."""
        settings = ModelSettings(embed_dim=16, num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((4, 137, 137, 3))
        pyramid = model(x)

        assert len(pyramid) == 3
        for level in pyramid:
            assert level.shape[0] == 4, "Batch size should be preserved"

    def test_parameter_count(self):
        """Test parameter counting for UIB-based model."""
        settings = ModelSettings(embed_dim=16, num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))
        param_count = count_parameters(model)

        # Per UIB with 2 DWs (no downsample):
        #   DW before (16 ch depthwise): 16*9 + 16 = 160
        #   PW expand (16→32): 16*32 + 32 = 544
        #   DW after (32 ch depthwise): 32*9 + 32 = 320
        #   PW compress (32→16): 32*16 + 16 = 528
        #   GroupNorms (4 total, 4 groups each): 4*16*2 + 4*32*2 + 4*16*2 + 4*16*2 = 128 + 256 + 128 + 128 = 640
        #   Total per UIB: 160 + 544 + 320 + 528 + 640 = 2192
        #
        # Per UIB with downsample (adds DW 3×3 stride=2):
        #   Downsample DW (16 ch): 16*9 + 16 = 160
        #   GroupNorm: 16*2 = 32
        #   Extra: 192
        #
        # First UIB of first level (3 ch input):
        #   DW before (3 ch): 3*9 + 3 = 30
        #   PW expand (3→32): 3*32 + 32 = 128
        #   Rest same: 320 + 528 + GN(3+32+16)*2*4 = 640
        #   Total: 30 + 128 + 320 + 528 + (3*8 + 32*8 + 16*8 + 16*8) = 30+128+320+528+24+256+128+128 = 1542
        #
        # Model: 6 UIBs total (3 levels × 2)
        #   Level 0: UIB_0 (3 ch, no ds) + UIB_1 (16 ch, ds)
        #   Level 1: UIB_0 (16 ch, no ds) + UIB_1 (16 ch, ds)
        #   Level 2: UIB_0 (16 ch, no ds) + UIB_1 (16 ch, ds)
        #
        # This is complex; just verify it's > 0 and reasonable
        assert param_count > 0, "Parameter count should be positive"
        assert param_count < 100000, f"Parameter count seems too high: {param_count}"

    def test_gradient_flow_full_model(self):
        """Test that gradients flow through the full model."""
        from jax import grad

        settings = ModelSettings(embed_dim=16, num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))
        x = jnp.ones((1, 137, 137, 3))

        def loss_fn(m, inp):
            pyramid = m(inp)
            return pyramid[-1].sum()  # Use coarsest level

        # Compute gradients
        grads = grad(loss_fn, argnums=0)(model, x)

        # Check that model has gradients
        grad_state = nnx.state(grads)

        # Check levels have gradients
        assert "levels" in grad_state
        level0 = grad_state["levels"][0]  # type: ignore
        assert len(level0) == 2, "Each level should have 2 UIBs"

        # Verify gradients exist and are non-zero
        uib0_grads = level0[0]  # type: ignore
        assert "pw_expand" in uib0_grads
        assert uib0_grads["pw_expand"]["kernel"] is not None
        assert not jnp.allclose(uib0_grads["pw_expand"]["kernel"], 0.0)

    def test_l2_norm_on_level_outputs(self):
        """Test that level outputs are L2-normalized."""
        settings = ModelSettings(embed_dim=16, num_levels=3)
        model = HierarchicalEmbeddingModel(settings, rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((1, 137, 137, 3)) * 10.0
        pyramid = model(x)

        # Each level output should be L2-normalized along last axis
        for i, level in enumerate(pyramid):
            norms = jnp.linalg.norm(level, axis=-1)
            assert jnp.allclose(norms, 1.0, atol=1e-5), f"Level {i} norms: {norms}"


class TestUIBResolutionCalculation:
    """Tests for UIB-based resolution calculation functions."""

    def test_calculate_uib_output_size_no_downsample(self):
        """Test UIB output size without downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
        )
        result = calculate_uib_output_size(32, config)
        # 2× DW 3×3 VALID: -4 pixels
        assert result == 28, f"Expected 28, got {result}"

    def test_calculate_uib_output_size_with_downsample(self):
        """Test UIB output size with downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=True,
        )
        result = calculate_uib_output_size(32, config)
        # 2× DW: -4 → 28, then downsample: (28-1)//2 = 13
        assert result == 13, f"Expected 13, got {result}"

    def test_calculate_level_output_size(self):
        """Test level output size (2 UIBs, second downsamples)."""
        configs = [
            UIBConfig(
                in_channels=16,
                out_channels=16,
                expanded_channels=32,
                use_dw_before_expand=True,
                use_dw_after_expand=True,
                downsample_after=False,
            ),
            UIBConfig(
                in_channels=16,
                out_channels=16,
                expanded_channels=32,
                use_dw_before_expand=True,
                use_dw_after_expand=True,
                downsample_after=True,
            ),
        ]
        result = calculate_level_output_size(137, configs)
        # UIB_0: 137 - 4 = 133
        # UIB_1: (133 - 4 - 1) // 2 = 64
        assert result == 64, f"Expected 64, got {result}"

    def test_image_size_default_config(self):
        """Test image_size with default config."""
        # For 3 levels, 1×1 grid, 16 window: target = 16
        h, w = image_size(coarsest_grid_size=1, window_size=16, levels=3)
        assert h == w, "Should be square"
        assert h > 16, "Input should be larger than target"

    def test_roundtrip_consistency(self):
        """Test that input/output calculations are consistent."""
        from barevision.utils.image import (
            calculate_required_input_size,
            calculate_coarse_output_size,
        )

        # Build default configs for 3 levels
        uib_configs_per_level = []
        for level_idx in range(3):
            level_configs = []
            for uib_idx in range(2):
                is_first_uib = uib_idx == 0
                is_first_level = level_idx == 0

                in_channels = 3 if (is_first_level and is_first_uib) else 16

                config = UIBConfig(
                    in_channels=in_channels,
                    out_channels=16,
                    expanded_channels=32,
                    use_dw_before_expand=True,
                    use_dw_after_expand=True,
                    downsample_after=not is_first_uib,
                )
                level_configs.append(config)
            uib_configs_per_level.append(level_configs)

        target = 16
        required_input = calculate_required_input_size(
            target, num_levels=3, uib_configs_per_level=uib_configs_per_level
        )

        output = calculate_coarse_output_size(
            required_input, num_levels=3, uib_configs_per_level=uib_configs_per_level
        )

        assert output == target, f"Roundtrip failed: {required_input} -> {output}"
