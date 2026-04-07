"""Unit tests for hierarchical embedding model with UIB architecture."""

import jax.numpy as jnp
import jax.random as jr
from flax import nnx

from barevision.embeddings.model import (
    HierarchicalEmbeddingModel,
    HierarchicalModelConfig,
    UniversalInvertedBlock,
    Level,
    LevelConfig,
    UIBConfig,
    count_parameters,
    make_default_model_config,
)


class TestUIBConfig:
    """Tests for UIBConfig size calculations."""

    def test_output_size_no_downsample(self):
        """Test forward size calculation without downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
        )
        result = config.output_size(32)
        # 2× DW 3×3 VALID: -4 pixels
        assert result == 28, f"Expected 28, got {result}"

    def test_output_size_with_downsample(self):
        """Test forward size calculation with downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=True,
        )
        result = config.output_size(32)
        # 2× DW: -4 → 28, then downsample: (28-1)//2 = 13
        assert result == 13, f"Expected 13, got {result}"

    def test_required_input_size_no_downsample(self):
        """Test inverse size calculation without downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=False,
        )
        result = config.required_input_size(28)
        # Inverse: +4 pixels
        assert result == 32, f"Expected 32, got {result}"

    def test_required_input_size_with_downsample(self):
        """Test inverse size calculation with downsampling."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=True,
        )
        result = config.required_input_size(13)
        # Inverse: downsample: 13*2+1=27, then +2+2=31
        assert result == 31, f"Expected 31, got {result}"

    def test_roundtrip_consistency(self):
        """Test that forward and inverse are consistent."""
        config = UIBConfig(
            in_channels=16,
            out_channels=16,
            expanded_channels=32,
            use_dw_before_expand=True,
            use_dw_after_expand=True,
            downsample_after=True,
        )
        input_size = 137
        output = config.output_size(input_size)
        recovered = config.required_input_size(output)
        assert recovered == input_size, f"Roundtrip failed: {input_size} → {output} → {recovered}"


class TestLevelConfig:
    """Tests for LevelConfig size calculations."""

    def test_output_size_two_uibs(self):
        """Test forward size calculation for level with 2 UIBs."""
        config = LevelConfig(
            level_idx=0,
            uib_configs=(
                UIBConfig(
                    in_channels=3,
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
            ),
        )
        result = config.output_size(137)
        # UIB_0: 137 - 2 - 2 = 133
        # UIB_1: 133 - 2 - 2 = 129, (129 - 1) // 2 = 64
        assert result == 64, f"Expected 64, got {result}"

    def test_required_input_size_two_uibs(self):
        """Test inverse size calculation for level with 2 UIBs."""
        config = LevelConfig(
            level_idx=0,
            uib_configs=(
                UIBConfig(
                    in_channels=3,
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
            ),
        )
        result = config.required_input_size(64)
        # UIB_1 (reversed): 64*2+1 = 129, +2+2 = 133
        # UIB_0 (reversed): 133 + 2 + 2 = 137
        assert result == 137, f"Expected 137, got {result}"

    def test_roundtrip_consistency(self):
        """Test that forward and inverse are consistent for level."""
        config = LevelConfig(
            level_idx=0,
            uib_configs=(
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
            ),
        )
        input_size = 137
        output = config.output_size(input_size)
        recovered = config.required_input_size(output)
        assert recovered == input_size, f"Roundtrip failed: {input_size} → {output} → {recovered}"


class TestHierarchicalModelConfig:
    """Tests for HierarchicalModelConfig size calculations."""

    def test_output_size_three_levels(self):
        """Test forward size calculation for 3-level model."""
        config = make_default_model_config()
        result = config.output_size(137)
        # Level 0: 137 → 133 → 64
        # Level 1: 64 → 60 → 27
        # Level 2: 27 → 23 → 9
        assert result == 9, f"Expected 9, got {result}"

    def test_required_input_size_three_levels(self):
        """Test inverse size calculation for 3-level model."""
        config = make_default_model_config()
        result = config.required_input_size(9)
        # Inverse of forward: should recover 137 (or close due to floor division)
        assert result <= 137, f"Expected <= 137, got {result}"
        # Verify it produces the right output
        assert config.output_size(result) == 9, f"Inverse doesn't produce correct output"

    def test_target_to_input(self):
        """Test target_to_input method."""
        config = make_default_model_config()
        h, w = config.target_to_input(coarsest_grid_size=1, window_size=16)
        assert h == w, "Should be square"
        assert h > 16, "Input should be larger than target coarse dim"

    def test_roundtrip_consistency(self):
        """Test that forward and inverse are consistent for full model."""
        config = make_default_model_config()
        input_size = 137
        output = config.output_size(input_size)
        recovered = config.required_input_size(output)
        # Due to floor division, recovered may be <= input_size
        assert config.output_size(recovered) == output, "Recovered doesn't produce same output"
        assert recovered <= input_size, "Recovered should be <= original"

    def test_build_model(self):
        """Test that config can build a model."""
        config = make_default_model_config()
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))
        assert isinstance(model, HierarchicalEmbeddingModel)
        assert len(model.levels) == 3


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


class TestLevel:
    """Tests for Level module."""

    def test_level_forward_pass(self):
        """Test Level forward pass with 2 UIBs."""
        config = LevelConfig(
            level_idx=0,
            uib_configs=(
                UIBConfig(
                    in_channels=3,
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
                    use_l2_norm=True,
                ),
            ),
        )
        level = Level(config, rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((1, 137, 137, 3))
        output = level(x)

        # UIB_0: 137 - 2 - 2 = 133
        # UIB_1: 133 - 2 - 2 = 129, (129 - 1) // 2 = 64
        assert output.shape == (1, 64, 64, 16), f"Got {output.shape}"


class TestHierarchicalEmbeddingModel:
    """Tests for HierarchicalEmbeddingModel forward pass."""

    def test_pyramid_output_shapes(self):
        """Test that model returns list of feature maps with correct shapes."""
        config = make_default_model_config()
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))

        # Default config: 2 UIBs per level, second downsamples
        # Level 0: 137 → 133 → 64
        # Level 1: 64 → 60 → 27
        # Level 2: 27 → 23 → 9
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
        # Build custom single-level config
        level_config = LevelConfig(
            level_idx=0,
            uib_configs=(
                UIBConfig(
                    in_channels=3,
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
                    use_l2_norm=True,
                ),
            ),
        )
        config = HierarchicalModelConfig(levels=(level_config,))
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))

        # Single level: UIB_0 (no downsample) → UIB_1 (downsample)
        # Input 37×37 → UIB_0: 37-4=33 → UIB_1: 33-4=29, (29-1)//2 = 14
        x = jnp.ones((1, 37, 37, 3))
        pyramid = model(x)

        assert len(pyramid) == 1
        assert pyramid[0].shape[-1] == 16
        assert pyramid[0].shape[1] == 14

    def test_batch_processing(self):
        """Test batch processing."""
        config = make_default_model_config()
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((4, 137, 137, 3))
        pyramid = model(x)

        assert len(pyramid) == 3
        for level in pyramid:
            assert level.shape[0] == 4, "Batch size should be preserved"

    def test_parameter_count(self):
        """Test parameter counting for UIB-based model."""
        config = make_default_model_config()
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))
        param_count = count_parameters(model)

        # Just verify it's > 0 and reasonable
        assert param_count > 0, "Parameter count should be positive"
        assert param_count < 100000, f"Parameter count seems too high: {param_count}"

    def test_gradient_flow_full_model(self):
        """Test that gradients flow through the full model."""
        from jax import grad

        config = make_default_model_config()
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))
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
        assert "uibs" in level0, "Level should have uibs"
        assert 0 in level0["uibs"], "Level should have at least one UIB"

        # Verify gradients exist and are non-zero
        uib0_grads = level0["uibs"][0]  # type: ignore
        assert "pw_expand" in uib0_grads
        assert "kernel" in uib0_grads["pw_expand"]
        assert uib0_grads["pw_expand"]["kernel"] is not None
        assert not jnp.allclose(uib0_grads["pw_expand"]["kernel"], 0.0)

    def test_l2_norm_on_level_outputs(self):
        """Test that level outputs are L2-normalized."""
        config = make_default_model_config()
        model = config.build_model(rngs=nnx.Rngs(jr.PRNGKey(0)))

        x = jnp.ones((1, 137, 137, 3)) * 10.0
        pyramid = model(x)

        # Each level output should be L2-normalized along last axis
        for i, level in enumerate(pyramid):
            norms = jnp.linalg.norm(level, axis=-1)
            assert jnp.allclose(norms, 1.0, atol=1e-5), f"Level {i} norms: {norms}"
