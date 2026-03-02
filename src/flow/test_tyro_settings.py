"""Test that tyro correctly handles the Settings dataclass.

This test ensures tyro can parse nested dataclasses with default values
and properly override them via command-line arguments.
"""

import sys
from io import StringIO

import pytest
import tyro

from flow.settings import (
    DatasetSettings,
    LoggingSettings,
    ModelSettings,
    Settings,
    TrainingSettings,
)


class TestTyroSettingsIntegration:
    """Test tyro's handling of nested Settings dataclass."""

    def test_tyro_uses_defaults_when_no_args(self):
        """Test that tyro uses default values when no CLI args provided."""

        def train(settings: Settings):
            return settings

        # Simulate calling with no arguments
        settings = tyro.cli(train, args=[])

        # Check defaults were applied
        assert settings.model.num_levels == 2
        assert settings.dataset.img_size == (384, 512)
        assert settings.training.learning_rate == 1e-4
        assert settings.logging.log_dir == "runs"

    def test_tyro_overrides_nested_values(self):
        """Test that tyro correctly overrides nested dataclass fields."""

        def train(settings: Settings):
            return settings

        # Override nested values - note: tyro prefixes with parameter name
        settings = tyro.cli(
            train,
            args=[
                "--settings.model.num-levels",
                "4",
                "--settings.dataset.img-size",
                "128",
                "128",
                "--settings.training.learning-rate",
                "0.001",
                "--settings.logging.log-dir",
                "custom_runs",
            ],
        )

        # Check overrides were applied
        assert settings.model.num_levels == 4
        assert settings.dataset.img_size == (128, 128)
        assert settings.training.learning_rate == 0.001
        assert settings.logging.log_dir == "custom_runs"

        # Check non-overridden defaults still present
        assert settings.model.embed_dim == 16
        assert settings.dataset.batch_size == 32  # Default is 32
        assert settings.training.epochs == 100

    def test_tyro_partial_override(self):
        """Test that tyro correctly handles partial overrides."""

        def train(settings: Settings):
            return settings

        # Override only one field per nested class
        settings = tyro.cli(
            train,
            args=[
                "--settings.model.embed-dim",
                "32",
                "--settings.dataset.batch-size",
                "64",
            ],
        )

        # Check overridden values
        assert settings.model.embed_dim == 32
        assert settings.dataset.batch_size == 64

        # Check other defaults still present
        assert settings.model.num_levels == 2
        assert settings.dataset.img_size == (384, 512)
        assert settings.training.learning_rate == 1e-4

    def test_settings_validation_still_works(self):
        """Test that Settings.validate() works on tyro-parsed settings."""

        def train(settings: Settings):
            return settings

        # Valid configuration
        settings = tyro.cli(
            train,
            args=[
                "--settings.model.num-levels",
                "2",
                "--settings.dataset.img-size",
                "64",
                "64",
            ],
        )

        is_valid, msg = settings.validate()
        assert is_valid is True
        assert "64x64" in msg

    def test_settings_validation_fails_on_invalid(self):
        """Test that Settings.validate() correctly fails on invalid config."""

        def train(settings: Settings):
            return settings

        # Invalid: 4 levels requires 256, but we pass 64
        settings = tyro.cli(
            train,
            args=[
                "--settings.model.num-levels",
                "4",  # requires 256
                "--settings.dataset.img-size",
                "64",  # too small
                "64",
            ],
        )

        # tyro should parse successfully (it doesn't validate logic)
        # But our validation should catch the error
        is_valid, msg = settings.validate()
        assert is_valid is False
        assert "must be multiple of" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
