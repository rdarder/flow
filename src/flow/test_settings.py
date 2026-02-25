"""Tests for settings validation.

Focus on validation logic that ensures model-dataset compatibility.
"""

import pytest
from flow.settings import (
    ModelSettings,
    DatasetSettings,
    TrainingSettings,
    LoggingSettings,
    Settings,
)


class TestCrossValidation:
    """Tests for model-dataset cross-compatibility using window_grid utilities."""
    
    def test_valid_configuration_passes(self):
        """Standard 2-level model with 64x64 images should validate."""
        settings = Settings(
            model=ModelSettings(num_levels=2),  # requires 64 (WINDOW_SIZE * 2^2)
            dataset=DatasetSettings(img_size=64),
            training=TrainingSettings(),
            logging=LoggingSettings(),
        )
        is_valid, msg = settings.validate()
        assert is_valid is True
        assert "64x64" in msg
    
    def test_invalid_size_fails(self):
        """Incompatible image size should fail validation."""
        settings = Settings(
            model=ModelSettings(num_levels=4),  # requires 256 (16 * 2^4)
            dataset=DatasetSettings(img_size=64),  # too small
            training=TrainingSettings(),
            logging=LoggingSettings(),
        )
        is_valid, msg = settings.validate()
        assert is_valid is False
        assert "too small" in msg.lower()
    
    def test_get_required_size_uses_window_grid(self):
        """Should return correct minimum size from window_grid.compute_valid_resolution."""
        settings = Settings(
            model=ModelSettings(num_levels=2),
            dataset=DatasetSettings(),
            training=TrainingSettings(),
            logging=LoggingSettings(),
        )
        # 2 levels with WINDOW_SIZE=16: 16 * 2^2 = 64
        assert settings.get_required_image_size() == 64


class TestInputValidation:
    """Tests for basic input validation in dataclasses."""

    def test_invalid_num_levels_raises(self):
        with pytest.raises(ValueError, match="num_levels"):
            ModelSettings(num_levels=0)

    def test_invalid_img_size_raises(self):
        with pytest.raises(ValueError, match="img_size"):
            DatasetSettings(img_size=4)

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            TrainingSettings(learning_rate=0)

    def test_invalid_log_views_raises(self):
        with pytest.raises(ValueError, match="Invalid log_views"):
            LoggingSettings(log_views=("invalid",))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
