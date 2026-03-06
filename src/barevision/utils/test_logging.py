"""Test shared logging utilities."""

import numpy as np

from barevision.utils.logging import JaxLogger


def test_jax_logger_smoke():
    """Test basic logger functionality."""
    logger = JaxLogger(log_dir="test_runs", run_name_prefix="test")

    # Test scalar logging
    logger.log_scalar("Test/loss", 0.5, 0)
    logger.log_scalar("Test/loss", 0.4, 1)

    # Test histogram logging
    logger.log_histogram("Test/values", np.random.randn(100), 0)

    # Test image logging
    logger.log_image("Test/image", np.random.rand(32, 32, 3), 0)

    # Test figure logging
    logger.log_figure("Test/figure", np.random.rand(64, 64, 3), 0)

    logger.close()
    print("✓ JaxLogger smoke test passed")


if __name__ == "__main__":
    test_jax_logger_smoke()
