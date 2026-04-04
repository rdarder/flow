"""Shared logging utilities for Barevision packages.

Provides TensorBoard logging with support for scalars, images, and histograms.
"""

import os
from datetime import datetime
from typing import Union

import numpy as np
from flax import nnx
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """TensorBoard logger for training metrics.

    Attributes:
        strict: If True, raise exceptions on logging errors instead of warnings
    """

    def __init__(
        self,
        log_dir: str = "runs",
        run_name_prefix: str = "barevision",
        run_name: str | None = None,
        strict: bool = False,
    ):
        """Initialize logger with automatic run naming.

        Args:
            log_dir: Directory for TensorBoard logs
            run_name_prefix: Prefix for run name (e.g., "barevision" -> "barevision_20240225_123045")
            run_name: Optional explicit run name. If None, auto-generated from prefix + timestamp
            strict: If True, raise exceptions on logging errors instead of printing warnings
        """
        if run_name is None:
            run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_path)
        self.log_dir = log_path
        self.strict = strict
        print(f"Logging to {log_path}")

    def _handle_error(self, error_type: str, e: Exception):
        """Handle logging errors based on strict mode."""
        if self.strict:
            raise RuntimeError(f"Logger error ({error_type}): {e}") from e
        else:
            print(f"Logger Warning ({error_type}): {e}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value.

        Args:
            tag: Metric name (e.g., "Loss/train_step")
            value: Scalar value to log
            step: Global step or epoch number
        """
        try:
            self.writer.add_scalar(tag, float(value), step)
        except Exception as e:
            self._handle_error("scalar", e)

    def log_image(
        self, tag: str, image: np.ndarray, step: int, dataformats: str = "HWC"
    ):
        """Log an image.

        Args:
            tag: Image name (e.g., "Visualization/Overview")
            image: Image array (numpy array)
            step: Global step or epoch number
            dataformats: Data format string, default "HWC" (Height, Width, Channels)
        """
        try:
            self.writer.add_image(tag, image, step, dataformats=dataformats)
        except Exception as e:
            self._handle_error("image", e)

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log a histogram of values.

        Args:
            tag: Histogram name (e.g., "Parameters/embeddings")
            values: Array of values to histogram
            step: Global step or epoch number
        """
        try:
            values_flat = np.array(values).flatten()
            if len(values_flat) > 0:
                self.writer.add_histogram(tag, values_flat, step)
        except Exception as e:
            self._handle_error("histogram", e)

    def log_figure(
        self, tag: str, figure_array: np.ndarray, step: int, dataformats: str = "HWC"
    ):
        """Log a pre-rendered figure array.

        Convenience wrapper around log_image for figure arrays.

        Args:
            tag: Figure name (e.g., "Visualization/Pyramid")
            figure_array: RGB numpy array from visualization functions
            step: Global step or epoch number
            dataformats: Data format string, default "HWC"
        """
        self.log_image(tag, figure_array, step, dataformats=dataformats)

    def close(self):
        """Close the logger and flush all pending writes."""
        self.writer.close()
