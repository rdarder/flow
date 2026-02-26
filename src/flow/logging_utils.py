"""Logging utilities for hierarchical optical flow training.

Provides TensorBoard logging with support for scalars, images, histograms,
and gradient/parameter tracking for diagnosing training issues.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from flax import nnx
from torch.utils.tensorboard import SummaryWriter


class JaxLogger:
    """TensorBoard logger for training metrics, visualizations, and diagnostics.

    Supports:
    - Scalar logging (loss, learning rate, etc.)
    - Image logging (flow visualizations, figures)
    - Histogram logging (parameter and gradient distributions)
    - Automatic run naming with timestamps
    """

    def __init__(self, log_dir: str = "runs", run_name_prefix: str = "flow"):
        """Initialize logger with automatic run naming.

        Args:
            log_dir: Directory for TensorBoard logs
            run_name_prefix: Prefix for run name (e.g., "flow" -> "flow_20240225_123045")
        """
        run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_path)
        self.log_dir = log_path
        print(f"Logging to {log_path}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value.

        Args:
            tag: Metric name (e.g., "Loss/train_step")
            value: Scalar value to log
            step: Global step or epoch number
        """
        try:
            value = float(value)
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            print(f"Logger Warning (scalar): {e}")

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
            print(f"Logger Warning (image): {e}")

    def log_figure(self, tag: str, figure_array: np.ndarray, step: int):
        """Log a pre-rendered figure array.

        Convenience wrapper around log_image for figure arrays.

        Args:
            tag: Figure name (e.g., "Visualization/Pyramid")
            figure_array: RGB numpy array from visualization functions
            step: Global step or epoch number
        """
        self.log_image(tag, figure_array, step, dataformats="HWC")

    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log a histogram of values.

        Args:
            tag: Histogram name (e.g., "Gradients/window_flow")
            values: Array of values to histogram
            step: Global step or epoch number
        """
        try:
            # Flatten if multi-dimensional
            values_flat = np.array(values).flatten()
            if len(values_flat) > 0:
                self.writer.add_histogram(tag, values_flat, step)
        except Exception as e:
            print(f"Logger Warning (histogram): {e}")

    def close(self):
        """Close the logger and flush all pending writes."""
        self.writer.close()


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/"
) -> Dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Key prefix for recursion
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _is_numeric_array(value) -> bool:
    """Check if value is a numeric numpy array or JAX array."""
    if hasattr(value, "value"):
        value = value.value

    # Check if it's an array-like with numeric dtype
    if hasattr(value, "dtype"):
        return np.issubdtype(value.dtype, np.number)

    return False


def _extract_numeric_arrays(state, parent_key=""):
    """Recursively extract numeric arrays from nested NNX State.

    Args:
        state: NNX State object (potentially nested)
        parent_key: Key prefix for nested entries

    Returns:
        List of (key_path, value) tuples for numeric parameters
    """
    results = []

    for key_path, value in state.items():
        # Build full key path
        if parent_key:
            full_key = (
                f"{parent_key}/{key_path[0]}"
                if isinstance(key_path, tuple)
                else f"{parent_key}/{key_path}"
            )
        else:
            full_key = key_path[0] if isinstance(key_path, tuple) else str(key_path)

        # If value is another State object, recurse
        if isinstance(value, nnx.State):
            results.extend(_extract_numeric_arrays(value, full_key))
        elif hasattr(value, "value") or hasattr(value, "grad"):
            # This is a Variable/Parameter
            results.append((full_key, value))

    return results


def log_gradient_histograms(
    logger: JaxLogger,
    model: nnx.Module,
    step: int,
    prefix: str = "Gradients",
):
    """Log histograms of gradients per layer.

    Extracts gradients from model state and logs histograms for each
    parameter group. Useful for detecting vanishing/exploding gradients.

    Args:
        logger: JaxLogger instance
        model: NNX model (must have been through backward pass)
        step: Global step
        prefix: Tag prefix for organization
    """
    try:
        # Get model state (Parameters only)
        state = nnx.state(model, nnx.Param)

        # Extract all numeric parameters recursively
        param_entries = _extract_numeric_arrays(state)

        # Group by layer
        layer_grads: Dict[str, list] = {}

        for key_path, value in param_entries:
            # Get gradient if available
            if hasattr(value, "grad") and value.grad is not None:
                grad = np.array(value.grad)

                # Skip non-numeric gradients
                if not np.issubdtype(grad.dtype, np.number):
                    continue

                # Extract layer name (first component of path)
                layer_name = key_path.split("/")[0] if "/" in key_path else key_path

                if layer_name not in layer_grads:
                    layer_grads[layer_name] = []
                layer_grads[layer_name].append(grad)

        # Log histograms per layer
        for layer_name, grads_list in layer_grads.items():
            if grads_list:
                all_grads = np.concatenate([g.flatten() for g in grads_list])
                tag = f"{prefix}/{layer_name}"
                logger.log_histogram(tag, all_grads, step)

        # Log overall gradient statistics
        all_grads_flat = []
        for grads_list in layer_grads.values():
            for g in grads_list:
                all_grads_flat.extend(g.flatten())

        if all_grads_flat:
            all_grads = np.array(all_grads_flat)
            logger.log_histogram(f"{prefix}/all", all_grads, step)
            logger.log_scalar(f"{prefix}/mean_abs", np.mean(np.abs(all_grads)), step)
            logger.log_scalar(f"{prefix}/max", np.max(np.abs(all_grads)), step)

    except Exception as e:
        print(f"Warning: Could not log gradient histograms: {e}")


def log_parameter_histograms(
    logger: JaxLogger,
    model: nnx.Module,
    step: int,
    prefix: str = "Parameters",
):
    """Log histograms of model parameters per layer.

    Logs parameter value distributions for each layer. Useful for detecting
    parameter collapse or initialization issues.

    Args:
        logger: JaxLogger instance
        model: NNX model
        step: Global step
        prefix: Tag prefix for organization
    """
    try:
        # Get model state (Parameters only)
        state = nnx.state(model, nnx.Param)

        # Extract all numeric parameters recursively
        param_entries = _extract_numeric_arrays(state)

        # Group by layer
        layer_params: Dict[str, list] = {}

        for key_path, value in param_entries:
            # Get parameter value
            if hasattr(value, "value"):
                param = np.array(value.value)
            else:
                param = np.array(value)

            # Skip non-numeric parameters
            if not np.issubdtype(param.dtype, np.number):
                continue

            # Extract layer name
            layer_name = key_path.split("/")[0] if "/" in key_path else key_path

            if layer_name not in layer_params:
                layer_params[layer_name] = []
            layer_params[layer_name].append(param)

        # Log histograms per layer
        for layer_name, params_list in layer_params.items():
            if params_list:
                all_params = np.concatenate([p.flatten() for p in params_list])
                tag = f"{prefix}/{layer_name}"
                logger.log_histogram(tag, all_params, step)

        # Log overall parameter statistics
        all_params_flat = []
        for params_list in layer_params.values():
            for p in params_list:
                all_params_flat.extend(p.flatten())

        if all_params_flat:
            all_params = np.array(all_params_flat)
            logger.log_histogram(f"{prefix}/all", all_params, step)
            logger.log_scalar(f"{prefix}/mean_abs", np.mean(np.abs(all_params)), step)
            logger.log_scalar(f"{prefix}/std", np.std(all_params), step)

    except Exception as e:
        print(f"Warning: Could not log parameter histograms: {e}")
