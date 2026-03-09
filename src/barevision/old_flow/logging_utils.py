"""Flow-specific logging utilities.

Re-exports shared logging utilities and provides flow-specific helpers.
"""

from barevision.utils.logging import JaxLogger, log_module_histograms

# Re-export for backward compatibility
__all__ = ["JaxLogger", "log_module_histograms"]


def log_gradient_histograms(
    logger: JaxLogger, model, step: int, prefix: str = "Gradients"
):
    """Log gradient histograms.

    Args:
        logger: JaxLogger instance
        model: NNX model
        step: Global step
        prefix: Tag prefix
    """
    log_module_histograms(logger, model, step, prefix=prefix, value_type="grads")


def log_parameter_histograms(
    logger: JaxLogger, model, step: int, prefix: str = "Parameters"
):
    """Log parameter histograms.

    Args:
        logger: JaxLogger instance
        model: NNX model
        step: Global step
        prefix: Tag prefix
    """
    log_module_histograms(logger, model, step, prefix=prefix, value_type="params")
