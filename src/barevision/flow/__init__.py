"""Barevision flow: Self-supervised embedding learning for optical flow.

This package contains models and training utilities for learning
embedding representations optimized for attention-based matching.
"""

from barevision.flow.optical_flow_model import OpticalFlowModel
from barevision.flow.model import (
    HierarchicalEmbeddingModel,
    count_parameters,
)
from barevision.flow.loss import compute_window_attention_losses

__version__ = "0.1.0"

__all__ = [
    "OpticalFlowModel",
    "HierarchicalEmbeddingModel",
    "count_parameters",
    "compute_window_attention_losses",
]
