"""Barevision flow: Self-supervised embedding learning for optical flow.

This package contains models and training utilities for learning
embedding representations optimized for attention-based matching.
"""

from barevision.flow.model import (
    HierarchicalEmbeddingModel,
    AttentionMaps,
    count_parameters,
)
from barevision.flow.loss import compute_embedding_losses

__version__ = "0.1.0"

__all__ = [
    "HierarchicalEmbeddingModel",
    "AttentionMaps",
    "count_parameters",
    "compute_embedding_losses",
]
