"""Barevision flow: Self-supervised embedding learning for optical flow.

This package contains models and training utilities for learning
embedding representations optimized for attention-based matching.

Subpackages:
- embeddings: Hierarchical feature pyramid extraction
- matching: Attention-based feature matching → flow prediction
- joint: Joint training and orchestration (Model, losses)
"""

from barevision.flow.embeddings.model import (
    HierarchicalEmbeddingModel,
    count_parameters,
)
from barevision.flow.joint.model import Model as OpticalFlowModel

__version__ = "0.1.0"

__all__ = [
    "OpticalFlowModel",
    "HierarchicalEmbeddingModel",
    "count_parameters",
]
