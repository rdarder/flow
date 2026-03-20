"""Embedding model and losses for Barevision.

Provides hierarchical embedding pyramid with entropy-based self-supervised training.
"""

from barevision.flow.embeddings.model import (
    HierarchicalEmbeddingModel,
    StemBlock,
    StandardBlock,
    count_parameters,
)
from barevision.flow.embeddings.losses import compute_hierarchical_entropy_loss

__all__ = [
    "HierarchicalEmbeddingModel",
    "StemBlock",
    "StandardBlock",
    "count_parameters",
    "compute_hierarchical_entropy_loss",
]
