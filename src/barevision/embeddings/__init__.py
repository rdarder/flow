"""Embedding model and losses for Barevision.

Provides hierarchical embedding pyramid with spatial variance-based self-supervised training.
"""

from barevision.embeddings.model import (
    HierarchicalEmbeddingModel,
    StemBlock,
    StandardBlock,
    count_parameters,
)
from barevision.embeddings.spatial_losses import (
    HierarchicalSpatialVarianceLoss,
    compute_hierarchical_spatial_variance_loss,
    windowed_spatial_variance_losses,
    self_attention_spatial_variance,
    cross_attention_spatial_variance,
)

__all__ = [
    "HierarchicalEmbeddingModel",
    "StemBlock",
    "StandardBlock",
    "count_parameters",
    "HierarchicalSpatialVarianceLoss",
    "compute_hierarchical_spatial_variance_loss",
    "windowed_spatial_variance_losses",
    "self_attention_spatial_variance",
    "cross_attention_spatial_variance",
]
