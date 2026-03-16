"""Embedding model and losses for Barevision.

Provides hierarchical embedding pyramid with entropy-based self-supervised training.
"""

from barevision.flow.embeddings.model import (
    HierarchicalEmbeddingModel,
    StemBlock,
    StandardBlock,
    count_parameters,
)
from barevision.flow.embeddings.losses import (
    self_attention_entropy_loss_core,
    cross_attention_entropy_loss_core,
    compute_window_attention_losses,
    compute_hierarchical_entropy_loss,
    crop_to_grid_aligned,
)

__all__ = [
    "HierarchicalEmbeddingModel",
    "StemBlock",
    "StandardBlock",
    "count_parameters",
    "self_attention_entropy_loss_core",
    "cross_attention_entropy_loss_core",
    "compute_window_attention_losses",
    "compute_hierarchical_entropy_loss",
    "crop_to_grid_aligned",
]