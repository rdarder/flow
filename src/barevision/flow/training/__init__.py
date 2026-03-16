"""Training model and losses for optical flow.

Combines embedding pyramid with flow matching for end-to-end training.
"""

from barevision.flow.training.model import Model
from barevision.flow.training.losses import compute_loss

__all__ = [
    "Model",
    "compute_loss",
]
