"""Training model and losses for optical flow.

Combines embedding pyramid with flow matching for end-to-end training.
"""

from barevision.flow.joint.model import JointEmbeddingFlotModel
from barevision.flow.joint.losses import combined_entropy_reconstruction_loss

__all__ = [
    "JointEmbeddingFlotModel",
    "combined_entropy_reconstruction_loss",
]
