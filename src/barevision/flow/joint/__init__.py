"""Training model and losses for optical flow.

Combines embedding pyramid with flow matching for end-to-end training.
"""

from barevision.flow.joint.model import JointEmbeddingFlowModel
from barevision.flow.joint.losses import combine_entropy_reconstruction_losses

__all__ = [
    "JointEmbeddingFlowModel",
    "combine_entropy_reconstruction_losses",
]
