"""Combined training loss for optical flow.

Combines embedding entropy loss with flow reconstruction loss.

Loss formulation:
    total_loss = entropy_loss + recon_weight * reconstruction_loss

Where:
    - entropy_loss: Primary objective (distinctive embeddings)
    - reconstruction_loss: Secondary objective (trackable embeddings)
    - recon_weight: Controls relative importance of reconstruction (default 0.1)
"""

import jax.numpy as jnp

from barevision.flow.embeddings.losses import HierarchicalEmbeddingLoss
from barevision.flow.matching.losses import HierarchicalReconstructionLoss
from barevision.flow.settings import JointEmbeddingFlowSettings


def combine_entropy_reconstruction_losses(
    entropy_loss: jnp.ndarray,
    reconstruction_loss: jnp.ndarray,
    entropy_aux: dict,
    reconstruction_aux: dict,
    reconstruction_loss_weight: float,
) -> tuple[jnp.ndarray, dict]:

    # Compute entropy loss (always returns aux with attention weights)

    weighted_recon_loss = reconstruction_loss_weight * reconstruction_loss
    # Combine losses: entropy is primary, reconstruction is secondary
    total_loss = entropy_loss + weighted_recon_loss

    loss_parts = dict(
        entropy=entropy_loss,
        weighted_reconstruction_loss=weighted_recon_loss,
        reconstruction=reconstruction_loss,
        total=total_loss,
    )
    aux = dict(
        entropy=entropy_aux,
        reconstruction=reconstruction_aux,
        loss=loss_parts,
    )

    return total_loss, aux


class JointEmbeddingReconstructionLoss:
    def __init__(
        self,
        embedding_loss: HierarchicalEmbeddingLoss,
        reconstruction_loss: HierarchicalReconstructionLoss,
        settings: JointEmbeddingFlowSettings,
    ):
        self.embedding_loss = embedding_loss
        self.reconstruction_loss = reconstruction_loss
        self.settings = settings

    def __call__(
        self,
        embedding_pyramid_pair: tuple[list[jnp.ndarray], list[jnp.ndarray]],
        flows: list[jnp.ndarray],
    ) -> tuple[jnp.ndarray, dict]:
        total_weight = self.settings.recon_weight + self.settings.entropy_weight
        norm_recon_weight = self.settings.recon_weight / total_weight
        norm_entropy_weight = self.settings.entropy_weight / total_weight

        entropy_loss, entropy_aux = self.embedding_loss(embedding_pyramid_pair)
        reconstruction_loss, reconstruction_aux = self.reconstruction_loss(
            embedding_pyramid_pair, flows
        )

        weighted_recon_loss = norm_recon_weight * reconstruction_loss
        weighted_entropy_loss = norm_entropy_weight * entropy_loss
        total_loss = weighted_recon_loss + weighted_entropy_loss
        aux = dict(
            entropy=entropy_aux,
            reconstruction=reconstruction_aux,
            loss=total_loss,
            weighted_recon_weight=weighted_recon_loss,
            weighted_entropy_weight=weighted_entropy_loss,
            recon_weight=norm_recon_weight,
            entropy_weight=norm_entropy_weight,
            embeddings=embedding_pyramid_pair,
            flows=flows,
        )
        return total_loss, aux
