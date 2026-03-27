"""Combined training loss for optical flow.

Combines embedding spatial variance loss with flow reconstruction loss.

Loss formulation:
    total_loss = spatial_variance_loss + recon_weight * reconstruction_loss

Where:
    - spatial_variance_loss: Primary objective (concentrated attention patterns)
    - reconstruction_loss: Secondary objective (trackable embeddings)
    - recon_weight: Controls relative importance of reconstruction (default 0.1)
"""

import jax.numpy as jnp

from barevision.flow.embeddings.spatial_losses import HierarchicalSpatialVarianceLoss
from barevision.flow.matching.losses import HierarchicalReconstructionLoss
from barevision.flow.settings import JointEmbeddingFlowSettings


def combine_spatial_variance_reconstruction_losses(
    spatial_variance_loss: jnp.ndarray,
    reconstruction_loss: jnp.ndarray,
    variance_aux: dict,
    reconstruction_aux: dict,
    reconstruction_loss_weight: float,
) -> tuple[jnp.ndarray, dict]:

    weighted_recon_loss = reconstruction_loss_weight * reconstruction_loss
    # Combine losses: spatial variance is primary, reconstruction is secondary
    total_loss = spatial_variance_loss + weighted_recon_loss

    loss_parts = dict(
        spatial_variance=spatial_variance_loss,
        weighted_reconstruction_loss=weighted_recon_loss,
        reconstruction=reconstruction_loss,
        total=total_loss,
    )
    aux = dict(
        spatial_variance=variance_aux,
        reconstruction=reconstruction_aux,
        loss=loss_parts,
    )

    return total_loss, aux


class JointEmbeddingReconstructionLoss:
    def __init__(
        self,
        embedding_loss: HierarchicalSpatialVarianceLoss,
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
        need_aux: bool = True,
    ) -> tuple[jnp.ndarray, dict]:
        total_weight = self.settings.recon_weight + self.settings.entropy_weight
        norm_recon_weight = self.settings.recon_weight / total_weight
        norm_entropy_weight = self.settings.entropy_weight / total_weight

        spatial_variance_loss, variance_aux = self.embedding_loss(
            embedding_pyramid_pair, need_aux=need_aux
        )
        reconstruction_loss, reconstruction_aux = self.reconstruction_loss(
            embedding_pyramid_pair, flows, need_aux=need_aux
        )

        weighted_recon_loss = norm_recon_weight * reconstruction_loss
        weighted_variance_loss = norm_entropy_weight * spatial_variance_loss
        total_loss = weighted_recon_loss + weighted_variance_loss

        if need_aux:
            aux = dict(
                entropy=variance_aux,  # Use 'entropy' key for backward compatibility with logging
                reconstruction=reconstruction_aux,
                loss=total_loss,
                weighted_recon_weight=weighted_recon_loss,
                weighted_entropy_weight=weighted_variance_loss,
                recon_weight=norm_recon_weight,
                entropy_weight=norm_entropy_weight,
                embeddings=embedding_pyramid_pair,
                flows=flows,
            )
        else:
            aux = {}

        return total_loss, aux
