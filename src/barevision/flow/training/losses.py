"""Combined training loss for optical flow.

Combines embedding entropy loss with flow reconstruction loss.

Loss formulation:
    total_loss = entropy_loss + recon_weight * reconstruction_loss

Where:
    - entropy_loss: Primary objective (distinctive embeddings)
    - reconstruction_loss: Secondary objective (trackable embeddings)
    - recon_weight: Controls relative importance of reconstruction (default 0.1)
"""

from typing import Tuple

import jax.numpy as jnp

from barevision.flow.embeddings.losses import compute_hierarchical_entropy_loss
from barevision.flow.matching.losses import warp_embeddings, reconstruction_loss_core


def compute_loss(
    pyramid1,
    pyramid2,
    warped_embeddings,
    target_embeddings,
    window_size: int,
    lambda_entropy: float,
    level_weight_decay: float,
    recon_weight: float,
    entropy_temperature: float,
    return_attention_weights: bool = False,
) -> Tuple[jnp.ndarray, dict]:
    """Compute combined entropy + reconstruction loss for optical flow training.

    Loss formulation:
        total = entropy_loss + recon_weight * reconstruction_loss

    This makes entropy the primary objective (ensuring distinctive embeddings)
    and reconstruction a secondary objective (ensuring embeddings are trackable).

    Args:
        pyramid1: List of feature maps from frame 1, one per level
        pyramid2: List of feature maps from frame 2, one per level
        warped_embeddings: (B, H, W, D) Frame 1 embeddings warped to F2
        target_embeddings: (B, H, W, D) Frame 2 embeddings (target)
        window_size: Size of attention windows
        lambda_entropy: Cross-attention loss weight in [0, 1]
        level_weight_decay: Weight multiplier per level
        recon_weight: Reconstruction loss weight
        entropy_temperature: Temperature for entropy loss
        return_attention_weights: If True, return attention weights and entropy maps

    Returns:
        Tuple of (total_loss, aux_dict)
    """
    # Compute entropy loss
    entropy_loss, entropy_aux = compute_hierarchical_entropy_loss(
        pyramid1,
        pyramid2,
        window_size=window_size,
        lambda_entropy=lambda_entropy,
        level_weight_decay=level_weight_decay,
        temperature=entropy_temperature,
        return_attention_weights=return_attention_weights,
    )

    # Compute reconstruction loss
    recon_loss = reconstruction_loss_core(warped_embeddings, target_embeddings)

    # Combine losses: entropy is primary, reconstruction is secondary
    total_loss = entropy_loss + recon_weight * recon_loss

    aux = dict(
        self_loss=entropy_aux["self_loss"],
        cross_loss=entropy_aux["cross_loss"],
        entropy_loss=entropy_loss,
        reconstruction_loss=recon_loss,
        total_loss=total_loss,
    )

    # Merge level-specific aux data if available
    if return_attention_weights:
        for key in [
            "level_self_attention_weights",
            "level_cross_attention_weights",
            "level_self_entropy_maps",
            "level_cross_entropy_maps",
        ]:
            if key in entropy_aux:
                aux[key] = entropy_aux[key]

    return total_loss, aux
