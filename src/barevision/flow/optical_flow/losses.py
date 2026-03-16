"""Optical flow training: combines embedding pyramid with flow estimation.

Full training objective combining entropy loss and reconstruction loss.
"""

from typing import Tuple

import jax.numpy as jnp

from barevision.flow.embeddings.losses import compute_hierarchical_entropy_loss
from barevision.flow.flow_estimator.losses import warp_embeddings, reconstruction_loss_core


def compute_training_loss(
    pyramid1,
    pyramid2,
    warped_embeddings,
    target_embeddings,
    window_size: int = 16,
    lambda_entropy: float = 0.5,
    level_weight_decay: float = 2.0,
    lambda_recon: float = 0.5,
    temperature: float = 0.2,
    return_attention_weights: bool = False,
) -> Tuple[jnp.ndarray, dict]:
    """Compute combined entropy + reconstruction loss for optical flow training.

    total = (1 - lambda_recon) * entropy_loss + lambda_recon * reconstruction_loss

    Args:
        pyramid1: List of feature maps from frame 1, one per level
        pyramid2: List of feature maps from frame 2, one per level
        warped_embeddings: (B, H, W, D) Frame 1 embeddings warped to F2
        target_embeddings: (B, H, W, D) Frame 2 embeddings (target)
        window_size: Size of attention windows (default 16)
        lambda_entropy: Cross-attention loss weight in [0, 1] (default 0.5)
        level_weight_decay: Weight multiplier per level (default 2.0)
        lambda_recon: Reconstruction loss weight in [0, 1] (default 0.5)
        temperature: Softmax temperature (default 0.2)
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
        temperature=temperature,
        return_attention_weights=return_attention_weights,
    )

    # Compute reconstruction loss
    recon_loss = reconstruction_loss_core(warped_embeddings, target_embeddings)

    # Combine losses
    total_loss = (1 - lambda_recon) * entropy_loss + lambda_recon * recon_loss

    aux = dict(
        self_loss=entropy_aux["self_loss"],
        cross_loss=entropy_aux["cross_loss"],
        entropy_loss=entropy_loss,
        reconstruction_loss=recon_loss,
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
