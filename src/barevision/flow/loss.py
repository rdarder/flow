"""Loss functions for self-supervised embedding training.

Simple entropy minimization for both self and cross attention:
- Self-attention: minimize entropy → sharp peak at self (natural advantage: q·q = ||q||²)
  This encourages UNIQUE embeddings (no other pixel competes with self)
- Cross-attention: minimize entropy → sharp peak at match location
  This encourages CONFIDENT matching

Design:
- Core functions: Pure math on (B, H, W, D) batches - no splitting, no vmap
- Wrapper functions: Handle window splitting, dimension rearranging, calling core, aggregating results

Phase 2 (Deep Supervision):
- Applies entropy loss at ALL pyramid levels simultaneously
- Crops each level to grid-aligned dimensions (divisible by window_size)
- Applies level weight decay (coarser levels get higher weight)
- Averages loss per-level first, then sums across levels
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp

from barevision.utils.grid import WindowGrid

# Temperature for softmax scaling. Low temperature sharpens attention distributions.
# Fixed at 0.05 for Phase 2 deep supervision.
TEMPERATURE = 0.05


def _compute_entropy(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution.

    Args:
        probabilities: (..., N) array where last dimension sums to 1

    Returns:
        Entropy values with shape (...)
    """
    eps = 1e-10
    return -jnp.sum(probabilities * jnp.log(probabilities + eps), axis=-1)


def self_attention_entropy_loss_core(
    windows: jnp.ndarray,
) -> jnp.ndarray:
    """Compute self-attention entropy loss on a batch of windows.

    Pure math - no splitting, no dimension rearranging. Just the loss computation.
    Returns POSITIVE entropy so minimizing loss = minimizing entropy.

    No masking, no penalties. Self naturally has highest attention (q·q = ||q||²).
    Low entropy means: "only self should dominate, no other pixel competes"
    This encourages unique embeddings.

    Args:
        windows: (B, H, W, D) batch of windows (already split and flattened)

    Returns:
        (B, H, W) per-pixel loss (positive entropy)
    """
    B, H, W, D = windows.shape
    N = H * W

    # Flatten spatial dimensions
    flat_windows = windows.reshape(B, N, D)

    # Compute attention logits: dot products (NO masking, NO penalty)
    logits = flat_windows @ flat_windows.transpose(0, 2, 1)  # (B, N, N)

    # Softmax and entropy (temperature scales logits for sharper distributions)
    attn_weights = jax.nn.softmax(logits / TEMPERATURE, axis=-1)
    entropy = _compute_entropy(attn_weights)

    # Return POSITIVE entropy (minimize entropy = encourage sharp/peaked attention)
    # Reshape back to spatial grid: (B, H, W)
    return entropy.reshape(B, H, W)


def cross_attention_entropy_loss_core(
    windows1: jnp.ndarray, windows2: jnp.ndarray
) -> jnp.ndarray:
    """Compute cross-attention entropy loss on a batch of windows.

    Pure math - no splitting, no dimension rearranging. Just the loss computation.

    Args:
        windows1: (B, H, W, D) batch of windows from frame 1
        windows2: (B, H, W, D) batch of windows from frame 2

    Returns:
        (B, H, W) per-pixel loss (positive entropy)
    """
    B, H, W, D = windows1.shape
    N = H * W

    # Flatten spatial dimensions
    flat1 = windows1.reshape(B, N, D)
    flat2 = windows2.reshape(B, N, D)

    # Compute cross-attention logits
    logits = flat1 @ flat2.transpose(0, 2, 1)  # (B, N, N)

    # Softmax and entropy (temperature scales logits for sharper distributions)
    attn_weights = jax.nn.softmax(logits / TEMPERATURE, axis=-1)
    entropy = _compute_entropy(attn_weights)

    # Reshape back to spatial grid: (B, H, W)
    return entropy.reshape(B, H, W)


def compute_embedding_losses(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    window_size: int = 16,
    lambda_entropy: float = 0.5,
) -> tuple[jnp.ndarray, dict]:
    """Compute combined self and cross attention losses.

    Handles window splitting and returns scalar loss values.
    Fails explicitly if input resolution is not aligned with window_size.

    Entropy is normalized by log(window_size²) to bring it into [0, 1] range,
    making it comparable to reconstruction loss.

    Args:
        emb1: (B, H, W, D) embeddings from frame 1
        emb2: (B, H, W, D) embeddings from frame 2
        window_size: Size of attention windows (default 16)
        lambda_entropy: Cross-attention loss weight in [0, 1] (default 0.5 = equal weighting)
                       combined = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss

    Returns:
        Tuple of (combined_loss, aux_dict) where:
            - combined_loss: scalar combined loss value (normalized to [0, 1])
            - aux_dict: {'self_loss': scalar, 'cross_loss': scalar}

    Raises:
        ValueError: If H or W is not divisible by window_size
    """
    B, H, W, D = emb1.shape

    # Validate resolution
    if H % window_size != 0:
        raise ValueError(f"Height {H} not divisible by window_size {window_size}")
    if W % window_size != 0:
        raise ValueError(f"Width {W} not divisible by window_size {window_size}")

    # Validate shapes match
    assert (
        emb2.shape == emb1.shape
    ), f"emb2 shape {emb2.shape} != emb1 shape {emb1.shape}"

    # Split into windows
    grid = WindowGrid(window_size=window_size)
    windows1 = grid.split(emb1)
    windows2 = grid.split(emb2)

    # Flatten batch and windows together for core functions
    num_windows = (H // window_size) * (W // window_size)
    flat_windows1 = windows1.reshape(B * num_windows, window_size, window_size, D)
    flat_windows2 = windows2.reshape(B * num_windows, window_size, window_size, D)

    # Compute core losses
    self_loss_flat = self_attention_entropy_loss_core(flat_windows1)
    cross_loss_flat = cross_attention_entropy_loss_core(flat_windows1, flat_windows2)

    # Reshape back to spatial grid: (B * num_windows, window_size, window_size) -> (B, H, W)
    def reshape_to_grid(loss_flat):
        loss = loss_flat.reshape(B, num_windows, window_size, window_size)
        loss = loss.reshape(
            B, H // window_size, W // window_size, window_size, window_size
        )
        loss = loss.transpose(0, 1, 3, 2, 4)
        return loss.reshape(B, H, W)

    # Mean and normalize by theoretical maximum entropy
    self_loss = reshape_to_grid(self_loss_flat).mean() / jnp.log(
        window_size * window_size
    )
    cross_loss = reshape_to_grid(cross_loss_flat).mean() / jnp.log(
        window_size * window_size
    )

    combined = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss

    aux = dict(
        self_loss=self_loss,
        cross_loss=cross_loss,
    )

    return combined, aux


def crop_to_grid_aligned(
    feature_map: jnp.ndarray, window_size: int = 16
) -> jnp.ndarray:
    """Crop feature map to dimensions divisible by window_size.

    Phase 2: Ensures each pyramid level can be cleanly split into 16x16 windows.
    Uses top-left crop to maintain spatial alignment between frame pairs.

    Args:
        feature_map: (B, H, W, D) feature map
        window_size: Window size for attention (default 16)

    Returns:
        Cropped feature map with H and W divisible by window_size
    """
    B, H, W, D = feature_map.shape

    # Calculate cropped dimensions
    crop_h = (H // window_size) * window_size
    crop_w = (W // window_size) * window_size

    # Top-left crop (same crop applied to both frames for alignment)
    return feature_map[:, :crop_h, :crop_w, :]


def compute_hierarchical_embedding_losses(
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    window_size: int = 16,
    lambda_entropy: float = 0.5,
    level_weight_decay: float = 2.0,
) -> tuple[jnp.ndarray, dict]:
    """Compute compound embedding loss across all pyramid levels.

    Phase 2 Deep Supervision:
    1. Crops each level to grid-aligned dimensions (divisible by window_size)
    2. Applies L2 normalization and temperature scaling per level
    3. Computes self + cross entropy loss per level
    4. Normalizes entropy by theoretical maximum (log of window size squared)
    5. Applies level-weighted loss (coarser levels get higher weight)
    6. Sums weighted per-level losses for final compound loss

    Level weighting: level_i_weight = level_weight_decay^i
    - Level 0 (finest): weight = decay^0 = 1
    - Level 1 (middle): weight = decay^1
    - Level 2 (coarsest): weight = decay^2
    With default decay=2.0, coarsest level gets 4x the weight of finest level.

    Entropy Normalization:
    - Per-level entropy is divided by log(window_size²) to normalize to [0, 1]
    - Final weighted sum is divided by sum(level_weights) to maintain [0, 1] range
    - This makes lambda_entropy and lambda_recon directly comparable

    Args:
        pyramid1: List of feature maps from frame 1, one per level
        pyramid2: List of feature maps from frame 2, one per level
        window_size: Size of attention windows (default 16)
        lambda_entropy: Cross-attention loss weight in [0, 1] (default 0.5 = equal weighting)
                       combined = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss
        level_weight_decay: Weight multiplier per level (default 2.0)
                           Coarser levels get: weight = decay^level_index

    Returns:
        Tuple of (total_loss, aux_dict) where:
            - total_loss: scalar sum of weighted per-level losses (normalized to [0, 1])
            - aux_dict: {'self_loss': scalar, 'cross_loss': scalar,
                        'level_losses': list of per-level weighted losses,
                        'level_weights': list of weight per level}

    Raises:
        ValueError: If pyramid levels don't match or crops result in zero-sized windows
    """
    if len(pyramid1) != len(pyramid2):
        raise ValueError(f"Pyramid level mismatch: {len(pyramid1)} vs {len(pyramid2)}")

    num_levels = len(pyramid1)
    level_losses = []
    level_weights = []
    total_loss = jnp.array(0.0)
    total_self_loss = jnp.array(0.0)
    total_cross_loss = jnp.array(0.0)

    # Theoretical maximum entropy for a window: log(N) where N = window_size²
    # This normalizes entropy to [0, 1] range
    max_entropy = jnp.log(window_size * window_size)

    for level_idx in range(num_levels):
        # Calculate level weight: coarser levels (higher index) get higher weight
        level_weight = level_weight_decay**level_idx

        emb1 = pyramid1[level_idx]
        emb2 = pyramid2[level_idx]

        # Crop to grid-aligned dimensions
        emb1_cropped = crop_to_grid_aligned(emb1, window_size)
        emb2_cropped = crop_to_grid_aligned(emb2, window_size)

        B, H, W, D = emb1_cropped.shape

        # Validate we have at least one window after cropping
        num_windows_h = H // window_size
        num_windows_w = W // window_size
        if num_windows_h == 0 or num_windows_w == 0:
            raise ValueError(
                f"Level {level_idx}: Cropped dimensions ({H}x{W}) too small "
                f"for window_size {window_size}"
            )

        # Split into windows
        grid = WindowGrid(window_size=window_size)
        windows1 = grid.split(emb1_cropped)
        windows2 = grid.split(emb2_cropped)

        # Flatten batch and windows together for core functions
        num_windows = num_windows_h * num_windows_w
        flat_windows1 = windows1.reshape(B * num_windows, window_size, window_size, D)
        flat_windows2 = windows2.reshape(B * num_windows, window_size, window_size, D)

        # Compute core losses
        self_loss_flat = self_attention_entropy_loss_core(flat_windows1)
        cross_loss_flat = cross_attention_entropy_loss_core(
            flat_windows1, flat_windows2
        )

        # Reshape back to spatial grid: (B * num_windows, window_size, window_size) -> (B, H, W)
        def reshape_to_grid(loss_flat):
            loss = loss_flat.reshape(B, num_windows, window_size, window_size)
            loss = loss.reshape(
                B, num_windows_h, num_windows_w, window_size, window_size
            )
            loss = loss.transpose(0, 1, 3, 2, 4)
            return loss.reshape(B, H, W)

        # Mean across batch and spatial dimensions for this level
        self_loss_level = reshape_to_grid(self_loss_flat).mean()
        cross_loss_level = reshape_to_grid(cross_loss_flat).mean()

        # Normalize entropy by theoretical maximum to bring into [0, 1] range
        self_loss_level = self_loss_level / max_entropy
        cross_loss_level = cross_loss_level / max_entropy

        # Per-level combined loss (unweighted) using lambda_entropy
        level_loss_unweighted = (
            1 - lambda_entropy
        ) * self_loss_level + lambda_entropy * cross_loss_level

        # Apply level weight
        level_loss_weighted = level_loss_unweighted * level_weight

        level_losses.append(level_loss_weighted)
        level_weights.append(level_weight)
        total_self_loss += self_loss_level * level_weight
        total_cross_loss += cross_loss_level * level_weight
        total_loss += level_loss_weighted

    # Normalize by total weight to maintain [0, 1] range regardless of num_levels
    total_weight = sum(level_weights)
    total_loss = total_loss / total_weight
    total_self_loss = total_self_loss / total_weight
    total_cross_loss = total_cross_loss / total_weight

    # Weighted per-level losses summed
    aux: dict = dict(
        self_loss=total_self_loss,
        cross_loss=total_cross_loss,
        level_losses=level_losses,
        level_weights=level_weights,
    )

    return total_loss, aux


def compute_combined_loss(
    pyramid1,
    pyramid2,
    warped_embeddings,
    target_embeddings,
    window_size: int = 16,
    lambda_entropy: float = 0.5,
    level_weight_decay: float = 2.0,
    lambda_recon: float = 0.5,
) -> tuple[jnp.ndarray, dict]:
    """Compute combined entropy + reconstruction loss.

    total = (1 - lambda_recon) * entropy_loss + lambda_recon * reconstruction_loss

    Args:
        pyramid1: List of feature maps from frame 1, one per level
        pyramid2: List of feature maps from frame 2, one per level
        warped_embeddings: (B, H, W, D) Frame 1 embeddings warped to F2 coordinate frame
        target_embeddings: (B, H, W, D) Frame 2 embeddings (target for reconstruction)
        window_size: Size of attention windows (default 16)
        lambda_entropy: Cross-attention loss weight in [0, 1] (default 0.5 = equal weighting)
                       entropy_loss = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss
        level_weight_decay: Weight multiplier per level (default 2.0)
        lambda_recon: Reconstruction loss weight in [0, 1] (default 0.5 = equal weighting)
                      total = (1 - lambda_recon) * entropy + lambda_recon * reconstruction

    Returns:
        Tuple of (total_loss, aux_dict) where:
            - total_loss: scalar combined loss
            - aux_dict: {'self_loss', 'cross_loss', 'entropy_loss', 'reconstruction_loss'}
    """
    # Compute entropy loss
    entropy_loss, entropy_aux = compute_hierarchical_embedding_losses(
        pyramid1,
        pyramid2,
        window_size=window_size,
        lambda_entropy=lambda_entropy,
        level_weight_decay=level_weight_decay,
    )

    # Compute reconstruction loss
    from barevision.flow.reconstruction_loss import reconstruction_loss_core

    recon_loss = reconstruction_loss_core(warped_embeddings, target_embeddings)

    # Combine losses with weighting
    # total = (1 - lambda_recon) * entropy + lambda_recon * reconstruction
    total_loss = (1 - lambda_recon) * entropy_loss + lambda_recon * recon_loss

    aux = dict(
        self_loss=entropy_aux["self_loss"],
        cross_loss=entropy_aux["cross_loss"],
        entropy_loss=entropy_loss,
        reconstruction_loss=recon_loss,
    )

    return total_loss, aux
