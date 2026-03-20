"""Entropy loss functions for embedding training.

Organization (top to bottom: coarse-grained to fine-grained):
1. compute_hierarchical_entropy_loss: Multi-level pyramid entropy loss
2. compute_window_attention_losses: Single-level window-based attention loss
3. cross_attention_entropy_loss_core: Core cross-attention entropy math
4. self_attention_entropy_loss_core: Core self-attention entropy math
5. _compute_entropy: Utility function
6. crop_to_grid_aligned: Utility function

Loss principles:
- Self-attention: minimize entropy → sharp peak at self (encourages UNIQUE embeddings)
- Cross-attention: minimize entropy → sharp peak at match (encourages CONFIDENT matching)

Phase 2 (Deep Supervision):
- Applies entropy loss at ALL pyramid levels simultaneously
- Crops each level to grid-aligned dimensions (divisible by window_size)
- Applies level weight decay (coarser levels get higher weight)
- Normalizes entropy by theoretical maximum to [0, 1] range
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp

from barevision.utils.grid import WindowGrid


def _compute_entropy(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution.

    Args:
        probabilities: (..., N) array where last dimension sums to 1

    Returns:
        Entropy values with shape (...)
    """
    eps = 1e-10
    return -jnp.sum(probabilities * jnp.log(probabilities + eps), axis=-1)


def self_attention_entropy_loss(
    windows: jnp.ndarray,
    temperature: float,
) -> tuple[jnp.ndarray, dict]:
    """Compute self-attention entropy loss on a batch of windows.

    Self attention (without masking) naturally has highest attention (q·q = ||q||²)
    on itself. This is expected.
    Low entropy means: "only self should dominate, there shouldn't be that many other
    positions with high attention"
    This encourages unique embeddings.
    What we really want is "an unique, small / tight region of embeddings with high
    attention that includes self". entropy is only a proxy for that.
    While one could expect to find the lowest possible entropy (self is the only
    position with attention, it concentrates all of it), in reality embeddings close
    to each other will have similar spatial traits so it's expected that they're
    also similar. we want as unique as possible without becoming
    so brittle that a small pose change makes the embedding for the same object
    change drastically. what we do want to avoid is to have scattered matches across
    a lookup window. this loss doesn't help much in disincentivizing that.

    Args:
        windows: (B, H, W, D) batch of windows (already split and flattened)
        temperature: Softmax temperature

    Returns:
        Tuple of (loss, aux_dict) where:
            - loss: (B, H, W) per-pixel loss
            - aux_dict: {'attention_weights': (B, N, N), 'entropy_map': (B, H, W)}
    """
    B, H, W, D = windows.shape
    N = H * W

    # Flatten spatial dimensions
    flat_windows = windows.reshape(B, N, D)

    # Compute attention logits
    logits = flat_windows @ flat_windows.transpose(0, 2, 1)  # (B, N, N)

    # Softmax and entropy
    attn_weights = jax.nn.softmax(logits / temperature, axis=-1)
    entropy = _compute_entropy(attn_weights)

    # Reshape back to spatial grid
    entropy_grid = entropy.reshape(B, H, W)

    return entropy_grid, {
        "attention_weights": attn_weights,
        "entropy_map": entropy_grid,
    }


def cross_attention_entropy_loss(
    windows1: jnp.ndarray,
    windows2: jnp.ndarray,
    temperature: float,
) -> tuple[jnp.ndarray, dict]:
    """Compute cross-attention entropy loss on a batch of windows.

    Similarly to self_attention, here we expect self to find an unique match on the corresponding
    other frame. This complements self attention. together they incentivize:
    - make unique embeddings for patterns within an image small area (self attention entropy low)
    - that are similar to the same pattern that will likely be present in the other frame (cross attention entropy low)

    without cross attention, self attention could just concentrate on noise: it's the most unique aspet of each position
    in an image. This forces it to also be matchable and robust across slight transformations.

    Args:
        windows1: (B, H, W, D) batch of windows from frame 1
        windows2: (B, H, W, D) batch of windows from frame 2
        temperature: Softmax temperature

    Returns:
        Tuple of (loss, aux_dict) where:
            - loss: (B, H, W) per-pixel loss
            - aux_dict: {'attention_weights': (B, N, N), 'entropy_map': (B, H, W)}
    """
    B, H, W, D = windows1.shape
    N = H * W

    # Flatten spatial dimensions
    flat1 = windows1.reshape(B, N, D)
    flat2 = windows2.reshape(B, N, D)

    # Compute cross-attention logits
    logits = flat1 @ flat2.transpose(0, 2, 1)  # (B, N, N)

    # Softmax and entropy
    attn_weights = jax.nn.softmax(logits / temperature, axis=-1)
    entropy = _compute_entropy(attn_weights)

    # Reshape back to spatial grid
    entropy_grid = entropy.reshape(B, H, W)

    return entropy_grid, {
        "attention_weights": attn_weights,
        "entropy_map": entropy_grid,
    }


def windowed_attention_losses(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    window_size: int,
    lambda_entropy: float,
    temperature: float,
) -> tuple[jnp.ndarray, dict]:
    """Compute combined self and cross attention losses for a pair of frames consisting of embeddings.
    The losses only happen in fixed size windows that make up the frames.

    Handles window splitting and returns scalar loss values per frame pair in the batch.
    Fails explicitly if input resolution is not aligned with window_size.

    Entropy is normalized by log(window_size²) to bring it into [0, 1] range.

    Args:
        emb1: (B, H, W, D) embeddings from frame 1
        emb2: (B, H, W, D) embeddings from frame 2
        window_size: Size of attention windows
        lambda_entropy: Cross-attention loss weight in [0, 1]
        temperature: Softmax temperature

    Returns:
        Tuple of (combined_loss, aux_dict) where:
            - combined_loss: scalar combined loss value (normalized to [0, 1]) shape (B, 1)
            - aux_dict: {'self_loss': scalar, 'cross_loss': scalar,
                        'self_attention_weights': (B*N, N, N), 'cross_attention_weights': (B*N, N, N),
                        'self_entropy_maps': (B, H, W), 'cross_entropy_maps': (B, H, W)}
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

    # Flatten batch and windows
    num_windows = (H // window_size) * (W // window_size)
    flat_windows1 = windows1.reshape(B * num_windows, window_size, window_size, D)
    flat_windows2 = windows2.reshape(B * num_windows, window_size, window_size, D)

    # Compute core losses (always return aux with attention weights)
    self_loss_flat, self_aux = self_attention_entropy_loss(
        flat_windows1,
        temperature=temperature,
    )
    cross_loss_flat, cross_aux = cross_attention_entropy_loss(
        flat_windows1,
        flat_windows2,
        temperature=temperature,
    )

    # Reshape back to spatial grid
    def reshape_to_grid(loss_flat):
        loss = loss_flat.reshape(B, num_windows, window_size, window_size)
        loss = loss.reshape(
            B, H // window_size, W // window_size, window_size, window_size
        )
        loss = loss.transpose(0, 1, 3, 2, 4)
        return loss.reshape(B, H, W)

    # Mean and normalize
    self_loss = reshape_to_grid(self_loss_flat).mean() / jnp.log(
        window_size * window_size
    )
    cross_loss = reshape_to_grid(cross_loss_flat).mean() / jnp.log(
        window_size * window_size
    )

    combined = (1 - lambda_entropy) * self_loss + lambda_entropy * cross_loss

    aux = dict(self_loss=self_loss, cross_loss=cross_loss)
    aux["self_attention_weights"] = self_aux["attention_weights"]
    aux["cross_attention_weights"] = cross_aux["attention_weights"]
    aux["self_entropy_maps"] = self_aux["entropy_map"]
    aux["cross_entropy_maps"] = cross_aux["entropy_map"]

    return combined, aux


def crop_to_grid_aligned(feature_map: jnp.ndarray, window_size: int) -> jnp.ndarray:
    """Crop feature map to dimensions divisible by window_size.

    Phase 2: Ensures each pyramid level can be cleanly split into windows.
    Uses centered crop to maximize spatial buffer on all sides for flow estimation.
    This provides symmetric buffer space for motion in any direction.

    Args:
        feature_map: (B, H, W, D) feature map
        window_size: Window size for attention

    Returns:
        Cropped feature map with H and W divisible by window_size
    """
    B, H, W, D = feature_map.shape

    # Calculate cropped dimensions
    crop_h = (H // window_size) * window_size
    crop_w = (W // window_size) * window_size

    # Centered crop: compute start position to center the crop
    # If H=79 and crop_h=64, start_h = (79-64)//2 = 7, end at 71 (removes 7 top, 8 bottom)
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2

    return feature_map[:, start_h : start_h + crop_h, start_w : start_w + crop_w, :]


def compute_hierarchical_entropy_loss(
    pyramid1: List[jnp.ndarray],
    pyramid2: List[jnp.ndarray],
    window_size: int,
    lambda_entropy: float,
    level_weight_decay: float,
    temperature: float,
) -> tuple[jnp.ndarray, dict]:
    """Compute compound entropy loss across all pyramid levels.

    Phase 2 Deep Supervision:
    1. Crops each level to grid-aligned dimensions
    2. Calls compute_window_attention_losses for each level
    3. Applies level-weighted loss (coarser levels get higher weight)
    4. Sums weighted per-level losses for final compound loss

    Level weighting: level_i_weight = level_weight_decay^i
    Default (1.0) gives uniform weighting across all levels.

    Args:
        pyramid1: List of feature maps from frame 1, one per level
        pyramid2: List of feature maps from frame 2, one per level
        window_size: Size of attention windows
        lambda_entropy: Cross-attention loss weight in [0, 1]
        level_weight_decay: Weight multiplier per level
        temperature: Softmax temperature

    Returns:
        Tuple of (total_loss, aux_dict) where:
            - total_loss: scalar sum of weighted per-level losses (normalized to [0, 1])
            - aux_dict: {'self_loss': scalar, 'cross_loss': scalar,
                        'level_self_attention_weights': [...], 'level_cross_attention_weights': [...],
                        'level_self_entropy_maps': [...], 'level_cross_entropy_maps': [...]}
    """
    if len(pyramid1) != len(pyramid2):
        raise ValueError(f"Pyramid level mismatch: {len(pyramid1)} vs {len(pyramid2)}")

    num_levels = len(pyramid1)
    level_losses = []
    level_weights = []
    total_loss = jnp.array(0.0)
    total_self_loss = jnp.array(0.0)
    total_cross_loss = jnp.array(0.0)

    # Aux data storage per level
    level_self_attn = []
    level_cross_attn = []
    level_self_entropy = []
    level_cross_entropy = []

    for level_idx in range(num_levels):
        level_weight = level_weight_decay**level_idx

        emb1 = pyramid1[level_idx]
        emb2 = pyramid2[level_idx]

        # Crop to grid-aligned dimensions
        emb1_cropped = crop_to_grid_aligned(emb1, window_size)
        emb2_cropped = crop_to_grid_aligned(emb2, window_size)

        B, H, W, D = emb1_cropped.shape

        # Validate we have at least one window
        num_windows_h = H // window_size
        num_windows_w = W // window_size
        if num_windows_h == 0 or num_windows_w == 0:
            raise ValueError(
                f"Level {level_idx}: Cropped dimensions ({H}x{W}) too small for window_size {window_size}"
            )

        # Delegate to single-level loss function (always returns aux with attention weights)
        level_loss, level_aux = windowed_attention_losses(
            emb1_cropped,
            emb2_cropped,
            window_size=window_size,
            lambda_entropy=lambda_entropy,
            temperature=temperature,
        )

        # Apply level weight
        level_loss_weighted = level_loss * level_weight

        level_losses.append(level_loss_weighted)
        level_weights.append(level_weight)
        total_self_loss += level_aux["self_loss"] * level_weight
        total_cross_loss += level_aux["cross_loss"] * level_weight
        total_loss += level_loss_weighted

        # Aggregate aux data
        level_self_attn.append(level_aux["self_attention_weights"])
        level_cross_attn.append(level_aux["cross_attention_weights"])
        level_self_entropy.append(level_aux["self_entropy_maps"])
        level_cross_entropy.append(level_aux["cross_entropy_maps"])

    # Normalize by total weight
    total_weight = sum(level_weights)
    total_loss = total_loss / total_weight
    total_self_loss = total_self_loss / total_weight
    total_cross_loss = total_cross_loss / total_weight

    aux = dict(
        self_loss=total_self_loss,
        cross_loss=total_cross_loss,
        level_losses=level_losses,
        level_weights=level_weights,
        level_self_attention_weights=level_self_attn,
        level_cross_attention_weights=level_cross_attn,
        level_self_entropy_maps=level_self_entropy,
        level_cross_entropy_maps=level_cross_entropy,
    )

    return total_loss, aux
