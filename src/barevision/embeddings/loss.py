"""Loss functions for self-supervised embedding training.

Simple entropy minimization for both self and cross attention:
- Self-attention: minimize entropy → sharp peak at self (natural advantage: q·q = ||q||²)
  This encourages UNIQUE embeddings (no other pixel competes with self)
- Cross-attention: minimize entropy → sharp peak at match location
  This encourages CONFIDENT matching

Design:
- Core functions: Pure math on (B, H, W, D) batches - no splitting, no vmap
- Wrapper functions: Handle window splitting, dimension rearranging, calling core, aggregating results
"""

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

    # Softmax and entropy
    attn_weights = jax.nn.softmax(logits / 0.05, axis=-1)
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

    # Softmax and entropy
    attn_weights = jax.nn.softmax(logits / 0.05, axis=-1)
    entropy = _compute_entropy(attn_weights)

    # Reshape back to spatial grid: (B, H, W)
    return entropy.reshape(B, H, W)


def combined_loss(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    window_size: int = 16,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> jnp.ndarray:
    """Compute combined self + cross attention loss.

    Wrapper that handles all window splitting, dimension rearranging, and aggregation.
    Calls core loss functions which only do the math.

    Fails explicitly if input resolution is not aligned with window_size.

    Args:
        emb1: (B, H, W, D) embeddings from frame 1
        emb2: (B, H, W, D) embeddings from frame 2
        window_size: Size of attention windows (default 16)
        alpha: Weight for self-attention loss (default 1.0)
        beta: Weight for cross-attention loss (default 1.0)

    Returns:
        (B, H, W) combined per-pixel loss

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
    windows1 = grid.split(emb1)  # (B, num_windows, window_size, window_size, D)
    windows2 = grid.split(emb2)

    # Flatten batch and windows together for core functions
    num_windows = (H // window_size) * (W // window_size)
    flat_windows1 = windows1.reshape(B * num_windows, window_size, window_size, D)
    flat_windows2 = windows2.reshape(B * num_windows, window_size, window_size, D)

    # Call core loss functions (pure math)
    self_loss = self_attention_entropy_loss_core(flat_windows1)
    cross_loss = cross_attention_entropy_loss_core(flat_windows1, flat_windows2)

    # Combine with weights
    combined = alpha * self_loss + beta * cross_loss

    # Reshape: (B * num_windows, window_size, window_size) -> (B, H, W)
    # First reshape to separate batch and windows
    combined = combined.reshape(B, num_windows, window_size, window_size)
    # Transpose to interleave windows spatially: (B, num_h, num_w, wh, ww) -> (B, num_h, wh, num_w, ww)
    num_h = H // window_size
    num_w = W // window_size
    combined = combined.reshape(B, num_h, num_w, window_size, window_size)
    combined = combined.transpose(0, 1, 3, 2, 4)
    combined = combined.reshape(B, num_h * window_size, num_w * window_size)

    return combined


def compute_embedding_losses(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    window_size: int = 16,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute combined self and cross attention losses.

    Handles window splitting and returns all three loss components.
    Fails explicitly if input resolution is not aligned with window_size.

    Args:
        emb1: (B, H, W, D) embeddings from frame 1
        emb2: (B, H, W, D) embeddings from frame 2
        window_size: Size of attention windows (default 16)
        alpha: Weight for self-attention loss (default 1.0)
        beta: Weight for cross-attention loss (default 0.1)

    Returns:
        Tuple of (combined_loss, self_loss, cross_loss), each (B, H, W)

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
        loss = loss.reshape(B, H // window_size, W // window_size, window_size, window_size)
        loss = loss.transpose(0, 1, 3, 2, 4)
        return loss.reshape(B, H, W)

    self_loss = reshape_to_grid(self_loss_flat)
    cross_loss = reshape_to_grid(cross_loss_flat)
    combined = alpha * self_loss + beta * cross_loss

    return combined, self_loss, cross_loss
