"""Loss functions for self-supervised embedding training.

Implements entropy-based objectives for learning sharp attention distributions:
- Self-attention entropy: Maximize after spatial penalty (discourage distant peaks)
- Cross-attention entropy: Minimize (encourage 1-2 sharp cross-frame matches)
"""

import jax
import jax.numpy as jnp


def _compute_entropy(probabilities: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of a probability distribution.

    Args:
        probabilities: (..., N) array where last dimension sums to 1

    Returns:
        Entropy values with shape (...) - positive values
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    return -jnp.sum(probabilities * jnp.log(probabilities + eps), axis=-1)


def _spatial_logits_matrix(
    window_size: int, scale: float = 10.0
) -> jnp.ndarray:
    """Create spatial score matrix using Gaussian kernel in log-space.

    Matches the SpatialScore approach from barevision.flow.token_attention:
        score_ij = -scale * ||pos_i - pos_j||^2

    For normalized coordinates [0, 1] within a window:
    - Nearby positions get small negative scores (higher attention)
    - Distant positions get large negative scores (suppressed attention)

    This is added to attention logits BEFORE softmax.

    Args:
        window_size: Size of the attention window (e.g., 16)
        scale: Scaling factor for the penalty (default 10.0, matching SpatialScore)

    Returns:
        (window_size*window_size, window_size*window_size) spatial score matrix
    """
    # Create normalized coordinate grid [0, 1]
    coords = jnp.linspace(0, 1, window_size, dtype=jnp.float32)
    y, x = jnp.meshgrid(coords, coords, indexing="ij")

    # Flatten to positions: (N, 2) where N = window_size^2
    positions = jnp.stack([x.ravel(), y.ravel()], axis=-1)

    # Compute pairwise squared distances using expanded square trick
    # ||pos_i - pos_j||^2 = ||pos_i||^2 + ||pos_j||^2 - 2 * pos_i · pos_j
    pos_norm_sq = jnp.sum(jnp.square(positions), axis=-1, keepdims=True)  # (N, 1)
    cross_term = positions @ positions.T  # (N, N)
    dist_sq = pos_norm_sq + pos_norm_sq.T - 2 * cross_term  # (N, N)

    # Clip negative values (numerical noise)
    dist_sq = jnp.maximum(dist_sq, 0.0)

    # Gaussian kernel in log-space: score = -scale * distance^2
    return -scale * dist_sq


def self_attention_entropy_loss(
    embeddings: jnp.ndarray, window_size: int = 16, spatial_scale: float = 10.0
) -> jnp.ndarray:
    """Compute self-attention entropy loss with spatial weighting.

    For each 16x16 window, computes self-attention with spatial weighting
    (nearby positions get higher attention) and then maximizes entropy.
    
    This encourages embeddings where distant pixels can still compete for attention
    despite the spatial penalty. Nearby attention is "forgiven" (doesn't count much
    toward entropy), so high entropy means the embedding creates multiple sharp peaks
    at various distances.

    The loss returned is negative entropy, so minimizing the loss maximizes entropy.

    Args:
        embeddings: (B, H, W, D) tensor of embeddings
        window_size: Size of attention windows (default 16)
        spatial_scale: Scale factor for spatial weighting (default 10.0, matching SpatialScore)

    Returns:
        Per-pixel loss of shape (B, H, W) - negative entropy values
    """
    B, H, W, D = embeddings.shape

    # Pad if necessary to make dimensions divisible by window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        embeddings = jnp.pad(
            embeddings,
            [(0, 0), (0, pad_h), (0, pad_w), (0, 0)],
            mode="edge",
        )
        H_padded, W_padded = embeddings.shape[1:3]
    else:
        H_padded, W_padded = H, W

    # Split into windows: (B, num_windows, window_size, window_size, D)
    num_h = H_padded // window_size
    num_w = W_padded // window_size

    # Reshape to windows
    emb_windows = embeddings.reshape(
        B, num_h, window_size, num_w, window_size, D
    )
    emb_windows = emb_windows.transpose(0, 1, 3, 2, 4, 5)
    emb_windows = emb_windows.reshape(
        B, num_h * num_w, window_size, window_size, D
    )

    # Precompute spatial score matrix (matches SpatialScore from flow)
    spatial_matrix = _spatial_logits_matrix(window_size, spatial_scale)

    def window_loss(window_emb: jnp.ndarray) -> jnp.ndarray:
        """Compute loss for a single window.

        Args:
            window_emb: (window_size * window_size, D) flattened window

        Returns:
            Per-position loss (window_size * window_size,)
        """
        # Compute dot products (attention logits)
        # Shape: (N, N) where N = window_size^2
        logits = window_emb @ window_emb.T

        # Mask self-attention (each position attending to itself)
        N = window_emb.shape[0]
        mask = jnp.eye(N, dtype=jnp.float32)
        logits = logits - mask * 1e9

        # Add spatial scores: nearby positions get boost, distant get suppressed
        # spatial_matrix contains negative values: -scale * distance^2
        logits = logits + spatial_matrix

        # Softmax to get attention weights
        # Shape: (N, N) - each row is a distribution
        attn_weights = jax.nn.softmax(logits, axis=-1)

        # Compute entropy for each position
        # Shape: (N,)
        entropy = _compute_entropy(attn_weights)

        # Return negative entropy (so minimizing loss = maximizing entropy)
        return -entropy

    # Vectorize over batch and windows
    # First flatten batch and windows together
    flat_windows = emb_windows.reshape(B * num_h * num_w, window_size * window_size, D)

    # Compute loss for each window
    per_window_loss = jax.vmap(window_loss)(flat_windows)

    # Reshape back to (B, num_h, num_w, window_size, window_size)
    per_window_loss = per_window_loss.reshape(
        B, num_h, num_w, window_size, window_size
    )

    # Transpose and reshape to original spatial dimensions
    per_window_loss = per_window_loss.transpose(0, 1, 3, 2, 4)
    per_window_loss = per_window_loss.reshape(B, H_padded, W_padded)

    # Crop back to original size if we padded
    if pad_h > 0 or pad_w > 0:
        per_window_loss = per_window_loss[:, :H, :W]

    return per_window_loss


def cross_attention_entropy_loss(
    emb1: jnp.ndarray,
    emb2: jnp.ndarray,
    window_size: int = 16,
) -> jnp.ndarray:
    """Compute cross-attention entropy loss.

    For each 16x16 window, computes cross-attention from frame1 to frame2
    and minimizes entropy to encourage sharp matches.

    Args:
        emb1: (B, H, W, D) embeddings from frame 1
        emb2: (B, H, W, D) embeddings from frame 2
        window_size: Size of attention windows (default 16)

    Returns:
        Per-pixel loss of shape (B, H, W) - positive entropy values
    """
    B, H, W, D = emb1.shape

    # Ensure emb2 has same shape
    assert emb2.shape == emb1.shape, f"emb2 shape {emb2.shape} != emb1 shape {emb1.shape}"

    # Pad if necessary
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        emb1 = jnp.pad(emb1, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], mode="edge")
        emb2 = jnp.pad(emb2, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], mode="edge")
        H_padded, W_padded = emb1.shape[1:3]
    else:
        H_padded, W_padded = H, W

    # Split into windows
    num_h = H_padded // window_size
    num_w = W_padded // window_size

    # Reshape both to windows
    emb1_windows = emb1.reshape(B, num_h, window_size, num_w, window_size, D)
    emb1_windows = emb1_windows.transpose(0, 1, 3, 2, 4, 5)
    emb1_windows = emb1_windows.reshape(B * num_h * num_w, window_size * window_size, D)

    emb2_windows = emb2.reshape(B, num_h, window_size, num_w, window_size, D)
    emb2_windows = emb2_windows.transpose(0, 1, 3, 2, 4, 5)
    emb2_windows = emb2_windows.reshape(B * num_h * num_w, window_size * window_size, D)

    def window_cross_loss(args) -> jnp.ndarray:
        """Compute cross-attention loss for a pair of windows.

        Args:
            args: Tuple of (emb1_window, emb2_window)
                  Each is (window_size * window_size, D)

        Returns:
            Per-position loss (window_size * window_size,)
        """
        q_emb, k_emb = args

        # Compute cross-attention logits
        # Shape: (N, N) where N = window_size^2
        logits = q_emb @ k_emb.T

        # Softmax over key positions (frame 2)
        # Shape: (N, N) - each row is a distribution over frame 2 positions
        attn_weights = jax.nn.softmax(logits, axis=-1)

        # Compute entropy for each query position
        # Shape: (N,)
        entropy = _compute_entropy(attn_weights)

        # Return entropy (minimize)
        return entropy

    # Vectorize over all windows
    per_window_loss = jax.vmap(window_cross_loss)((emb1_windows, emb2_windows))

    # Reshape back to (B, num_h, num_w, window_size, window_size)
    per_window_loss = per_window_loss.reshape(B, num_h, num_w, window_size, window_size)

    # Transpose and reshape to original spatial dimensions
    per_window_loss = per_window_loss.transpose(0, 1, 3, 2, 4)
    per_window_loss = per_window_loss.reshape(B, H_padded, W_padded)

    # Crop back to original size if we padded
    if pad_h > 0 or pad_w > 0:
        per_window_loss = per_window_loss[:, :H, :W]

    return per_window_loss


def combined_loss(
    self_loss: jnp.ndarray,
    cross_loss: jnp.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> jnp.ndarray:
    """Compute weighted combination of self and cross attention losses.

    Args:
        self_loss: Per-pixel self-attention loss (negative entropy)
        cross_loss: Per-pixel cross-attention loss (positive entropy)
        alpha: Weight for self-attention loss (default 1.0)
        beta: Weight for cross-attention loss (default 1.0)

    Returns:
        Combined per-pixel loss
    """
    assert self_loss.shape == cross_loss.shape, (
        f"Loss shapes mismatch: self={self_loss.shape}, cross={cross_loss.shape}"
    )

    return alpha * self_loss + beta * cross_loss
