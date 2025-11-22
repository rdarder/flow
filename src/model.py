import jax.numpy as jnp
from flax import nnx


class PatchLookup(nnx.Module):
    """
    Core 'Line 4' Attention Module.
    - Operates on pixels (flattened H*W).
    - Separates Visual Similarity from Spatial Proximity.
    - Uses Gaussian Kernel for Spatial Score.
    - Returns Residual Flow and Consensus.
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        # Learnable parameters for mixing the scores
        # visual_scale: Temperature for feature dot product
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))

        # spatial_scale: The 'beta' in exp(-beta * distance^2)
        # Initialized to something reasonable (e.g., 0.1) so it doesn't kill gradients early
        self.spatial_scale = nnx.Param(0.1)

    def __call__(
            self,
            q_features: jnp.ndarray,  # (B, N, C)
            k_features: jnp.ndarray,  # (B, N, C)
            q_pos: jnp.ndarray,  # (B, N, 2) - Global Coords
            k_pos: jnp.ndarray  # (B, N, 2) - Global Coords
    ):
        """
        Args:
            q_features: Query features (Frame 1)
            k_features: Key features (Frame 2)
            q_pos: Absolute coordinates of Query pixels (e.g., expected location)
            k_pos: Absolute coordinates of Key pixels (e.g., grid location)

        Returns:
            flow: (B, N, 2) - The estimated flow vector (target - source)
            consensus: (B, N, 1) - Confidence score (max attention weight)
        """
        B, N, C = q_features.shape

        # --- 1. Visual Similarity ---
        # Standard Dot Product Attention
        # (B, N, C) @ (B, C, N) -> (B, N, N)
        visual_logits = q_features @ jnp.swapaxes(k_features, -2, -1)
        visual_score = visual_logits * self.visual_scale.value

        # --- 2. Spatial Proximity (Gaussian Kernel) ---
        # We want: -scale * ||q_pos - k_pos||^2
        # Expand: ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A . B)

        # Squared Norms
        q_pos_sq = jnp.sum(jnp.square(q_pos), axis=-1, keepdims=True)  # (B, N, 1)
        k_pos_sq = jnp.sum(jnp.square(k_pos), axis=-1, keepdims=True)  # (B, N, 1)
        k_pos_sq_T = jnp.swapaxes(k_pos_sq, -2, -1)  # (B, 1, N)

        # Cross Term (2 * A . B)
        # (B, N, 2) @ (B, 2, N) -> (B, N, N)
        pos_cross = 2.0 * (q_pos @ jnp.swapaxes(k_pos, -2, -1))

        # Combine
        # (B, N, 1) + (B, 1, N) - (B, N, N) -> (B, N, N)
        dist_sq = q_pos_sq + k_pos_sq_T - pos_cross

        # Convert distance to log-probability (Gaussian Kernel logit)
        # We use a negative scale because distance is a penalty
        spatial_score = -jnp.abs(self.spatial_scale.value) * dist_sq

        # --- 3. Combine & Softmax ---
        # We simply add the logits.
        # This acts as a multiplicative filter in probability space:
        # Prob ~ exp(Visual) * exp(Spatial)
        logits = visual_score + spatial_score

        attn_weights = nnx.softmax(logits, axis=-1)

        # --- 4. Value Aggregation (Center of Mass) ---
        # We compute the weighted average of the KEY POSITIONS.
        # (B, N, N) @ (B, N, 2) -> (B, N, 2)
        target_pos_est = attn_weights @ k_pos

        # --- 5. Output Calculation ---
        # Flow = Target - Source
        flow = target_pos_est - q_pos

        # Consensus = Max attention weight (How sharp is the decision?)
        # (B, N, 1)
        consensus = jnp.max(attn_weights, axis=-1, keepdims=True)

        return flow, consensus
