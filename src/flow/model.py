import jax.numpy as jnp
from flax import nnx


class SpatialScore(nnx.Module):
    """
    Computes the Gaussian Kernel spatial score between two sets of positions.
    Score = -scale * ||pos1 - pos2||^2
    Uses the expanded square trick for efficiency: ||A-B||^2 = A^2 + B^2 - 2AB
    """

    def __init__(self, initial_scale: float = 10.0, *, rngs: nnx.Rngs):
        # Initialized to 10.0 because normalized distances (0-1) are small.
        self.scale = nnx.Param(initial_scale)

    def __call__(self, pos1: jnp.ndarray, pos2: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            pos1: (B, N, 2)
            pos2: (B, N, 2)
        Returns:
            scores: (B, N, N) pairwise spatial scores
        """
        # Squared Norms
        p1_sq = jnp.sum(jnp.square(pos1), axis=-1, keepdims=True)  # (B, N, 1)
        p2_sq = jnp.sum(jnp.square(pos2), axis=-1, keepdims=True)  # (B, N, 1)
        p2_sq_T = jnp.swapaxes(p2_sq, -2, -1)  # (B, 1, N)

        # Cross Term (2 * A . B)
        # (B, N, 2) @ (B, 2, N) -> (B, N, N)
        cross = 2.0 * (pos1 @ jnp.swapaxes(pos2, -2, -1))

        # Combine
        dist_sq = p1_sq + p2_sq_T - cross

        # Clip negative values (numerical noise from float32 subtraction)
        dist_sq = jnp.maximum(dist_sq, 0.0)

        # Convert distance to log-probability (Gaussian Kernel logit)
        return -jnp.abs(self.scale.get_value()) * dist_sq


class PatchLookup(nnx.Module):
    """
    Core attention Module with Prior-Guided Spatial Search.

    The attention weight is determined by a combination of embedding Similarity
    and distance. Distance means how far apart it is from the expected flow. Expected
    flow comes from a prior, be that a guess or a coarser estimation of the same
    frame pair.

    The prior flow shifts query positions, and prior confidence modulates
    the spatial distance penalty (high confidence = stronger spatial penalty).
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))
        self.spatial_score = SpatialScore(initial_scale=10.0, rngs=rngs)
        self.prior_spatial_scale = nnx.Param(
            1.0
        )  # Scales confidence effect on distance
        self.outside_spatial_scale = nnx.Param(
            1.0
        )  # Scales penalty for prior pointing outside window

    def __call__(
        self,
        q_features: jnp.ndarray,  # (B, N, C)
        k_features: jnp.ndarray,  # (B, N, C)
        q_pos: jnp.ndarray,  # (B, N, 2) - Normalized [0, 1]
        k_pos: jnp.ndarray,  # (B, N, 2) - Normalized [0, 1]
        prior_flow: jnp.ndarray,  # (B, N, 2) - Normalized flow from coarser level
        prior_confidence: jnp.ndarray,  # (B, N, 1) - Confidence in prior
    ):
        B, N, C = q_features.shape

        # --- 1. Visual Similarity ---
        visual_logits = q_features @ jnp.swapaxes(k_features, -2, -1)
        visual_score = visual_logits * self.visual_scale.get_value()

        # --- 2. Spatial Proximity with Prior Guidance ---
        # Shift query positions by prior flow
        q_pos_adjusted = q_pos + prior_flow  # (B, N, 2)

        # Compute spatial distance from adjusted positions
        spatial_score_raw = self.spatial_score(q_pos_adjusted, k_pos)  # (B, N, N)

        # Modulate spatial penalty by prior confidence
        # High confidence -> larger effective distance -> stronger penalty for deviation
        # Low confidence -> smaller effective distance -> more permissive search
        # prior_confidence is (B, N, 1), broadcasts to (B, N, N) across keys
        spatial_score = (
            spatial_score_raw * prior_confidence * self.prior_spatial_scale.get_value()
        )

        # --- 2b. Outside Window Penalty ---
        # When prior flow points outside the lookup window, increase penalty
        # to compensate for the fact that we can't verify against true prior neighborhood
        # Manhattan distance to [0, 1] bounds in normalized coordinates
        outside_x = jnp.maximum(0.0, q_pos_adjusted[..., 0] - 1.0) + jnp.maximum(
            0.0, 0.0 - q_pos_adjusted[..., 0]
        )
        outside_y = jnp.maximum(0.0, q_pos_adjusted[..., 1] - 1.0) + jnp.maximum(
            0.0, 0.0 - q_pos_adjusted[..., 1]
        )
        outside_distance = outside_x + outside_y  # (B, N)
        outside_distance = outside_distance[..., None]  # (B, N, 1) for broadcasting

        # Penalty scales with: how far outside, prior confidence, and learned scale
        outside_penalty = (
            outside_distance * prior_confidence * self.outside_spatial_scale.get_value()
        )  # (B, N, 1) -> broadcasts to (B, N, N)

        # Combine spatial scores
        spatial_score = spatial_score - outside_penalty

        # --- 3. Combine & Softmax ---
        logits = visual_score + spatial_score
        attn_weights = nnx.softmax(logits, axis=-1)

        # --- 4. Value Aggregation ---
        target_pos_est = attn_weights @ k_pos

        # --- 5. Output Calculation ---
        flow = target_pos_est - q_pos  # Flow relative to original query position
        consensus = jnp.max(attn_weights, axis=-1, keepdims=True)

        return flow, consensus, attn_weights


class PeerPropagation(nnx.Module):
    """
    Module 2: Peer Propagation (V2) - Normalized Coords
    """

    def __init__(self, embed_dim: int, *, rngs: nnx.Rngs):
        self.visual_scale = nnx.Param(1.0 / jnp.sqrt(embed_dim))
        self.spatial_score = SpatialScore(initial_scale=10.0, rngs=rngs)
        self.consensus_bias_scale = nnx.Param(5.0)

    def __call__(
        self,
        features: jnp.ndarray,  # (B, N, C)
        pos: jnp.ndarray,  # (B, N, 2) - Normalized
        flow_v1: jnp.ndarray,  # (B, N, 2) - Normalized Flow
        consensus_v1: jnp.ndarray,  # (B, N, 1)
    ):
        B, N, C = features.shape

        # --- 1. Visual Similarity ---
        visual_logits = features @ jnp.swapaxes(features, -2, -1)
        visual_score = visual_logits * self.visual_scale.get_value()

        # --- 2. Spatial Proximity ---
        spatial_score = self.spatial_score(pos, pos)

        # --- 3. Consensus Bias ---
        consensus_key = jnp.swapaxes(consensus_v1, -2, -1)
        consensus_score = consensus_key * self.consensus_bias_scale.get_value()

        # --- 4. Combine & Mask ---
        logits = visual_score + spatial_score + consensus_score

        mask = jnp.eye(N, dtype=bool)
        logits = logits + (mask * -1e9)

        attn_weights = nnx.softmax(logits, axis=-1)

        # --- 5. Value Aggregation ---
        flow_peer = attn_weights @ flow_v1

        # --- 6. Peer Consensus ---
        peer_consensus = attn_weights @ consensus_v1

        return flow_peer, attn_weights, peer_consensus
