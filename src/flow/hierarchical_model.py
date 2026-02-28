"""Hierarchical optical flow model with prior-guided attention.

Integrates embedding pyramid and window processing with prior-guided attention.
Coarse flow estimates guide spatial search at finer levels.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from flow.embedding_pyramid import EmbeddingPyramid
from flow.flow_blender import upsample_confidence_2x, upsample_flow_2x
from flow.grid_flow import GridFlowEstimator
from flow.window_grid import compute_valid_resolution, crop_to_valid


class HierarchicalFlowModel(nnx.Module):
    """Complete hierarchical optical flow model with prior-guided attention.

    Processes input images through a pyramid, estimates flow at each level
    using windowed attention with prior guidance from coarser levels.

    Architecture:
    1. Generate embedding pyramid for both frames
    2. Process each level with prior-guided attention
       - Level 0: Zero prior flow, neutral confidence (0.5)
       - Level N: Prior from level N-1 (upsampled 2x)
    3. Return final flow at finest resolution

    For 2-level pyramid with 64×64 input:
    - Level 0: 16×16 embeddings → 16×16 flow (with zero prior)
    - Level 1: 32×32 embeddings → 32×32 flow (guided by upsampled level 0)
    - Output: 32×32 flow field
    """

    def __init__(
        self,
        num_levels: int = 2,
        embed_dim: int = 16,
        in_channels: int = 3,
        window_size: int = 16,
        auto_crop: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            num_levels: Number of pyramid levels (default 2 for 64×64 → 32×32 flow)
            embed_dim: Embedding dimension at each level
            in_channels: Number of input channels (3 for RGB, 1 for grayscale)
            window_size: Size of attention windows (default 16)
            auto_crop: If True, automatically crop inputs to valid resolution
            rngs: NNX RNGs
        """
        self.num_levels = num_levels
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.window_size = window_size
        self.auto_crop = auto_crop

        # Compute required input size
        self.required_h, self.required_w = compute_valid_resolution(
            num_levels, window_size
        )

        # Pyramid generator
        self.pyramid = EmbeddingPyramid(
            num_levels=num_levels,
            embed_dim=embed_dim,
            in_channels=in_channels,
            rngs=rngs,
        )

        # Grid flow estimator (for all levels)
        self.grid_flow_estimator = GridFlowEstimator(
            embed_dim=embed_dim,
            window_size=window_size,
            rngs=rngs,
        )

    def _validate_or_crop_inputs(
        self, img1: jnp.ndarray, img2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Validate input resolution or auto-crop if enabled.

        Args:
            img1: Frame 1 (B, H, W, C)
            img2: Frame 2 (B, H, W, C)

        Returns:
            Cropped or validated images at required resolution
        """
        _, H, W, _ = img1.shape

        if H == self.required_h and W == self.required_w:
            return img1, img2

        if self.auto_crop:
            if H >= self.required_h and W >= self.required_w:
                # Can crop
                img1_cropped = crop_to_valid(
                    img1,
                    self.num_levels,
                    self.window_size,
                    self.required_h,
                    self.required_w,
                )
                img2_cropped = crop_to_valid(
                    img2,
                    self.num_levels,
                    self.window_size,
                    self.required_h,
                    self.required_w,
                )
                return img1_cropped, img2_cropped
            else:
                raise ValueError(
                    f"Input size ({H}, {W}) is smaller than required "
                    f"({self.required_h}, {self.required_w}). "
                    f"Cannot auto-crop. Set auto_crop=False and provide correct size."
                )
        else:
            raise ValueError(
                f"Input size ({H}, {W}) doesn't match required size "
                f"({self.required_h}, {self.required_w}) for {self.num_levels} pyramid levels. "
                f"Set auto_crop=True to enable automatic cropping."
            )

    def _process_level(
        self,
        emb1: jnp.ndarray,
        emb2: jnp.ndarray,
        level_idx: int,
        prior_flow: jnp.ndarray,
        prior_confidence: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Process a single pyramid level with prior-guided attention.

        Args:
            emb1: Embeddings from frame 1 at this level
            emb2: Embeddings from frame 2 at this level
            level_idx: Index of this level (0 = coarsest, num_levels-1 = finest)
            prior_flow: Flow estimate from coarser level (B, H, W, 2)
            prior_confidence: Confidence in prior flow (B, H, W, 1)

        Returns:
            flow: Flow estimate at this level (B, H, W, 2) in normalized coords
            confidence: Confidence scores (B, H, W, 1)
            aux: Auxiliary outputs
        """
        # Process with grid flow estimator and prior guidance
        flow, confidence, aux = self.grid_flow_estimator(
            emb1, emb2, prior_flow, prior_confidence
        )

        # Add level info to aux
        aux["level_idx"] = level_idx
        aux["level_shape"] = emb1.shape[1:3]

        return flow, confidence, aux

    def __call__(
        self, img1: jnp.ndarray, img2: jnp.ndarray, return_intermediates: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Estimate optical flow between two frames using prior-guided attention.

        Args:
            img1: Frame 1 (B, H, W, C) - will be cropped to valid size if needed
            img2: Frame 2 (B, H, W, C)
            return_intermediates: If True, include all intermediate outputs

        Returns:
            flow: Final flow estimate (B, H', W', 2) in pixel coordinates
                  H', W' = required_size / 2 (finest level after pyramid)
            aux: Dictionary with intermediate outputs for debugging
        """
        # Validate/crop inputs
        img1, img2 = self._validate_or_crop_inputs(img1, img2)

        # Generate pyramids (coarsest to finest)
        pyramid1 = self.pyramid(img1)  # List of [level_0, level_1, ...]
        pyramid2 = self.pyramid(img2)

        # Initialize priors for level 0 (hardcoded: zero flow, neutral confidence)
        B = img1.shape[0]
        H0, W0 = pyramid1[0].shape[1], pyramid1[0].shape[2]
        prior_flow = jnp.zeros((B, H0, W0, 2))  # Zero prior flow
        prior_confidence = jnp.full((B, H0, W0, 1), 0.5)  # Neutral confidence

        # Process each level with prior-guided attention
        flow_levels = []
        conf_levels = []
        aux_levels = []

        for level_idx in range(self.num_levels):
            emb1 = pyramid1[level_idx]
            emb2 = pyramid2[level_idx]

            # Process this level with prior guidance
            flow, conf, aux = self._process_level(
                emb1, emb2, level_idx, prior_flow, prior_confidence
            )

            flow_levels.append(flow)
            conf_levels.append(conf)
            aux_levels.append(aux)

            # Prepare prior for next level (upsample 2x)
            # Stop gradient to prevent backpropagation through hierarchy
            # Each level trains independently on its own objective
            if level_idx < self.num_levels - 1:
                prior_flow = upsample_flow_2x(flow)
                prior_confidence = upsample_confidence_2x(conf)

        # Final flow is from the finest level (already includes prior guidance)
        flow_final = flow_levels[-1]
        conf_final = conf_levels[-1]

        # Convert to pixel coordinates
        finest_H = self.required_h // 2
        finest_W = self.required_w // 2
        # Use average of H and W for pixel scale to handle non-square
        finest_scale = float((self.required_h + self.required_w) / 2)

        flow_pixels = flow_final * finest_scale

        # Prepare output
        aux = {
            "flow_normalized": flow_final,
            "confidence": conf_final,
            "pyramid_shapes": [p.shape for p in pyramid1],
            "num_levels": self.num_levels,
            "finest_resolution": (finest_H, finest_W),
            "pixel_scale": finest_scale,
        }

        if return_intermediates:
            aux["level_flows"] = flow_levels
            aux["level_confidences"] = conf_levels
            aux["level_aux"] = aux_levels
            aux["pyramid1"] = pyramid1
            aux["pyramid2"] = pyramid2

        return flow_pixels, aux
