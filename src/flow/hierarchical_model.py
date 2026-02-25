"""Hierarchical optical flow model.

Integrates embedding pyramid, window processing, and flow blending
into a complete end-to-end model for 64×64 (and beyond) images.
"""

from typing import Tuple, Dict, Any, Optional
import jax.numpy as jnp
from flax import nnx

from flow.embedding_pyramid import EmbeddingPyramid
from flow.window_flow import WindowFlowProcessor
from flow.flow_blender import FlowBlender
from flow.window_grid import crop_to_valid, compute_valid_resolution


class HierarchicalFlowModel(nnx.Module):
    """Complete hierarchical optical flow model.

    Processes input images through a pyramid, estimates flow at each level
    using windowed attention, and blends results using confidence scores.

    Architecture:
    1. Generate embedding pyramid for both frames
    2. Process coarsest level directly (single window)
    3. Process finer levels using windowed attention
    4. Blend coarse flow into fine flow using confidence weights
    5. Return final flow at finest resolution

    For 2-level pyramid with 64×64 input:
    - Level 0 (coarse): 16×16 embeddings → 16×16 flow
    - Level 1 (fine): 32×32 embeddings → 32×32 flow
    - Blend: 16×16 flow upsampled to 32×32, blended with fine flow
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
        self.required_size = compute_valid_resolution(num_levels)

        # Pyramid generator
        self.pyramid = EmbeddingPyramid(
            num_levels=num_levels,
            embed_dim=embed_dim,
            in_channels=in_channels,
            rngs=rngs,
        )

        # Window flow processor (for all levels)
        self.window_processor = WindowFlowProcessor(
            embed_dim=embed_dim,
            window_size=window_size,
            rngs=rngs,
        )

        # Flow blender (for all levels except finest)
        self.blender = FlowBlender()

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

        if H == self.required_size and W == self.required_size:
            return img1, img2

        if self.auto_crop:
            if H >= self.required_size and W >= self.required_size:
                # Can crop
                img1_cropped = crop_to_valid(img1, self.num_levels)
                img2_cropped = crop_to_valid(img2, self.num_levels)
                return img1_cropped, img2_cropped
            else:
                raise ValueError(
                    f"Input size ({H}, {W}) is smaller than required "
                    f"({self.required_size}, {self.required_size}). "
                    f"Cannot auto-crop. Set auto_crop=False and provide correct size."
                )
        else:
            raise ValueError(
                f"Input size ({H}, {W}) doesn't match required size "
                f"({self.required_size}, {self.required_size}) for {self.num_levels} pyramid levels. "
                f"Set auto_crop=True to enable automatic cropping."
            )

    def _process_level(
        self, emb1: jnp.ndarray, emb2: jnp.ndarray, level_idx: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """Process a single pyramid level to estimate flow.

        Args:
            emb1: Embeddings from frame 1 at this level
            emb2: Embeddings from frame 2 at this level
            level_idx: Index of this level (0 = coarsest, num_levels-1 = finest)

        Returns:
            flow: Flow estimate at this level (B, H, W, 2) in normalized coords
            confidence: Confidence scores (B, H, W, 1)
            aux: Auxiliary outputs
        """
        # Use window processor for all levels (handles single or multiple windows)
        flow, confidence, aux = self.window_processor(emb1, emb2)

        # Add level info to aux
        aux["level_idx"] = level_idx
        aux["level_shape"] = emb1.shape[1:3]

        return flow, confidence, aux

    def __call__(
        self, img1: jnp.ndarray, img2: jnp.ndarray, return_intermediates: bool = False
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Estimate optical flow between two frames.

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

        # Process coarsest level first
        flow_levels = []
        conf_levels = []
        aux_levels = []

        for level_idx in range(self.num_levels):
            emb1 = pyramid1[level_idx]
            emb2 = pyramid2[level_idx]

            flow, conf, aux = self._process_level(emb1, emb2, level_idx)

            flow_levels.append(flow)
            conf_levels.append(conf)
            aux_levels.append(aux)

        # Now blend from coarse to fine
        # Start with coarsest level as base
        flow_current = flow_levels[0]
        conf_current = conf_levels[0]

        for level_idx in range(1, self.num_levels):
            flow_fine = flow_levels[level_idx]
            conf_fine = conf_levels[level_idx]

            # Blend coarse into fine
            flow_blended, conf_blended, blend_aux = self.blender.blend_pyramid_levels(
                flow_fine, conf_fine, flow_current, conf_current
            )

            flow_current = flow_blended
            conf_current = conf_blended
            aux_levels[level_idx]["blend"] = blend_aux

        # flow_current is now at finest resolution in normalized coordinates
        # Convert to pixel coordinates
        # The finest level has shape (required_size / 2, required_size / 2)
        finest_H = self.required_size // 2
        finest_scale = float(self.required_size)  # Scale by original image size

        flow_pixels = flow_current * finest_scale

        # Prepare output
        aux = {
            "flow_normalized": flow_current,
            "confidence": conf_current,
            "pyramid_shapes": [p.shape for p in pyramid1],
            "num_levels": self.num_levels,
            "finest_resolution": (finest_H, finest_H),
            "pixel_scale": finest_scale,
        }

        if return_intermediates:
            aux["level_flows"] = flow_levels
            aux["level_confidences"] = conf_levels
            aux["level_aux"] = aux_levels
            aux["pyramid1"] = pyramid1
            aux["pyramid2"] = pyramid2

        return flow_pixels, aux
