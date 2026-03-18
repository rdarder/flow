"""Data augmentation functions for training.

All augmentations are applied on-the-fly during training with per-sample deterministic seeding.
This ensures the same sample always gets the same augmentation across epochs, while different
samples get different augmentations for diversity.

Augmentations are designed for self-supervised learning with reconstruction loss:
- Both img1 and img2 receive the same geometric transformation
- No ground truth flow is used, so no flow negation is needed
- Frame swap is valid since we optimize for reconstruction, not flow direction
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image


def apply_horizontal_flip(
    img1: np.ndarray, img2: np.ndarray, rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply horizontal flip (left-right mirror) to both images.

    Args:
        img1: First image array (H, W, 3) in [0, 1]
        img2: Second image array (H, W, 3) in [0, 1]
        rng: Random number generator (unused, operation is deterministic)

    Returns:
        Tuple of (flipped_img1, flipped_img2)
    """
    flipped_img1 = np.fliplr(img1).copy()
    flipped_img2 = np.fliplr(img2).copy()
    return flipped_img1, flipped_img2


def apply_vertical_flip(
    img1: np.ndarray, img2: np.ndarray, rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply vertical flip (up-down mirror) to both images.

    Args:
        img1: First image array (H, W, 3) in [0, 1]
        img2: Second image array (H, W, 3) in [0, 1]
        rng: Random number generator (unused, operation is deterministic)

    Returns:
        Tuple of (flipped_img1, flipped_img2)
    """
    flipped_img1 = np.flipud(img1).copy()
    flipped_img2 = np.flipud(img2).copy()
    return flipped_img1, flipped_img2


def apply_rotation(
    img1: np.ndarray, img2: np.ndarray, rng: random.Random, max_angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random rotation to both images.

    Args:
        img1: First image array (H, W, 3) in [0, 1]
        img2: Second image array (H, W, 3) in [0, 1]
        rng: Random number generator for sampling rotation angle
        max_angle: Maximum rotation angle in degrees (rotation is uniform in [-angle, +angle])

    Returns:
        Tuple of (rotated_img1, rotated_img2)
    """
    angle = rng.uniform(-max_angle, max_angle)

    # Convert to PIL for rotation
    img1_pil = Image.fromarray((img1 * 255).astype(np.uint8))
    img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))

    # Rotate with bilinear interpolation, expand to fit entire rotated image
    img1_rotated = img1_pil.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False
    )
    img2_rotated = img2_pil.rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=False
    )

    # Convert back to numpy array and normalize to [0, 1]
    img1_rotated = np.array(img1_rotated).astype(np.float32) / 255.0
    img2_rotated = np.array(img2_rotated).astype(np.float32) / 255.0

    return img1_rotated, img2_rotated


def apply_color_augmentation(
    img1: np.ndarray, img2: np.ndarray, rng: random.Random, strength: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random brightness and contrast augmentation to both images.

    Args:
        img1: First image array (H, W, 3) in [0, 1]
        img2: Second image array (H, W, 3) in [0, 1]
        rng: Random number generator for sampling augmentation parameters
        strength: Strength of color jitter (multiplier for brightness/contrast changes)

    Returns:
        Tuple of (augmented_img1, augmented_img2)
    """
    # Sample brightness and contrast factors
    # Brightness: multiply by factor in [1 - strength, 1 + strength]
    # Contrast: multiply by factor in [1 - strength, 1 + strength]
    brightness_factor = 1.0 + rng.uniform(-strength, strength)
    contrast_factor = 1.0 + rng.uniform(-strength, strength)

    def augment_image(img: np.ndarray) -> np.ndarray:
        # Apply brightness
        img = img * brightness_factor

        # Apply contrast: interpolate between mean and current value
        mean = img.mean()
        img = mean + (img - mean) * contrast_factor

        # Clip to valid range [0, 1]
        img = np.clip(img, 0.0, 1.0)

        return img

    return augment_image(img1), augment_image(img2)


def apply_swap_frames(
    img1: np.ndarray, img2: np.ndarray, rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap the order of the two frames.

    Note: For self-supervised reconstruction loss, this is valid without any flow negation.
    The network learns to reconstruct img2 from img1 regardless of temporal order.

    Args:
        img1: First image array (H, W, 3) in [0, 1]
        img2: Second image array (H, W, 3) in [0, 1]
        rng: Random number generator (unused, operation is deterministic)

    Returns:
        Tuple of (img2, img1) - swapped order
    """
    return img2, img1


def compose_augmentations(
    img1: np.ndarray,
    img2: np.ndarray,
    rng: random.Random,
    horizontal_flip_prob: float = 0.0,
    vertical_flip_prob: float = 0.0,
    rotation_prob: float = 0.0,
    rotation_max_angle: float = 15.0,
    color_augmentation_prob: float = 0.0,
    color_jitter_strength: float = 0.1,
    swap_frames_prob: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a composition of augmentations to both images.

    Each augmentation is applied independently with its specified probability.
    Augmentations are applied in a fixed order:
    1. Geometric transformations (flips, rotation)
    2. Color augmentation
    3. Frame swap (always last since it changes semantics)

    Args:
        img1: First image array (H, W, 3) in [0, 1]
        img2: Second image array (H, W, 3) in [0, 1]
        rng: Random number generator for probabilistic decisions
        horizontal_flip_prob: Probability of horizontal flip
        vertical_flip_prob: Probability of vertical flip
        rotation_prob: Probability of random rotation
        rotation_max_angle: Maximum rotation angle in degrees
        color_augmentation_prob: Probability of color augmentation
        color_jitter_strength: Strength of color jitter
        swap_frames_prob: Probability of swapping frames

    Returns:
        Tuple of (augmented_img1, augmented_img2)
    """
    # Apply geometric augmentations (order: flip, flip, rotate)
    if rng.random() < horizontal_flip_prob:
        img1, img2 = apply_horizontal_flip(img1, img2, rng)

    if rng.random() < vertical_flip_prob:
        img1, img2 = apply_vertical_flip(img1, img2, rng)

    if rng.random() < rotation_prob:
        img1, img2 = apply_rotation(img1, img2, rng, rotation_max_angle)

    # Apply color augmentation
    if rng.random() < color_augmentation_prob:
        img1, img2 = apply_color_augmentation(img1, img2, rng, color_jitter_strength)

    # Apply frame swap (always last)
    if rng.random() < swap_frames_prob:
        img1, img2 = apply_swap_frames(img1, img2, rng)

    return img1, img2
