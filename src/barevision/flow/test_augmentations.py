"""Tests for data augmentation functions."""

import random

import numpy as np
import pytest

from barevision.flow.augmentations import (
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_rotation,
    apply_color_augmentation,
    apply_swap_frames,
    compose_augmentations,
)


def create_test_images():
    """Create simple test images for augmentation testing.

    Returns:
        Tuple of (img1, img2) with distinct patterns for verification
    """
    # Create img1 with a gradient pattern
    img1 = np.zeros((64, 64, 3), dtype=np.float32)
    img1[:, :, 0] = np.linspace(0, 1, 64)  # Red gradient (left to right)
    img1[:, :, 1] = np.linspace(1, 0, 64)[:, None]  # Green gradient (top to bottom)
    img1[:, :, 2] = 0.5  # Constant blue

    # Create img2 with different pattern
    img2 = np.zeros((64, 64, 3), dtype=np.float32)
    img2[:, :, 0] = 0.5  # Constant red
    img2[:, :, 1] = np.linspace(0, 1, 64)  # Green gradient (left to right)
    img2[:, :, 2] = np.linspace(1, 0, 64)[:, None]  # Blue gradient (top to bottom)

    return img1, img2


class TestHorizontalFlip:
    """Tests for horizontal flip augmentation."""

    def test_horizontal_flip_changes_image(self):
        """Horizontal flip should change the image content."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        flipped_img1, flipped_img2 = apply_horizontal_flip(img1, img2, rng)

        # Flip should change the image (gradient direction reverses)
        assert not np.array_equal(img1, flipped_img1)
        assert not np.array_equal(img2, flipped_img2)

    def test_horizontal_flip_preserves_shape(self):
        """Horizontal flip should preserve image shape."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        flipped_img1, flipped_img2 = apply_horizontal_flip(img1, img2, rng)

        assert flipped_img1.shape == img1.shape
        assert flipped_img2.shape == img2.shape

    def test_horizontal_flip_preserves_value_range(self):
        """Horizontal flip should preserve [0, 1] value range."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        flipped_img1, flipped_img2 = apply_horizontal_flip(img1, img2, rng)

        assert np.all(flipped_img1 >= 0) and np.all(flipped_img1 <= 1)
        assert np.all(flipped_img2 >= 0) and np.all(flipped_img2 <= 1)

    def test_horizontal_flip_deterministic(self):
        """Horizontal flip should be deterministic (same input = same output)."""
        img1, img2 = create_test_images()
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        flipped1_img1, flipped1_img2 = apply_horizontal_flip(img1, img2, rng1)
        flipped2_img1, flipped2_img2 = apply_horizontal_flip(img1, img2, rng2)

        assert np.array_equal(flipped1_img1, flipped2_img1)
        assert np.array_equal(flipped1_img2, flipped2_img2)


class TestVerticalFlip:
    """Tests for vertical flip augmentation."""

    def test_vertical_flip_changes_image(self):
        """Vertical flip should change the image content."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        flipped_img1, flipped_img2 = apply_vertical_flip(img1, img2, rng)

        assert not np.array_equal(img1, flipped_img1)
        assert not np.array_equal(img2, flipped_img2)

    def test_vertical_flip_preserves_shape(self):
        """Vertical flip should preserve image shape."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        flipped_img1, flipped_img2 = apply_vertical_flip(img1, img2, rng)

        assert flipped_img1.shape == img1.shape
        assert flipped_img2.shape == img2.shape


class TestRotation:
    """Tests for rotation augmentation."""

    def test_rotation_changes_image(self):
        """Rotation should change the image content."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        rotated_img1, rotated_img2 = apply_rotation(img1, img2, rng, max_angle=30.0)

        # Rotation should change the image
        assert not np.array_equal(img1, rotated_img1)
        assert not np.array_equal(img2, rotated_img2)

    def test_rotation_preserves_shape(self):
        """Rotation should preserve image shape."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        rotated_img1, rotated_img2 = apply_rotation(img1, img2, rng, max_angle=30.0)

        assert rotated_img1.shape == img1.shape
        assert rotated_img2.shape == img2.shape

    def test_rotation_preserves_value_range(self):
        """Rotation should preserve [0, 1] value range."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        rotated_img1, rotated_img2 = apply_rotation(img1, img2, rng, max_angle=30.0)

        assert np.all(rotated_img1 >= 0) and np.all(rotated_img1 <= 1)
        assert np.all(rotated_img2 >= 0) and np.all(rotated_img2 <= 1)

    def test_rotation_zero_angle(self):
        """Rotation with zero angle should return nearly identical image."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        # Force zero rotation
        rotated_img1, rotated_img2 = apply_rotation(img1, img2, rng, max_angle=0.0)

        # Should be very close (PIL conversion to uint8 and back introduces quantization)
        # Difference is ~1/255 ≈ 0.004 from uint8 round-trip
        assert np.allclose(rotated_img1, img1, atol=0.005)
        assert np.allclose(rotated_img2, img2, atol=0.005)


class TestColorAugmentation:
    """Tests for color augmentation."""

    def test_color_augmentation_changes_image(self):
        """Color augmentation should change the image content."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        aug_img1, aug_img2 = apply_color_augmentation(img1, img2, rng, strength=0.2)

        # Color augmentation should change the image
        assert not np.array_equal(img1, aug_img1)
        assert not np.array_equal(img2, aug_img2)

    def test_color_augmentation_preserves_shape(self):
        """Color augmentation should preserve image shape."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        aug_img1, aug_img2 = apply_color_augmentation(img1, img2, rng, strength=0.2)

        assert aug_img1.shape == img1.shape
        assert aug_img2.shape == img2.shape

    def test_color_augmentation_preserves_value_range(self):
        """Color augmentation should preserve [0, 1] value range."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        aug_img1, aug_img2 = apply_color_augmentation(img1, img2, rng, strength=0.2)

        assert np.all(aug_img1 >= 0) and np.all(aug_img1 <= 1)
        assert np.all(aug_img2 >= 0) and np.all(aug_img2 <= 1)

    def test_color_augmentation_zero_strength(self):
        """Color augmentation with zero strength should return identical image."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        aug_img1, aug_img2 = apply_color_augmentation(img1, img2, rng, strength=0.0)

        # With zero strength, images should be nearly identical
        # (minor floating point differences from multiplication by 1.0)
        assert np.allclose(aug_img1, img1, atol=1e-6)
        assert np.allclose(aug_img2, img2, atol=1e-6)


class TestSwapFrames:
    """Tests for frame swap augmentation."""

    def test_swap_frames_exchanges_images(self):
        """Frame swap should exchange img1 and img2."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        swapped_img1, swapped_img2 = apply_swap_frames(img1, img2, rng)

        # Swap should exchange the images
        assert np.array_equal(swapped_img1, img2)
        assert np.array_equal(swapped_img2, img1)

    def test_swap_frames_preserves_shape(self):
        """Frame swap should preserve image shapes."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        swapped_img1, swapped_img2 = apply_swap_frames(img1, img2, rng)

        assert swapped_img1.shape == img1.shape
        assert swapped_img2.shape == img2.shape


class TestComposeAugmentations:
    """Tests for composed augmentations."""

    def test_compose_with_no_augmentations(self):
        """Composition with all probabilities at 0 should return unchanged images."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        result_img1, result_img2 = compose_augmentations(
            img1,
            img2,
            rng,
            horizontal_flip_prob=0.0,
            vertical_flip_prob=0.0,
            rotation_prob=0.0,
            color_augmentation_prob=0.0,
            swap_frames_prob=0.0,
        )

        assert np.array_equal(result_img1, img1)
        assert np.array_equal(result_img2, img2)

    def test_compose_with_horizontal_flip(self):
        """Composition with horizontal flip should apply flip."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        result_img1, result_img2 = compose_augmentations(
            img1,
            img2,
            rng,
            horizontal_flip_prob=1.0,
            vertical_flip_prob=0.0,
            rotation_prob=0.0,
            color_augmentation_prob=0.0,
            swap_frames_prob=0.0,
        )

        # Should be flipped
        assert not np.array_equal(result_img1, img1)

    def test_compose_with_swap(self):
        """Composition with swap should exchange images."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        result_img1, result_img2 = compose_augmentations(
            img1,
            img2,
            rng,
            horizontal_flip_prob=0.0,
            vertical_flip_prob=0.0,
            rotation_prob=0.0,
            color_augmentation_prob=0.0,
            swap_frames_prob=1.0,
        )

        # Should be swapped
        assert np.array_equal(result_img1, img2)
        assert np.array_equal(result_img2, img1)

    def test_compose_deterministic_with_same_seed(self):
        """Composition with same seed should produce identical results."""
        img1, img2 = create_test_images()
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        result1_img1, result1_img2 = compose_augmentations(
            img1, img2, rng1, horizontal_flip_prob=0.5, rotation_prob=0.5
        )

        # Reset and run again with same seed
        result2_img1, result2_img2 = compose_augmentations(
            img1, img2, rng2, horizontal_flip_prob=0.5, rotation_prob=0.5
        )

        assert np.array_equal(result1_img1, result2_img1)
        assert np.array_equal(result1_img2, result2_img2)

    def test_compose_preserves_value_range(self):
        """Composition should preserve [0, 1] value range."""
        img1, img2 = create_test_images()
        rng = random.Random(42)

        result_img1, result_img2 = compose_augmentations(
            img1,
            img2,
            rng,
            horizontal_flip_prob=1.0,
            rotation_prob=1.0,
            color_augmentation_prob=1.0,
        )

        assert np.all(result_img1 >= 0) and np.all(result_img1 <= 1)
        assert np.all(result_img2 >= 0) and np.all(result_img2 <= 1)
