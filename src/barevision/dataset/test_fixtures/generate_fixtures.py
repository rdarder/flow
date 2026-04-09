#!/usr/bin/env python
"""Generate synthetic test fixtures for dataset tests.

Creates small video folders with a few frames each for testing.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "frames"


def generate_solid_color_image(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    """Generate a solid color RGB image.

    Args:
        color: RGB tuple (0-255)
        size: Image size in pixels

    Returns:
        (size, size, 3) uint8 array
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def generate_gradient_image(color1: tuple[int, int, int], color2: tuple[int, int, int], size: int = 64) -> np.ndarray:
    """Generate a horizontal gradient image.

    Args:
        color1: Left edge RGB tuple
        color2: Right edge RGB tuple
        size: Image size in pixels

    Returns:
        (size, size, 3) uint8 array
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(size):
        t = x / (size - 1)
        color = tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))
        img[:, x] = color
    return img


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Video A: 4 frames with solid colors
    video_a_dir = FIXTURES_DIR / "video_a"
    video_a_dir.mkdir(exist_ok=True)

    colors_a = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
    ]

    for i, color in enumerate(colors_a):
        img = generate_solid_color_image(color)
        img_path = video_a_dir / f"frame_{i:06d}.jpg"
        Image.fromarray(img).save(img_path)
        print(f"Created {img_path}")

    # Video B: 5 frames with gradients
    video_b_dir = FIXTURES_DIR / "video_b"
    video_b_dir.mkdir(exist_ok=True)

    gradients_b = [
        ((255, 0, 0), (0, 0, 0)),      # Red to black
        ((0, 255, 0), (0, 0, 0)),      # Green to black
        ((0, 0, 255), (0, 0, 0)),      # Blue to black
        ((255, 255, 0), (0, 0, 0)),    # Yellow to black
        ((255, 0, 255), (0, 0, 0)),    # Magenta to black
    ]

    for i, (c1, c2) in enumerate(gradients_b):
        img = generate_gradient_image(c1, c2)
        img_path = video_b_dir / f"frame_{i:06d}.jpg"
        Image.fromarray(img).save(img_path)
        print(f"Created {img_path}")

    print(f"\nGenerated fixtures in {FIXTURES_DIR}")
    print(f"  video_a: {len(colors_a)} frames")
    print(f"  video_b: {len(gradients_b)} frames")


if __name__ == "__main__":
    main()
