from typing import Tuple

import jax.numpy as jnp

WINDOW_SIZE = 16


def compute_valid_resolution(num_levels: int) -> int:
    """
    Compute the required image resolution for a given number of pyramid levels.

    For num_levels pyramid levels, we need:
    - Level N-1 (finest): WINDOW_SIZE * 2^(num_levels-1) embeddings
    - Each embedding comes from a 2x2 region

    From the doc:
    - Level 1 (finest): 64x64 image -> 32x32 embeddings (2x2 patchify)
    - Level 0 (coarse): 32x32 embeddings -> 16x16 embeddings (2x2 patchify)

    So for 2 levels:
    - Coarse level has 16x16 embeddings
    - Fine level has 32x32 embeddings
    - Image is 64x64

    Pattern: image_size = WINDOW_SIZE * 2^num_levels
    - 1 level: 16 * 2^1 = 32 -> but we'd want 16x16 attention on 16x16 embeddings
      Actually for 1 level, we have 16x16 embeddings, which means 32x32 image
    - 2 levels: 16 * 2^2 = 64 -> 64x64 image, 32x32 fine embeddings

    So: valid_resolution = WINDOW_SIZE * 2^num_levels
    """
    return WINDOW_SIZE * (2**num_levels)


def validate_resolution(image_size: int, num_levels: int) -> Tuple[bool, str]:
    """
    Validate that image resolution is compatible with pyramid + windowing.

    Returns:
        (is_valid, message)
    """
    expected_size = compute_valid_resolution(num_levels)

    if image_size == expected_size:
        return (
            True,
            f"Valid: {image_size}x{image_size} is exactly right for {num_levels} level(s)",
        )
    elif image_size > expected_size:
        # Check if it's croppable
        if image_size >= expected_size:
            return (
                True,
                f"Croppable: {image_size}x{image_size} can be "
                "cropped to {expected_size}x{expected_size}",
            )
        else:
            return (
                False,
                f"Invalid: {image_size}x{image_size} is too small. "
                "Minimum for {num_levels} level(s) is {expected_size}x{expected_size}",
            )
    else:
        return (
            False,
            f"Invalid: {image_size}x{image_size} is too small. "
            "Expected {expected_size}x{expected_size} for {num_levels} level(s)",
        )


def crop_to_valid(img: jnp.ndarray, num_levels: int) -> jnp.ndarray:
    """
    Center-crop an image to the valid resolution for the given pyramid depth.

    Args:
        img: Image tensor (B, H, W, C) or (H, W, C)
        num_levels: Number of pyramid levels

    Returns:
        Cropped image to valid size
    """
    target_size = compute_valid_resolution(num_levels)

    if img.ndim == 3:
        h, w, c = img.shape
        if h == target_size and w == target_size:
            return img

        # Center crop
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return img[start_h : start_h + target_size, start_w : start_w + target_size, :]

    elif img.ndim == 4:
        b, h, w, c = img.shape
        if h == target_size and w == target_size:
            return img

        # Center crop
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return img[
            :, start_h : start_h + target_size, start_w : start_w + target_size, :
        ]

    else:
        raise ValueError(f"Expected 3D or 4D array, got {img.ndim}D")


class WindowGrid:
    """
    Handles splitting and stitching of embedding grids into 16x16 windows.

    For example, with WINDOW_SIZE=16:
    - 32x32 embeddings -> 4 windows (2x2 grid)
    - 64x64 embeddings -> 16 windows (4x4 grid)
    - 16x16 embeddings -> 1 window (1x1 grid)
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size

    def __repr__(self) -> str:
        return f"WindowGrid(window_size={self.window_size})"

    def compute_num_windows(self, h: int, w: int) -> int:
        """Compute the number of windows in a grid of size (H, W)."""
        if h % self.window_size != 0:
            raise ValueError(
                f"Height {h} is not divisible by window size {self.window_size}. "
                f"Expected image size compatible with WINDOW_SIZE * 2^n."
            )
        if w % self.window_size != 0:
            raise ValueError(
                f"Width {w} is not divisible by window size {self.window_size}. "
                f"Expected image size compatible with WINDOW_SIZE * 2^n."
            )

        num_h = h // self.window_size
        num_w = w // self.window_size
        return num_h * num_w

    def split(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Split embeddings into non-overlapping windows.

        Args:
            embeddings: (B, H, W, C) tensor

        Returns:
            (B, num_windows, window_size, window_size, C) tensor
        """
        batch_size, h, w, channels = embeddings.shape

        if h % self.window_size != 0:
            raise ValueError(
                f"Height {h} is not divisible by window size {self.window_size}. "
                f"Cannot split {h}x{w} grid into {self.window_size}x{self.window_size} windows."
            )
        if w % self.window_size != 0:
            raise ValueError(
                f"Width {w} is not divisible by window size {self.window_size}. "
                f"Cannot split {h}x{w} grid into {self.window_size}x{self.window_size} windows."
            )

        num_h = h // self.window_size
        num_w = w // self.window_size
        num_windows = num_h * num_w

        # Reshape: (B, H, W, C) -> (B, num_h, window_size, num_w, window_size, C)
        # First, split H and W dimensions
        windows = embeddings.reshape(
            batch_size, num_h, self.window_size, num_w, self.window_size, channels
        )

        # Rearrange: (B, num_h, window_size, num_w, window_size, C)
        #         -> (B, num_h, num_w, window_size, window_size, C)
        #         -> (B, num_h * num_w, window_size, window_size, C)
        windows = windows.transpose(0, 1, 3, 2, 4, 5)
        windows = windows.reshape(
            batch_size, num_windows, self.window_size, self.window_size, channels
        )

        return windows

    def stitch(self, windows: jnp.ndarray, grid_h: int, grid_w: int) -> jnp.ndarray:
        """
        Stitch windows back into a grid.

        Args:
            windows: (B, num_windows, window_size, window_size, C) tensor
            grid_h: Number of windows along height
            grid_w: Number of windows along width

        Returns:
            (B, H, W, C) tensor where H = grid_h * window_size, W = grid_w * window_size
        """
        batch_size, num_windows, win_h, win_w, channels = windows.shape

        expected_windows = grid_h * grid_w
        if num_windows != expected_windows:
            raise ValueError(
                f"Number of windows {num_windows} doesn't match "
                "expected grid {grid_h}x{grid_w} = {expected_windows}"
            )

        if win_h != self.window_size or win_w != self.window_size:
            raise ValueError(
                f"Window size ({win_h}, {win_w}) doesn't match "
                "expected ({self.window_size}, {self.window_size})"
            )

        # Reshape: (B, num_windows, W, W, C) -> (B, grid_h, grid_w, W, W, C)
        windows = windows.reshape(
            batch_size, grid_h, grid_w, self.window_size, self.window_size, channels
        )

        # Rearrange: (B, grid_h, grid_w, W, W, C) -> (B, grid_h, W, grid_w, W, C)
        windows = windows.transpose(0, 1, 3, 2, 4, 5)

        # Reshape: (B, grid_h, W, grid_w, W, C) -> (B, grid_h * W, grid_w * W, C)
        h = grid_h * self.window_size
        w = grid_w * self.window_size
        embeddings = windows.reshape(batch_size, h, w, channels)

        return embeddings
