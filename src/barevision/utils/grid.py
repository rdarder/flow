"""Grid and window utilities for spatial operations.

Provides window splitting/stitching for attention-based operations.
"""

import jax.numpy as jnp


class WindowGrid:
    """Handles splitting and stitching of embedding grids into windows.

    For example, with window_size=16:
    - 32x32 embeddings -> 4 windows (2x2 grid)
    - 64x64 embeddings -> 16 windows (4x4 grid)
    - 16x16 embeddings -> 1 window (1x1 grid)
    """

    def __init__(self, window_size: int = 16):
        self.window_size = window_size

    def __repr__(self) -> str:
        return f"WindowGrid(window_size={self.window_size})"

    def compute_num_windows(self, h: int, w: int) -> int:
        """Compute the number of windows in a grid of size (H, W)."""
        if h % self.window_size != 0:
            raise ValueError(
                f"Height {h} is not divisible by window size {self.window_size}. "
                f"Expected image size compatible with window_size * 2^n."
            )
        if w % self.window_size != 0:
            raise ValueError(
                f"Width {w} is not divisible by window size {self.window_size}. "
                f"Expected image size compatible with window_size * 2^n."
            )

        num_h = h // self.window_size
        num_w = w // self.window_size
        return num_h * num_w

    def split(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Split embeddings into non-overlapping windows.

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
        """Stitch windows back into a grid.

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
                f"expected grid {grid_h}x{grid_w} = {expected_windows}"
            )

        if win_h != self.window_size or win_w != self.window_size:
            raise ValueError(
                f"Window size ({win_h}, {win_w}) doesn't match "
                f"expected ({self.window_size}, {self.window_size})"
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
