from typing import Optional, Tuple

import jax.numpy as jnp


def compute_valid_resolution(
    num_levels: int,
    window_size: int = 16,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Compute the required image resolution for a given number of pyramid levels.

    For num_levels pyramid levels and given window_size, we need:
    - Level N-1 (finest): window_size * 2^(num_levels-1) embeddings
    - Each embedding comes from a 2x2 region

    Pattern: valid_size = window_size * 2^num_levels
    - 1 level: window_size * 2^1 = window_size * 2
    - 2 levels: window_size * 2^2 = window_size * 4
    - 3 levels: window_size * 2^3 = window_size * 8

    Args:
        num_levels: Number of pyramid levels
        window_size: Size of attention windows (default 16)
        height: Target height (if None, uses square resolution)
        width: Target width (if None, uses square resolution)

    Returns:
        Required image size as (height, width) tuple
    """
    min_size = window_size * (2**num_levels)

    if height is None:
        height = min_size
    if width is None:
        width = min_size

    return (height, width)


def validate_resolution(
    height: int, width: int, num_levels: int, window_size: int = 16
) -> Tuple[bool, str]:
    """
    Validate that image resolution is compatible with pyramid + windowing.

    Args:
        height: Input image height
        width: Input image width
        num_levels: Number of pyramid levels
        window_size: Size of attention windows (default 16)

    Returns:
        (is_valid, message)
    """
    min_size = window_size * (2**num_levels)

    # Check if dimensions are valid (multiples of min_size)
    h_valid = height % min_size == 0
    w_valid = width % min_size == 0

    if h_valid and w_valid:
        if height == min_size and width == min_size:
            return (
                True,
                f"Valid: {height}x{width} is minimum size for {num_levels} level(s)",
            )
        else:
            return (
                True,
                f"Valid: {height}x{width} is compatible with {num_levels} level(s)",
            )
    else:
        issues = []
        if not h_valid:
            issues.append(f"height {height} (must be multiple of {min_size})")
        if not w_valid:
            issues.append(f"width {width} (must be multiple of {min_size})")
        return (False, f"Invalid dimensions: {', '.join(issues)}")


def crop_to_valid(
    img: jnp.ndarray,
    num_levels: int = 2,
    window_size: int = 16,
    target_height: Optional[int] = None,
    target_width: Optional[int] = None,
) -> jnp.ndarray:
    """
    Center-crop an image to the valid resolution for the given pyramid depth.

    Args:
        img: Image tensor (B, H, W, C) or (H, W, C)
        num_levels: Number of pyramid levels (used to compute target if not provided)
        window_size: Size of attention windows (default 16)
        target_height: Target height (if provided, overrides num_levels computation)
        target_width: Target width (if provided, overrides num_levels computation)

    Returns:
        Cropped image to valid size
    """
    # Compute target size if not provided
    if target_height is None or target_width is None:
        target_height, target_width = compute_valid_resolution(num_levels, window_size)

    if img.ndim == 3:
        h, w, c = img.shape
        if h == target_height and w == target_width:
            return img

        # Center crop
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        return img[
            start_h : start_h + target_height, start_w : start_w + target_width, :
        ]

    elif img.ndim == 4:
        b, h, w, c = img.shape
        if h == target_height and w == target_width:
            return img

        # Center crop
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        return img[
            :, start_h : start_h + target_height, start_w : start_w + target_width, :
        ]

    else:
        raise ValueError(f"Expected 3D or 4D array, got {img.ndim}D")


def create_coordinate_grid(h: int, w: int) -> jnp.ndarray:
    """Create normalized coordinate grid [0, 1] for a spatial grid.

    Args:
        h: Height of grid
        w: Width of grid

    Returns:
        Grid of shape (h, w, 2) with (x, y) coordinates in [0, 1]
    """
    # Create coordinate grid
    y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")

    # Normalize to [0, 1]
    x_norm = (
        x.astype(jnp.float32) / float(w - 1)
        if w > 1
        else jnp.zeros_like(x, dtype=jnp.float32)
    )
    y_norm = (
        y.astype(jnp.float32) / float(h - 1)
        if h > 1
        else jnp.zeros_like(y, dtype=jnp.float32)
    )

    # Stack to get (h, w, 2)
    grid = jnp.stack([x_norm, y_norm], axis=-1)
    return grid


def grid_to_tokens(grid: jnp.ndarray) -> jnp.ndarray:
    """Convert spatial grid to token format by flattening H and W dimensions.

    Args:
        grid: (B, H, W, C) tensor

    Returns:
        (B, H*W, C) tokens
    """
    B, H, W, C = grid.shape
    return grid.reshape(B, H * W, C)


def tokens_to_grid(tokens: jnp.ndarray, h: int, w: int) -> jnp.ndarray:
    """Convert tokens back to spatial grid format.

    Args:
        tokens: (B, H*W, C) tensor
        h: Target height
        w: Target width

    Returns:
        (B, H, W, C) grid tensor
    """
    B, _, C = tokens.shape
    return tokens.reshape(B, h, w, C)


class WindowGrid:
    """
    Handles splitting and stitching of embedding grids into windows.

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
