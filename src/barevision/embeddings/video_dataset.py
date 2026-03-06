"""Video frame dataset for self-supervised embedding training.

Loads video frames with sparse pairing for self-supervised training.
Generates pairs (frame_t, frame_{t+k}) for k in [1, max_distance].

Train/val split:
- Training: First 13 videos alphabetically (backyard through table)
- Validation: Last 2 videos (toys, workshop)
"""

import os
from pathlib import Path
from typing import List, Tuple, NamedTuple, Optional

from PIL import Image
import numpy as np

from .utils import get_datasets_dir


class FramePair(NamedTuple):
    """A pair of frames from the same video."""

    video_name: str
    frame_t_idx: int
    frame_tk_idx: int
    distance: int
    img1_path: str
    img2_path: str


class VideoFrameDataset:
    """Dataset for loading paired video frames.

    Generates frame pairs (t, t+k) for self-supervised training where:
    - t ranges over all valid frame indices
    - k ranges from 1 to max_frame_distance
    - Pairs are only created within the same video

    The dataset uses contiguous train/val splits per video to prevent leakage.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = "train",
        max_frame_distance: int = 5,
        img_size: Tuple[int, int] = (190, 190),
    ):
        """Initialize the dataset.

        Args:
            data_root: Root directory containing video subdirectories.
                      If None, uses project's datasets/frames directory.
            split: 'train' or 'val'
            max_frame_distance: Maximum temporal distance k for frame pairs (t, t+k)
            img_size: Target image size (height, width)
        """
        # Use project datasets directory if not specified
        if data_root is None:
            self.data_root = str(get_datasets_dir())
        else:
            self.data_root = data_root

        self.split = split
        self.max_frame_distance = max_frame_distance
        self.img_size = img_size

        assert split in ["train", "val"], f"Invalid split: {split}"
        assert os.path.isdir(self.data_root), f"Dataset root not found: {self.data_root}"

        # Get video directories
        all_videos = sorted(os.listdir(self.data_root))

        # Train/val split: first 13 train, last 2 val
        if split == "train":
            self.videos = all_videos[:13]
        else:
            self.videos = all_videos[13:]

        # Build frame pairs index
        self.frame_pairs: List[FramePair] = []
        self._build_index()

    def _build_index(self):
        """Build index of all frame pairs."""
        self.frame_pairs = []

        for video_name in self.videos:
            video_path = os.path.join(self.data_root, video_name)

            if not os.path.isdir(video_path):
                continue

            # Get all frame files
            frame_files = sorted(
                [f for f in os.listdir(video_path) if f.endswith((".jpg", ".png"))]
            )
            num_frames = len(frame_files)

            if num_frames < 2:
                continue

            # Generate pairs (t, t+k) for all valid t and k in [1, max_distance]
            for t in range(num_frames):
                max_k = min(self.max_frame_distance, num_frames - t - 1)
                for k in range(1, max_k + 1):
                    self.frame_pairs.append(
                        FramePair(
                            video_name=video_name,
                            frame_t_idx=t,
                            frame_tk_idx=t + k,
                            distance=k,
                            img1_path=os.path.join(video_path, frame_files[t]),
                            img2_path=os.path.join(video_path, frame_files[t + k]),
                        )
                    )

    def __len__(self) -> int:
        """Return number of frame pairs."""
        return len(self.frame_pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load a frame pair.

        Args:
            idx: Index into the dataset

        Returns:
            Tuple of (img1, img2, metadata) where:
                - img1: (H, W, 3) float32 array in [0, 1]
                - img2: (H, W, 3) float32 array in [0, 1]
                - metadata: dict with video_name, frame indices, distance
        """
        pair = self.frame_pairs[idx]

        # Load and preprocess images
        img1 = self._load_image(pair.img1_path)
        img2 = self._load_image(pair.img2_path)

        metadata = {
            "video_name": pair.video_name,
            "frame_t": pair.frame_t_idx,
            "frame_tk": pair.frame_tk_idx,
            "distance": pair.distance,
        }

        return img1, img2, metadata

    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess a single image.

        Args:
            path: Path to image file

        Returns:
            (H, W, 3) float32 array in [0, 1]
        """
        img = Image.open(path)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize to target size (use Resampling.LANCZOS if available, else LANCZOS)
        try:
            from PIL.Image import Resampling
            resample = Resampling.LANCZOS
        except ImportError:
            resample = Image.LANCZOS

        # PIL resize expects (width, height), so swap from (height, width)
        img = img.resize((self.img_size[1], self.img_size[0]), resample)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0

        return img_array

    def get_video_stats(self) -> dict:
        """Get statistics about the dataset.

        Returns:
            Dict with video names, frame counts, and pair counts
        """
        stats = {"videos": {}, "total_pairs": len(self.frame_pairs)}

        for pair in self.frame_pairs:
            if pair.video_name not in stats["videos"]:
                stats["videos"][pair.video_name] = {"frames": 0, "pairs": 0}
            stats["videos"][pair.video_name]["pairs"] += 1

        # Count unique frames per video
        for video_name in self.videos:
            video_path = os.path.join(self.data_root, video_name)
            if os.path.isdir(video_path):
                frames = [
                    f
                    for f in os.listdir(video_path)
                    if f.endswith((".jpg", ".png"))
                ]
                if video_name in stats["videos"]:
                    stats["videos"][video_name]["frames"] = len(frames)

        return stats


def create_train_val_datasets(
    data_root: Optional[str] = None,
    max_frame_distance: int = 5,
    img_size: Tuple[int, int] = (190, 190),
) -> Tuple[VideoFrameDataset, VideoFrameDataset]:
    """Create train and validation datasets.

    Args:
        data_root: Root directory containing video subdirectories.
                  If None, uses project's datasets/frames directory.
        max_frame_distance: Maximum temporal distance for frame pairs
        img_size: Target image size (height, width)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = VideoFrameDataset(
        data_root=data_root,
        split="train",
        max_frame_distance=max_frame_distance,
        img_size=img_size,
    )

    val_dataset = VideoFrameDataset(
        data_root=data_root,
        split="val",
        max_frame_distance=max_frame_distance,
        img_size=img_size,
    )

    return train_dataset, val_dataset
