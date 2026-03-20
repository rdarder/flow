"""Video frame dataset for self-supervised embedding training.

Loads video frames with sparse pairing for self-supervised training.
Generates pairs (frame_t, frame_{t+k}) for k in [1, max_distance].

Train/val split:
- Training: 85% of videos (rounded down)
- Validation: 15% of videos (rounded up)
- Split uses JAX PRNG with configurable seed for reproducibility
"""

import os
import random
from typing import Iterator, List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from PIL import Image

from barevision.utils import image
from barevision.utils.path import get_datasets_dir


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
    - k ranges from min_frame_distance to max_frame_distance
    - Pairs are only created within the same video

    The dataset uses video-based train/val splits (85/15) with configurable seed.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = "train",
        min_frame_distance: int = 1,
        max_frame_distance: int = 5,
        img_size: Tuple[int, int] = (190, 190),
        seed: int = 42,
        train_ratio: float = 0.85,
    ):
        """Initialize the dataset.

        Args:
            data_root: Root directory containing video subdirectories.
                      If None, uses project's datasets/frames directory.
            split: 'train' or 'val'
            min_frame_distance: Minimum temporal distance k for frame pairs (t, t+k)
            max_frame_distance: Maximum temporal distance k for frame pairs (t, t+k)
            img_size: Target image size (height, width)
            seed: Random seed for reproducible train/val split
            train_ratio: Ratio of videos for training (default 0.85 = 85%)
        """
        # Use project datasets directory if not specified
        if data_root is None:
            self.data_root = str(get_datasets_dir())
        else:
            self.data_root = data_root

        self.split = split
        self.min_frame_distance = min_frame_distance
        self.max_frame_distance = max_frame_distance
        self.img_size = img_size
        self.seed = seed
        self.train_ratio = train_ratio

        assert split in ["train", "val"], f"Invalid split: {split}"
        assert os.path.isdir(
            self.data_root
        ), f"Dataset root not found: {self.data_root}"

        # Get video directories
        all_videos = sorted(os.listdir(self.data_root))

        # Train/val split: percentage-based with JAX PRNG
        self.videos = self._split_videos(all_videos)

        # Build frame pairs index
        self.frame_pairs: List[FramePair] = []
        self._build_index()

    def _split_videos(self, all_videos: List[str]) -> List[str]:
        """Split videos into train/val using percentage-based approach.

        Args:
            all_videos: Sorted list of all video directory names

        Returns:
            List of video names for this dataset's split
        """
        num_videos = len(all_videos)
        num_train = int(num_videos * self.train_ratio)  # Round down for train

        # Use JAX PRNG for reproducible shuffling
        key = jr.PRNGKey(self.seed)
        shuffle_key = jr.fold_in(key, 0)  # Deterministic shuffle

        # Create indices and shuffle
        indices = jnp.arange(num_videos)
        shuffled_indices = jr.permutation(shuffle_key, indices)

        # Split indices
        if self.split == "train":
            selected_indices = shuffled_indices[:num_train]
        else:
            selected_indices = shuffled_indices[num_train:]

        # Convert back to list and get video names
        selected_indices_list = sorted(selected_indices.tolist())
        return [all_videos[i] for i in selected_indices_list]

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

            # Generate pairs (t, t+k) for all valid t and k in [min_k, max_k]
            for t in range(num_frames):
                min_k = self.min_frame_distance
                max_k = min(self.max_frame_distance, num_frames - t - 1)
                if min_k <= max_k:
                    for k in range(min_k, max_k + 1):
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

        resample = Image.Resampling.LANCZOS

        # PIL resize expects (width, height), so swap from (height, width)
        img = img.resize((self.img_size[1], self.img_size[0]), resample)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0

        return img_array


def _shuffle_indices(
    indices: list[int],
    shuffle: bool,
    max_frames: int | None,
    random_seed: int | None,
) -> list[int]:
    """Shuffle and/or sample indices.

    Args:
        indices: List of indices to shuffle
        shuffle: Whether to shuffle
        max_frames: Maximum number of frames to sample (None = use all)
        random_seed: Random seed for reproducibility

    Returns:
        Shuffled and/or sampled list of indices
    """
    if max_frames is not None and max_frames > 0:
        # Sampling mode: take max_frames samples
        if shuffle:
            rng = random.Random(random_seed)
            return rng.sample(indices, min(max_frames, len(indices)))
        return indices[:max_frames]

    if not shuffle:
        return indices

    # Full shuffle
    rng = random.Random(random_seed)
    rng.shuffle(indices)
    return indices


def create_dataloader(
    dataset_settings,
    split: str,
    shuffle: bool = True,
    random_seed: int | None = None,
    augmentation_settings=None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, list[dict]]]:
    """Yield batches of frame pairs.

    Args:
        dataset_settings: DatasetSettings object with batch_size, img_size, max_samples
        split: 'train' or 'val'
        shuffle: Whether to shuffle the dataset (default True for train)
        random_seed: Random seed for shuffling (for reproducibility)
        augmentation_settings: AugmentationSettings object for on-the-fly augmentation.
                              If None, no augmentation is applied.

    Yields:
        Tuple of (img1_batch, img2_batch, metadata_batch)
    """
    from barevision.flow.dataset.augmentations import compose_augmentations

    image_size = image.image_size(
        dataset_settings.coarse_grid_size,
        dataset_settings.window_size,
        dataset_settings.num_levels,
    )
    dataset = VideoFrameDataset(
        split=split,
        min_frame_distance=dataset_settings.min_frame_distance,
        max_frame_distance=dataset_settings.max_frame_distance,
        img_size=image_size,
        seed=random_seed,
    )

    max_samples = (
        dataset_settings.max_samples if dataset_settings.max_samples > 0 else None
    )

    # Shuffle and/or sample indices
    indices = _shuffle_indices(
        list(range(len(dataset))), shuffle, max_samples, random_seed
    )

    # Create base RNG for augmentation seeding
    base_rng = random.Random(random_seed)

    # Yield batches
    for i in range(0, len(indices), dataset_settings.batch_size):
        batch_indices = indices[i : i + dataset_settings.batch_size]
        if len(batch_indices) < dataset_settings.batch_size:
            continue

        # Load and augment batch
        batch_imgs1 = []
        batch_imgs2 = []
        batch_metadata = []

        for idx in batch_indices:
            img1, img2, metadata = dataset[idx]

            # Apply augmentations on-the-fly with per-sample deterministic seeding
            if augmentation_settings is not None and split == "train":
                # Create deterministic RNG for this sample
                # Seed is derived from sample index to ensure same augmentation across epochs
                sample_rng = random.Random(base_rng.randint(0, 2**31 - 1))

                img1, img2 = compose_augmentations(
                    img1,
                    img2,
                    sample_rng,
                    horizontal_flip_prob=augmentation_settings.horizontal_flip_prob,
                    vertical_flip_prob=augmentation_settings.vertical_flip_prob,
                    rotation_prob=augmentation_settings.rotation_prob,
                    rotation_max_angle=augmentation_settings.rotation_max_angle,
                    color_augmentation_prob=augmentation_settings.color_augmentation_prob,
                    color_jitter_strength=augmentation_settings.color_jitter_strength,
                    swap_frames_prob=augmentation_settings.swap_frames_prob,
                )

            batch_imgs1.append(img1)
            batch_imgs2.append(img2)
            batch_metadata.append(metadata)

        yield jnp.stack(batch_imgs1), jnp.stack(batch_imgs2), batch_metadata
