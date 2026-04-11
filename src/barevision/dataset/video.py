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
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, NamedTuple, Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from barevision.utils.console import ConsoleLogger
from barevision.utils.path import get_datasets_dir


class DatasetConfig(BaseModel):
    """Dataset configuration for video frame loading.

    Attributes:
        batch_size: Training batch size
        coarse_grid_size: Target coarse-level grid dimension (default 1 for 1×1 grid)
        window_size: Window size at coarse level (default 16)
        num_levels: Number of pyramid levels (used to calculate required input size)
        min_frame_distance: Minimum temporal distance for frame pairs (default 1)
        max_frame_distance: Maximum temporal distance for frame pairs
        max_samples: Maximum samples per epoch (-1 for full dataset)
        frame_cache_max_mb: Maximum memory for frame cache in MB (-1 for unlimited)
    """

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(default=8, ge=1, description="Training batch size")
    coarse_grid_size: int = Field(
        default=1, ge=1, description="Target coarse-level grid dimension"
    )
    window_size: int = Field(
        default=16, ge=1, description="Window size at coarse level"
    )
    num_levels: int = Field(default=3, ge=1, description="Number of pyramid levels")
    min_frame_distance: int = Field(
        default=1, ge=1, description="Minimum temporal distance for frame pairs"
    )
    max_frame_distance: int = Field(
        default=3, ge=1, description="Maximum temporal distance for frame pairs"
    )
    max_samples: int = Field(
        default=-1, description="Maximum samples per epoch (-1 for full dataset)"
    )
    frame_cache_max_mb: int = Field(
        default=500,
        description="Maximum memory for frame cache in MB (-1 for unlimited)",
    )


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
        logger: ConsoleLogger,
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
        self.logger = logger
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

    # def __(self, idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    #     """Load a frame pair.
    #
    #     Args:
    #         idx: Index into the dataset
    #
    #     Returns:
    #         Tuple of (img1, img2, metadata) where:
    #             - img1: (H, W, 3) float32 array in [0, 1]
    #             - img2: (H, W, 3) float32 array in [0, 1]
    #             - metadata: dict with video_name, frame indices, distance
    #     """
    #     pair = self.frame_pairs[idx]
    #
    #     # Load and preprocess images
    #     img1 = self.load_image(pair.img1_path)
    #     img2 = self.load_image(pair.img2_path)
    #
    #     metadata = {
    #         "video_name": pair.video_name,
    #         "frame_t": pair.frame_t_idx,
    #         "frame_tk": pair.frame_tk_idx,
    #         "distance": pair.distance,
    #     }
    #
    #     return img1, img2, metadata

    def load_image(self, path: str) -> np.ndarray:
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


class PreloadedFrameDataset:

    def __init__(
        self,
        logger: ConsoleLogger,
        video_dataset: VideoFrameDataset,
        indices: list[int],
        frame_cache_max_mb: int = 500,
    ):
        self.logger = logger
        self.video_dataset = video_dataset
        self.indices = indices
        self.frame_pairs = [video_dataset.frame_pairs[i] for i in indices]

        unique_frames_set = set()
        for pair in self.frame_pairs:
            unique_frames_set.add((pair.video_name, pair.frame_t_idx))
            unique_frames_set.add((pair.video_name, pair.frame_tk_idx))

        self.unique_frames = list(unique_frames_set)
        self.frame_lookup = {
            (v, idx): i for i, (v, idx) in enumerate(self.unique_frames)
        }

        # Memory Check
        H, W = video_dataset.img_size
        total_mb = (len(self.unique_frames) * H * W * 3 * 4) / (1024 * 1024)
        if 0 <= frame_cache_max_mb < total_mb:
            raise MemoryError(
                f"Required {total_mb:.1f}MB > limit {frame_cache_max_mb}MB"
            )

        msg = f"Pre-loading {len(self.unique_frames)} frames ({total_mb:.1f}MB)..."
        with self.logger.task(msg):
            self.frames = self._preload_frames()

    def _preload_frames(self) -> jax.Array:
        # Cache video file lists to avoid O(N^2) directory scans
        video_files_cache = {}
        for video_name in set(v for v, _ in self.unique_frames):
            v_path = os.path.join(self.video_dataset.data_root, video_name)
            video_files_cache[video_name] = sorted(
                [f for f in os.listdir(v_path) if f.endswith((".jpg", ".png"))]
            )

        # Build flat list of paths for parallel mapper
        task_paths = [
            os.path.join(self.video_dataset.data_root, v, video_files_cache[v][idx])
            for v, idx in self.unique_frames
        ]

        # Parallelize decoding and resizing
        with ThreadPoolExecutor() as executor:
            frames_list = list(executor.map(self.video_dataset.load_image, task_paths))

        frames_np = np.stack(frames_list)
        return jax.device_put(frames_np, jax.devices("cpu")[0])

    def __len__(self) -> int:
        return len(self.frame_pairs)

    def __getitem__(self, idx: int):
        pair = self.frame_pairs[idx]
        img1 = self.frames[self.frame_lookup[(pair.video_name, pair.frame_t_idx)]]
        img2 = self.frames[self.frame_lookup[(pair.video_name, pair.frame_tk_idx)]]

        metadata = {
            "video_name": pair.video_name,
            "frame_t": pair.frame_t_idx,
            "frame_tk": pair.frame_tk_idx,
            "distance": pair.distance,
        }
        return img1, img2, metadata


class PreparedDataset:
    """Dataset with pre-loaded frames, ready for multi-epoch training.

    This class holds pre-loaded frames in memory for the lifetime of training.
    It provides epoch-specific iterators that shuffle indices without re-loading data.

    Usage:
        prepared = PreparedDataset(logger, config, split="train", image_size=(190, 190))
        for epoch in range(epochs):
            loader = prepared.get_epoch_iterator(epoch, shuffle=True, batch_size=8)
            for batch in loader:
                # train...
    """

    def __init__(
        self,
        logger: ConsoleLogger,
        dataset_config: DatasetConfig,
        split: str,
        image_size: Tuple[int, int],
        base_seed: int = 42,
    ):
        """Initialize prepared dataset with pre-loaded frames.

        Args:
            logger: Console logger
            dataset_config: Dataset configuration
            split: 'train' or 'val'
            image_size: Target image size (height, width)
            base_seed: Base random seed for shuffling
        """
        self.logger = logger
        self.dataset_config = dataset_config
        self.split = split
        self.base_seed = base_seed
        self.batch_size = dataset_config.batch_size

        # Build video dataset (frame pairs index)
        self.video_dataset = VideoFrameDataset(
            logger=logger,
            split=split,
            min_frame_distance=dataset_config.min_frame_distance,
            max_frame_distance=dataset_config.max_frame_distance,
            img_size=image_size,
            seed=base_seed,
        )

        # Determine which indices to use (sampling or full dataset)
        max_samples = (
            dataset_config.max_samples if dataset_config.max_samples > 0 else None
        )
        indices = list(range(len(self.video_dataset)))
        if max_samples is not None:
            indices = indices[:max_samples]

        # Pre-load frames into memory (one-time cost)
        self.preloaded = PreloadedFrameDataset(
            logger,
            self.video_dataset,
            indices,
            frame_cache_max_mb=dataset_config.frame_cache_max_mb,
        )

    def get_epoch_iterator(
        self,
        epoch: int,
        shuffle: bool = True,
        batch_size: Optional[int] = None,
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, list[dict]]]:
        """Return a fresh iterator for the given epoch.

        Args:
            epoch: Epoch number (used for shuffle seed)
            shuffle: Whether to shuffle indices
            batch_size: Batch size (defaults to config batch_size)

        Returns:
            Iterator yielding (img1_batch, img2_batch, metadata_list)
        """
        if batch_size is None:
            batch_size = self.batch_size

        epoch_seed = self.base_seed + epoch if shuffle else self.base_seed
        indices = _shuffle_indices(
            list(range(len(self.preloaded))),
            shuffle=shuffle,
            max_frames=None,
            random_seed=epoch_seed,
        )
        return _BatchIterator(self.preloaded, indices, batch_size)


class _BatchIterator:
    """Iterator that yields batches from a pre-loaded dataset with given indices."""

    def __init__(
        self,
        preloaded_dataset: PreloadedFrameDataset,
        indices: list[int],
        batch_size: int,
    ):
        self.preloaded = preloaded_dataset
        self.indices = indices
        self.batch_size = batch_size
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[jnp.ndarray, jnp.ndarray, list[dict]]:
        if self.pos >= len(self.indices):
            raise StopIteration

        batch_imgs1 = []
        batch_imgs2 = []
        batch_metadata = []

        while len(batch_imgs1) < self.batch_size and self.pos < len(self.indices):
            idx = self.indices[self.pos]
            img1, img2, metadata = self.preloaded[idx]
            batch_imgs1.append(img1)
            batch_imgs2.append(img2)
            batch_metadata.append(metadata)
            self.pos += 1

        if len(batch_imgs1) < self.batch_size:
            raise StopIteration

        return jnp.stack(batch_imgs1), jnp.stack(batch_imgs2), batch_metadata


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
    logger: ConsoleLogger,
    dataset_settings,
    image_size: Tuple[int, int],
    split: str,
    shuffle: bool = True,
    random_seed: int | None = None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, list[dict]]]:
    """Yield batches of frame pairs.

    Pre-loads all unique frames into memory for fast batch generation.
    JPEG decoding happens once upfront, then batches are sliced from memory.

    Note: This function creates a new PreparedDataset on each call.
    For multi-epoch training, use PreparedDataset directly and call
    get_epoch_iterator() for each epoch to avoid re-loading frames.

    Args:
        logger:
        dataset_settings: DatasetSettings object with batch_size, img_size, max_samples
        image_size: Target image size (height, width) - calculated by caller
        split: 'train' or 'val'
        shuffle: Whether to shuffle the dataset (default True for train)
        random_seed: Random seed for shuffling (for reproducibility)

    Yields:
        Tuple of (img1_batch, img2_batch, metadata_batch)
    """
    # Convert dataset_settings to DatasetConfig if needed
    if isinstance(dataset_settings, DatasetConfig):
        config = dataset_settings
    else:
        # Backwards compatibility: assume it has the required attributes
        config = DatasetConfig(
            batch_size=dataset_settings.batch_size,
            coarse_grid_size=getattr(dataset_settings, "coarse_grid_size", 1),
            window_size=getattr(dataset_settings, "window_size", 16),
            num_levels=getattr(dataset_settings, "num_levels", 3),
            min_frame_distance=dataset_settings.min_frame_distance,
            max_frame_distance=dataset_settings.max_frame_distance,
            max_samples=dataset_settings.max_samples,
            frame_cache_max_mb=dataset_settings.frame_cache_max_mb,
        )

    prepared = PreparedDataset(
        logger=logger,
        dataset_config=config,
        split=split,
        image_size=image_size,
        base_seed=random_seed or 42,
    )

    # For backwards compatibility, return a single epoch iterator
    return prepared.get_epoch_iterator(epoch=0, shuffle=shuffle)
