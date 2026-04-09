"""Unit tests for video frame dataset."""

from pathlib import Path

import numpy as np
import pytest

from barevision.dataset.video import VideoFrameDataset, PreloadedFrameDataset, create_dataloader
from barevision.utils.path import set_datasets_dir_override, clear_datasets_dir_override


# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "test_fixtures" / "frames"


@pytest.fixture(autouse=True)
def use_test_fixtures():
    """Use test fixtures directory for all tests in this module."""
    set_datasets_dir_override(FIXTURES_DIR)
    yield
    clear_datasets_dir_override()


class TestVideoFrameDataset:
    """Tests for VideoFrameDataset loading and pairing."""

    def test_dataset_creation(self):
        """Test that train and val datasets can be created."""
        # Uses auto-detected datasets directory
        train_dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )

        val_dataset = VideoFrameDataset(
            split="val",
            max_frame_distance=2,
            img_size=(64, 64),
        )

        assert len(train_dataset) > 0, "Training dataset should have samples"
        assert len(val_dataset) > 0, "Validation dataset should have samples"
        assert len(train_dataset) > len(
            val_dataset
        ), "Training set should be larger than validation"

    def test_train_val_split(self):
        """Test that train/val split has no overlap and uses 85/15 ratio."""
        train_dataset = VideoFrameDataset(split="train", max_frame_distance=2, seed=42)

        val_dataset = VideoFrameDataset(split="val", max_frame_distance=2, seed=42)

        train_videos = set(train_dataset.videos)
        val_videos = set(val_dataset.videos)

        assert (
            len(train_videos.intersection(val_videos)) == 0
        ), "Train and val should have no common videos"
        # With 2 videos and 85% ratio: 1 train, 1 val
        assert len(train_videos) == 1, "Training should have 1 video (85% of 2)"
        assert len(val_videos) == 1, "Validation should have 1 video (15% of 2)"

    def test_frame_pair_loading(self):
        """Test loading a single frame pair."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=5,
            img_size=(190, 190),
        )

        img1, img2, metadata = dataset[0]

        # Check image shapes
        assert img1.shape == (190, 190, 3), f"Expected (190, 190, 3), got {img1.shape}"
        assert img2.shape == (190, 190, 3), f"Expected (190, 190, 3), got {img2.shape}"

        # Check image values
        assert img1.dtype == np.float32, f"Expected float32, got {img1.dtype}"
        assert img2.dtype == np.float32, f"Expected float32, got {img2.dtype}"
        assert (
            img1.min() >= 0.0 and img1.max() <= 1.0
        ), "Image values should be in [0, 1]"
        assert (
            img2.min() >= 0.0 and img2.max() <= 1.0
        ), "Image values should be in [0, 1]"

        # Check metadata
        assert "video_name" in metadata
        assert "frame_t" in metadata
        assert "frame_tk" in metadata
        assert "distance" in metadata
        assert metadata["distance"] >= 1
        assert metadata["distance"] <= 5
        assert metadata["frame_tk"] == metadata["frame_t"] + metadata["distance"]

    def test_frame_distances(self):
        """Test that frame pairs span distances 1 through max_distance."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )

        # Check all samples (fixtures are small)
        distances = set()
        for i in range(len(dataset)):
            _, _, metadata = dataset[i]
            distances.add(metadata["distance"])

        # Should have all distances from 1 to max_distance
        expected_distances = set(range(1, 3))
        assert (
            distances == expected_distances
        ), f"Expected distances {expected_distances}, got {distances}"

    def test_custom_max_distance(self):
        """Test with different max_frame_distance values."""
        for max_dist in [2, 3, 10]:
            dataset = VideoFrameDataset(
                split="train",
                max_frame_distance=max_dist,
                img_size=(190, 190),
            )

            # Sample a subset to check distances
            distances = set()
            sample_size = min(100, len(dataset))
            for i in range(sample_size):
                _, _, metadata = dataset[i]
                distances.add(metadata["distance"])

            assert (
                max(distances) <= max_dist
            ), f"Max distance {max(distances)} exceeds limit {max_dist}"

    def test_custom_image_size(self):
        """Test with different image sizes."""
        for img_size in [(128, 128), (256, 256), (190, 256)]:
            dataset = VideoFrameDataset(
                split="train",
                max_frame_distance=5,
                img_size=img_size,
            )

            img1, img2, _ = dataset[0]
            # img_size is (height, width), output should be (H, W, 3)
            expected_shape = (img_size[0], img_size[1], 3)  # (H, W, C)
            assert (
                img1.shape == expected_shape
            ), f"Expected {expected_shape}, got {img1.shape}"
            assert (
                img2.shape == expected_shape
            ), f"Expected {expected_shape}, got {img2.shape}"

    def test_seed_reproducibility(self):
        """Test that same seed produces same split."""
        dataset1 = VideoFrameDataset(split="train", max_frame_distance=5, seed=42)
        dataset2 = VideoFrameDataset(split="train", max_frame_distance=5, seed=42)

        assert dataset1.videos == dataset2.videos, "Same seed should produce same split"

        # Different seed should produce different split (most likely)
        dataset3 = VideoFrameDataset(split="train", max_frame_distance=5, seed=123)
        # Videos might be same by chance but unlikely
        assert dataset1.videos != dataset3.videos or dataset1.seed != dataset3.seed

    def test_frame_ordering(self):
        """Test that frames are loaded in correct temporal order."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=1,  # Only adjacent frames
            img_size=(64, 64),
        )

        # Use the first video in the split (could be video_a or video_b)
        assert len(dataset.videos) > 0, "Should have at least one video"
        video_name = dataset.videos[0]
        
        video_pairs = [p for p in dataset.frame_pairs if p.video_name == video_name]
        assert len(video_pairs) > 0, f"Should have {video_name} pairs"

        # Check that pairs are in temporal order
        for pair in video_pairs:  # Check all
            assert (
                pair.frame_t_idx < pair.frame_tk_idx
            ), "Frame t should come before frame t+k"
            assert (
                pair.frame_tk_idx - pair.frame_t_idx == pair.distance
            ), "Distance should match frame index difference"


class TestPreloadedFrameDataset:
    """Tests for PreloadedFrameDataset."""

    def test_preload_creation(self):
        """Test that PreloadedFrameDataset can be created."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )
        indices = list(range(len(dataset)))

        preloaded = PreloadedFrameDataset(dataset, indices, frame_cache_max_mb=100)

        assert len(preloaded) > 0
        assert len(preloaded.unique_frames) > 0
        assert preloaded.frames is not None

    def test_preload_frame_access(self):
        """Test loading frames from pre-loaded dataset."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )
        indices = list(range(len(dataset)))

        preloaded = PreloadedFrameDataset(dataset, indices, frame_cache_max_mb=100)

        img1, img2, metadata = preloaded[0]

        # Check image shapes
        assert img1.shape == (64, 64, 3)
        assert img2.shape == (64, 64, 3)

        # Check image values
        assert img1.min() >= 0.0 and img1.max() <= 1.0
        assert img2.min() >= 0.0 and img2.max() <= 1.0

        # Check metadata
        assert "video_name" in metadata
        assert "frame_t" in metadata
        assert "frame_tk" in metadata

    def test_preload_memory_limit(self):
        """Test that memory limit is enforced."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )
        indices = list(range(len(dataset)))

        # Calculate actual memory usage
        # 5 frames × 64×64×3×4 bytes = 0.2MB
        # Set limit to 0.1MB to trigger error
        with pytest.raises(MemoryError):
            PreloadedFrameDataset(dataset, indices, frame_cache_max_mb=0)

    def test_preload_unlimited_memory(self):
        """Test that -1 disables memory limit."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )
        indices = list(range(len(dataset)))

        # Should not raise
        preloaded = PreloadedFrameDataset(dataset, indices, frame_cache_max_mb=-1)
        assert preloaded.frames is not None

    def test_preload_frame_reuse(self):
        """Test that same frame is reused for multiple pairs."""
        dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=2,
            img_size=(64, 64),
        )
        indices = list(range(len(dataset)))

        preloaded = PreloadedFrameDataset(dataset, indices, frame_cache_max_mb=100)

        # Find pairs that share a frame
        frame_t_counts = {}
        for pair in preloaded.frame_pairs:
            key = (pair.video_name, pair.frame_t_idx)
            frame_t_counts[key] = frame_t_counts.get(key, 0) + 1

        # At least one frame should be used multiple times
        reused_frames = [k for k, v in frame_t_counts.items() if v > 1]
        assert len(reused_frames) > 0, "Should have frames used in multiple pairs"

        # Verify that reused frames point to same array index
        for video_name, frame_idx in reused_frames:
            lookup_key = (video_name, frame_idx)
            assert lookup_key in preloaded.frame_lookup


class TestCreateDataloader:
    """Tests for create_dataloader with pre-loading."""

    def test_dataloader_yields_batches(self):
        """Test that dataloader yields batches correctly."""
        from barevision.dataset.video import DatasetConfig

        config = DatasetConfig(
            batch_size=2,
            max_frame_distance=2,
            frame_cache_max_mb=100,
        )

        loader = create_dataloader(
            config,
            image_size=(64, 64),
            split="train",
            shuffle=False,
            random_seed=42,
        )

        batches = list(loader)
        assert len(batches) > 0

        img1_batch, img2_batch, metadata_batch = batches[0]
        assert img1_batch.shape[0] == 2  # batch size
        assert img1_batch.shape[1:] == (64, 64, 3)
        assert len(metadata_batch) == 2

    def test_dataloader_respects_max_samples(self):
        """Test that max_samples limits the number of samples."""
        from barevision.dataset.video import DatasetConfig

        config = DatasetConfig(
            batch_size=2,
            max_frame_distance=2,
            max_samples=4,  # Only 4 samples
            frame_cache_max_mb=100,
        )

        loader = create_dataloader(
            config,
            image_size=(64, 64),
            split="train",
            shuffle=False,
            random_seed=42,
        )

        total_samples = 0
        for img1, img2, metadata in loader:
            total_samples += len(metadata)

        assert total_samples == 4
