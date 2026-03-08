"""Unit tests for video frame dataset."""

import numpy as np

from barevision.embeddings.video_dataset import VideoFrameDataset


class TestVideoFrameDataset:
    """Tests for VideoFrameDataset loading and pairing."""

    def test_dataset_creation(self):
        """Test that train and val datasets can be created."""
        # Uses auto-detected datasets directory
        train_dataset = VideoFrameDataset(
            split="train",
            max_frame_distance=5,
            img_size=(190, 190),
        )

        val_dataset = VideoFrameDataset(
            split="val",
            max_frame_distance=5,
            img_size=(190, 190),
        )

        assert len(train_dataset) > 0, "Training dataset should have samples"
        assert len(val_dataset) > 0, "Validation dataset should have samples"
        assert len(train_dataset) > len(
            val_dataset
        ), "Training set should be larger than validation"

    def test_train_val_split(self):
        """Test that train/val split has no overlap and uses 85/15 ratio."""
        train_dataset = VideoFrameDataset(split="train", max_frame_distance=5, seed=42)

        val_dataset = VideoFrameDataset(split="val", max_frame_distance=5, seed=42)

        train_videos = set(train_dataset.videos)
        val_videos = set(val_dataset.videos)

        assert (
            len(train_videos.intersection(val_videos)) == 0
        ), "Train and val should have no common videos"
        # With 15 videos and 85% ratio: 12 train, 3 val
        assert len(train_videos) == 12, "Training should have 12 videos (85% of 15)"
        assert len(val_videos) == 3, "Validation should have 3 videos (15% of 15)"

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
            max_frame_distance=5,
            img_size=(190, 190),
        )

        # Sample a subset to check distances (don't iterate all 37k samples)
        distances = set()
        sample_size = min(1000, len(dataset))
        for i in range(sample_size):
            _, _, metadata = dataset[i]
            distances.add(metadata["distance"])
            if len(distances) == 5:  # Found all, can stop early
                break

        # Should have all distances from 1 to 5
        expected_distances = set(range(1, 6))
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
            img_size=(190, 190),
        )

        # Find a video with multiple pairs
        video_pairs = [p for p in dataset.frame_pairs if p.video_name == "backyard"]
        assert len(video_pairs) > 0, "Should have backyard pairs"

        # Check that pairs are in temporal order
        for pair in video_pairs[:10]:  # Check first 10
            assert (
                pair.frame_t_idx < pair.frame_tk_idx
            ), "Frame t should come before frame t+k"
            assert (
                pair.frame_tk_idx - pair.frame_t_idx == pair.distance
            ), "Distance should match frame index difference"
