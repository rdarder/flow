"""Smoke test for video frame dataset.

Verifies dataset structure and can be used for quick validation.

Run:
    python -m barevision.embeddings.check_dataset
"""

import os
from PIL import Image
import numpy as np


def check_dataset_structure(data_root: str = "datasets/frames"):
    """Verify dataset structure and report statistics."""
    print("=" * 60)
    print("DATASET STRUCTURE CHECK")
    print("=" * 60)
    print()

    if not os.path.exists(data_root):
        print(f"❌ Dataset root not found: {data_root}")
        return False

    # List all videos
    videos = sorted(os.listdir(data_root))
    print(f"Total videos found: {len(videos)}")
    print()

    # Statistics per video
    total_frames = 0
    video_stats = []

    for video in videos:
        video_path = os.path.join(data_root, video)
        if not os.path.isdir(video_path):
            continue

        frames = sorted([f for f in os.listdir(video_path) if f.endswith((".jpg", ".png"))])
        num_frames = len(frames)
        total_frames += num_frames

        # Check first frame
        if num_frames > 0:
            first_frame = os.path.join(video_path, frames[0])
            img = Image.open(first_frame)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            video_stats.append({
                'name': video,
                'frames': num_frames,
                'size': img.size,
                'mode': img.mode
            })

            print(f"✓ {video:15s}: {num_frames:4d} frames, {img.size[0]}x{img.size[1]}, {img.mode}")

    print()
    print(f"Total frames: {total_frames}")
    print()

    # Verify train/val split
    train_videos = videos[:13]
    val_videos = videos[13:]

    train_frames = sum(s['frames'] for s in video_stats if s['name'] in train_videos)
    val_frames = sum(s['frames'] for s in video_stats if s['name'] in val_videos)

    print("Train/Val Split:")
    print(f"  Training:   {len(train_videos):2d} videos, {train_frames:5d} frames")
    print(f"  Validation: {len(val_videos):2d} videos, {val_frames:5d} frames")
    print()

    return True


def test_image_loading(data_root: str = "datasets/frames", img_size=(190, 190)):
    """Test loading and preprocessing a few sample images."""
    print("=" * 60)
    print("IMAGE LOADING TEST")
    print("=" * 60)
    print()

    videos = sorted(os.listdir(data_root))

    # Test with first 3 videos
    test_videos = videos[:3]

    for video in test_videos:
        video_path = os.path.join(data_root, video)
        frames = sorted([f for f in os.listdir(video_path) if f.endswith((".jpg", ".png"))])

        if len(frames) < 2:
            print(f"⚠ {video}: Not enough frames for pairing test")
            continue

        # Load first frame
        frame1_path = os.path.join(video_path, frames[0])
        img1 = Image.open(frame1_path)
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')

        # Load second frame
        frame2_path = os.path.join(video_path, frames[1])
        img2 = Image.open(frame2_path)
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')

        # Resize using LANCZOS resampling
        try:
            from PIL.Image import Resampling
            resample = Resampling.LANCZOS
        except ImportError:
            resample = Image.LANCZOS

        img1_resized = img1.resize(img_size, resample)
        img2_resized = img2.resize(img_size, resample)

        arr1 = np.array(img1_resized).astype(np.float32) / 255.0
        arr2 = np.array(img2_resized).astype(np.float32) / 255.0

        print(f"✓ {video}:")
        print(f"    Frame 1: {frames[0]} → shape={arr1.shape}, range=[{arr1.min():.3f}, {arr1.max():.3f}]")
        print(f"    Frame 2: {frames[1]} → shape={arr2.shape}, range=[{arr2.min():.3f}, {arr2.max():.3f}]")

    print()
    print("✓ Image loading verified")
    print()


def test_frame_pairing(data_root: str = "datasets/frames", max_distance: int = 5):
    """Verify frame pairing logic generates expected number of pairs."""
    print("=" * 60)
    print("FRAME PAIRING TEST")
    print("=" * 60)
    print()

    videos = sorted(os.listdir(data_root))
    train_videos = videos[:13]

    total_pairs = 0

    for video in train_videos[:3]:  # Test with first 3 training videos
        video_path = os.path.join(data_root, video)
        frames = sorted([f for f in os.listdir(video_path) if f.endswith((".jpg", ".png"))])
        num_frames = len(frames)

        # Count pairs (same logic as dataset will use)
        pairs = []
        for t in range(num_frames):
            for k in range(1, min(max_distance + 1, num_frames - t)):
                pairs.append((t, t + k, k))

        total_pairs += len(pairs)
        print(f"✓ {video:15s}: {num_frames:4d} frames → {len(pairs):5d} pairs (distances 1-{max_distance})")

    print()
    print(f"Sample from first 3 videos: {total_pairs} pairs")
    print("✓ Frame pairing logic verified")
    print()


def main():
    """Run all dataset checks."""
    print()

    success = check_dataset_structure()
    if not success:
        print("❌ Dataset structure check failed")
        return 1

    test_image_loading()
    test_frame_pairing()

    print("=" * 60)
    print("ALL DATASET CHECKS PASSED")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    exit(main())
