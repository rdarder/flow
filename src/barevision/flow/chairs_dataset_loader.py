"""ChairsSDHom dataset loader for optical flow training.

Loads PNG images and PFM flow files directly from the ChairsSDHom dataset structure.
"""

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .pfm_utils import read_pfm


class ChairsSDHomDataset(Dataset):
    """ChairsSDHom optical flow dataset.

    Loads image pairs and flow fields directly from the dataset directory.
    The dataset structure should be:
        data_root/
            train/
                t0/          # First frame images (.png)
                t1/          # Second frame images (.png)
                flow/        # Flow fields (.pfm)
            test/
                t0/
                t1/
                flow/

    Args:
        root: Path to dataset root directory (e.g., "datasets/ChairsSDHom/data")
        split: Dataset split ("train" or "test")
        target_size: Target image size as (height, width) tuple
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_size: Tuple[int, int] = (384, 512),
    ):
        self.root = root
        self.split = split
        self.target_size = target_size
        self.target_h, self.target_w = target_size

        # Validate split
        if split not in ["train", "test"]:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        # Set up directories
        split_dir = os.path.join(root, split)
        self.t0_dir = os.path.join(split_dir, "t0")
        self.t1_dir = os.path.join(split_dir, "t1")
        self.flow_dir = os.path.join(split_dir, "flow")

        # Check directories exist
        for dir_path, dir_name in [
            (self.t0_dir, "t0"),
            (self.t1_dir, "t1"),
            (self.flow_dir, "flow"),
        ]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}. "
                    f"Please ensure the ChairsSDHom dataset is properly extracted."
                )

        # Get list of sample names (from t0 directory)
        t0_files = sorted([f for f in os.listdir(self.t0_dir) if f.endswith(".png")])
        self.sample_names = [f.replace(".png", "") for f in t0_files]

        print(f"ChairsSDHom {split}: Found {len(self.sample_names)} samples")

        # Image transforms: resize and convert to tensor [0, 1]
        self.img_transform = T.Compose(
            [
                T.Resize((self.target_h, self.target_w), antialias=True),
                T.ToTensor(),  # Converts to [0, 1] and reorders to (C, H, W)
            ]
        )

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (img1, img2, flow) where:
                - img1: First frame (H, W, 3) in [0, 1]
                - img2: Second frame (H, W, 3) in [0, 1]
                - flow: Optical flow (H, W, 2) in pixel coordinates
        """
        sample_name = self.sample_names[idx]

        # Load images
        t0_path = os.path.join(self.t0_dir, f"{sample_name}.png")
        t1_path = os.path.join(self.t1_dir, f"{sample_name}.png")
        flow_path = os.path.join(self.flow_dir, f"{sample_name}.pfm")

        # Open and transform images
        img1_pil = Image.open(t0_path).convert("RGB")
        img2_pil = Image.open(t1_path).convert("RGB")

        # Get original size for flow scaling
        orig_w, orig_h = img1_pil.size

        # Apply transforms (resizes and converts to tensor)
        img1_tensor = self.img_transform(img1_pil)  # (3, H, W)
        img2_tensor = self.img_transform(img2_pil)  # (3, H, W)

        # Load and process flow
        flow_data, _ = read_pfm(flow_path)  # (H, W, 2) in original resolution
        flow_data = flow_data.copy()  # Fix negative stride issue from np.flipud

        # Scale flow to match resized images
        scale_h = self.target_h / orig_h
        scale_w = self.target_w / orig_w
        flow_data[:, :, 0] *= scale_w  # x component
        flow_data[:, :, 1] *= scale_h  # y component

        # Convert flow to tensor and resize
        flow_tensor = torch.from_numpy(flow_data).permute(2, 0, 1)  # (2, H, W)

        # Use bilinear interpolation for flow resizing
        flow_resized = TF.resize(
            flow_tensor,
            (self.target_h, self.target_w),
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=False,
        )

        # Convert to HWC format (matching synthetic dataset)
        img1 = img1_tensor.permute(1, 2, 0)  # (H, W, 3)
        img2 = img2_tensor.permute(1, 2, 0)  # (H, W, 3)
        flow = flow_resized.permute(1, 2, 0)  # (H, W, 2)

        return img1, img2, flow


if __name__ == "__main__":
    # Test script
    print("Testing ChairsSDHom dataset loader...")

    try:
        dataset = ChairsSDHomDataset(
            root="datasets/ChairsSDHom/data",
            split="train",
            target_size=(384, 512),
        )

        print(f"\nDataset loaded: {len(dataset)} samples")

        # Test a few samples
        for i in range(min(3, len(dataset))):
            img1, img2, flow = dataset[i]
            print(f"\nSample {i}:")
            print(
                f"  img1 shape: {img1.shape}, range: [{img1.min():.3f}, {img1.max():.3f}]"
            )
            print(
                f"  img2 shape: {img2.shape}, range: [{img2.min():.3f}, {img2.max():.3f}]"
            )
            print(
                f"  flow shape: {flow.shape}, range: [{flow.min():.3f}, {flow.max():.3f}]"
            )

        print("\n--- TEST PASSED ---")

    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
