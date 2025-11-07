import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import center_crop


def readPFM(file):
    """ Read a PFM file. """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[..., :2], scale

class ChairsV0Dataset(Dataset):
    def __init__(self, dataset_root, split='train', img_size=32, patch_size=4):
        """
        Args:
            dataset_root (str): Path to the ChairsSDHom root directory.
            split (str): 'train' or 'test'.
            img_size (int): Final downscaled image size (e.g., 32).
            patch_size (int): Patch size of our V0 model (e.g., 4).
        """
        self.split_dir = os.path.join(dataset_root, split)
        self.t0_dir = os.path.join(self.split_dir, 't0')
        self.flow_dir = os.path.join(self.split_dir, 'flow')

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # e.g., 32 // 4 = 8

        # Get all sample names
        self.sample_names = [
            f.split('.')[0] for f in os.listdir(self.t0_dir) if f.endswith('.png')
        ]

        # Define image transforms
        # 1. Resize to target size (32x32)
        # 2. Convert to tensor (scales to [0, 1])
        # 3. Normalize (standard for computer vision)
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]

        # --- Load Files ---
        t0_path = os.path.join(self.t0_dir, f"{sample_name}.png")
        # We need t1 to be consistent, so we load from the same split
        t1_path = os.path.join(self.split_dir, 't1', f"{sample_name}.png")
        flow_path = os.path.join(self.flow_dir, f"{sample_name}.pfm")

        img_t0_orig = Image.open(t0_path).convert('RGB')
        img_t1_orig = Image.open(t1_path).convert('RGB')
        flow_orig, _ = readPFM(flow_path)  # Shape: (H_orig, W_orig, 2)

        # --- 1. Center Crop to Square (384x384) ---
        # The smallest dimension is 384
        crop_size = 384
        img_t0_cropped = center_crop(img_t0_orig, [crop_size, crop_size])
        img_t1_cropped = center_crop(img_t1_orig, [crop_size, crop_size])

        # We must also crop the flow (as numpy array)
        h, w, _ = flow_orig.shape
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        flow_cropped = flow_orig[top:top + crop_size, left:left + crop_size]

        # --- 2. Process Images ---
        # Apply Resize, ToTensor, and Normalize
        img_t0 = self.img_transform(img_t0_cropped)
        img_t1 = self.img_transform(img_t1_cropped)

        # --- 3. Process Flow (The V0 Target) ---

        # a. Scale flow vectors
        # Scale factor = (target_size / original_size)
        scale_factor = self.img_size / crop_size  # 32 / 384
        flow_scaled = flow_cropped * scale_factor

        # b. Average Pool to get (8, 8, 2) target
        # We need to use torch for easy avg_pool
        # (H, W, C) -> (C, H, W)
        flow_tensor = torch.from_numpy(flow_scaled).permute(2, 0, 1)

        # (C, H, W) -> (B, C, H, W) for pooling
        flow_tensor = flow_tensor.unsqueeze(0)

        # The "kernel size" for pooling is the original patch size
        # 384px / 8 patches = 48px per patch
        pool_kernel = crop_size // self.grid_size  # 384 // 8 = 48

        flow_target = F.avg_pool2d(
            flow_tensor,
            kernel_size=pool_kernel,
            stride=pool_kernel
        )

        # (B, C, H_grid, W_grid) -> (C, H_grid, W_grid)
        flow_target = flow_target.squeeze(0)  # Shape: [2, 8, 8]

        return img_t0, img_t1, flow_target


def test_dataset_loading(dataset_path):
    #
    # !!! IMPORTANT !!!
    # Set this path to the root of your ChairsSDHom dataset
    #
    BATCH_SIZE = 4

    try:
        # Create the Dataset
        train_dataset = ChairsV0Dataset(dataset_path, split='train')

        # Create a DataLoader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Fetch one batch
        img1_batch, img2_batch, flow_batch = next(iter(train_loader))

        print("--- V0 Data Pipeline Test ---")
        print(f"Successfully loaded 1 batch.")

        # Check shapes
        print(f"\nInput Image 1 Batch Shape: {img1_batch.shape}")
        print(f"Expected: (B, 3, 32, 32)")
        assert img1_batch.shape == (BATCH_SIZE, 3, 32, 32)

        print(f"\nInput Image 2 Batch Shape: {img2_batch.shape}")
        print(f"Expected: (B, 3, 32, 32)")
        assert img2_batch.shape == (BATCH_SIZE, 3, 32, 32)

        print(f"\nGround Truth Flow Batch Shape: {flow_batch.shape}")
        print(f"Expected: (B, 2, 8, 8)")
        assert flow_batch.shape == (BATCH_SIZE, 2, 8, 8)

        print("\n\nShape tests PASSED.")
        print("This 'on-the-fly' pipeline is ready for training.")

    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"Could not find dataset at: {DATASET_PATH}")
        print("Please update the DATASET_PATH variable in this script.")

def flow_to_color(flow, max_flow=None):
    """
    Converts a 2D optical flow field (dx, dy) into a color-coded RGB image.

    Args:
        flow (np.ndarray): Flow field of shape (H, W, 2).
        max_flow (float, optional): Maximum flow magnitude for normalization.
                                    If None, it's computed from the flow itself.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) in [0, 1] range.
    """
    H, W, C = flow.shape

    # Separate flow into (dx, dy)
    # The PFM reader returns (H, W, 3) for 'PF' files.
    # We assume flow is in the first two channels.
    dx = flow[..., 0]
    dy = flow[..., 1]

    # --- Convert cartesian (dx, dy) to polar (magnitude, angle) ---

    # Magnitude (speed)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    # Angle (direction)
    angle = np.arctan2(dy, dx)  # Range [-pi, pi]

    # --- Map polar coordinates to HSV color space ---

    # Hue: Map angle from [-pi, pi] to [0, 1]
    # (angle / (2*pi)) + 0.5
    h = (angle + np.pi) / (2 * np.pi)

    # Saturation: Always 1 (full color)
    s = np.ones_like(h)

    # Value (brightness): Map magnitude to [0, 1]
    if max_flow is None:
        # Normalize by the 99th percentile for better visualization
        # (max() can be skewed by a single outlier)
        max_mag = np.percentile(magnitude, 99)
    else:
        max_mag = max_flow

    v = np.clip(magnitude / (max_mag + 1e-6), 0, 1)

    # --- Convert HSV to RGB ---
    hsv = np.stack([h, s, v], axis=-1)
    rgb = plt.cm.hsv(hsv[..., 0])  # Use matplotlib's hsv colormap

    # Apply brightness (Value)
    rgb[..., :3] *= v[..., np.newaxis]

    return rgb[..., :3]


# --- 3. Main Visualization Function ---

def visualize_sample(dataset_root, sample_id, split='train'):
    """
    Loads and displays img_t0, img_t1, and the flow for a given sample.
    """

    # Format the sample name
    sample_name = f"{sample_id:05d}"  # e.g., 5 -> "00005"

    # Define file paths
    t0_path = os.path.join(dataset_root, split, 't0', f"{sample_name}.png")
    t1_path = os.path.join(dataset_root, split, 't1', f"{sample_name}.png")
    flow_path = os.path.join(dataset_root, split, 'flow', f"{sample_name}.pfm")

    # --- Load Data ---
    try:
        img_t0 = Image.open(t0_path)
        img_t1 = Image.open(t1_path)
        flow_data, _ = readPFM(flow_path)
    except FileNotFoundError as e:
        print(f"Error loading files for sample {sample_name}: {e}")
        print("Please check your DATASET_PATH.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Convert flow to a color image
    # Note: We pass flow_data directly, which is (H, W, 3)
    # The flow_to_color function is designed to handle this by only
    # using the first two channels (flow_data[..., :2])
    flow_img = flow_to_color(flow_data)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_t0)
    axes[0].set_title(f"Image t0 (Sample {sample_name})")
    axes[0].axis('off')

    axes[1].imshow(img_t1)
    axes[1].set_title(f"Image t1 (Sample {sample_name})")
    axes[1].axis('off')

    axes[2].imshow(flow_img)
    axes[2].set_title("Optical Flow (t0 -> t1)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    #
    # !!! IMPORTANT !!!
    # Set this path to the root of your ChairsSDHom dataset
    #
    DATASET_PATH = "../datasets/ChairsSDHom/data"

    # Visualize a sample
    visualize_sample(DATASET_PATH, sample_id=1023, split='train')
    # test_dataset_loading(DATASET_PATH)

    # You can try other samples
    # visualize_sample(DATASET_PATH, sample_id=100, split='train')
    # visualize_sample(DATASET_PATH, sample_id=1, split='test')