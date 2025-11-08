import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import noise  # Make sure you have 'pip install noise'


class SyntheticFlowDataset(Dataset):
    """
    Generates a synthetic dataset of multiple, occluding,
    textured blobs on a Perlin noise background.
    """

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 num_blobs_range=(1, 2),
                 blob_size_range=(6, 10),
                 noise_scale_range=(4.0, 6.0),
                 blob_threshold=0.2,
                 max_flow=5,
                 # --- NEW PARAMETERS ---
                 bg_noise_scale=4.0,  # Scale for the background (larger=slower noise)
                 frame_noise_std=0.01):  # Std dev of Gaussian noise to add at the end

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_blobs_range = num_blobs_range
        self.blob_size_range = blob_size_range
        self.noise_scale_range = noise_scale_range
        self.blob_threshold = blob_threshold
        self.max_flow = max_flow

        # --- NEW ---
        self.bg_noise_scale = bg_noise_scale
        self.frame_noise_std = frame_noise_std

    def __len__(self):
        return 5000  # Virtual epoch size

    def _generate_blob_map(self, size, scale):
        """(Unchanged) Generates a single (size, size) noise map [0, 1]."""
        map = np.zeros((size, size))
        x_off, y_off = np.random.rand(2) * 1000
        for y in range(size):
            for x in range(size):
                map[y, x] = noise.pnoise2(
                    (x + x_off) / scale, (y + y_off) / scale,
                    octaves=2, persistence=0.5, lacunarity=2.0
                )
        # Normalize from approx [-0.7, 0.7] to [0, 1]
        return torch.from_numpy((map + 0.7) / 1.4).float()

    def __getitem__(self, idx):

        # --- 1. Generate Perlin Background (NEW) ---
        # We generate a dim, colored Perlin noise background
        # by creating three separate noise maps.
        bg_r = self._generate_blob_map(self.img_size, self.bg_noise_scale)
        bg_g = self._generate_blob_map(self.img_size, self.bg_noise_scale)
        bg_b = self._generate_blob_map(self.img_size, self.bg_noise_scale)

        # Scale noise to a dim [0.25, 0.75] range
        bg_noise = (torch.stack([bg_r, bg_g, bg_b], dim=0) * 0.5) + 0.25

        # Both images start with the *same* static background
        img1 = bg_noise.clone()
        img2 = bg_noise.clone()

        # The background's flow is zero (for now)
        flow_fullres = torch.zeros(2, self.img_size, self.img_size)

        # --- 2. Generate and "Paint" Blobs (Unchanged) ---
        # This logic is identical, and will "paint" blobs
        # *over* the noise background.
        num_blobs = np.random.randint(self.num_blobs_range[0], self.num_blobs_range[1] + 1)
        blobs = []
        for _ in range(num_blobs):
            blobs.append({
                'z_index': np.random.rand(),
                'color': torch.rand(3, 1, 1),
                'flow': (torch.rand(2, 1, 1) * 2 * self.max_flow) - self.max_flow,
                'size': np.random.randint(self.blob_size_range[0], self.blob_size_range[1] + 1),
                'scale': np.random.uniform(self.noise_scale_range[0], self.noise_scale_range[1]),
                'pos1': (
                    np.random.randint(0, self.img_size - self.blob_size_range[1]),
                    np.random.randint(0, self.img_size - self.blob_size_range[1])
                )
            })

        blobs.sort(key=lambda b: b['z_index'], reverse=True)

        for blob in blobs:
            size, y1, x1 = blob['size'], blob['pos1'][0], blob['pos1'][1]
            dx, dy = blob['flow'][0, 0, 0].item(), blob['flow'][1, 0, 0].item()
            x2, y2 = int(np.clip(x1 + dx, 0, self.img_size - size)), int(np.clip(y1 + dy, 0, self.img_size - size))
            actual_flow = torch.tensor([x2 - x1, y2 - y1]).float().view(2, 1, 1)
            noise_map = self._generate_blob_map(size, blob['scale'])
            shape_mask = (noise_map > self.blob_threshold).unsqueeze(0)
            final_color = (blob['color'] + (noise_map - 0.5) * 0.4).clamp(0, 1)

            blob_mask_1 = torch.zeros(1, self.img_size, self.img_size, dtype=torch.bool)
            blob_color_1 = torch.zeros(3, self.img_size, self.img_size)
            blob_flow_1 = torch.zeros(2, self.img_size, self.img_size)
            blob_mask_2 = torch.zeros(1, self.img_size, self.img_size, dtype=torch.bool)
            blob_color_2 = torch.zeros(3, self.img_size, self.img_size)

            blob_mask_1[:, y1:y1 + size, x1:x1 + size] = shape_mask
            blob_color_1[:, y1:y1 + size, x1:x1 + size] = final_color
            blob_flow_1[:, y1:y1 + size, x1:x1 + size] = actual_flow
            blob_mask_2[:, y2:y2 + size, x2:x2 + size] = shape_mask
            blob_color_2[:, y2:y2 + size, x2:x2 + size] = final_color

            # This "where" operation paints the blob over the background
            img1 = torch.where(blob_mask_1, blob_color_1, img1)
            flow_fullres = torch.where(blob_mask_1, blob_flow_1, flow_fullres)
            img2 = torch.where(blob_mask_2, blob_color_2, img2)

        # --- 3. Process Flow Target (Unchanged) ---
        flow_fullres_scaled = flow_fullres / self.patch_size
        flow_target = F.avg_pool2d(
            flow_fullres_scaled.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(0)

        # --- 4. Add Dynamic Frame Noise (NEW) ---
        # This breaks pixel-perfect matches, forcing
        # the model to learn robust *features*.
        noise1 = torch.randn_like(img1) * self.frame_noise_std
        noise2 = torch.randn_like(img2) * self.frame_noise_std

        img1_final = (img1 + noise1).clamp(0, 1)
        img2_final = (img2 + noise2).clamp(0, 1)

        # --- 5. Return JAX-compatible format (Unchanged) ---
        P = (self.img_size // self.patch_size) ** 2
        return (
            img1_final,  # (3, 32, 32)
            img2_final,  # (3, 32, 32)
            flow_target.reshape(2, P).T  # (P, 2)
        )

if __name__ == '__main__':
    """
    Test script to run the dataset directly and find errors
    hidden by the DataLoader's multiprocessing.
    """

    print("Starting dataset test run...")

    # --- 1. Instantiate the dataset ---
    # We can use all the default parameters
    try:
        dataset = SyntheticFlowDataset(
            img_size=32,
            patch_size=4,
            bg_noise_scale=16.0,
            frame_noise_std=0.05
        )
        print(f"Successfully instantiated SyntheticFlowDataset.")
        print(f"Virtual dataset size: {len(dataset)}")
    except Exception as e:
        print(f"--- FAILED during dataset __init__ ---")
        print(f"ERROR: {e}")
        # Re-raise to get the full traceback
        raise e

    # --- 2. Create a DataLoader with num_workers=0 ---
    # This is the KEY to debugging. It forces
    # __getitem__ to run in the main process.
    try:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # <-- The Debug Trick
        )
        print(f"Successfully instantiated DataLoader with num_workers=0.")
    except Exception as e:
        print(f"--- FAILED during DataLoader __init__ ---")
        print(f"ERROR: {e}")
        raise e

    # --- 3. Try to fetch a few batches ---
    print("\nAttempting to fetch 5 batches...")
    try:
        for i, (img1_batch, img2_batch, flow_batch) in enumerate(loader):
            if i >= 5:
                break

            # Print shapes to verify
            print(f"  Batch {i}:")
            print(f"    img1 shape: {img1_batch.shape}")
            print(f"    flow shape: {flow_batch.shape}")

            # Check for NaNs
            if torch.isnan(img1_batch).any() or torch.isnan(flow_batch).any():
                print(f"    WARNING: NaN detected in batch {i}!")

        print("\n--- TEST PASSED ---")
        print("Successfully fetched 5 batches. The dataset is working correctly.")

    except Exception as e:
        print(f"\n--- FAILED during __getitem__ (data loading) ---")
        print("This is the real error message:")
        print(f"ERROR: {e}")
        # Re-raise to get the full traceback
        raise e