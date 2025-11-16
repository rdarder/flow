import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import noise


class SyntheticFlowDataset(Dataset):
    """
    Generates a synthetic dataset of multiple, occluding,
    textured blobs on a Perlin noise background.
    """

    def __init__(self,
                 img_size=18,
                 num_blobs_range=(1, 2),
                 blob_size_range=(3, 10),
                 noise_scale_range=(4.0, 6.0),
                 blob_threshold=0.2,
                 max_flow=8,
                 bg_noise_scale=4.0,
                 frame_noise_std=0.01,
                 length: int = 5000):

        self.img_size = img_size
        self.num_blobs_range = num_blobs_range
        self.blob_size_range = blob_size_range
        self.noise_scale_range = noise_scale_range
        self.blob_threshold = blob_threshold
        self.max_flow = max_flow
        self.length = length
        self.bg_noise_scale = bg_noise_scale
        self.frame_noise_std = frame_noise_std

    def __len__(self):
        return self.length

    def _generate_blob_map(self, size, scale):
        """Generates a single (size, size) noise map [0, 1]."""
        map = np.zeros((size, size))
        x_off, y_off = np.random.rand(2) * 1000
        for y in range(size):
            for x in range(size):
                map[y, x] = noise.pnoise2(
                    (x + x_off) / scale, (y + y_off) / scale,
                    octaves=2, persistence=0.5, lacunarity=2.0
                )
        return torch.from_numpy((map + 0.7) / 1.4).float()

    def __getitem__(self, idx):

        # --- 1. Generate Background ---
        bg_r = self._generate_blob_map(self.img_size, self.bg_noise_scale)
        bg_g = self._generate_blob_map(self.img_size, self.bg_noise_scale)
        bg_b = self._generate_blob_map(self.img_size, self.bg_noise_scale)
        bg_noise = (torch.stack([bg_r, bg_g, bg_b], dim=0) * 0.5) + 0.25

        img1 = bg_noise.clone()
        img2 = bg_noise.clone()
        flow_fullres = torch.zeros(2, self.img_size, self.img_size)

        # --- 2. Paint Blobs ---
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

            img1 = torch.where(blob_mask_1, blob_color_1, img1)
            flow_fullres = torch.where(blob_mask_1, blob_flow_1, flow_fullres)
            img2 = torch.where(blob_mask_2, blob_color_2, img2)

        # --- 3. Add Noise ---
        noise1 = torch.randn_like(img1) * self.frame_noise_std
        noise2 = torch.randn_like(img2) * self.frame_noise_std
        img1_final = (img1 + noise1).clamp(0, 1)
        img2_final = (img2 + noise2).clamp(0, 1)

        # --- 4. Return ---
        return (
            img1_final.permute(1, 2, 0),  # (18, 18, 3)
            img2_final.permute(1, 2, 0),  # (18, 18, 3)
            flow_fullres.permute(1, 2, 0)  # (18, 18, 2) DENSE PIXEL FLOW
        )


if __name__ == '__main__':
    """Test script to run the dataset directly."""
    print("Starting dataset test run...")

    try:
        dataset = SyntheticFlowDataset(
            img_size=18,  # <-- MODIFIED
            bg_noise_scale=16.0,
            frame_noise_std=0.05
        )
        print(f"Successfully instantiated SyntheticFlowDataset (img_size=18).")
    except Exception as e:
        print(f"--- FAILED during dataset __init__ ---")
        raise e

    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        print(f"Successfully instantiated DataLoader with num_workers=0.")
    except Exception as e:
        print(f"--- FAILED during DataLoader __init__ ---")
        raise e

    print("\nAttempting to fetch 5 batches...")
    try:
        for i, (img1_batch, img2_batch, flow_batch) in enumerate(loader):
            if i >= 5:
                break
            print(f"  Batch {i}:")
            print(f"    img1 shape: {img1_batch.shape}")
            print(f"    flow shape: {flow_batch.shape}")
            if torch.isnan(img1_batch).any() or torch.isnan(flow_batch).any():
                print(f"    WARNING: NaN detected in batch {i}!")
        print("\n--- TEST PASSED ---")
    except Exception as e:
        print(f"\n--- FAILED during __getitem__ (data loading) ---")
        raise e
