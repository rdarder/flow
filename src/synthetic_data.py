import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import noise  # <-- You will need to pip install noise


class SyntheticFlowDataset(Dataset):
    """
    Generates a synthetic dataset of multiple, occluding,
    textured blobs, each with its own flow vector.
    """

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 num_blobs_range=(1, 4),
                 blob_size_range=(8, 14),
                 noise_scale_range=(4.0, 8.0),
                 blob_threshold=0.3,
                 max_flow=5):

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_blobs_range = num_blobs_range
        self.blob_size_range = blob_size_range
        self.noise_scale_range = noise_scale_range
        self.blob_threshold = blob_threshold
        self.max_flow = max_flow

    def __len__(self):
        return 10000  # Virtual epoch size

    def _generate_blob_map(self, size, scale):
        """Generates a (size, size) noise map."""
        map = np.zeros((size, size))
        # Use a random offset for variety
        x_off, y_off = np.random.rand(2) * 1000

        for y in range(size):
            for x in range(size):
                map[y, x] = noise.pnoise2(
                    (x + x_off) / scale,
                    (y + y_off) / scale,
                    octaves=2,
                    persistence=0.5,
                    lacunarity=2.0
                )
        # Normalize from approx [-0.7, 0.7] to [0, 1]
        map = (map + 0.7) / 1.4
        return torch.from_numpy(map).float()

    def __getitem__(self, idx):
        # 1. Create blank "canvases"
        img1 = torch.zeros(3, self.img_size, self.img_size)
        img2 = torch.zeros(3, self.img_size, self.img_size)
        flow_fullres = torch.zeros(2, self.img_size, self.img_size)

        # 2. Generate blob "specs"
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

        # 3. Sort blobs by z_index (back-to-front)
        blobs.sort(key=lambda b: b['z_index'], reverse=True)

        # 4. "Render" blobs in order
        for blob in blobs:
            size = blob['size']
            y1, x1 = blob['pos1']

            # Calculate end position (and clamp to screen)
            # Note: flow is (dx, dy) but tensor coords are (y, x)
            dx, dy = blob['flow'][0, 0, 0].item(), blob['flow'][1, 0, 0].item()
            x2 = int(np.clip(x1 + dx, 0, self.img_size - size))
            y2 = int(np.clip(y1 + dy, 0, self.img_size - size))

            # Recalculate actual flow after clamping
            actual_flow = torch.tensor([x2 - x1, y2 - y1]).float().view(2, 1, 1)

            # Generate the blob's texture and shape
            noise_map = self._generate_blob_map(size, blob['scale'])
            shape_mask = (noise_map > self.blob_threshold).unsqueeze(0)  # [1, H, W]

            # Create color with shades
            color_variation = (noise_map - 0.5) * 0.4  # Add +/- 20% variation
            final_color = (blob['color'] + color_variation).clamp(0, 1)  # [3, H, W]

            # Create full-size canvases for this blob
            blob_mask_1 = torch.zeros(1, self.img_size, self.img_size, dtype=torch.bool)
            blob_color_1 = torch.zeros(3, self.img_size, self.img_size)
            blob_flow_1 = torch.zeros(2, self.img_size, self.img_size)

            blob_mask_2 = torch.zeros(1, self.img_size, self.img_size, dtype=torch.bool)
            blob_color_2 = torch.zeros(3, self.img_size, self.img_size)

            # "Paste" blob data onto its canvases
            blob_mask_1[:, y1:y1 + size, x1:x1 + size] = shape_mask
            blob_color_1[:, y1:y1 + size, x1:x1 + size] = final_color
            blob_flow_1[:, y1:y1 + size, x1:x1 + size] = actual_flow

            blob_mask_2[:, y2:y2 + size, x2:x2 + size] = shape_mask
            blob_color_2[:, y2:y2 + size, x2:x2 + size] = final_color

            # 5. Composite (overwrite) onto main canvases
            # This is the occlusion step.
            img1 = torch.where(blob_mask_1, blob_color_1, img1)
            flow_fullres = torch.where(blob_mask_1, blob_flow_1, flow_fullres)

            img2 = torch.where(blob_mask_2, blob_color_2, img2)

        # --- 6. Create V0 Target (same as before) ---

        # Scale flow from pixel-space to patch-space
        flow_fullres_scaled = flow_fullres / self.patch_size

        flow_target = F.avg_pool2d(
            flow_fullres_scaled.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(0)

        return img1, img2, flow_target