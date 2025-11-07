import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


class SyntheticFlowDataset(Dataset):
    """
    Generates a synthetic dataset of "blobs" moving.
    - A "blob" is 1-3 overlapping, randomly-sized squares.
    - Each blob has one random color.
    - The entire blob moves with one flow vector.
    """

    def __init__(self,
                 img_size=32,
                 patch_size=4,
                 min_shape_size=4,
                 max_shape_size=10,
                 max_flow=5,
                 max_shapes_per_blob=3):
        """
        Args:
            img_size (int): Final image size (e.g., 32).
            patch_size (int): Patch size of our V0 model (e.g., 4).
            min_shape_size (int): Smallest size for a sub-square.
            max_shape_size (int): Largest size for a sub-square.
            max_flow (int): Max pixel displacement.
            max_shapes_per_blob (int): Max sub-squares to make a blob (e.g., 3).
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.min_shape_size = min_shape_size
        self.max_shape_size = max_shape_size
        self.max_flow = max_flow
        self.max_shapes_per_blob = max_shapes_per_blob

    def __len__(self):
        # Virtual epoch size
        return 10000

    def __getitem__(self, idx):
        # 1. Create two blank images
        img1 = torch.zeros(3, self.img_size, self.img_size)
        img2 = torch.zeros(3, self.img_size, self.img_size)

        # 2. Create the full-res ground truth flow field
        flow_fullres = torch.zeros(2, self.img_size, self.img_size)

        # 3. Define ONE flow vector for the entire blob
        dx = np.random.randint(-self.max_flow, self.max_flow + 1)
        dy = np.random.randint(-self.max_flow, self.max_flow + 1)

        # 5. Define how many sub-shapes this blob has
        num_shapes = np.random.randint(1, self.max_shapes_per_blob + 1)

        for _ in range(num_shapes):
            # 4. Define ONE color for the entire blob
            # (Shape [3, 1, 1] for broadcasting)
            color = torch.rand(3, 1, 1)

            # a. Create a random size for this sub-shape
            shape_size = np.random.randint(self.min_shape_size, self.max_shape_size + 1)

            # b. Define random start position
            x1 = np.random.randint(0, self.img_size - shape_size)
            y1 = np.random.randint(0, self.img_size - shape_size)

            # c. Calculate end position (and clamp)
            x2 = np.clip(x1 + dx, 0, self.img_size - shape_size)
            y2 = np.clip(y1 + dy, 0, self.img_size - shape_size)

            # d. Re-calculate *actual* flow for this sub-shape
            # (This is needed if clamping happened)
            actual_dx = x2 - x1
            actual_dy = y2 - y1

            # e. Draw the shapes and flow
            img1[:, y1:y1 + shape_size, x1:x1 + shape_size] = color
            img2[:, y2:y2 + shape_size, x2:x2 + shape_size] = color

            # Write this sub-shape's flow to the GT field
            flow_fullres[0, y1:y1 + shape_size, x1:x1 + shape_size] = actual_dx
            flow_fullres[1, y1:y1 + shape_size, x1:x1 + shape_size] = actual_dy

        # --- 6. Create the V0 Target (same as before) ---

        # a. Scale flow vectors from pixel-space to patch-space
        flow_fullres_scaled = flow_fullres / self.patch_size

        # b. Average Pool to get (8, 8, 2) target
        flow_target = F.avg_pool2d(
            flow_fullres_scaled.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(0)  # Shape: [2, 8, 8]

        return img1, img2, flow_target