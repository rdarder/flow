import numpy as np
from PIL import Image
import os
import re
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
from tqdm import tqdm # A handy progress bar: pip install tqdm

# --- Copied from your chairs_dataset.py ---
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
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[..., :2], scale
# --- End of copied function ---


def preprocess_dataset(dataset_root, img_size=32, patch_size=4):
    """
    Reads the entire Chairs dataset, processes it, and saves
    the results to fast-loading .pt files.
    """
    print("Starting preprocessing...")
    
    # --- 1. Define the JAX-compatible transforms ---
    # We remove T.Normalize to keep data in [0, 1]
    img_transform = T.Compose([
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(), # Scales to [0, 1]
    ])
    
    grid_size = img_size // patch_size # 8
    P = grid_size * grid_size # 64
    
    for split in ['train', 'test']:
        print(f"\nProcessing '{split}' split...")
        
        split_dir = os.path.join(dataset_root, split)
        t0_dir = os.path.join(split_dir, 't0')
        flow_dir = os.path.join(split_dir, 'flow')
        
        sample_names = [
            f.split('.')[0] for f in os.listdir(t0_dir) if f.endswith('.png')
        ]
        
        # This list will hold all the (img1, img2, flow) tuples in memory
        processed_data_list = []
        
        # Use tqdm for a nice progress bar
        for sample_name in tqdm(sample_names, desc=f'Processing {split}'):
            # --- 2. Load all files (slow I/O) ---
            t0_path = os.path.join(t0_dir, f"{sample_name}.png")
            t1_path = os.path.join(split_dir, 't1', f"{sample_name}.png")
            flow_path = os.path.join(flow_dir, f"{sample_name}.pfm")

            try:
                img_t0_orig = Image.open(t0_path).convert('RGB')
                img_t1_orig = Image.open(t1_path).convert('RGB')
                flow_orig, _ = readPFM(flow_path) # (H, W, 2)
            except Exception as e:
                print(f"Warning: Skipping sample {sample_name} due to error: {e}")
                continue

            # --- 3. Process data (slow CPU) ---
            crop_size = 384
            img_t0_cropped = center_crop(img_t0_orig, [crop_size, crop_size])
            img_t1_cropped = center_crop(img_t1_orig, [crop_size, crop_size])
            
            h, w, _ = flow_orig.shape
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            flow_cropped = flow_orig[top:top + crop_size, left:left + crop_size]

            img_t0_torch = img_transform(img_t0_cropped) # (3, 32, 32)
            img_t1_torch = img_transform(img_t1_cropped) # (3, 32, 32)

            scale_factor = img_size / crop_size
            flow_scaled = flow_cropped * scale_factor

            flow_tensor = torch.from_numpy(flow_scaled).permute(2, 0, 1).unsqueeze(0)
            
            flow_resized = F.interpolate(
                flow_tensor, size=(img_size, img_size), 
                mode='bilinear', align_corners=False
            )
            
            flow_target_torch = F.avg_pool2d(
                flow_resized, kernel_size=patch_size, stride=patch_size
            ).squeeze(0) # Shape: (2, 8, 8)

            # --- 4. Format for JAX (as in your file) ---
            flow_target_jax = flow_target_torch.reshape(2, P).T # (64, 2)
            img_t0_jax = img_t0_torch.permute(1, 2, 0) # (32, 32, 3)
            img_t1_jax = img_t1_torch.permute(1, 2, 0) # (32, 32, 3)

            # Add the final tensors to our list
            processed_data_list.append((img_t0_jax, img_t1_jax, flow_target_jax))

        # --- 5. Save the entire split to one file ---
        output_filename = f"chairs_v0_{split}.pt"
        torch.save(processed_data_list, output_filename)
        print(f"Successfully processed and saved {len(processed_data_list)} samples to {output_filename}")


if __name__ == "__main__":
    #
    # !!! IMPORTANT !!!
    # Set this path to the root of your ChairsSDHom dataset
    # (The one that contains the 'train' and 'test' folders)
    #
    DATASET_PATH = "../datasets/ChairsSDHom/data" 
    
    preprocess_dataset(DATASET_PATH, img_size=32, patch_size=4)
