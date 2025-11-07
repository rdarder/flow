from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Import our custom modules ---
from model import V0MicroFlow
from datasets import ChairsV0Dataset, flow_to_color
from synthetic_data import SyntheticFlowDataset

# --- 1. Hyperparameters & Setup ---
DATASET_PATH = "../datasets/ChairsSDHom/data"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
IMG_SIZE = 32
PATCH_SIZE = 4
DEBUG_MAX_STEPS = None
NOISE_LEVEL = 0.05

# --- 2. Visualization/Logging Helpers ---

# (Un-normalization for plotting images)
# These are the std and mean from the dataset pipeline
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def unnormalize_image(tensor):
    """Reverses the T.Normalize operation for a single image."""
    img = tensor.clone()  # [C, H, W]
    for t, m, s in zip(img, NORM_MEAN, NORM_STD):
        t.mul_(s).add_(m)  # (t * s) + m
    img = img.permute(1, 2, 0)  # [H, W, C]
    img = torch.clamp(img, 0, 1)
    return img.cpu().numpy()


def create_flow_figure(img1, img2, flow_gt, flow_pred):
    """Creates a 4-panel matplotlib figure for logging to TensorBoard."""
    # We'll just show the first item in the batch
    img1_un = unnormalize_image(img1[0])
    img2_un = unnormalize_image(img2[0])

    # [2, H, W] -> [H, W, 2]
    flow_gt_np = flow_gt[0].permute(1, 2, 0).cpu().numpy()
    flow_pred_np = flow_pred[0].permute(1, 2, 0).cpu().numpy()

    flow_gt_img = flow_to_color(flow_gt_np)
    flow_pred_img = flow_to_color(flow_pred_np)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img1_un)
    axes[0].set_title("Image 1 (t0)")
    axes[0].axis('off')

    axes[1].imshow(img2_un)
    axes[1].set_title("Image 2 (t1)")
    axes[1].axis('off')

    axes[2].imshow(flow_gt_img)
    axes[2].set_title("Ground Truth Flow")
    axes[2].axis('off')

    axes[3].imshow(flow_pred_img)
    axes[3].set_title("Predicted Flow")
    axes[3].axis('off')

    fig.tight_layout()
    return fig


# --- 3. Setup Device, Data, Model, etc. ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading ---
# train_dataset = ChairsV0Dataset(DATASET_PATH, split='train', img_size=IMG_SIZE, patch_size=PATCH_SIZE)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#
# test_dataset = ChairsV0Dataset(DATASET_PATH, split='test', img_size=IMG_SIZE, patch_size=PATCH_SIZE)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

train_dataset = SyntheticFlowDataset()
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle, it's all random
    num_workers=4
)

test_dataset = SyntheticFlowDataset() # Use a separate instance for val
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)
print(f"Loaded {len(train_dataset)} training samples.")
print(f"Loaded {len(test_dataset)} test samples.")

# --- Model, Loss, Optimizer ---
model = V0MicroFlow(img_size=IMG_SIZE, patch_size=PATCH_SIZE).to(device)
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- TensorBoard Logger ---
RUN_NAME = "V0.2_PostNorm_LR_1e-3"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"runs/{RUN_NAME}_{timestamp}"
writer = SummaryWriter(log_dir)
print(f"Logging to TensorBoard. Run: tensorboard --logdir=runs")

# --- 4. The Training Loop ---
global_step = 0
for epoch in range(EPOCHS):

    # --- Training Phase ---
    model.train()
    total_train_loss = 0.0

    for i, (img1, img2, flow_gt) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        flow_gt = flow_gt.to(device)

        if NOISE_LEVEL > 0:
            # Create different noise for each image
            noise_a = torch.randn_like(img1) * NOISE_LEVEL
            noise_b = torch.randn_like(img2) * NOISE_LEVEL

            img1 = img1 + noise_a
            img2 = img2 + noise_b

        flow_pred = model(img1, img2)
        loss = criterion(flow_pred, flow_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        if (i + 1) % 20 == 0:
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.flush()

        global_step += 1

        if DEBUG_MAX_STEPS is not None and (i + 1) >= DEBUG_MAX_STEPS:
            print(f"DEBUG: Reached {DEBUG_MAX_STEPS} steps, ending mini-epoch.")
            break

    avg_train_loss = total_train_loss / (i + 1)

    # --- Validation & Logging Phase ---
    model.eval()
    total_val_loss = 0.0
    logged_images_this_epoch = False
    log_batch_idx = epoch % len(test_loader)

    with torch.no_grad():
        for i, (img1_val, img2_val, flow_gt_val) in enumerate(test_loader):
            img1_val = img1_val.to(device)
            img2_val = img2_val.to(device)
            flow_gt_val = flow_gt_val.to(device)

            flow_pred_val = model(img1_val, img2_val)
            loss = criterion(flow_pred_val, flow_gt_val)
            total_val_loss += loss.item()

            # *** NEW: Log image sample on the first validation batch of each epoch ***
            if i == log_batch_idx:
                fig = create_flow_figure(
                    img1_val.cpu(),
                    img2_val.cpu(),
                    flow_gt_val.cpu(),
                    flow_pred_val.cpu()
                )
                writer.add_figure('Validation/prediction_sample', fig, epoch)
                plt.close(fig)  # Close the figure to save memory
                logged_images_this_epoch = True

        if DEBUG_MAX_STEPS is not None and (i + 1) >= DEBUG_MAX_STEPS:
            print(f"DEBUG: Reached {DEBUG_MAX_STEPS} steps, ending validation.")
            break

    avg_val_loss = total_val_loss / (i + 1)

    # --- End of Epoch Logging ---
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)

    # 1. Log Layer 1 Kernels (Input: 3 -> 32)
    kernels_l1 = model.patch_embed[0].weight.detach().cpu()
    # kernels_l1 shape is (32, 3, 3, 3)
    # Normalize them to [0, 1]
    k_min, k_max = kernels_l1.min(), kernels_l1.max()
    kernels_l1_norm = (kernels_l1 - k_min) / (k_max - k_min)

    # Create a grid (make_grid handles the 3-channel RGB aspect)
    grid_l1 = torchvision.utils.make_grid(kernels_l1_norm, nrow=8)
    writer.add_image('PatchEmbed/kernels_layer1', grid_l1, epoch)

    # 2. Log Layer 2 Kernels (Input: 32 -> 64)
    kernels_l2 = model.patch_embed[3].weight.detach().cpu()
    # kernels_l2 shape is (64, 32, 3, 3)

    # We can't visualize all 32 input channels.
    # As a simple V0 visualization, let's just see the
    # kernels for the *first input channel* (grayscale).
    kernels_l2_slice = kernels_l2[:, 0:1, :, :]  # Shape: (64, 1, 3, 3)

    # Normalize this slice
    k_min, k_max = kernels_l2_slice.min(), kernels_l2_slice.max()
    kernels_l2_norm = (kernels_l2_slice - k_min) / (k_max - k_min)

    # Create a grid (make_grid will handle grayscale)
    grid_l2 = torchvision.utils.make_grid(kernels_l2_norm, nrow=8)
    writer.add_image('PatchEmbed/kernels_layer2_slice', grid_l2, epoch)

    # *** NEW: Log Gradients and Weights ***
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        if param.data is not None:
            writer.add_histogram(f'Weights/{name}', param.data, epoch)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Logs updated.")

# --- 5. Cleanup ---
writer.close()
print("Training complete.")

model_save_path = "v0_micro_flow_final_logged.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Final model saved to {model_save_path}")
