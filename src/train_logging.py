import matplotlib.pyplot as plt
import numpy as np

def flow_to_color(flow, max_flow=None):
    """Converts a 2D optical flow field (H, W, 2) into a color-coded RGB image."""
    # (This function is from our PyTorch script)
    H, W, C = flow.shape
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    h = (angle + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    if max_flow is None:
        max_mag = np.percentile(magnitude, 99)
    else:
        max_mag = max_flow
    v = np.clip(magnitude / (max_mag + 1e-6), 0, 1)
    hsv = np.stack([h, s, v], axis=-1)
    rgb = plt.cm.hsv(hsv[..., 0])
    rgb[..., :3] *= v[..., np.newaxis]
    return rgb[..., :3]

def create_flow_figure_jax(img1, img2, flow_gt, flow_pred, grid_size):
    """Creates a 3-panel matplotlib figure for JAX data."""
    # We'll just show the first item in the batch
    # img1 is (B, C, H, W) or (B, H, W, C). We need to get it to (H, W, C)
    
    # Get the first sample from the batch
    img1_sample = np.array(img1[0])
    img2_sample = np.array(img2[0])
    flow_gt_sample = np.array(flow_gt[0])
    flow_pred_sample = np.array(flow_pred[0])
    
    # --- 1. Handle Image ---
    # Data is (C, H, W) from our fixed dataloader
    if img1_sample.shape[0] == 3:
        img1_sample = img1_sample.transpose(1, 2, 0) # (H, W, C)
    if img2_sample.shape[0] == 3:
        img2_sample = img2_sample.transpose(1, 2, 0) # (H, W, C)
    # Clip to [0, 1] (our synthetic data is just 0s and 1s)
    img1_plot = np.clip(img1_sample, 0, 1)
    img2_plot = np.clip(img2_sample, 0, 1)

    # --- 2. Handle Flows ---
    # Flows are (P, 2), e.g., (64, 2). Reshape to (H, W, 2).
    flow_gt_img = flow_to_color(
        flow_gt_sample.reshape(grid_size, grid_size, 2)
    )
    flow_pred_img = flow_to_color(
        flow_pred_sample.reshape(grid_size, grid_size, 2)
    )
    
    # --- 3. Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(img1_plot)
    axes[0].set_title("Image 1 (t0)")
    axes[0].axis('off')

    axes[1].imshow(img2_plot)
    axes[1].set_title("Image 2 (t0)")
    axes[1].axis('off')

    axes[2].imshow(flow_gt_img)
    axes[2].set_title("Ground Truth Flow")
    axes[2].axis('off')
    
    axes[3].imshow(flow_pred_img)
    axes[3].set_title("Predicted Flow")
    axes[3].axis('off')
    
    fig.tight_layout()
    return fig
