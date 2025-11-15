import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from flax import nnx
from jax import numpy as jnp
from torch.utils.tensorboard import SummaryWriter


def flow_to_color(flow, max_flow=None):
    """Converts a 2D optical flow field (H, W, 2) into a color-coded RGB image."""
    # (This function is from our PyTorch script)
    H, W, C = flow.shape
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)
    h = (angle + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    if max_flow is None:
        max_mag = np.percentile(magnitude, 99.9)
    else:
        max_mag = max_flow
    v = np.clip(magnitude / (max_mag + 1e-6), 0, 1)
    hsv = np.stack([h, s, v], axis=-1)
    rgb = plt.cm.hsv(hsv[..., 0])
    rgb[..., :3] *= v[..., np.newaxis]
    return rgb[..., :3]


class JaxLogger:
    def __init__(self, version: str, log_dir='runs'):
        run_name = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_path)
        print(f"Logging to {log_path}")

    def log(self, tag, value, step):
        # We need to .item() to get a Python scalar
        # or .to_py() for JAX DeviceArrays
        try:
            value = float(value)
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            print(f"Logger Warning: {e}")

    def close(self):
        self.writer.close()


def create_diagnostic_figure_jax(
        img1, img2, flow_gt,
        f_final, f_cross, f_peer,
        c_cross_grid, a_cross, a_peer
):
    """
    Creates the new 9-panel diagnostic figure.
    Expects all inputs to be JAX arrays from a *single* batch item (e.g., index 0).
    """

    # --- 1. Prepare Data (Convert JAX to NumPy for plotting) ---
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # Convert flows to color images
    gt_img = flow_to_color(np.array(flow_gt))
    f_final_img = flow_to_color(np.array(f_final))
    f_cross_img = flow_to_color(np.array(f_cross))
    f_peer_img = flow_to_color(np.array(f_peer))

    # --- 2. Process Attention & Confidence Maps ---
    # Find the most occluded patch (lowest confidence)
    # c_cross_grid is (64, 1)
    worst_patch_idx = jnp.argmin(c_cross_grid[:, 0])

    # Get the 1D attention vectors for that patch and reshape to 8x8
    h, w = 8, 8  # We know this from our grid size
    a_cross_map = np.array(a_cross[worst_patch_idx, :]).reshape(h, w)
    a_peer_map = np.array(a_peer[worst_patch_idx, :]).reshape(h, w)

    # Reshape confidence map to 8x8 for viewing
    c_cross_map = np.array(c_cross_grid).reshape(h, w)

    # --- 3. Create Plot ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Flow Diagnostics", fontsize=16)

    # Row 1: Inputs & Ground Truth
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title("Image 1 (f0)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title("Image 2 (f1)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt_img)
    axes[0, 2].set_title("Ground Truth Flow")
    axes[0, 2].axis('off')

    # Row 2: Flow Components
    axes[1, 0].imshow(f_cross_img)
    axes[1, 0].set_title("F_cross (V1 Flow)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(f_peer_img)
    axes[1, 1].set_title("F_peer (V2 Flow)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(f_final_img)
    axes[1, 2].set_title("F_final (Blended)")
    axes[1, 2].axis('off')

    # Row 3: Attention & Confidence
    im_c = axes[2, 0].imshow(c_cross_map, cmap='viridis', vmin=0.0, vmax=1.0)
    axes[2, 0].set_title("C_cross (V1 Confidence)")
    plt.colorbar(im_c, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im_a1 = axes[2, 1].imshow(a_cross_map, cmap='hot')
    axes[2, 1].set_title(f"A_cross (for patch {worst_patch_idx})")
    plt.colorbar(im_a1, ax=axes[2, 1], fraction=0.046, pad=0.04)

    im_a2 = axes[2, 2].imshow(a_peer_map, cmap='hot')
    axes[2, 2].set_title(f"A_peer (for patch {worst_patch_idx})")
    plt.colorbar(im_a2, ax=axes[2, 2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

VMIN, VMAX = -1.0, 1.0


def log_single_kernel(logger, kernel_jax, name, epoch):
    """Helper to convert, normalize, grid, and log one kernel."""
    kernels_torch = torch.from_numpy(np.array(kernel_jax))
    kernels_permuted = kernels_torch.permute(3, 2, 0, 1)
    kernels_clamped = torch.clamp(kernels_permuted, VMIN, VMAX)
    kernels_norm = (kernels_clamped - VMIN) / (VMAX - VMIN + 1e-6)
    grid = torchvision.utils.make_grid(kernels_norm, nrow=8, padding=1)
    logger.writer.add_image(f'Kernels/{name}', grid, epoch)


def log_kernels_to_tensorboard(model, logger, epoch):
    """
    Logs the *spatial, depthwise* (dw) kernels from the 'stem'
    to TensorBoard as an image grid, using a *fixed*
    normalization range to make changes visible.
    """
    log_single_kernel(logger, model.stem.dw1.kernel.value, 'dw1_W (stem)', epoch)


def log_gradients(grads_tree, logger, epoch):
    """(This function replaces our manual logging)"""
    # Get just the learnable param gradients
    param_grads = grads_tree.filter(nnx.Param)
    param_grads_flat = nnx.to_flat_state(param_grads)
    for path, grad_val in param_grads_flat:
        # 'path' is a tuple of keys, e.g., ('stem', 'dw1_W', 'kernel')
        # 'grad_val' is the raw jax array

        # We join the path tuple to make a clean name
        param_name = ".".join(path)

        grad_mag = jnp.mean(jnp.abs(grad_val))
        logger.log(f'GradMag/{param_name}', grad_mag, epoch)