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
        max_mag = np.percentile(magnitude, 99)
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
