import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision
from flax import nnx
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import BarebonesFlowModel
from synthetic_dataset import SyntheticFlowDataset


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


class Training:
    def __init__(self, epochs: int, batch_size: int, learning_rate: float, img_size_hw: tuple[int, int],
                 embed_dim: int, log_every_steps: int, logger: JaxLogger,
                 train_dataset: Dataset):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_size = img_size_hw
        self.embed_dim = embed_dim
        key = jax.random.PRNGKey(135)
        self.rngs = nnx.Rngs(params=key, sample=0)
        self.log_every_steps = log_every_steps
        self.model = BarebonesFlowModel(
            img_size_hw=img_size_hw,
            embed_dim=embed_dim,
            rngs=self.rngs
        )
        self.optimizer = nnx.Optimizer(self.model, optax.sgd(learning_rate=learning_rate), wrt=nnx.Param)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.logger = logger

    def log_metrics(self, step: int, metrics_bag: dict):
        for metric_category, metrics in metrics_bag.items():
            for metric_name, metric_value in metrics.items():
                if hasattr(metric_value, 'ndim') and metric_value.ndim > 1:
                    continue
                self.logger.log(f'{metric_category}/{metric_name}', metric_value, step)

    def log_epoch_end(self, epoch: int, total_epoch_loss: float):
        avg_loss = total_epoch_loss / len(self.train_loader)
        self.logger.log('Loss/train_epoch', avg_loss, epoch)
        self.logger.log('params/patch_lookup/attn_temperature', self.model.patch_lookup.attn_temperature, epoch)
        self.logger.log('params/patch_lookup/pos_scale', self.model.patch_lookup.pos_scale, epoch)
        # Log new denoiser params
        self.logger.log('params/denoiser/attn_temperature', self.model.denoiser.attn_temperature, epoch)
        self.logger.log('params/denoiser/denoise_factor', self.model.denoiser.denoise_factor, epoch)
        self.logger.log('params/peer_prop/pos_scale', self.model.peer_prop.pos_scale, epoch)
        self.logger.log('params/peer_prop/attn_temperature', self.model.peer_prop.attn_temperature, epoch)
        print(f"Epoch [{epoch + 1}/{self.epochs}] | Avg Loss: {avg_loss:.6f}")

    def log_gradients(self, gradients, epoch: int):
        log_gradients(gradients, self.logger, epoch)

    def log_conv_kernels(self, epoch: int):
        log_kernels_to_tensorboard(self.model, self.logger, epoch)

    def log_evaluation_sample(self, epoch: int):
        """Grabs one batch, runs inference, and logs the new diagnostic figure."""
        try:
            # 1. Get one random batch from the *shuffled* loader
            raw_frame1, raw_frame2, flow_gt_pt = next(iter(self.train_loader))

            # 2. Convert to JAX arrays
            img1_batch = jnp.array(raw_frame1.numpy())
            img2_batch = jnp.array(raw_frame2.numpy())
            flow_gt_batch = jnp.array(flow_gt_pt.numpy())

            # 3. Run inference
            # We now need the 'aux' bag to get our trace data
            dense_flow_pred, aux = self.model(img1_batch, img2_batch, flow_gt_batch)

            # 4. Get all the debug data from the trace
            trace = aux['trace']

            # 5. Create and log the new diagnostic figure
            # We'll just log the first item in the batch (index 0)
            fig = create_diagnostic_figure_jax(
                img1_batch[0],
                img2_batch[0],
                flow_gt_batch[0],
                dense_flow_pred[0],  # F_final (blended)
                trace['F_cross_dense'][0],  # F_cross (V1 flow)
                trace['F_peer_dense'][0],  # F_peer (V2 flow)
                trace['C_cross_grid'][0],  # (64, 1) confidence
                trace['A_cross'][0],  # (64, 64) V1 attention
                trace['A_peer'][0]  # (64, 64) V2 attention
            )
            self.logger.writer.add_figure('Flow_Diagnostics/epoch', fig, epoch)

        except Exception as e:
            print(f"Warning: Could not log evaluation sample. Error: {e}")
            import traceback
            traceback.print_exc()

    def train_one_epoch(self, global_step: int, epoch: int):
        total_epoch_loss = 0.0
        grads = None
        frame1_batch, frame2_batch, flow_gt_batch, aux_bag = None, None, None, None
        for i, (raw_frame1, raw_frame2, flow_gt_pt) in enumerate(self.train_loader):
            frame1_batch = jnp.array(raw_frame1.numpy())
            frame2_batch = jnp.array(raw_frame2.numpy())
            flow_gt_batch = jnp.array(flow_gt_pt.numpy())

            loss, aux_bag, grads = update_step(self.model, self.optimizer, frame1_batch, frame2_batch,
                                               flow_gt_batch)
            total_epoch_loss += loss
            if (global_step + 1) % self.log_every_steps == 0:
                self.log_metrics(global_step, aux_bag)

            global_step += 1

        self.log_epoch_end(epoch, total_epoch_loss)
        self.log_gradients(grads, epoch)
        self.log_conv_kernels(epoch)
        self.log_evaluation_sample(epoch)
        return global_step

    def run(self):
        print("Starting training with Flax.nnx...")
        global_step = 0
        for epoch in range(self.epochs):
            global_step = self.train_one_epoch(global_step, epoch)
        self.logger.close()
        print("Training complete.")

    @classmethod
    def default_build(cls):
        # train_dataset = PreprocessedChairsDataset('../datasets/chairs_32/')
        train_dataset = SyntheticFlowDataset(img_size=18, length=10_000)
        version = 'v1'
        jax_logger = JaxLogger(version)
        training = Training(
            epochs=200,
            batch_size=64,
            learning_rate=1e-3,
            img_size_hw=(18, 18),
            embed_dim=16,
            log_every_steps=50,
            train_dataset=train_dataset,
            logger=jax_logger,
        )
        return training


def loss_fn(model, img1_batch, img2_batch, flow_gt_batch):
    _, aux = model(img1_batch, img2_batch, flow_gt_batch)

    loss_components = aux['loss']
    total_loss = loss_components['flow']
    loss_components['total'] = total_loss
    return total_loss, aux


@nnx.jit
def update_step(model, optimizer, img1, img2, flow_gt):
    """Performs one update step using nnx and optax."""

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, img1, img2, flow_gt
    )
    optimizer.update(model, grads)
    return loss, aux, grads


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


if __name__ == "__main__":
    training = Training.default_build()
    training.run()