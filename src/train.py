import os
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision
from flax import nnx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import the new m2 model
from model import BarebonesFlowModel
from synthetic_dataset import SyntheticFlowDataset


# --- 1. Logger Utility ---
class JaxLogger:
    def __init__(self, version: str, log_dir='runs'):
        run_name = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_path)
        print(f"Logging to {log_path}")

    def log(self, tag, value, step):
        try:
            value = float(value)
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            print(f"Logger Warning: {e}")

    def close(self):
        self.writer.close()


# --- 2. Visualization Utils ---
def flow_to_color(flow, max_flow=None):
    """Converts flow (H, W, 2) to RGB image."""
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


def create_diagnostic_figure(
        img1, img2, flow_gt,
        f_final, f_cross, f_peer,
        c_cross, c_peer,
        a_cross, a_peer
):
    """
    Creates a rich 3x3 diagnostic grid showing inputs, outputs, and internal traces.
    """
    # Prepare standard images
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    # Prepare Flows
    gt_color = flow_to_color(np.array(flow_gt))
    final_color = flow_to_color(np.array(f_final))
    cross_color = flow_to_color(np.array(f_cross))
    peer_color = flow_to_color(np.array(f_peer))

    # Prepare Confidence Maps (H, W, 1) -> (H, W)
    c_cross_map = np.array(c_cross).squeeze()
    c_peer_map = np.array(c_peer).squeeze()

    # Prepare Attention Samples
    # a_cross is (B, N, N). We take one sample (index 0) and reshape.
    # We visualize the attention *from* the center pixel *to* all other pixels.
    # N = H*W. Center index approx N/2 + W/2.
    H, W = c_cross_map.shape
    N = H * W
    center_idx = N // 2 + W // 2

    # Extract row for center pixel: (1, N) -> reshape to (H, W)
    att_cross_sample = np.array(a_cross[center_idx]).reshape(H - 2, W - 2)
    att_peer_sample = np.array(a_peer[center_idx]).reshape(H - 2, W - 2)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # Row 1: The Task & Result
    axes[0, 0].imshow(img1_np)
    axes[0, 0].set_title("Frame 1")
    axes[0, 1].imshow(img2_np)
    axes[0, 1].set_title("Frame 2")
    axes[0, 2].imshow(gt_color)
    axes[0, 2].set_title("GT Flow")

    # Row 2: V1 (Patch Lookup) Internals
    axes[1, 0].imshow(cross_color)
    axes[1, 0].set_title("V1 Flow (F_cross)")
    im_c1 = axes[1, 1].imshow(c_cross_map, vmin=0, vmax=1, cmap='viridis')
    axes[1, 1].set_title("V1 Confidence (C_cross)")
    plt.colorbar(im_c1, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 2].imshow(att_cross_sample, cmap='plasma')
    axes[1, 2].set_title("V1 Attn (Center Pixel)")

    # Row 3: V2 (Peer Prop) Internals + Final
    axes[2, 0].imshow(peer_color)
    axes[2, 0].set_title("V2 Flow (F_peer)")
    im_c2 = axes[2, 1].imshow(c_peer_map, cmap='viridis')
    axes[2, 1].set_title("V2 Consensus (C_peer)")
    plt.colorbar(im_c2, ax=axes[2, 1], fraction=0.046, pad=0.04)

    # Last spot: Final Result
    axes[2, 2].imshow(final_color)
    axes[2, 2].set_title("Final Pred (Blended)")

    # Clean up axes
    for ax in axes.flat:
        ax.axis('off')

    # Convert to array for TensorBoard
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_array = buffer.reshape(height, width, 4)[..., :3]
    plt.close(fig)

    return image_array


# --- 3. Training Logic ---

# Simple L2 Loss on Flow
def loss_fn(model, img1, img2, flow_gt):
    # Run model
    # Returns: flow_pred, aux_dict
    flow_pred, aux = model(img1, img2)

    # L2 Loss
    diff = flow_pred - flow_gt
    loss = jnp.mean(jnp.square(diff))

    return loss, aux


@nnx.jit
def train_step(model, optimizer, img1, img2, flow_gt):
    """
    Performs a single update step.
    Note: We pass 'model' for the state update, but 'model' is also inside 'optimizer'.
    The optimizer handles the param updates in place on the model instance.
    """
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, img1, img2, flow_gt)
    optimizer.update(model, grads)
    return loss, aux


class Training:
    def __init__(
            self,
            model,
            learning_rate=1e-3,
            log_dir='runs',
            train_dataset=None
    ):
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)

        # Logging
        self.logger = JaxLogger("m2_cartesian", log_dir)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    def log_visuals(self, epoch, img1, img2, flow_gt, aux, f_final):
        """Extracts traces and logs the figure."""
        # Extract JAX arrays from the batch (take index 0)
        # All aux outputs are (B, H, W, C) due to the model's reshaping/padding
        try:
            img = create_diagnostic_figure(
                img1[0], img2[0], flow_gt[0],
                f_final[0],
                aux['F_cross'][0],
                aux['F_peer'][0],
                aux['C_cross'][0],
                aux['C_peer'][0],
                aux['A_cross'][0],  # (64, 64) - we handle reshaping in plotting func
                aux['A_peer'][0]  # (64, 64)
            )
            self.logger.writer.add_image("Diagnostics", img, epoch, dataformats='HWC')
        except Exception as e:
            print(f"Error logging visuals: {e}")
            import traceback
            traceback.print_exc()

    def log_kernels(self, epoch):
        """Logs the stem kernels to TensorBoard."""
        try:
            log_kernels_to_tensorboard(self.model, self.logger, epoch)
        except Exception as e:
            print(f"Error logging kernels: {e}")

    def train(self, epochs=200, log_every_steps=50):
        print(f"Starting training for {epochs} epochs...")
        global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            # Store last batch for visualization
            last_batch = None
            last_aux = None
            last_pred = None

            for i, (img1_pt, img2_pt, flow_gt_pt) in enumerate(self.train_loader):
                # Convert to JAX
                img1 = jnp.array(img1_pt.numpy())
                img2 = jnp.array(img2_pt.numpy())
                flow_gt = jnp.array(flow_gt_pt.numpy())

                # Step
                loss, aux = train_step(self.model, self.optimizer, img1, img2, flow_gt)

                # Track
                epoch_loss += loss
                batch_count += 1
                global_step += 1

                if i % log_every_steps == 0:
                    self.logger.log("Loss/train_step", loss, global_step)

                # Keep for logging
                last_batch = (img1, img2, flow_gt)
                last_aux = aux

                # We need the prediction too, which isn't returned by train_step directly
                # But we can infer it or re-run inference for the log sample

            # End of Epoch Logging
            avg_loss = epoch_loss / batch_count
            self.logger.log("Loss/train_epoch", avg_loss, epoch)

            # Log Visuals (Re-run inference on last batch item to be clean)
            if last_batch is not None:
                img1, img2, flow_gt = last_batch
                # Run inference mode (same as train for this simple model)
                f_final, aux = self.model(img1, img2)
                self.log_visuals(epoch, img1, img2, flow_gt, aux, f_final)

            # Log Kernels
            self.log_kernels(epoch)

            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

            # Log Params
            model = self.model
            self.logger.log("Params/lookup/Visual/Scale", model.patch_lookup.visual_scale.value, epoch)
            self.logger.log("Params/lookup/Spatial/Scale", model.patch_lookup.spatial_score.scale.value, epoch)
            self.logger.log("Params/propagate/Visual/Scale", model.peer_prop.visual_scale.value, epoch)
            self.logger.log("Params/propagate/Spatial/Scale", model.peer_prop.spatial_score.scale.value, epoch)
            self.logger.log("Params/propagate/ConsensusBias", model.peer_prop.consensus_bias_scale.value, epoch)
            self.logger.log("Params/Blend", model.lookup_blend.value, epoch)

        self.logger.close()


VMIN = -1
VMAX = 1


def log_single_kernel(logger, kernel_jax, name, epoch):
    """Helper to convert, normalize, grid, and log one kernel."""
    # Flax Conv kernel shape: (H, W, In, Out)
    # PyTorch make_grid expects: (B, C, H, W) -> We treat Out as Batch, In as Channel
    kernels_torch = torch.from_numpy(np.array(kernel_jax))

    # Permute to (Out, In, H, W)
    kernels_permuted = kernels_torch.permute(3, 2, 0, 1)

    # Normalize for visualization (-1 to 1 range assumption for "middle gray" zero)
    # We clamp to ensure no outliers blow out the grid contrast
    kernels_clamped = torch.clamp(kernels_permuted, VMIN, VMAX)
    kernels_norm = (kernels_clamped - VMIN) / (VMAX - VMIN + 1e-6)

    # Make grid
    grid = torchvision.utils.make_grid(kernels_norm, nrow=8, padding=1)
    logger.writer.add_image(f'Kernels/{name}', grid, epoch)


def log_kernels_to_tensorboard(model, logger, epoch):
    """
    Logs the *spatial, depthwise* (dw) kernels from the 'stem'
    to TensorBoard as an image grid, using a *fixed*
    normalization range to make changes visible.
    """
    # Access the kernel value from the NNX module
    if hasattr(model.stem, 'dw1'):
        log_single_kernel(logger, model.stem.dw1.kernel.value, 'dw1_W (stem)', epoch)


if __name__ == "__main__":
    # Setup
    dataset = SyntheticFlowDataset(
        img_size=18,
        length=5000,
        blob_size_range=(2, 6),
    )  # 18x18 matches model expectation

    # Initialize Model
    key = jax.random.PRNGKey(42)
    model = BarebonesFlowModel(img_size_hw=(18, 18), embed_dim=16, rngs=nnx.Rngs(key))

    # Run
    trainer = Training(model, train_dataset=dataset)
    trainer.train(epochs=100)
