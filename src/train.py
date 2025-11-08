from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from chairs_dataset import ChairsV0Dataset
from model import apply_model, init_params, create_location_tensor
from synthetic_dataset import SyntheticFlowDataset
from train_logging import create_flow_figure_jax


# --- 2. Simple Logger Class ---
class JaxLogger:
    def __init__(self, log_dir='runs'):
        run_name = f"v0_jax_barebones_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


# --- 4. JAX Training Functions (Loss & Update) ---

# We use vmap to "batch" our single-sample apply_model function.
# (None, 0, 0, None, None, None) tells vmap how to handle the args:
# - params: Don't batch (broadcast)
# - img1: Batch along axis 0
# - img2: Batch along axis 0
# - L: Don't batch (broadcast)
# - patch_size: Don't batch (static)
# - embed_dim: Don't batch (static)
batch_apply_model = vmap(
    apply_model, in_axes=(None, 0, 0, None, None, None)
)

LAMBDA_VARIANCE = 1e-4  # A small "nudge" to encourage variance


def loss_fn(params, img1_batch, img2_batch, flow_gt_batch, L, patch_size, embed_dim):
    """
    Calculates the total loss (L1 + variance reward) and returns
    auxiliary metrics for logging.
    """

    # 1. Get model outputs
    # batch_apply_model now returns a tuple of batched tensors
    flow_pred_batch, F1_batch = batch_apply_model(
        params, img1_batch, img2_batch, L, patch_size, embed_dim
    )

    # --- 2. Calculate L1 Loss ---
    l1_loss = jnp.mean(jnp.abs(flow_pred_batch - flow_gt_batch))

    # --- 3. Calculate Variance Reward ---
    # F1_batch is (B, P, C). We want var across P (axis 1).
    feature_variance = jnp.var(F1_batch, axis=1)  # (B, C)
    mean_feature_variance = jnp.mean(feature_variance)

    # We reward variance by *subtracting* its negative (i.e., adding a penalty
    # for low variance).
    variance_reward = -LAMBDA_VARIANCE * mean_feature_variance

    # --- 4. Total Loss ---
    total_loss = l1_loss + variance_reward

    # --- 5. Auxiliary data for logging ---
    metrics = {
        'l1_loss': l1_loss,
        'variance_reward': variance_reward,
        'mean_feature_variance': mean_feature_variance,
        'total_loss': total_loss
    }

    # Return (loss, aux_data)
    # jax.grad will *only* differentiate with respect to 'total_loss'
    return total_loss, metrics


@partial(jit, static_argnums=(5, 6))  # <-- RE-ENABLE @jit
def update_step(params, img1_batch, img2_batch, flow_gt_batch, L, patch_size, embed_dim, lr):
    """Performs one update step."""

    # 1. Use has_aux=True
    # This returns ((loss, metrics), grads)
    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, img1_batch, img2_batch, flow_gt_batch, L, patch_size, embed_dim
    )

    # 2. Optimizer step (unchanged)
    updated_params = jax.tree_util.tree_map(
        lambda p, g: p - lr * g,
        params,
        grads
    )

    # 3. Return everything for logging
    return updated_params, total_loss, metrics


def log_kernels_to_tensorboard(params, logger, epoch):
    """
    Logs the patch_embed kernels to TensorBoard as an image grid.
    """
    VMIN = -1.0
    VMAX = 1.0
    for name, kernels_jax in params['stem'].items():
        if not name.startswith('dw') or not name.endswith('_W'):
            continue  # Skip biases, norms, and pointwise (pw) kernels
        try:
            # 2. Convert to Torch tensor for easy grid visualization
            # JAX -> NumPy -> Torch
            kernels_torch = torch.from_numpy(np.array(kernels_jax))

            kernels_clamped = torch.clamp(kernels_torch, VMIN, VMAX)
            kernels_norm = (kernels_clamped - VMIN) / (VMAX - VMIN + 1e-6)

            # 3. Normalize them to [0, 1] to be visible
            # k_min = kernels_torch.min()
            # k_max = kernels_torch.max()
            # kernels_norm = (kernels_torch - k_min) / (k_max - k_min)

            # 4. Create an 8x8 grid of our 64 RGB filters
            # (nrow=8 for an 8x8 grid)
            grid = torchvision.utils.make_grid(kernels_norm, nrow=8, padding=1)

            # 5. Log to the writer
            logger.writer.add_image(f'Kernels/{name}', grid, epoch)
            # logger.flush()

        except Exception as e:
            print(f"Kernel Logging Warning: {e}")


grad_snapshot_fn = jax.grad(loss_fn, has_aux=True)

# --- 5. The Main Training Loop ---
if __name__ == "__main__":

    # --- Hyperparameters ---
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    IMG_SIZE = 32
    PATCH_SIZE = 4

    grid_size = IMG_SIZE // PATCH_SIZE

    # --- JAX Setup ---
    key = jr.PRNGKey(42)
    params, (p_size, e_dim) = init_params(key)
    L = create_location_tensor(IMG_SIZE // PATCH_SIZE)

    # --- Data Setup ---
    # train_dataset = ChairsV0Dataset(dataset_root='../datasets/ChairsSDHom/data')
    train_dataset = SyntheticFlowDataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # --- Logger Setup ---
    logger = JaxLogger()

    print("Starting training...")
    global_step = 0
    for epoch in range(EPOCHS):
        total_epoch_loss = 0.0

        for i, (img1_pt, img2_pt, flow_gt_pt) in enumerate(train_loader):

            # --- 1. Data Conversion (PyTorch -> JAX) ---
            # We just convert the numpy arrays to jax.numpy
            img1_batch = jnp.array(img1_pt.numpy())
            img2_batch = jnp.array(img2_pt.numpy())
            flow_gt_batch = jnp.array(flow_gt_pt.numpy())

            # --- 2. Run the JIT-compiled Update Step ---
            params, loss, metrics = update_step(
                params, img1_batch, img2_batch, flow_gt_batch,
                L, p_size, e_dim, LEARNING_RATE
            )

            total_epoch_loss += loss

            # --- 3. Log to TensorBoard ---
            if global_step % 20 == 0:
                for metric_name, metric_value in metrics.items():
                    logger.log(f'Loss/{metric_name}', metric_value, global_step)

            global_step += 1

        # --- 4. End of Epoch Logging (NEW) ---
        avg_loss = total_epoch_loss / len(train_loader)
        logger.log('Loss/train_epoch', avg_loss, epoch)

        # --- Generate the Gradient Snapshot ---
        # (We use the *last* batch of data for this)
        grads_snapshot, _ = grad_snapshot_fn(
            params, img1_batch, img2_batch, flow_gt_batch,
            L, p_size, e_dim
        )
        #
        # # Calculate and log the gradient magnitudes
        grad_mag_w = jnp.mean(jnp.abs(grads_snapshot['stem']['dw1_W']))
        # grad_mag_b = jnp.mean(jnp.abs(grads_snapshot['stem']['dw1_b']))
        grad_mag_temp = jnp.mean(jnp.abs(grads_snapshot['log_temp']))
        #
        logger.log('GradMag/W_embed', grad_mag_w, epoch)
        # logger.log('GradMag/b_embed', grad_mag_b, epoch)
        logger.log('GradMag/log_temp', grad_mag_temp, epoch)
        logger.log('params/log_temp', params['log_temp'], epoch)

        # --- NEW: Generate Prediction Snapshot ---
        # (We use the *last* batch of data again)
        flow_pred_snapshot = batch_apply_model(
            params, img1_batch, img2_batch, L, p_size, e_dim
        )

        fig = create_flow_figure_jax(
            img1_pt.numpy(),  # The (B, C, H, W) torch tensor
            img2_pt.numpy(),  # The (B, C, H, W) torch tensor
            flow_gt_batch,  # The (B, P, 2) jax array
            flow_pred_snapshot,  # The (B, P, 2) jax array
            grid_size
        )
        logger.writer.add_figure('Validation/prediction_sample', fig, global_step)
        plt.close(fig)  # Close the figure to save memory
        log_kernels_to_tensorboard(params, logger, epoch)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Avg Loss: {avg_loss:.6f} | "
              f"W_grad: {grad_mag_w:.2e} "
              # f"| b_grad: {grad_mag_b:.2e}"
              )

    logger.close()
    print("Training complete.")
