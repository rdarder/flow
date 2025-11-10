import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import jit
from torch.utils.data import DataLoader

from model import BarebonesFlowModel
from chairs_dataset import PreprocessedChairsDataset
from synthetic_dataset import SyntheticFlowDataset
from train_logging import JaxLogger, create_flow_figure_jax, log_kernels_to_tensorboard, log_gradients


def compute_l1_loss(flow_pred, flow_gt):
    """Calculates the main L1 flow loss."""
    return jnp.mean(jnp.abs(flow_pred - flow_gt))


def compute_decorrelation_loss(F_batch):
    """
    Calculates the Barlow Twins/VICReg-style loss for a
    batch of features (B, P, C).
    """
    B, P, C = F_batch.shape

    # 1. Center features
    F_mean = F_batch.mean(axis=(0, 1))  # (C,)
    F_centered = F_batch - F_mean

    # 2. Calculate Covariance Matrix
    F_flat = F_centered.reshape(-1, C)  # (B*P, C)
    N = F_flat.shape[0]
    Cov = (F_flat.T @ F_flat) / (N - 1)  # (C, C)

    # 3. Variance Loss (on-diagonal)
    # We want diagonal to be 1.0. Penalize if it's < 1.0.
    loss_variance = jnp.mean(jax.nn.relu(1.0 - jnp.diag(Cov)))

    # 4. Covariance Loss (off-diagonal)
    # We want off-diagonal to be 0.0.
    Cov_off_diag = Cov.at[jnp.diag_indices(C)].set(0)
    loss_covariance = jnp.mean(Cov_off_diag ** 2)

    return loss_variance, loss_covariance


def loss_fn(model, img1_batch, img2_batch, flow_gt_batch):
    """Calculates the total combined loss."""

    flow_pred, extra = model(img1_batch, img2_batch)

    l1_loss = compute_l1_loss(flow_pred, flow_gt_batch)

    # Aux Loss for F1
    loss_var_F1, loss_cov_F1 = compute_decorrelation_loss(extra['f1'])

    # Aux Loss for F2 (as you suggested)
    loss_var_F2, loss_cov_F2 = compute_decorrelation_loss(extra['f2'])

    # Average the auxiliary losses
    total_var_loss = (loss_var_F1 + loss_var_F2) / 2.0
    total_cov_loss = (loss_cov_F1 + loss_cov_F2) / 2.0

    # --- 3. Combine them (with weights) ---
    # (We can tune these lambda weights later)
    LAMBDA_VAR = 1e-2
    LAMBDA_COV = 1e-2

    total_loss = l1_loss + (LAMBDA_VAR * total_var_loss) + (LAMBDA_COV * total_cov_loss)

    # --- 4. Return metrics for logging ---
    metrics = {
        'total_loss': total_loss,
        'l1_flow_loss': l1_loss,
        'variance_loss': total_var_loss,
        'covariance_loss': total_cov_loss
    }
    return total_loss, metrics

# --- 2. The New "Update Step" (using nnx and optax) ---
@jit
def update_step(model, optimizer, img1_batch, img2_batch, flow_gt_batch):
    """Performs one update step using nnx and optax."""

    # 1. Get grads (nnx.value_and_grad is the new tool)
    # It knows to differentiate *only* the nnx.Param parts of 'model'
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, img1_batch, img2_batch, flow_gt_batch
    )
    optimizer.update(model, grads)
    return model, loss, metrics, grads


# --- 3. Our Refactored Logging Helpers ---


if __name__ == "__main__":

    # --- Hyperparameters ---
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    IMG_SIZE = 32
    PATCH_SIZE = 4
    EMBED_DIM = 64
    LAMBDA_VARIANCE = 1e-4

    grid_size = IMG_SIZE // PATCH_SIZE

    # --- JAX / Model Setup ---
    key = jax.random.PRNGKey(42)
    # nnx.split is the new way to get a "param" key
    # It creates an Rngs object
    rngs = nnx.Rngs(params=key)

    # Model initialization is now one line
    model = BarebonesFlowModel(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM, rngs=rngs
    )

    # --- Optax Setup ---
    optimizer = nnx.Optimizer(model,
                              optax.adam(learning_rate=LEARNING_RATE),
                              wrt=nnx.Param)

    # --- Data Setup ---
    train_dataset = PreprocessedChairsDataset('../datasets/chairs_32/')  # (Our V0.2 with noise)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    logger = JaxLogger()
    print("Starting training with Flax.nnx...")
    global_step = 0

    for epoch in range(EPOCHS):
        total_epoch_loss = 0.0

        for i, (img1_pt, img2_pt, flow_gt_pt) in enumerate(train_loader):

            img1_batch = jnp.array(img1_pt.numpy())
            img2_batch = jnp.array(img2_pt.numpy())
            flow_gt_batch = jnp.array(flow_gt_pt.numpy())

            # --- 2. Run the JIT-compiled Update Step ---
            model, loss, metrics, grads = update_step(
                model, optimizer, img1_batch, img2_batch, flow_gt_batch
            )

            total_epoch_loss += loss

            if global_step % 20 == 0:
                for metric_name, metric_value in metrics.items():
                    logger.log(f'Loss/{metric_name}', metric_value, global_step)
            global_step += 1

        # --- 4. End of Epoch Logging ---
        avg_loss = total_epoch_loss / len(train_loader)
        logger.log('Loss/train_epoch', avg_loss, epoch)
        logger.log('params/zero_boost', model.log_w_zero_boost, epoch)

        # --- Log Gradients ---
        log_gradients(grads, logger, epoch)

        # --- Log Kernel Images ---
        log_kernels_to_tensorboard(model, logger, epoch)

        # --- Log Prediction Snapshot ---
        # We just call the model (it's vmapped in the class)
        flow_pred_snapshot, extra = model(img1_batch, img2_batch)

        fig = create_flow_figure_jax(
            img1_pt.numpy(), img2_pt.numpy(),
            flow_gt_batch, flow_pred_snapshot,
            grid_size
        )
        logger.writer.add_figure('Validation/prediction_sample', fig, global_step)
        plt.close(fig)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Avg Loss: {avg_loss:.6f}")

    logger.close()
    print("Training complete.")
