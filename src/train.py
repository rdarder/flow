import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import jit
from torch.utils.data import DataLoader

from model import BarebonesFlowModel
from synthetic_dataset import SyntheticFlowDataset
from train_logging import JaxLogger, create_flow_figure_jax, log_kernels_to_tensorboard, log_gradients


# It's now clean and just takes the model as the first arg.
def loss_fn(model, img1_batch, img2_batch, flow_gt_batch):
    """Calculates the total loss (L1 + variance reward)."""

    # 1. Get model outputs
    flow_pred_batch = model(img1_batch, img2_batch)

    # 2. L1 Loss
    total_loss = jnp.mean(jnp.abs(flow_pred_batch - flow_gt_batch))

    metrics = {
        'total_loss': total_loss
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
    train_dataset = SyntheticFlowDataset()  # (Our V0.2 with noise)
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

        # --- Log Gradients ---
        log_gradients(grads, logger, epoch)

        # --- Log Kernel Images ---
        log_kernels_to_tensorboard(model, logger, epoch)

        # --- Log Prediction Snapshot ---
        # We just call the model (it's vmapped in the class)
        flow_pred_snapshot = model(img1_batch, img2_batch)

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
