import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from torch.utils.data import DataLoader, Dataset
from model import BarebonesFlowModel
from synthetic_dataset import SyntheticFlowDataset
from train_logging import JaxLogger, log_kernels_to_tensorboard, log_gradients, create_flow_figure_jax


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
        self.logger.log('params/zero_boost_radius', self.model.zero_boost_radius, epoch)
        self.logger.log('params/attn_temperature', self.model.attn_temperature, epoch)
        print(f"Epoch [{epoch + 1}/{self.epochs}] | Avg Loss: {avg_loss:.6f}")

    def log_gradients(self, gradients, epoch: int):
        log_gradients(gradients, self.logger, epoch)

    def log_conv_kernels(self, epoch: int):
        log_kernels_to_tensorboard(self.model, self.logger, epoch)

    def log_evaluation_sample(self, epoch: int):
        """Grabs one batch, runs inference, and logs a flow image."""
        try:
            # 1. Get one random batch from the *shuffled* loader
            raw_frame1, raw_frame2, flow_gt_pt = next(iter(self.train_loader))

            # 2. Convert to JAX arrays
            img1_batch = jnp.array(raw_frame1.numpy())
            img2_batch = jnp.array(raw_frame2.numpy())
            flow_gt_batch = jnp.array(flow_gt_pt.numpy())

            # 3. Run inference (forward pass only, no jit, no update)
            # We must pass the GT for the model's loss_fn to work,
            # even though we only want the prediction.
            dense_flow_pred, _ = self.model(img1_batch, img2_batch, flow_gt_batch)

            # 4. Create and log the figure
            fig = create_flow_figure_jax(
                img1_batch,
                img2_batch,
                flow_gt_batch,
                dense_flow_pred
            )
            self.logger.writer.add_figure('Flow_Comparison/epoch', fig, epoch)

        except Exception as e:
            print(f"Warning: Could not log evaluation sample. Error: {e}")

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

    LAMBDA_VAR = 0.  # 1e-3
    LAMBDA_COV = 0.  # 1e-3

    loss_components = aux['loss']
    total_loss = (
            loss_components['flow'] +
            (LAMBDA_VAR * loss_components['variance']) +
            (LAMBDA_COV * loss_components['covariance'])
    )
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


if __name__ == "__main__":
    training = Training.default_build()
    training.run()
