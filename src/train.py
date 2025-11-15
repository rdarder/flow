import jax
import jax.numpy as jnp
import optax
from flax import nnx
from torch.utils.data import DataLoader, Dataset
from model import BarebonesFlowModel
from synthetic_dataset import SyntheticFlowDataset
from train_logging import JaxLogger, log_kernels_to_tensorboard, log_gradients, create_diagnostic_figure_jax

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


if __name__ == "__main__":
    training = Training.default_build()
    training.run()