"""Training script for hierarchical optical flow model.

Uses tyro for CLI parsing with nested dataclass support.
Example usage:
    python train.py  # Uses all defaults
    python train.py --settings.model.num-levels 3 --settings.dataset.img-size 128
    python train.py --settings.training.epochs 10 --settings.training.steps-per-epoch 50
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import torchvision
import tyro
from flax import nnx
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from flow.hierarchical_model import HierarchicalFlowModel
from flow.settings import (
    DatasetSettings,
    LoggingSettings,
    ModelSettings,
    Settings,
    TrainingSettings,
)
from flow.synthetic_dataset import SyntheticFlowDataset


# --- 1. Logger Utility ---
class JaxLogger:
    """TensorBoard logger for training metrics and visualizations."""

    def __init__(self, log_dir: str = "runs", run_name_prefix: str = "flow"):
        run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(log_path)
        print(f"Logging to {log_path}")

    def log(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        try:
            value = float(value)
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            print(f"Logger Warning: {e}")

    def log_image(
        self, tag: str, image: np.ndarray, step: int, dataformats: str = "HWC"
    ):
        """Log an image."""
        try:
            self.writer.add_image(tag, image, step, dataformats=dataformats)
        except Exception as e:
            print(f"Logger Warning (image): {e}")

    def close(self):
        """Close the logger."""
        self.writer.close()


# --- 2. Visualization Utils ---
def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """Converts flow (H, W, 2) to RGB image."""
    H, W, C = flow.shape
    dx, dy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    h = (angle + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    if max_flow is None:
        max_mag = np.percentile(magnitude, 99.9)
    else:
        max_mag = max_flow
    v = np.clip(magnitude / (max_mag + 1e-6), 0, 1)
    hsv = np.stack([h, s, v], axis=-1)
    cmap = matplotlib.colormaps.get_cmap("hsv")
    rgb = cmap(hsv[..., 0])
    rgb[..., :3] *= v[..., np.newaxis]
    return rgb[..., :3]


def create_flow_comparison_figure(
    img1: np.ndarray,
    img2: np.ndarray,
    flow_gt: np.ndarray,
    flow_pred: np.ndarray,
    level_flows: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Creates a diagnostic figure showing inputs, ground truth, and predicted flow.

    For hierarchical model, optionally shows flow at each pyramid level.
    Handles shape mismatch by downsampling ground truth if needed.
    """
    # Handle shape mismatch - downsample ground truth if needed
    if flow_pred.shape[:2] != flow_gt.shape[:2]:
        from jax.image import resize

        target_h, target_w = flow_pred.shape[:2]
        scale_h = target_h / flow_gt.shape[0]
        scale_w = target_w / flow_gt.shape[1]
        # Scale flow values
        flow_gt_scaled = flow_gt * np.array([scale_w, scale_h])
        # Downsample
        flow_gt = np.array(
            resize(
                jnp.array(flow_gt_scaled),
                (target_h, target_w, flow_gt.shape[-1]),
                method="bilinear",
            )
        )

    # Determine grid size
    num_levels = len(level_flows) if level_flows else 0
    n_rows = max(2, 2 + (num_levels + 1) // 2)  # At least 2 rows
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif not isinstance(axes, np.ndarray):
        axes = np.array(axes).reshape(n_rows, n_cols)
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    # Row 1: Inputs and GT
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title("Frame 1")
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("Frame 2")
    axes[0, 2].imshow(flow_to_color(flow_gt))
    axes[0, 2].set_title("GT Flow")

    # Row 2: Predictions
    axes[1, 0].imshow(flow_to_color(flow_pred))
    axes[1, 0].set_title("Predicted Flow")

    # Error magnitude
    error = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=-1))
    im_err = axes[1, 1].imshow(error, cmap="hot")
    axes[1, 1].set_title("Flow Error")
    plt.colorbar(im_err, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Error histogram
    axes[1, 2].hist(error.flatten(), bins=50, color="blue", alpha=0.7)
    axes[1, 2].set_title(f"Error Distribution (mean={np.mean(error):.2f})")
    axes[1, 2].set_xlabel("Error magnitude")
    axes[1, 2].set_ylabel("Count")

    # Additional rows: Pyramid levels
    if level_flows:
        level_items = list(level_flows.items())
        for i, (level_name, level_flow) in enumerate(level_items):
            row = 2 + i // 3
            col = i % 3
            if row < n_rows and col < n_cols:
                axes[row, col].imshow(flow_to_color(level_flow))
                axes[row, col].set_title(f"Level: {level_name}")

    # Clean up axes
    for ax in axes.flat:
        ax.axis("off")

    # Convert to array for TensorBoard
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    # Get ARGB buffer from Agg backend
    buf = fig.canvas.tostring_argb()
    buffer = np.frombuffer(buf, dtype=np.uint8)
    # ARGB to RGB - skip alpha channel (first byte)
    image_array = buffer.reshape(height, width, 4)[:, :, 1:]
    plt.close(fig)

    return image_array


# --- 3. Loss Functions ---
def endpoint_error_loss(flow_pred: jnp.ndarray, flow_gt: jnp.ndarray) -> jnp.ndarray:
    """Endpoint error (EPE) loss - standard for optical flow.

    Handles shape mismatch by downsampling ground truth if needed.
    """
    # Check if shapes match, if not downsample ground truth
    if flow_pred.shape[1:3] != flow_gt.shape[1:3]:
        # Downsample ground truth to match prediction
        target_h, target_w = flow_pred.shape[1:3]
        scale_h = target_h / flow_gt.shape[1]
        scale_w = target_w / flow_gt.shape[2]
        # Scale flow values to match new resolution
        flow_gt_scaled = flow_gt * jnp.array([scale_w, scale_h])
        # Downsample using interpolation
        from jax.image import resize

        flow_gt_down = resize(
            flow_gt_scaled,
            (flow_gt.shape[0], target_h, target_w, flow_gt.shape[-1]),
            method="bilinear",
        )
        flow_gt = flow_gt_down

    return jnp.mean(jnp.sqrt(jnp.sum((flow_pred - flow_gt) ** 2, axis=-1) + 1e-8))


def loss_fn(
    model: HierarchicalFlowModel,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
):
    """Compute loss and auxiliary outputs."""
    flow_pred, aux = model(img1, img2)
    loss = endpoint_error_loss(flow_pred, flow_gt)
    return loss, (flow_pred, aux)


# --- 4. Training Step ---
@nnx.jit
def train_step(
    model: HierarchicalFlowModel,
    optimizer: nnx.Optimizer,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
):
    """Performs a single training step."""
    (loss, (flow_pred, aux)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, img1, img2, flow_gt
    )
    optimizer.update(model, grads)
    return loss, flow_pred, aux


# --- 5. Training Class ---
class Trainer:
    """Main training loop manager."""

    def __init__(self, model: HierarchicalFlowModel, settings: Settings):
        self.model = model
        self.settings = settings

        # Optimizer with optional gradient clipping
        if settings.training.grad_clip_norm > 0:
            optimizer_chain = optax.chain(
                optax.clip_by_global_norm(settings.training.grad_clip_norm),
                optax.adamw(settings.training.learning_rate),
            )
        else:
            optimizer_chain = optax.adamw(settings.training.learning_rate)

        self.optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

        # Logger
        self.logger = JaxLogger(
            log_dir=settings.logging.log_dir,
            run_name_prefix=settings.logging.run_name_prefix,
        )

        # Dataset and loader
        self.train_dataset = SyntheticFlowDataset(
            img_size=settings.dataset.img_size,
            length=settings.dataset.length,
            max_flow=settings.dataset.max_flow,
            blob_size_range=settings.dataset.blob_size_range,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=settings.dataset.batch_size,
            shuffle=True,
            num_workers=settings.dataset.num_workers,
            drop_last=True,
        )

        self.global_step = 0

    def log_visuals(
        self,
        epoch: int,
        img1: jnp.ndarray,
        img2: jnp.ndarray,
        flow_gt: jnp.ndarray,
        flow_pred: jnp.ndarray,
        aux: Dict[str, Any],
    ):
        """Log visualization figures."""
        try:
            # Extract first sample from batch
            img1_np = np.array(img1[0])
            img2_np = np.array(img2[0])
            flow_gt_np = np.array(flow_gt[0])
            flow_pred_np = np.array(flow_pred[0])

            # Get pyramid level flows if available
            level_flows = {}
            if "pyramid_flows" in aux:
                for level_name, level_flow in aux["pyramid_flows"].items():
                    level_flows[level_name] = np.array(level_flow[0])

            # Create and log figure
            fig = create_flow_comparison_figure(
                img1_np, img2_np, flow_gt_np, flow_pred_np, level_flows
            )
            self.logger.log_image("Flow/Comparison", fig, epoch, dataformats="HWC")

        except Exception as e:
            print(f"Error logging visuals: {e}")
            import traceback

            traceback.print_exc()

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        epoch_loss = 0.0
        batch_count = 0

        # Store last batch for visualization
        last_batch = None
        last_outputs = None

        # Determine number of steps
        max_steps = self.settings.training.steps_per_epoch
        if max_steps < 0:
            max_steps = len(self.train_loader)

        for i, (img1_pt, img2_pt, flow_gt_pt) in enumerate(self.train_loader):
            if i >= max_steps:
                break

            # Convert to JAX
            img1 = jnp.array(img1_pt.numpy())
            img2 = jnp.array(img2_pt.numpy())
            flow_gt = jnp.array(flow_gt_pt.numpy())

            # Training step
            loss, flow_pred, aux = train_step(
                self.model, self.optimizer, img1, img2, flow_gt
            )

            # Track metrics
            epoch_loss += float(loss)
            batch_count += 1
            self.global_step += 1

            # Log step-level metrics
            if i % self.settings.training.log_every_steps == 0:
                self.logger.log("Loss/train_step", float(loss), self.global_step)

            # Store for visualization
            last_batch = (img1, img2, flow_gt)
            last_outputs = (flow_pred, aux)

        # Compute average loss
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0

        # Log epoch-level metrics and visuals
        self.logger.log("Loss/train_epoch", avg_loss, epoch)

        if last_batch is not None and last_outputs is not None:
            img1, img2, flow_gt = last_batch
            flow_pred, aux = last_outputs
            self.log_visuals(epoch, img1, img2, flow_gt, flow_pred, aux)

        return avg_loss

    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.settings.training.epochs} epochs...")
        print(
            f"Dataset size: {len(self.train_dataset)}, Batch size: {self.settings.dataset.batch_size}"
        )
        print(
            f"Steps per epoch: {self.settings.training.steps_per_epoch if self.settings.training.steps_per_epoch > 0 else 'full dataset'}"
        )

        for epoch in range(self.settings.training.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        self.logger.close()
        print("Training complete!")


# --- 6. Main Entry Point ---
def main(settings: Settings):
    """Main training entry point with tyro CLI support.

    Args:
        settings: Complete experiment configuration (parsed from CLI by tyro)
    """
    # Validate settings
    is_valid, message = settings.validate()
    if not is_valid:
        print(f"Settings validation failed: {message}")
        print(
            f"Required image size for num_levels={settings.model.num_levels}: {settings.get_required_image_size()}"
        )
        print(f"Current img_size: {settings.dataset.img_size}")
        return 1

    print("Settings validated successfully!")
    print(
        f"Model: {settings.model.num_levels} levels, embed_dim={settings.model.embed_dim}"
    )
    print(f"Dataset: {settings.dataset.img_size}x{settings.dataset.img_size} images")
    print(
        f"Training: {settings.training.epochs} epochs, lr={settings.training.learning_rate}"
    )

    # Initialize RNG
    key = jax.random.PRNGKey(settings.training.seed)

    # Initialize model
    model = HierarchicalFlowModel(
        num_levels=settings.model.num_levels,
        embed_dim=settings.model.embed_dim,
        in_channels=settings.model.in_channels,
        window_size=settings.model.window_size,
        auto_crop=settings.model.auto_crop,
        rngs=nnx.Rngs(key),
    )

    print(
        f"Model initialized (required size: {model.required_size}x{model.required_size})"
    )

    # Create trainer and run
    trainer = Trainer(model, settings)
    trainer.train()

    return 0


def create_smoke_test_settings() -> Settings:
    """Create settings for a quick smoke test (fast execution)."""
    return Settings(
        model=ModelSettings(
            num_levels=2,
            embed_dim=8,  # Smaller for speed
            in_channels=3,
            window_size=16,
            auto_crop=True,
        ),
        dataset=DatasetSettings(
            img_size=64,
            length=100,  # Small dataset
            max_flow=5,
            batch_size=4,
            num_workers=0,  # No multiprocessing for small runs
            blob_size_range=(2, 6),
        ),
        training=TrainingSettings(
            learning_rate=1e-3,
            epochs=2,  # Just 2 epochs
            steps_per_epoch=10,  # Only 10 steps per epoch
            log_every_steps=5,
            checkpoint_freq=0,  # Disable checkpointing
            grad_clip_norm=0.0,
            seed=42,
        ),
        logging=LoggingSettings(
            log_dir="runs",
            run_name_prefix="smoke_test",
            num_visualization_samples=2,
            log_views=("overview",),
        ),
    )


if __name__ == "__main__":
    # Check for smoke test mode
    import sys

    if "--smoke-test" in sys.argv:
        sys.argv.remove("--smoke-test")
        settings = create_smoke_test_settings()
        exit(main(settings))
    else:
        # Use tyro to parse CLI arguments into Settings
        # tyro automatically handles nested dataclasses with prefixed args
        # e.g., --settings.model.num-levels 3
        settings = tyro.cli(Settings)
        exit(main(settings))
