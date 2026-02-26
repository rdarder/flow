"""Training script for hierarchical optical flow model.

Uses tyro for CLI parsing with nested dataclass support.
Example usage:
    python train.py  # Uses all defaults
    python train.py --settings.model.num-levels 3 --settings.dataset.img-size 128
    python train.py --settings.training.epochs 10 --settings.training.steps-per-epoch 50
"""

import os
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import nnx
from torch.utils.data import DataLoader

from flow.checkpoint_manager import (
    AbstractCheckpointManager,
    create_checkpoint_manager,
)
from flow.hierarchical_model import HierarchicalFlowModel
from flow.logging_utils import (
    JaxLogger,
    log_gradient_histograms,
    log_parameter_histograms,
)
from flow.settings import (
    DatasetSettings,
    LoggingSettings,
    ModelSettings,
    Settings,
    TrainingSettings,
    VisualizationSettings,
)
from flow.synthetic_dataset import SyntheticFlowDataset
from flow.visualization import (
    create_blending_figure,
    create_components_figure,
    create_confidence_analysis_figure,
    create_overview_figure,
    create_pyramid_detail_figure,
)


# --- 1. Loss Functions ---
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


# --- 2. Training Steps ---
@nnx.jit
def train_step_fast(
    model: HierarchicalFlowModel,
    optimizer: nnx.Optimizer,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
):
    """Fast training step - JIT optimizes away unused aux values.

    This wrapper function doesn't return the aux dict, so JIT can optimize
    away its computation during training for better performance.
    """
    (loss, (flow_pred, _)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, img1, img2, flow_gt
    )
    optimizer.update(model, grads)
    return loss, flow_pred, grads


@nnx.jit
def train_step_with_aux(
    model: HierarchicalFlowModel,
    optimizer: nnx.Optimizer,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
):
    """Training step that returns full aux for visualization.

    Used once per epoch for logging detailed diagnostics.
    """
    (loss, (flow_pred, aux)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, img1, img2, flow_gt
    )
    optimizer.update(model, grads)
    return loss, flow_pred, grads, aux


# --- 3. Training Class ---
class Trainer:
    """Main training loop manager."""

    def __init__(
        self,
        model: HierarchicalFlowModel,
        settings: Settings,
        checkpoint_manager: AbstractCheckpointManager,
        start_epoch: int = 0,
        start_step: int = 0,
    ):
        self.model = model
        self.settings = settings
        self.checkpoint_manager = checkpoint_manager
        self.start_epoch = start_epoch
        self.global_step = start_step

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
            num_blobs_range=settings.dataset.num_blobs_range,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=settings.dataset.batch_size,
            shuffle=True,
            num_workers=settings.dataset.num_workers,
            drop_last=True,
        )

    def log_all_visualizations(
        self, epoch: int, img1: jnp.ndarray, img2: jnp.ndarray, flow_gt: jnp.ndarray
    ):
        """Log all visualization figures once per epoch with full intermediates.

        This method calls the model with return_intermediates=True to get
        detailed diagnostic information for all visualization views.

        Args:
            epoch: Current epoch number
            img1: First frame batch (B, H, W, C)
            img2: Second frame batch (B, H, W, C)
            flow_gt: Ground truth flow batch (B, H, W, 2)
        """
        try:
            # Get full aux with intermediates (not jitted, called once per epoch)
            flow_pred, aux = self.model(img1, img2, return_intermediates=True)

            # Convert to numpy for visualization
            img1_np = np.array(img1[0])
            img2_np = np.array(img2[0])
            flow_gt_np = np.array(flow_gt[0])
            flow_pred_np = np.array(flow_pred[0])

            # 1. Overview figure
            level_flows = {}
            if "level_flows" in aux:
                for i, flow in enumerate(aux["level_flows"]):
                    level_flows[f"Level {i}"] = np.array(flow[0])

            fig_overview = create_overview_figure(
                img1_np, img2_np, flow_gt_np, flow_pred_np, level_flows,
                flow_max_percent=self.settings.visualization.flow_max_percent
            )
            self.logger.log_figure("Visualization/Overview", fig_overview, epoch)

            # 2. Pyramid detail (if intermediates available)
            if "level_flows" in aux and "level_confidences" in aux:
                level_flows_dict = {
                    f"L{i}": np.array(f[0]) for i, f in enumerate(aux["level_flows"])
                }
                level_conf_dict = {
                    f"L{i}": np.array(c[0])
                    for i, c in enumerate(aux["level_confidences"])
                }
                fig_pyramid = create_pyramid_detail_figure(
                    level_flows_dict, level_conf_dict
                )
                self.logger.log_figure("Visualization/Pyramid", fig_pyramid, epoch)

            # 3. Blending analysis (if blend aux available)
            # The blend data is in level_aux[1] (second level) since blending happens
            # from coarse (level 0) into fine (level 1)
            if "level_aux" in aux and len(aux.get("level_flows", [])) >= 2:
                level_aux_list = aux["level_aux"]
                # Find a level that has blend data (usually level 1 for 2-level pyramid)
                blend_data = None
                blend_level_idx = None
                for idx, level_aux in enumerate(level_aux_list):
                    if "blend" in level_aux:
                        blend_data = level_aux["blend"]
                        blend_level_idx = idx
                        break

                if blend_data is not None:
                    # For 2-level: coarse_idx = 0, fine_idx = 1
                    coarse_idx = blend_level_idx - 1 if blend_level_idx > 0 else 0
                    fine_idx = blend_level_idx

                    fig_blending = create_blending_figure(
                        flow_fine=np.array(aux["level_flows"][fine_idx][0]),
                        conf_fine=np.array(aux["level_confidences"][fine_idx][0]),
                        flow_coarse_upsampled=np.array(
                            blend_data["flow_coarse_upsampled"][0]
                        ),
                        conf_coarse_upsampled=np.array(
                            blend_data["conf_coarse_upsampled"][0]
                        ),
                        weight_fine=np.array(blend_data["weight_fine"][0]),
                        weight_coarse=np.array(blend_data["weight_coarse"][0]),
                        flow_final=flow_pred_np,
                        flow_gt=flow_gt_np,
                    )
                    self.logger.log_figure(
                        "Visualization/Blending", fig_blending, epoch
                    )

            # 4. Components figure (if window_flow aux available)
            if "level_aux" in aux:
                # Extract from first level's window_flow aux
                level0_aux = aux["level_aux"][0]
                if "flow_lookup" in level0_aux:
                    fig_components = create_components_figure(
                        flow_lookup=np.array(level0_aux["flow_lookup"][0]),
                        flow_peer=np.array(level0_aux["flow_peer"][0]),
                        conf_lookup=np.array(level0_aux["conf_lookup"][0]),
                        conf_peer=np.array(level0_aux["conf_peer"][0]),
                        flow_blended=np.array(aux["level_flows"][0][0]),
                        conf_blended=np.array(aux["level_confidences"][0][0]),
                    )
                    self.logger.log_figure(
                        "Visualization/Components", fig_components, epoch
                    )

            # 5. Confidence analysis
            fig_confidence = create_confidence_analysis_figure(
                flow_pred_np, flow_gt_np, np.array(aux["confidence"][0])
            )
            self.logger.log_figure("Visualization/Confidence", fig_confidence, epoch)

        except Exception as e:
            print(f"Error logging visualizations: {e}")
            import traceback

            traceback.print_exc()

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch. Returns average loss."""
        epoch_loss = 0.0
        batch_count = 0

        # Store last batch and gradients for visualization
        last_batch = None
        last_grads = None

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

            # Training step (fast path - JIT optimizes away aux)
            loss, flow_pred, grads = train_step_fast(
                self.model, self.optimizer, img1, img2, flow_gt
            )

            # Track metrics
            epoch_loss += float(loss)
            batch_count += 1
            self.global_step += 1

            # Log step-level metrics
            if i % self.settings.training.log_every_steps == 0:
                self.logger.log_scalar("Loss/train_step", float(loss), self.global_step)

            # Save checkpoint if the manager says we should (and step > 0)
            if self.global_step > 0 and self.checkpoint_manager.should_save(
                self.global_step
            ):
                self.checkpoint_manager.save(
                    step=self.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                )

            # Store for visualization and histograms
            last_batch = (img1, img2, flow_gt)
            last_grads = grads

        # Compute average loss
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0

        # Log epoch-level metrics and diagnostics
        self.logger.log_scalar("Loss/train_epoch", avg_loss, epoch)

        if last_batch is not None:
            img1, img2, flow_gt = last_batch

            # Log gradient and parameter histograms
            if last_grads is not None:
                log_gradient_histograms(self.logger, self.model, self.global_step)
                log_parameter_histograms(self.logger, self.model, self.global_step)

            # Log all visualization figures (calls model with intermediates)
            self.log_all_visualizations(epoch, img1, img2, flow_gt)

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
        print(f"Checkpoints will be saved to: {self.settings.training.checkpoint_dir}")
        print(
            f"Checkpoint frequency: every {self.settings.training.checkpoint_freq} steps"
        )

        for epoch in range(self.start_epoch, self.settings.training.epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        # Save final checkpoint (only if we haven't saved at this step already)
        if self.checkpoint_manager.should_save(self.global_step):
            self.checkpoint_manager.save(
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.settings.training.epochs - 1,
            )

        self.checkpoint_manager.close()
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

    # Initialize optimizer for potential checkpoint loading
    if settings.training.grad_clip_norm > 0:
        optimizer_chain = optax.chain(
            optax.clip_by_global_norm(settings.training.grad_clip_norm),
            optax.adamw(settings.training.learning_rate),
        )
    else:
        optimizer_chain = optax.adamw(settings.training.learning_rate)
    optimizer = nnx.Optimizer(model, optimizer_chain, wrt=nnx.Param)

    # Create checkpoint manager using factory pattern
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=settings.training.checkpoint_dir,
        save_interval_steps=settings.training.checkpoint_freq,
        max_to_keep=settings.training.keep_last_n_checkpoints,
        enabled=settings.training.checkpoint_freq > 0,
    )

    # Handle checkpoint loading/resume
    start_epoch = 0
    start_step = 0

    if settings.training.resume:
        # Resume from latest checkpoint in checkpoint_dir
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Resuming from checkpoint at step {latest_step}")
            start_epoch, start_step = checkpoint_manager.restore(
                model=model,
                optimizer=optimizer,
            )
        else:
            print("Warning: No checkpoint found to resume from")
            print("Starting fresh training...")

    # Create trainer and run
    trainer = Trainer(
        model,
        settings,
        checkpoint_manager,
        start_epoch=start_epoch,
        start_step=start_step,
    )
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
            checkpoint_freq=5,  # Save every 5 steps for testing
            checkpoint_dir="test_checkpoints",
            keep_last_n_checkpoints=2,
            grad_clip_norm=0.0,
            seed=42,
            resume=False,
        ),
        logging=LoggingSettings(
            log_dir="runs",
            run_name_prefix="smoke_test",
            num_visualization_samples=2,
            log_views=("overview",),
        ),
        visualization=VisualizationSettings(
            flow_max_percent=0.1,
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
