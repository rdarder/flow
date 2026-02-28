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

from flow.chairs_dataset_loader import ChairsSDHomDataset
from flow.checkpoint_manager import AbstractCheckpointManager, create_checkpoint_manager
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
from flow.visualization import (
    create_components_figure,
    create_confidence_analysis_figure,
    create_overview_figure,
    create_prior_effect_figure,
    create_pyramid_detail_figure,
)


# --- 0. Benchmarking Utilities ---
class StepsPerSecondTracker:
    """Tracks training speed in steps per second with running average.

    Simple wall-clock timer that reports steps/sec and ETA.
    """

    def __init__(self, log_interval: int = 10):
        """Initialize tracker.

        Args:
            log_interval: Report speed every N steps
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.start_time = None
        self.last_report_time = None

    def start(self):
        """Start timing. Call at beginning of training."""
        import time

        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time
        self.step_count = 0

    def step(self) -> Optional[Dict[str, float]]:
        """Record a completed step.

        Returns:
            Dictionary with metrics if it's time to report, else None
        """
        import time

        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            current_time = time.perf_counter()

            # Guard against timing before start() is called
            if self.start_time is None:
                return None

            elapsed_since_start = current_time - self.start_time

            if self.last_report_time is not None:
                elapsed_since_last = current_time - self.last_report_time
                steps_per_sec = (
                    self.log_interval / elapsed_since_last
                    if elapsed_since_last > 0
                    else 0.0
                )
                ms_per_step = (elapsed_since_last / self.log_interval) * 1000
            else:
                steps_per_sec = 0.0
                ms_per_step = 0.0

            overall_steps_per_sec = (
                self.step_count / elapsed_since_start
                if elapsed_since_start > 0
                else 0.0
            )

            self.last_report_time = current_time

            return {
                "steps": self.step_count,
                "steps_per_sec": steps_per_sec,
                "overall_steps_per_sec": overall_steps_per_sec,
                "elapsed_sec": elapsed_since_start,
                "ms_per_step": ms_per_step,
            }

        return None

    def get_eta(self, total_steps: int) -> Optional[str]:
        """Get estimated time to completion.

        Args:
            total_steps: Total steps expected

        Returns:
            ETA string or None if not enough data
        """
        import time

        if self.step_count < 5 or self.start_time is None:
            return None

        current_time = time.perf_counter()
        elapsed = current_time - self.start_time
        steps_per_sec = self.step_count / elapsed

        remaining_steps = total_steps - self.step_count
        remaining_sec = remaining_steps / steps_per_sec

        # Format as HH:MM:SS
        hours = int(remaining_sec // 3600)
        minutes = int((remaining_sec % 3600) // 60)
        seconds = int(remaining_sec % 60)

        if hours > 0:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        else:
            return f"{minutes}m {seconds:02d}s"


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
        self.train_dataset = ChairsSDHomDataset(
            root=settings.dataset.data_root,
            split=settings.dataset.split,
            target_size=settings.dataset.img_size,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=settings.dataset.batch_size,
            shuffle=True,
            num_workers=settings.dataset.num_workers,
            drop_last=True,
        )

        # Speed tracker for benchmarking
        self.speed_tracker = StepsPerSecondTracker(
            log_interval=settings.training.log_every_steps
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
            flow_pred, aux = self.model(img1, img2)

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
                img1_np,
                img2_np,
                flow_gt_np,
                flow_pred_np,
                level_flows,
                flow_max_percent=self.settings.visualization.flow_max_percent,
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

            # 3. Prior guidance visualization (for levels 1+)
            # Show how prior from coarser level guided the search
            if "level_aux" in aux and len(aux.get("level_flows", [])) >= 2:
                # Show prior effect for level 1 (first level with actual prior)
                for level_idx in range(1, len(aux["level_flows"])):
                    level_aux = aux["level_aux"][level_idx]
                    if "prior_flow" in level_aux:
                        fig_prior = create_prior_effect_figure(
                            prior_flow=np.array(level_aux["prior_flow"][0]),
                            level_flow=np.array(aux["level_flows"][level_idx][0]),
                            prior_confidence=np.array(level_aux["prior_confidence"][0]),
                            level_confidence=np.array(
                                aux["level_confidences"][level_idx][0]
                            ),
                            original_resolution=self.settings.dataset.img_size[
                                0
                            ],  # Use height
                            level_name=f"Level {level_idx}",
                            flow_max_percent=self.settings.visualization.flow_max_percent,
                        )
                        self.logger.log_figure(
                            f"Visualization/Prior_Effect/Level_{level_idx}",
                            fig_prior,
                            epoch,
                        )

            # 4. Components figure (if window_flow aux available)
            if "level_aux" in aux:
                # Extract from first level's window_flow aux (Level 0)
                level0_aux = aux["level_aux"][0]
                if "flow_lookup" in level0_aux and "flow_mixed" in level0_aux:
                    fig_components = create_components_figure(
                        flow_lookup=np.array(level0_aux["flow_lookup"][0]),
                        flow_peer=np.array(level0_aux["flow_peer"][0]),
                        conf_lookup=np.array(level0_aux["conf_lookup"][0]),
                        conf_peer=np.array(level0_aux["conf_peer"][0]),
                        flow_mixed=np.array(level0_aux["flow_mixed"][0]),
                        conf_mixed=np.array(level0_aux["conf_mixed"][0]),
                        original_resolution=self.settings.dataset.img_size[
                            0
                        ],  # Use height
                        level_name="Level 0",
                        flow_max_percent=self.settings.visualization.flow_max_percent,
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

        # Start timing for this epoch
        self.speed_tracker.start()

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

            # Track speed and log periodically
            speed_metrics = self.speed_tracker.step()
            if speed_metrics:
                # Log to TensorBoard
                self.logger.log_scalar(
                    "Speed/steps_per_sec",
                    speed_metrics["steps_per_sec"],
                    self.global_step,
                )
                self.logger.log_scalar(
                    "Speed/ms_per_step", speed_metrics["ms_per_step"], self.global_step
                )

                # Print to console
                eta = self.speed_tracker.get_eta(max_steps)
                eta_str = f", ETA: {eta}" if eta else ""
                print(
                    f"  Step {i}/{max_steps} | {speed_metrics['steps_per_sec']:.2f} steps/sec | {speed_metrics['ms_per_step']:.1f} ms/step{eta_str}"
                )

            # Log step-level loss metrics
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
    h, w = settings.dataset.img_size
    print(
        f"Model: {settings.model.num_levels} levels, embed_dim={settings.model.embed_dim}"
    )
    print(f"Dataset: {h}x{w} images")
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

    print(f"Model initialized (required size: {model.required_h}x{model.required_w})")

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
            img_size=(128, 128),  # Small resolution for smoke test
            data_root="datasets/ChairsSDHom/data",
            split="train",
            batch_size=4,
            num_workers=0,  # No multiprocessing for small runs
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
        # Parse any remaining CLI args to override smoke test defaults
        if len(sys.argv) > 1:
            settings = tyro.cli(Settings, default=settings)
        exit(main(settings))
    else:
        # Use tyro to parse CLI arguments into Settings
        # tyro automatically handles nested dataclasses with prefixed args
        # e.g., --settings.model.num-levels 3
        settings = tyro.cli(Settings)
        exit(main(settings))
