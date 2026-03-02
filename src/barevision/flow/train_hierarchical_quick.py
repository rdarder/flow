"""Quick training test for hierarchical flow model.

This is a minimal training script to verify the hierarchical model
works end-to-end. For full training, use train.py instead.
"""

import argparse
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from .hierarchical_model import HierarchicalFlowModel
from .synthetic_dataset import SyntheticFlowDataset


def train_step(model, optimizer, batch):
    """Single training step."""
    img1, img2, flow_gt = batch

    def loss_fn(model):
        flow_pred, aux = model(img1, img2)

        # The model output is at finest pyramid resolution (e.g., 32x32 for 64x64 input)
        # Need to downsample ground truth to match
        B, H_gt, W_gt, C = flow_gt.shape
        H_pred, W_pred = flow_pred.shape[1:3]

        if (H_gt, W_gt) != (H_pred, W_pred):
            # Downsample GT to match prediction
            # Simple average pooling by reshaping
            flow_gt_down = flow_gt.reshape(
                B, H_pred, H_gt // H_pred, W_pred, W_gt // W_pred, C
            )
            flow_gt_down = flow_gt_down.mean(axis=(2, 4))
        else:
            flow_gt_down = flow_gt

        # Simple L2 loss on flow
        loss = jnp.mean((flow_pred - flow_gt_down) ** 2)

        return loss, {"flow_pred": flow_pred, "aux": aux}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)

    return loss, metrics


def quick_test(
    num_steps: int = 10,
    batch_size: int = 2,
    img_size: int = 64,
    num_levels: int = 2,
):
    """Run a quick training test."""
    print(f"\n{'='*60}")
    print(f"Quick Training Test - Hierarchical Flow Model")
    print(f"{'='*60}")
    print(f"Config: img_size={img_size}, num_levels={num_levels}")
    print(f"Steps: {num_steps}, batch_size: {batch_size}")
    print(f"{'='*60}\n")

    # Create model
    rngs = nnx.Rngs(0)
    model = HierarchicalFlowModel(
        num_levels=num_levels,
        embed_dim=16,
        in_channels=3,  # RGB images from dataset
        rngs=rngs,
    )

    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)

    # Create dataset
    dataset = SyntheticFlowDataset(
        img_size=img_size,
        length=batch_size * num_steps,
        max_flow=5,  # Must be int
    )

    print(f"Dataset: {len(dataset)} samples")
    print(f"\nStarting training...\n")

    # Training loop
    losses = []
    for step in range(num_steps):
        # Get batch - convert torch tensors to jax arrays
        start_idx = step * batch_size
        batch_torch = [dataset[start_idx + i] for i in range(batch_size)]

        # Convert torch tensors to jax arrays - dataset already returns HWC format
        batch = (
            jnp.stack([jnp.array(b[0].numpy()) for b in batch_torch]),
            jnp.stack([jnp.array(b[1].numpy()) for b in batch_torch]),
            jnp.stack([jnp.array(b[2].numpy()) for b in batch_torch]),
        )

        # Train step
        loss, metrics = train_step(model, optimizer, batch)
        losses.append(float(loss))

        if step % 5 == 0 or step == num_steps - 1:
            print(f"Step {step}: loss = {loss:.4f}")

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"{'='*60}\n")

    return model, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick training test")
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--img-size", type=int, default=64, help="Image size")
    parser.add_argument(
        "--num-levels", type=int, default=2, help="Number of pyramid levels"
    )

    args = parser.parse_args()

    model, losses = quick_test(
        num_steps=args.steps,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_levels=args.num_levels,
    )
