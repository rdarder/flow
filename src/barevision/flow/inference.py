"""Inference script for optical flow estimation.

Loads a trained model from checkpoint and estimates flow between two images.

Usage:
    python -m barevision.flow.inference --checkpoint_path checkpoints/flow_20260317_143052/final \
                                        --image1 frame1.png \
                                        --image2 frame2.png \
                                        --output flow.npy
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from barevision.flow.checkpoint_utils import (
    config_from_checkpoint,
    restore_model_from_checkpoint,
)
from barevision.flow.training.model import Model as OpticalFlowModel


def load_image(path: Path, target_size: tuple[int, int]) -> jnp.ndarray:
    """Load and preprocess image for model input.

    Args:
        path: Path to image file
        target_size: (height, width) to resize to

    Returns:
        Preprocessed image array (1, H, W, 3) in range [0, 1]
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    img_array = np.array(img).astype(np.float32) / 255.0
    return jnp.expand_dims(img_array, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate optical flow between two images"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/flow_20260317_143052/final)",
    )
    parser.add_argument(
        "--image1", type=str, required=True, help="Path to first image (frame t)"
    )
    parser.add_argument(
        "--image2", type=str, required=True, help="Path to second image (frame t+k)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flow.npy",
        help="Output path for flow field (default: flow.npy)",
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Also save flow visualization as PNG",
    )

    args = parser.parse_args()

    # Load checkpoint configuration
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    config = config_from_checkpoint(checkpoint_path)

    # Extract model configuration
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config.get("training", {})

    print(f"Model configuration:")
    print(f"  - Window size: {model_config['window_size']}")
    print(f"  - Num levels: {model_config['num_levels']}")
    print(f"  - Embed dim: {model_config['embed_dim']}")
    print(f"  - Flow hidden dim: {model_config['flow_hidden_dim']}")

    # Handle img_size which may be a list
    img_size_list = dataset_config["img_size"]
    if isinstance(img_size_list, list):
        img_size = int(img_size_list[0])  # Assume square
    else:
        img_size = int(img_size_list)

    print(f"  - Expected input size: {img_size}x{img_size}")

    # Create model with same architecture
    # Use seed from checkpoint config for reproducibility (weights will be overwritten)
    # Try training.seed first (new checkpoints), fall back to dataset.seed (old checkpoints)
    model_seed = training_config.get("seed", dataset_config.get("seed", 42))
    model = OpticalFlowModel(
        hidden_dim=32,
        embed_dim=model_config["embed_dim"],
        num_groups=8,
        num_levels=model_config["num_levels"],
        flow_hidden_dim=model_config["flow_hidden_dim"],
        window_size=model_config["window_size"],
        rngs=nnx.Rngs(model_seed),
    )

    # Restore model weights
    step = restore_model_from_checkpoint(checkpoint_path, model)
    print(f"Restored model from step {step}")

    # Load and preprocess images
    print(f"\nLoading images (resizing to {img_size}x{img_size})...")
    img1 = load_image(Path(args.image1), (img_size, img_size))
    img2 = load_image(Path(args.image2), (img_size, img_size))

    # Run inference
    print("Running inference...")
    flow, pyramid1, pyramid2 = model(
        img1, img2, temperature=model_config["temperature"]
    )

    # Extract flow (remove batch dimension)
    flow_np = np.array(flow[0])
    print(f"\nFlow field shape: {flow_np.shape}")

    # Detailed flow statistics
    magnitude = np.linalg.norm(flow_np, axis=-1)
    u_component = flow_np[..., 0]
    v_component = flow_np[..., 1]

    print(f"\nFlow magnitude statistics:")
    print(f"  - Mean: {magnitude.mean():.4f}")
    print(f"  - Std:  {magnitude.std():.4f}")
    print(
        f"  - Max:  {magnitude.max():.4f} at pixel {np.unravel_index(magnitude.argmax(), magnitude.shape)}"
    )
    print(f"  - Min:  {magnitude.min():.4f}")

    print(f"\nFlow component statistics (normalized window coordinates):")
    print(f"  U component (horizontal):")
    print(f"    - Mean: {u_component.mean():.4f}")
    print(f"    - Std:  {u_component.std():.4f}")
    print(f"    - Range: [{u_component.min():.4f}, {u_component.max():.4f}]")
    print(f"  V component (vertical):")
    print(f"    - Mean: {v_component.mean():.4f}")
    print(f"    - Std:  {v_component.std():.4f}")
    print(f"    - Range: [{v_component.min():.4f}, {v_component.max():.4f}]")

    # Diagnostic: Check for bias in flow predictions
    print(f"\nFlow bias diagnostic:")
    print(
        f"  - Pixels with |flow| > 0.01: {(magnitude > 0.01).sum()} / {magnitude.size} ({100 * (magnitude > 0.01).sum() / magnitude.size:.1f}%)"
    )
    print(
        f"  - Pixels with |flow| > 0.1:  {(magnitude > 0.1).sum()} / {magnitude.size} ({100 * (magnitude > 0.1).sum() / magnitude.size:.1f}%)"
    )
    print(
        f"  - Pixels with |flow| > 0.3:  {(magnitude > 0.3).sum()} / {magnitude.size} ({100 * (magnitude > 0.3).sum() / magnitude.size:.1f}%)"
    )

    # Check if flow is suspiciously uniform (potential bias issue)
    flow_variance = magnitude.var()
    print(f"  - Flow variance: {flow_variance:.6f}")
    if flow_variance < 0.001 and magnitude.mean() > 0.05:
        print(f"  ⚠️  WARNING: Low variance with non-zero mean suggests model bias!")
        print(
            f"     This may indicate the FlowEstimator MLP has learned a biased output."
        )
        print(
            f"     Consider: adding bias initialization, output activation, or checking training data."
        )

    # Save flow field
    output_path = Path(args.output)
    np.save(output_path, flow_np)
    print(f"\nFlow saved to: {output_path}")

    # Optionally save visualization
    if args.save_viz:
        from barevision.flow.matching.visualization import (
            flow_to_arrows,
            flow_to_colorwheel,
        )

        # Colorwheel visualization with adaptive scaling for better contrast
        flow_viz = flow_to_colorwheel(flow_np, max_flow=0.3, adaptive=True)
        viz_path = output_path.with_suffix(".colorwheel.png")
        Image.fromarray((flow_viz * 255).astype(np.uint8)).save(viz_path)
        print(f"Colorwheel visualization saved to: {viz_path}")

        # Arrow visualization
        arrows_viz = flow_to_arrows(
            flow_np,
            max_flow=0.3,
            window_size=model_config["window_size"],
            grid_density=8,
        )
        arrows_path = output_path.with_suffix(".arrows.png")
        Image.fromarray(arrows_viz).save(arrows_path)
        print(f"Arrow visualization saved to: {arrows_path}")


if __name__ == "__main__":
    main()
