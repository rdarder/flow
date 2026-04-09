#!/usr/bin/env python3
"""Inspect the structure of aux data from a training step.

Run this to understand what's available in aux for logging/visualization.
"""

import jax.numpy as jnp
from flax import nnx
import optax

from barevision.config import RootConfig
from barevision.embeddings.model import HierarchicalEmbeddingModel
from barevision.embeddings.spatial_losses import HierarchicalSpatialVarianceLoss


def print_structure(obj, indent=0, max_depth=3, max_items=5):
    """Recursively print the structure of a nested object."""
    prefix = "  " * indent
    
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys:")
        if indent >= max_depth:
            print(f"{prefix}  ... (max depth reached)")
            return
        for i, (key, value) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj) - max_items} more keys)")
                break
            print(f"{prefix}  {key!r}:")
            print_structure(value, indent + 2, max_depth, max_items)
    
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} with {len(obj)} items:")
        if indent >= max_depth:
            print(f"{prefix}  ... (max depth reached)")
            return
        for i, item in enumerate(obj):
            if i >= max_items:
                print(f"{prefix}  ... ({len(obj) - max_items} more items)")
                break
            print(f"{prefix}  [{i}]:")
            print_structure(item, indent + 2, max_depth, max_items)
    
    elif hasattr(obj, "shape"):
        # JAX/numpy array
        print(f"{prefix}{type(obj).__name__}: shape={obj.shape}, dtype={obj.dtype}")
    
    else:
        print(f"{prefix}{type(obj).__name__}: {obj!r}")


def main():
    # Load minimal config for inspection
    config = RootConfig(**{
        "name": "inspect_aux",
        "model": {
            "levels": [
                {
                    "uib_configs": [
                        {
                            "in_channels": 3,
                            "out_channels": 8,
                            "expanded_channels": 16,
                            "use_dw_before_expand": True,
                            "use_dw_after_expand": True,
                            "downsample_after": False,
                            "use_l2_norm": False,
                        },
                        {
                            "in_channels": 8,
                            "out_channels": 16,
                            "expanded_channels": 32,
                            "use_dw_before_expand": True,
                            "use_dw_after_expand": True,
                            "downsample_after": False,
                            "use_l2_norm": True,
                        },
                    ]
                },
                {
                    "uib_configs": [
                        {
                            "in_channels": 16,
                            "out_channels": 16,
                            "expanded_channels": 32,
                            "use_dw_before_expand": True,
                            "use_dw_after_expand": False,
                            "downsample_after": True,
                            "use_l2_norm": True,
                        },
                    ]
                },
                {
                    "uib_configs": [
                        {
                            "in_channels": 16,
                            "out_channels": 16,
                            "expanded_channels": 32,
                            "use_dw_before_expand": True,
                            "use_dw_after_expand": False,
                            "downsample_after": True,
                            "use_l2_norm": True,
                        },
                    ]
                },
            ],
        },
        "dataset": {
            "batch_size": 2,
            "coarse_grid_size": 1,
            "window_size": 8,
            "num_levels": 3,
            "min_frame_distance": 1,
            "max_frame_distance": 3,
            "max_samples": -1,
            "frame_cache_max_mb": 100,
        },
        "loss": {
            "spatial_variance": {
                "window_size": 8,
                "level_weight_decay": 1.0,
                "lambda_self": 0.5,
                "self_temperature": 0.25,
                "cross_temperature": 0.25,
            }
        },
        "training": {
            "seed": 42,
            "epochs": 1,
            "learning_rate": 0.001,
        },
        "logging": {
            "tensorboard_dir": "/tmp/inspect_aux_tb",
            "every_steps": 100,
            "visualizations_every_steps": 100,
        },
        "checkpoint": {
            "every_epochs": 1,
            "location": "/tmp/inspect_aux_ckpt",
            "keep_best_n": 3,
        },
        "validation": {
            "every_epochs": 1,
        },
    })
    
    # Create model
    rngs = nnx.Rngs(config.training.seed)
    model = config.model.build_model(rngs=rngs)
    loss_fn_obj = HierarchicalSpatialVarianceLoss(config.loss.spatial_variance)
    
    # Create dummy batch
    img_size = config.model.target_to_input(
        config.dataset.coarse_grid_size,
        config.dataset.window_size,
    )
    img1 = jnp.ones((config.dataset.batch_size, *img_size, 3))
    img2 = jnp.ones((config.dataset.batch_size, *img_size, 3))
    
    # Run forward pass with aux
    pyramid1 = model(img1)
    pyramid2 = model(img2)
    loss, aux = loss_fn_obj((pyramid1, pyramid2), need_aux=True)
    
    print("=" * 60)
    print("AUX STRUCTURE FROM loss_fn_obj()")
    print("=" * 60)
    print_structure(aux, max_depth=3, max_items=10)
    
    # Also show what training.py adds
    print("\n" + "=" * 60)
    print("AUX STRUCTURE AS SEEN IN training.py (after _compute_loss_and_grads)")
    print("=" * 60)
    aux_with_extras = dict(aux)
    aux_with_extras["pyramids"] = (pyramid1, pyramid2)
    aux_with_extras["img_pair"] = (img1, img2)
    print_structure(aux_with_extras, max_depth=3, max_items=10)
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("- aux['pyramids']: (pyramid1, pyramid2) - tuple of lists, one per level")
    print("- aux['self_loss'], aux['cross_loss']: scalar loss components")
    print("- aux['level_losses'], aux['level_weights']: per-level breakdown")
    print("- aux['level_*_attention_weights']: attention weights per level (if need_aux)")
    print("- aux['level_*_variance_maps']: variance maps per level (if need_aux)")
    print("- aux['img_pair']: (img1, img2) - added by training.py for visualization")


if __name__ == "__main__":
    main()
