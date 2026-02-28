"""Gradient complexity analyzer for the hierarchical optical flow model.

Analyzes gradient flow through the model architecture to understand:
- Which parameters receive gradients from which loss terms
- Gradient magnitudes per parameter group
- Effect of stop_gradient on hierarchical training
- Computational cost of gradients vs forward pass

Usage:
    python -m flow.gradient_analyzer
    python -m flow.gradient_analyzer --num-levels 3 --embed-dim 32
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flow.hierarchical_model import HierarchicalFlowModel
from flow.settings import ModelSettings


@dataclass
class GradientConfig:
    """Configuration for gradient analysis."""

    num_levels: int = 2
    embed_dim: int = 16
    in_channels: int = 3
    window_size: int = 16
    batch_size: int = 2
    img_size: Tuple[int, int] = (128, 128)
    seed: int = 42


@dataclass
class ParameterInfo:
    """Information about a parameter group."""

    name: str
    path: str
    shape: Tuple[int, ...]
    count: int
    module_type: str


@dataclass
class GradientStats:
    """Statistics for gradients of a parameter group."""

    param_name: str
    has_gradient: bool
    mean_abs: float
    std: float
    max_abs: float
    min_abs: float
    l2_norm: float
    loss_source: str  # "level_0", "level_1", "all_levels", "none"


def create_test_model(config: GradientConfig) -> HierarchicalFlowModel:
    """Create a HierarchicalFlowModel for testing.

    Args:
        config: Gradient analysis configuration

    Returns:
        Initialized model
    """
    rngs = nnx.Rngs(jax.random.PRNGKey(config.seed))
    model = HierarchicalFlowModel(
        num_levels=config.num_levels,
        embed_dim=config.embed_dim,
        in_channels=config.in_channels,
        window_size=config.window_size,
        auto_crop=True,
        rngs=rngs,
    )
    return model


def create_test_batch(
    config: GradientConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create synthetic test data.

    Args:
        config: Gradient analysis configuration

    Returns:
        (img1, img2, flow_gt) tensors
    """
    key = jax.random.PRNGKey(config.seed)
    key1, key2, key3 = jax.random.split(key, 3)

    h, w = config.img_size

    img1 = jax.random.normal(key1, (config.batch_size, h, w, config.in_channels))
    img2 = jax.random.normal(key2, (config.batch_size, h, w, config.in_channels))

    # Ground truth flow at finest resolution (model output resolution)
    # Model outputs at required_h // 2, required_w // 2
    required_h, required_w = model.required_h, model.required_w
    finest_h = required_h // 2
    finest_w = required_w // 2
    flow_gt = jax.random.normal(key3, (config.batch_size, finest_h, finest_w, 2)) * 0.1

    return img1, img2, flow_gt


def get_parameter_info(model: HierarchicalFlowModel) -> List[ParameterInfo]:
    """Extract parameter information from model.

    Args:
        model: HierarchicalFlowModel instance

    Returns:
        List of parameter information
    """
    params_info = []

    # Get all parameters using nnx.state - returns a State object with nested structure
    try:
        params_dict = nnx.state(model)

        # State is a dict-like object where values are either Params or nested States
        def extract_params(state_obj, path_prefix=""):
            for key, value in state_obj.items():
                current_path = f"{path_prefix}/{key}" if path_prefix else key

                if isinstance(value, nnx.Param):
                    # This is a parameter
                    path_str = current_path

                    # Determine module type from path
                    if "pyramid" in path_str:
                        module_type = "EmbeddingPyramid"
                    elif (
                        "grid_flow_estimator" in path_str
                        or "token_cross_attn" in path_str
                        or "token_self_attn" in path_str
                    ):
                        module_type = "GridFlowEstimator"
                    elif "spatial_score" in path_str:
                        module_type = "SpatialScore"
                    elif "prior_blender" in path_str:
                        module_type = "PriorBlender"
                    else:
                        module_type = "Unknown"

                    info = ParameterInfo(
                        name=key,
                        path=path_str,
                        shape=tuple(value.shape),
                        count=int(np.prod(value.shape)),
                        module_type=module_type,
                    )
                    params_info.append(info)
                elif hasattr(value, "items"):
                    # Nested State - recurse
                    extract_params(value, current_path)

        extract_params(params_dict)
    except Exception as e:
        print(f"Warning: Could not extract parameter info: {e}")
        # Fallback: try the old method
        try:
            params_dict = nnx.state(model, nnx.Param)
            for path, param in params_dict.items():
                path_str = "/".join(str(p) for p in path)

                if "pyramid" in path_str:
                    module_type = "EmbeddingPyramid"
                elif (
                    "grid_flow_estimator" in path_str
                    or "token_cross_attn" in path_str
                    or "token_self_attn" in path_str
                ):
                    module_type = "GridFlowEstimator"
                elif "spatial_score" in path_str:
                    module_type = "SpatialScore"
                elif "prior_blender" in path_str:
                    module_type = "PriorBlender"
                else:
                    module_type = "Unknown"

                info = ParameterInfo(
                    name=path[-1] if path else "unknown",
                    path=path_str,
                    shape=tuple(param.value.shape),
                    count=int(np.prod(param.value.shape)),
                    module_type=module_type,
                )
                params_info.append(info)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

    return params_info


def count_parameters_by_module(params_info: List[ParameterInfo]) -> Dict[str, int]:
    """Count parameters per module type.

    Args:
        params_info: List of parameter information

    Returns:
        Dictionary mapping module type to parameter count
    """
    counts = {}
    for info in params_info:
        counts[info.module_type] = counts.get(info.module_type, 0) + info.count
    return counts


def compute_level_specific_loss(
    model: HierarchicalFlowModel,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
    level_idx: int,
) -> jnp.ndarray:
    """Compute loss for a specific pyramid level only.

    Args:
        model: HierarchicalFlowModel
        img1: First frame
        img2: Second frame
        flow_gt: Ground truth flow
        level_idx: Which level to compute loss for (0 = coarsest)

    Returns:
        Loss value for that level
    """
    flow_pred, aux = model(img1, img2)

    if "level_flows" not in aux:
        raise ValueError("Model did not return intermediate level flows")

    level_flows = aux["level_flows"]
    if level_idx >= len(level_flows):
        raise ValueError(f"Level {level_idx} not found, only {len(level_flows)} levels")

    level_flow = level_flows[level_idx]

    # Downsample ground truth if needed
    if level_flow.shape[1:3] != flow_gt.shape[1:3]:
        target_h, target_w = level_flow.shape[1:3]
        scale_h = target_h / flow_gt.shape[1]
        scale_w = target_w / flow_gt.shape[2]
        flow_gt_scaled = flow_gt * jnp.array([scale_w, scale_h])

        from jax.image import resize

        flow_gt_down = resize(
            flow_gt_scaled,
            (flow_gt.shape[0], target_h, target_w, flow_gt.shape[-1]),
            method="bilinear",
        )
        flow_gt = flow_gt_down

    # Endpoint error loss
    epe = jnp.sqrt(jnp.sum((level_flow - flow_gt) ** 2, axis=-1) + 1e-8)
    loss = jnp.mean(epe)

    return loss


def analyze_gradients_for_level(
    model: HierarchicalFlowModel,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
    level_idx: int,
) -> Dict[str, Optional[jnp.ndarray]]:
    """Compute gradients for a specific level's loss.

    Args:
        model: HierarchicalFlowModel
        img1: First frame
        img2: Second frame
        flow_gt: Ground truth flow
        level_idx: Which level to compute loss for

    Returns:
        Dictionary mapping parameter paths to gradients
    """
    # Compute loss and gradients for this level
    loss, grads = nnx.value_and_grad(
        lambda m: compute_level_specific_loss(m, img1, img2, flow_gt, level_idx)
    )(model)

    return grads


def compute_gradient_stats(grads_dict: Dict[str, Any]) -> Dict[str, GradientStats]:
    """Compute statistics for gradients.

    Args:
        grads_dict: Dictionary mapping parameter paths to gradient arrays

    Returns:
        Dictionary mapping parameter paths to gradient statistics
    """
    stats = {}

    for path, grad in grads_dict.items():
        path_str = "/".join(str(p) for p in path)

        if grad is None:
            stats[path_str] = GradientStats(
                param_name=path_str,
                has_gradient=False,
                mean_abs=0.0,
                std=0.0,
                max_abs=0.0,
                min_abs=0.0,
                l2_norm=0.0,
                loss_source="none",
            )
        else:
            grad_flat = grad.flatten()
            abs_grad = jnp.abs(grad_flat)

            stats[path_str] = GradientStats(
                param_name=path_str,
                has_gradient=True,
                mean_abs=float(jnp.mean(abs_grad)),
                std=float(jnp.std(grad_flat)),
                max_abs=float(jnp.max(abs_grad)),
                min_abs=float(jnp.min(abs_grad)),
                l2_norm=float(jnp.linalg.norm(grad_flat)),
                loss_source="unknown",  # Will be filled in later
            )

    return stats


def measure_timing(
    model: HierarchicalFlowModel,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
    num_runs: int = 10,
) -> Dict[str, float]:
    """Measure forward pass and gradient computation time.

    Args:
        model: HierarchicalFlowModel
        img1: First frame
        img2: Second frame
        flow_gt: Ground truth flow
        num_runs: Number of runs for averaging

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    _ = model(img1, img2)
    loss, grads = nnx.value_and_grad(
        lambda m: compute_level_specific_loss(m, img1, img2, flow_gt, 0)
    )(model)
    jax.block_until_ready(loss)

    # Time forward pass
    forward_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(img1, img2)
        jax.block_until_ready(None)
        end = time.perf_counter()
        forward_times.append((end - start) * 1000)  # Convert to ms

    # Time gradient computation (level 0)
    grad_times_level0 = []
    for _ in range(num_runs):
        start = time.perf_counter()
        loss, grads = nnx.value_and_grad(
            lambda m: compute_level_specific_loss(m, img1, img2, flow_gt, 0)
        )(model)
        jax.block_until_ready(loss)
        end = time.perf_counter()
        grad_times_level0.append((end - start) * 1000)

    # Time gradient computation (level 1 - if exists)
    grad_times_level1 = []
    if model.num_levels > 1:
        for _ in range(num_runs):
            start = time.perf_counter()
            loss, grads = nnx.value_and_grad(
                lambda m: compute_level_specific_loss(m, img1, img2, flow_gt, 1)
            )(model)
            jax.block_until_ready(loss)
            end = time.perf_counter()
            grad_times_level1.append((end - start) * 1000)

    return {
        "forward_mean_ms": np.mean(forward_times),
        "forward_std_ms": np.std(forward_times),
        "grad_level0_mean_ms": np.mean(grad_times_level0),
        "grad_level0_std_ms": np.std(grad_times_level0),
        "grad_level1_mean_ms": np.mean(grad_times_level1) if grad_times_level1 else 0.0,
        "grad_level1_std_ms": np.std(grad_times_level1) if grad_times_level1 else 0.0,
    }


def analyze_stop_gradient_effect(
    model: HierarchicalFlowModel,
    img1: jnp.ndarray,
    img2: jnp.ndarray,
    flow_gt: jnp.ndarray,
) -> Dict[str, Any]:
    """Analyze the effect of stop_gradient on priors.

    This tests whether gradients flow back through the hierarchy by comparing
    gradients from Level 0 loss vs Level 1 loss.

    Args:
        model: HierarchicalFlowModel
        img1: First frame
        img2: Second frame
        flow_gt: Ground truth flow

    Returns:
        Analysis results dictionary
    """
    results = {
        "level_affected_params": {},
        "hierarchy_isolation": False,
        "analysis": [],
    }

    # Get gradients from Level 0 loss
    grads_level0 = analyze_gradients_for_level(model, img1, img2, flow_gt, 0)
    stats_level0 = compute_gradient_stats(grads_level0)

    # Get gradients from Level 1 loss (finest level)
    finest_level = model.num_levels - 1
    grads_level1 = analyze_gradients_for_level(model, img1, img2, flow_gt, finest_level)
    stats_level1 = compute_gradient_stats(grads_level1)

    # Analyze which parameters get gradients from which level
    pyramid_params_level0 = []
    pyramid_params_level1 = []
    grid_params_level0 = []
    grid_params_level1 = []

    for path_str, stats in stats_level0.items():
        if stats.has_gradient:
            if "pyramid" in path_str:
                pyramid_params_level0.append(path_str)
            elif "grid_flow" in path_str or "token_attn" in path_str:
                grid_params_level0.append(path_str)

    for path_str, stats in stats_level1.items():
        if stats.has_gradient:
            if "pyramid" in path_str:
                pyramid_params_level1.append(path_str)
            elif "grid_flow" in path_str or "token_attn" in path_str:
                grid_params_level1.append(path_str)

    # Check if pyramid parameters only get gradients from their corresponding level
    # Level 0 loss should primarily affect coarse pyramid layers
    # Level 1 loss should primarily affect fine pyramid layers

    results["level_affected_params"]["level_0"]["pyramid"] = pyramid_params_level0
    results["level_affected_params"]["level_0"]["grid_flow"] = grid_params_level0
    results["level_affected_params"][f"level_{finest_level}"][
        "pyramid"
    ] = pyramid_params_level1
    results["level_affected_params"][f"level_{finest_level}"][
        "grid_flow"
    ] = grid_params_level1

    # If stop_gradient is working, pyramid params should have limited overlap
    pyramid_overlap = set(pyramid_params_level0) & set(pyramid_params_level1)
    results["pyramid_param_overlap"] = len(pyramid_overlap)
    results["pyramid_overlap_ratio"] = len(pyramid_overlap) / max(
        len(pyramid_params_level0), 1
    )

    return results


def print_header(text: str, width: int = 80) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def print_section(text: str) -> None:
    """Print a section header."""
    print("\n" + "-" * 60)
    print(text)
    print("-" * 60)


def print_parameter_summary(params_info: List[ParameterInfo]) -> None:
    """Print summary of model parameters."""
    print_section("Parameter Count by Module")

    module_counts = count_parameters_by_module(params_info)
    total = sum(module_counts.values())

    print(f"{'Module Type':<30} {'Count':>12} {'Percentage':>12}")
    print("-" * 54)

    for module_type, count in sorted(module_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        print(f"{module_type:<30} {count:>12,} {pct:>11.1f}%")

    print("-" * 54)
    print(f"{'TOTAL':<30} {total:>12,} {'100.0%':>12}")

    # Print detailed breakdown
    print_section("Detailed Parameter List")
    print(f"{'Path':<50} {'Shape':<25} {'Count':>10}")
    print("-" * 85)

    for info in sorted(params_info, key=lambda x: x.path):
        shape_str = str(info.shape)
        print(f"{info.path:<50} {shape_str:<25} {info.count:>10,}")


def print_gradient_analysis(
    stats_level0: Dict[str, GradientStats],
    stats_level1: Dict[str, GradientStats],
    model: HierarchicalFlowModel,
) -> None:
    """Print gradient analysis report."""
    print_section("Gradient Analysis by Level")

    finest_level = model.num_levels - 1

    # Group by module type
    def group_by_module(stats_dict):
        groups = {}
        for path, stats in stats_dict.items():
            if "pyramid" in path:
                module = "Pyramid"
            elif "token_cross_attn" in path:
                module = "TokenCrossAttn"
            elif "token_self_attn" in path:
                module = "TokenSelfAttn"
            elif "spatial_score" in path:
                module = "SpatialScore"
            elif "prior_blender" in path:
                module = "PriorBlender"
            else:
                module = "Other"

            if module not in groups:
                groups[module] = []
            groups[module].append((path, stats))
        return groups

    groups_l0 = group_by_module(stats_level0)
    groups_l1 = group_by_module(stats_level1)

    # Print summary per module
    print(
        f"\n{'Module':<20} {'Level 0':<15} {'Level ' + str(finest_level):<15} {'Overlap'}"
    )
    print("-" * 60)

    all_modules = set(groups_l0.keys()) | set(groups_l1.keys())

    for module in sorted(all_modules):
        l0_has = sum(1 for _, s in groups_l0.get(module, []) if s.has_gradient)
        l0_total = len(groups_l0.get(module, []))
        l1_has = sum(1 for _, s in groups_l1.get(module, []) if s.has_gradient)
        l1_total = len(groups_l1.get(module, []))

        # Count overlap (parameters that get gradients from both)
        l0_paths = {p for p, s in groups_l0.get(module, []) if s.has_gradient}
        l1_paths = {p for p, s in groups_l1.get(module, []) if s.has_gradient}
        overlap = len(l0_paths & l1_paths)

        print(f"{module:<20} {l0_has}/{l0_total:<12} {l1_has}/{l1_total:<12} {overlap}")

    # Print gradient magnitude statistics
    print_section("Gradient Magnitude Statistics")

    def compute_module_stats(stats_dict):
        module_stats = {}
        for module in ["Pyramid", "TokenCrossAttn", "TokenSelfAttn", "SpatialScore"]:
            grads = [
                s.mean_abs
                for p, s in stats_dict.items()
                if module.lower() in p.lower() and s.has_gradient
            ]
            if grads:
                module_stats[module] = {
                    "mean": np.mean(grads),
                    "std": np.std(grads),
                    "max": np.max(grads),
                    "count": len(grads),
                }
        return module_stats

    stats_l0_by_module = compute_module_stats(stats_level0)
    stats_l1_by_module = compute_module_stats(stats_level1)

    print(
        f"\n{'Module':<20} {'Level':<8} {'Mean':>12} {'Std':>12} {'Max':>12} {'Count':>8}"
    )
    print("-" * 72)

    for module in sorted(stats_l0_by_module.keys()):
        s0 = stats_l0_by_module[module]
        print(
            f"{module:<20} {'0':<8} {s0['mean']:>12.6f} {s0['std']:>12.6f} {s0['max']:>12.6f} {s0['count']:>8}"
        )

        if module in stats_l1_by_module:
            s1 = stats_l1_by_module[module]
            print(
                f"{'':<20} {str(finest_level):<8} {s1['mean']:>12.6f} {s1['std']:>12.6f} {s1['max']:>12.6f} {s1['count']:>8}"
            )


def print_timing_report(timing: Dict[str, float]) -> None:
    """Print timing analysis report."""
    print_section("Computation Time Analysis")

    print(f"\n{'Operation':<30} {'Mean (ms)':>12} {'Std (ms)':>12}")
    print("-" * 54)

    print(
        f"{'Forward Pass':<30} {timing['forward_mean_ms']:>12.3f} {timing['forward_std_ms']:>12.3f}"
    )
    print(
        f"{'Gradient (Level 0)':<30} {timing['grad_level0_mean_ms']:>12.3f} {timing['grad_level0_std_ms']:>12.3f}"
    )

    if timing["grad_level1_mean_ms"] > 0:
        print(
            f"{'Gradient (Level 1)':<30} {timing['grad_level1_mean_ms']:>12.3f} {timing['grad_level1_std_ms']:>12.3f}"
        )

        # Compute overhead
        overhead_l0 = (
            (timing["grad_level0_mean_ms"] / timing["forward_mean_ms"]) - 1
        ) * 100
        overhead_l1 = (
            (timing["grad_level1_mean_ms"] / timing["forward_mean_ms"]) - 1
        ) * 100

        print(f"\n{'Gradient Overhead vs Forward Pass:'}")
        print(f"  Level 0: {overhead_l0:.1f}%")
        print(f"  Level 1: {overhead_l1:.1f}%")


def print_stop_gradient_analysis(
    stats_level0: Dict[str, GradientStats],
    stats_level1: Dict[str, GradientStats],
    model: HierarchicalFlowModel,
) -> None:
    """Print stop_gradient effect analysis."""
    print_section("Stop Gradient Effect Analysis")

    finest_level = model.num_levels - 1

    # Find parameters that only get gradients from specific levels
    pyramid_l0_only = []
    pyramid_l1_only = []
    pyramid_both = []
    grid_l0_only = []
    grid_l1_only = []
    grid_both = []

    for path_str in set(stats_level0.keys()) | set(stats_level1.keys()):
        has_l0 = stats_level0.get(
            path_str, GradientStats("", False, 0, 0, 0, 0, 0, "")
        ).has_gradient
        has_l1 = stats_level1.get(
            path_str, GradientStats("", False, 0, 0, 0, 0, 0, "")
        ).has_gradient

        if "pyramid" in path_str:
            if has_l0 and not has_l1:
                pyramid_l0_only.append(path_str)
            elif has_l1 and not has_l0:
                pyramid_l1_only.append(path_str)
            elif has_l0 and has_l1:
                pyramid_both.append(path_str)
        elif "grid_flow" in path_str or "token_attn" in path_str:
            if has_l0 and not has_l1:
                grid_l0_only.append(path_str)
            elif has_l1 and not has_l0:
                grid_l1_only.append(path_str)
            elif has_l0 and has_l1:
                grid_both.append(path_str)

    print(
        f"\n{'Parameter Group':<30} {'Level 0 Only':>15} {'Level {finest_level} Only':>15} {'Both Levels':>15}"
    )
    print("-" * 75)
    print(
        f"{'Pyramid Params':<30} {len(pyramid_l0_only):>15} {len(pyramid_l1_only):>15} {len(pyramid_both):>15}"
    )
    print(
        f"{'Grid Flow Params':<30} {len(grid_l0_only):>15} {len(grid_l1_only):>15} {len(grid_both):>15}"
    )

    # Interpretation
    print("\n" + "-" * 60)
    print("Interpretation:")
    print("-" * 60)

    if len(pyramid_both) == 0 and len(grid_both) == 0:
        print("✓ Stop-gradient is working: Parameters are isolated by level")
        print("  Each pyramid level trains independently")
    elif len(pyramid_both) < 3 and len(grid_both) < 3:
        print("⚠ Partial isolation: Most parameters are level-specific")
        print(
            f"  Only {len(pyramid_both)} pyramid and {len(grid_both)} grid params shared"
        )
    else:
        print(
            "✗ Weak isolation: Many parameters receive gradients from multiple levels"
        )
        print(
            f"  {len(pyramid_both)} pyramid and {len(grid_both)} grid params are shared"
        )
        print("  Consider adding stop_gradient between levels")

    # Level-specific parameter breakdown
    if pyramid_l0_only or pyramid_l1_only:
        print(f"\n{'Pyramid Level-Specific Parameters:'}")
        if pyramid_l0_only:
            print(f"  Level 0 (coarse): {len(pyramid_l0_only)} params")
            for p in pyramid_l0_only[:3]:  # Show first 3
                print(f"    - {p.split('/')[-1]}")
            if len(pyramid_l0_only) > 3:
                print(f"    ... and {len(pyramid_l0_only) - 3} more")

        if pyramid_l1_only:
            print(f"  Level {finest_level} (fine): {len(pyramid_l1_only)} params")
            for p in pyramid_l1_only[:3]:
                print(f"    - {p.split('/')[-1]}")
            if len(pyramid_l1_only) > 3:
                print(f"    ... and {len(pyramid_l1_only) - 3} more")


def run_analysis(config: Optional[GradientConfig] = None) -> Dict[str, Any]:
    """Run full gradient complexity analysis.

    Args:
        config: Optional configuration, uses defaults if not provided

    Returns:
        Analysis results dictionary
    """
    if config is None:
        config = GradientConfig()

    print_header("HIERARCHICAL FLOW MODEL - GRADIENT COMPLEXITY ANALYZER")

    print(f"Configuration:")
    print(f"  Num Levels: {config.num_levels}")
    print(f"  Embed Dim: {config.embed_dim}")
    print(f"  Window Size: {config.window_size}")
    print(f"  Image Size: {config.img_size}")
    print(f"  Batch Size: {config.batch_size}")

    # Create model and data
    print("\nInitializing model...")
    global model  # Make accessible for create_test_batch
    model = create_test_model(config)
    img1, img2, flow_gt = create_test_batch(config)

    print(f"Model requires input size: {model.required_h}x{model.required_w}")
    print(f"Ground truth flow shape: {flow_gt.shape}")

    # Get parameter info
    params_info = get_parameter_info(model)
    print_parameter_summary(params_info)

    # Compute gradients for each level
    print("\nComputing gradients...")
    grads_level0 = analyze_gradients_for_level(model, img1, img2, flow_gt, 0)
    stats_level0 = compute_gradient_stats(grads_level0)

    finest_level = config.num_levels - 1
    grads_level1 = analyze_gradients_for_level(model, img1, img2, flow_gt, finest_level)
    stats_level1 = compute_gradient_stats(grads_level1)

    print_gradient_analysis(stats_level0, stats_level1, model)

    # Stop gradient analysis
    print_stop_gradient_analysis(stats_level0, stats_level1, model)

    # Timing analysis
    print("\nMeasuring computation times...")
    timing = measure_timing(model, img1, img2, flow_gt, num_runs=10)
    print_timing_report(timing)

    # Summary
    print_header("ANALYSIS SUMMARY")

    total_params = sum(p.count for p in params_info)
    grad_params_l0 = sum(1 for s in stats_level0.values() if s.has_gradient)
    grad_params_l1 = sum(1 for s in stats_level1.values() if s.has_gradient)

    print(f"Total Parameters: {total_params:,}")
    print(f"Parameters with gradients (Level 0): {grad_params_l0}")
    print(f"Parameters with gradients (Level {finest_level}): {grad_params_l1}")
    print(f"Forward pass time: {timing['forward_mean_ms']:.2f} ms")
    print(f"Gradient time (Level 0): {timing['grad_level0_mean_ms']:.2f} ms")

    ratio = timing["grad_level0_mean_ms"] / timing["forward_mean_ms"]
    print(f"Gradient/Forward ratio: {ratio:.2f}x")

    return {
        "config": config,
        "total_params": total_params,
        "params_info": params_info,
        "stats_level0": stats_level0,
        "stats_level1": stats_level1,
        "timing": timing,
    }


def main():
    """Main entry point with CLI support."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze gradient complexity in hierarchical optical flow model"
    )
    parser.add_argument(
        "--num-levels", type=int, default=2, help="Number of pyramid levels"
    )
    parser.add_argument("--embed-dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument(
        "--window-size", type=int, default=16, help="Attention window size"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[128, 128],
        metavar=("H", "W"),
        help="Image size (height width)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for testing"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = GradientConfig(
        num_levels=args.num_levels,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    run_analysis(config)


if __name__ == "__main__":
    main()
