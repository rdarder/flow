import jax
import jax.numpy as jnp
from typing import List, Tuple
from pydantic import BaseModel, ConfigDict, Field
from barevision.utils.grid import (
    crop_to_grid_aligned,
    WindowGrid,
    generate_normalized_coordinates,
)


class SpatialVarianceLossConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    window_size: int = 16
    level_weight_decay: float = 1.1
    lambda_self: float = 0.6
    self_temperature: float = 0.25
    cross_temperature: float = 0.25


def _compute_spatial_variance(weights: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    # Optimized E[X²] - (E[X])² avoids O(N²) intermediate tensor (B, N, N, 2)
    mean_pos = jnp.einsum("bnk,kd->bnd", weights, coords)
    mean_sq = jnp.einsum("bnk,k->bn", weights, jnp.sum(coords**2, axis=-1))
    variance = mean_sq - jnp.sum(mean_pos**2, axis=-1)
    return jnp.maximum(variance, 0.0)


def self_attention_spatial_variance(windows, temperature, coords):
    B, H, W, D = windows.shape
    N = H * W
    flat = windows.reshape(B, N, D)
    logits = flat @ flat.transpose(0, 2, 1)
    weights = jax.nn.softmax(logits / temperature, axis=-1)
    variance = _compute_spatial_variance(weights, coords)
    return variance.reshape(B, H, W), weights


def cross_attention_spatial_variance(w1, w2, temperature, coords):
    B, H, W, D = w1.shape
    N = H * W
    f1, f2 = w1.reshape(B, N, D), w2.reshape(B, N, D)
    logits = f1 @ f2.transpose(0, 2, 1)
    weights = jax.nn.softmax(logits / temperature, axis=-1)
    variance = _compute_spatial_variance(weights, coords)
    return variance.reshape(B, H, W), weights


def windowed_spatial_variance_losses(emb1, emb2, config, coords, need_aux=True):
    B, H, W, D = emb1.shape
    grid = WindowGrid(window_size=config.window_size)
    w1, w2 = grid.split(emb1), grid.split(emb2)

    num_windows = (H // config.window_size) * (W // config.window_size)
    fw1 = w1.reshape(B * num_windows, config.window_size, config.window_size, D)
    fw2 = w2.reshape(B * num_windows, config.window_size, config.window_size, D)

    v_self, weights_self = self_attention_spatial_variance(
        fw1, config.self_temperature, coords
    )
    v_cross, weights_cross = cross_attention_spatial_variance(
        fw1, fw2, config.cross_temperature, coords
    )

    def to_grid(v):
        v = v.reshape(
            B,
            H // config.window_size,
            W // config.window_size,
            config.window_size,
            config.window_size,
        )
        return v.transpose(0, 1, 3, 2, 4).reshape(B, H, W)

    v_self_grid, v_cross_grid = to_grid(v_self), to_grid(v_cross)
    l_self, l_cross = v_self_grid.mean(), v_cross_grid.mean()
    loss = (1 - config.lambda_self) * l_cross + config.lambda_self * l_self

    aux = {"self_loss": l_self, "cross_loss": l_cross}
    if need_aux:
        aux.update(
            {
                "self_attention_weights": weights_self,
                "cross_attention_weights": weights_cross,
                "self_variance_maps": v_self_grid,
                "cross_variance_maps": v_cross_grid,
            }
        )
    return loss, aux


def compute_hierarchical_spatial_variance_loss(
    pyramid1, pyramid2, config, need_aux=True
):
    # Precomputing coords once avoids L redundant calls inside level loop
    coords = generate_normalized_coordinates(config.window_size)
    total_loss = total_self = total_cross = total_weight = 0.0

    aux = {"level_losses": [], "level_weights": [], "self_loss": 0.0, "cross_loss": 0.0}
    if need_aux:
        aux.update(
            {
                k: []
                for k in [
                    "level_self_attention_weights",
                    "level_cross_attention_weights",
                    "level_self_variance_maps",
                    "level_cross_variance_maps",
                ]
            }
        )

    for i, (e1, e2) in enumerate(zip(pyramid1, pyramid2)):
        weight = config.level_weight_decay**i
        l, a = windowed_spatial_variance_losses(
            crop_to_grid_aligned(e1, config.window_size),
            crop_to_grid_aligned(e2, config.window_size),
            config,
            coords,
            need_aux,
        )
        total_loss += l * weight
        total_self += a["self_loss"] * weight
        total_cross += a["cross_loss"] * weight
        total_weight += weight

        aux["level_losses"].append(l * weight)
        aux["level_weights"].append(weight)
        if need_aux:
            aux["level_self_attention_weights"].append(a["self_attention_weights"])
            aux["level_cross_attention_weights"].append(a["cross_attention_weights"])
            aux["level_self_variance_maps"].append(a["self_variance_maps"])
            aux["level_cross_variance_maps"].append(a["cross_variance_maps"])

    aux["self_loss"] = total_self / total_weight
    aux["cross_loss"] = total_cross / total_weight
    return total_loss / total_weight, aux


class HierarchicalSpatialVarianceLoss:
    def __init__(self, config: SpatialVarianceLossConfig):
        self.config = config

    def __call__(
        self,
        pyramid_pair: Tuple[List[jnp.ndarray], List[jnp.ndarray]],
        need_aux: bool = True,
    ):
        return compute_hierarchical_spatial_variance_loss(
            pyramid_pair[0], pyramid_pair[1], self.config, need_aux
        )
