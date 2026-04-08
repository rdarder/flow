"""FLOP counter for hierarchical embedding models.

Computes theoretical FLOPs for a given model configuration and input size.
Useful for architecture exploration and optimization.
"""

from dataclasses import dataclass
from typing import List, Tuple

from barevision.embeddings.model import (
    HierarchicalModelConfig,
    LevelConfig,
    UIBConfig,
)


@dataclass
class LayerFlops:
    """FLOP count for a single layer."""

    name: str
    input_size: Tuple[int, int, int]  # (H, W, C)
    output_size: Tuple[int, int, int]  # (H, W, C)
    flops: int

    @property
    def flops_m(self) -> float:
        """FLOPs in millions."""
        return self.flops / 1_000_000


@dataclass
class BlockFlops:
    """FLOP count for a UIB block."""

    block_name: str
    layers: List[LayerFlops]

    @property
    def total_flops(self) -> int:
        return sum(layer.flops for layer in self.layers)

    @property
    def total_flops_m(self) -> float:
        return self.total_flops / 1_000_000


@dataclass
class LevelFlops:
    """FLOP count for a pyramid level."""

    level_name: str
    blocks: List[BlockFlops]

    @property
    def total_flops(self) -> int:
        return sum(block.total_flops for block in self.blocks)

    @property
    def total_flops_m(self) -> float:
        return self.total_flops / 1_000_000


@dataclass
class FlopsReport:
    """Complete FLOP report for a model."""

    input_size: Tuple[int, int, int]  # (H, W, C)
    levels: List[LevelFlops]

    @property
    def total_flops(self) -> int:
        return sum(level.total_flops for level in self.levels)

    @property
    def total_flops_m(self) -> float:
        return self.total_flops / 1_000_000

    @property
    def total_flops_g(self) -> float:
        return self.total_flops / 1_000_000_000


def _conv_flops(
    h: int,
    w: int,
    c_in: int,
    c_out: int,
    kernel_size: int,
    stride: int = 1,
    groups: int = 1,
) -> int:
    """Compute FLOPs for a convolution layer.

    FLOPs = 2 * kernel_volume * (C_in / groups) * C_out * output_spatial_size

    Args:
        h: Input height
        w: Input width
        c_in: Input channels
        c_out: Output channels
        kernel_size: Square kernel size
        stride: Convolution stride
        groups: Group convolution factor (c_in for depthwise)

    Returns:
        Total FLOPs (multiply + add counted separately)
    """
    out_h = (h - kernel_size) // stride + 1
    out_w = (w - kernel_size) // stride + 1
    if out_h <= 0 or out_w <= 0:
        return 0

    kernel_volume = kernel_size * kernel_size
    output_size = out_h * out_w

    # 2 for multiply + add
    return 2 * kernel_volume * (c_in // groups) * c_out * output_size


def _l2_norm_flops(h: int, w: int, c: int) -> int:
    """Compute FLOPs for L2 normalization.

    Per position: square (C), sum (C-1), sqrt (1), divide (C)
    ≈ 4C FLOPs per spatial position

    Args:
        h: Height
        w: Width
        c: Channels

    Returns:
        Total FLOPs
    """
    return 4 * c * h * w


def _uib_output_size(h: int, w: int, config: UIBConfig) -> Tuple[int, int]:
    """Calculate output spatial size for a UIB block."""
    out_h, out_w = h, w

    if config.use_dw_before_expand:
        out_h -= 2
        out_w -= 2
    if config.use_dw_after_expand:
        out_h -= 2
        out_w -= 2
    if config.downsample_after:
        out_h = (out_h - 1) // 2
        out_w = (out_w - 1) // 2

    return max(1, out_h), max(1, out_w)


def _compute_uib_flops(
    h: int, w: int, c_in: int, config: UIBConfig, block_idx: int
) -> Tuple[BlockFlops, int, int, int]:
    """Compute FLOPs for a UIB block.

    Args:
        h: Input height
        w: Input width
        c_in: Input channels
        config: UIB configuration
        block_idx: Block index for naming

    Returns:
        Tuple of (BlockFlops, output_h, output_w, output_c)
    """
    layers = []
    curr_h, curr_w, curr_c = h, w, c_in

    # DW before expand
    if config.use_dw_before_expand:
        out_h, out_w = curr_h - 2, curr_w - 2
        flops = _conv_flops(curr_h, curr_w, curr_c, curr_c, 3, groups=curr_c)
        layers.append(
            LayerFlops(
                name=f"DW_before (3×3, C={curr_c})",
                input_size=(curr_h, curr_w, curr_c),
                output_size=(out_h, out_w, curr_c),
                flops=flops,
            )
        )
        curr_h, curr_w = out_h, out_w

    # PW expand
    out_h, out_w = curr_h - 0, curr_w - 0  # 1×1 conv, no spatial change
    flops = _conv_flops(curr_h, curr_w, curr_c, config.expanded_channels, 1)
    layers.append(
        LayerFlops(
            name=f"PW_expand (1×1, {curr_c}→{config.expanded_channels})",
            input_size=(curr_h, curr_w, curr_c),
            output_size=(out_h, out_w, config.expanded_channels),
            flops=flops,
        )
    )
    curr_h, curr_w = out_h, out_w
    curr_c = config.expanded_channels

    # DW after expand
    if config.use_dw_after_expand:
        out_h, out_w = curr_h - 2, curr_w - 2
        flops = _conv_flops(curr_h, curr_w, curr_c, curr_c, 3, groups=curr_c)
        layers.append(
            LayerFlops(
                name=f"DW_after (3×3, C={curr_c})",
                input_size=(curr_h, curr_w, curr_c),
                output_size=(out_h, out_w, curr_c),
                flops=flops,
            )
        )
        curr_h, curr_w = out_h, out_w

    # PW compress
    out_h, out_w = curr_h - 0, curr_w - 0
    flops = _conv_flops(curr_h, curr_w, curr_c, config.out_channels, 1)
    layers.append(
        LayerFlops(
            name=f"PW_compress (1×1, {curr_c}→{config.out_channels})",
            input_size=(curr_h, curr_w, curr_c),
            output_size=(out_h, out_w, config.out_channels),
            flops=flops,
        )
    )
    curr_h, curr_w = out_h, out_w
    curr_c = config.out_channels

    # Downsample
    if config.downsample_after:
        out_h, out_w = (curr_h - 1) // 2, (curr_w - 1) // 2
        flops = _conv_flops(curr_h, curr_w, curr_c, curr_c, 3, stride=2, groups=curr_c)
        layers.append(
            LayerFlops(
                name=f"Downsample (3×3 s=2, C={curr_c})",
                input_size=(curr_h, curr_w, curr_c),
                output_size=(out_h, out_w, curr_c),
                flops=flops,
            )
        )
        curr_h, curr_w = out_h, out_w

    # L2 norm
    if config.use_l2_norm:
        flops = _l2_norm_flops(curr_h, curr_w, curr_c)
        layers.append(
            LayerFlops(
                name=f"L2_norm (C={curr_c})",
                input_size=(curr_h, curr_w, curr_c),
                output_size=(curr_h, curr_w, curr_c),
                flops=flops,
            )
        )

    block_name = f"UIB_{block_idx}"
    block_flops = BlockFlops(block_name=block_name, layers=layers)

    return block_flops, curr_h, curr_w, curr_c


def _compute_level_flops(
    h: int, w: int, c_in: int, config: LevelConfig, level_idx: int
) -> Tuple[LevelFlops, int, int, int]:
    """Compute FLOPs for a pyramid level.

    Args:
        h: Input height
        w: Input width
        c_in: Input channels
        config: Level configuration
        level_idx: Level index for naming

    Returns:
        Tuple of (LevelFlops, output_h, output_w, output_c)
    """
    blocks = []
    curr_h, curr_w, curr_c = h, w, c_in

    for block_idx, uib_config in enumerate(config.uib_configs):
        block_flops, curr_h, curr_w, curr_c = _compute_uib_flops(
            curr_h, curr_w, curr_c, uib_config, block_idx
        )
        blocks.append(block_flops)

    level_name = f"Level_{level_idx}"
    level_flops = LevelFlops(level_name=level_name, blocks=blocks)

    return level_flops, curr_h, curr_w, curr_c


def compute_flops(
    config: HierarchicalModelConfig,
    input_h: int,
    input_w: int,
    input_c: int = 3,
) -> FlopsReport:
    """Compute FLOPs for a hierarchical embedding model.

    Args:
        config: Model configuration
        input_h: Input height
        input_w: Input width
        input_c: Input channels (default 3 for RGB)

    Returns:
        FlopsReport with detailed breakdown
    """
    levels = []
    curr_h, curr_w, curr_c = input_h, input_w, input_c

    for level_idx, level_config in enumerate(config.levels):
        level_flops, curr_h, curr_w, curr_c = _compute_level_flops(
            curr_h, curr_w, curr_c, level_config, level_idx
        )
        levels.append(level_flops)

    return FlopsReport(
        input_size=(input_h, input_w, input_c),
        levels=levels,
    )


def print_flops_report(report: FlopsReport) -> None:
    """Print a formatted FLOPs report.

    Args:
        report: FLOPs report to print
    """
    print("=" * 70)
    print("FLOPs Report - Hierarchical Embedding Model")
    print("=" * 70)
    print(f"Input size: {report.input_size[0]}×{report.input_size[1]}×{report.input_size[2]}")
    print(f"Total FLOPs: {report.total_flops_g:.3f}G ({report.total_flops_m:.2f}M)")
    print()

    for level in report.levels:
        print(f"{level.level_name}: {level.total_flops_m:.2f}M FLOPs")
        print("-" * 50)

        for block in level.blocks:
            print(f"  {block.block_name}: {block.total_flops_m:.2f}M")

            for layer in block.layers:
                h_in, w_in, c_in = layer.input_size
                h_out, w_out, c_out = layer.output_size
                print(
                    f"    {layer.name:35s} "
                    f"{h_in}×{w_in}×{c_in} → {h_out}×{w_out}×{c_out} "
                    f"{layer.flops_m:6.2f}M"
                )

        print()

    # Summary by level
    print("=" * 70)
    print("Summary by Level")
    print("=" * 70)

    total = report.total_flops
    for level in report.levels:
        pct = 100 * level.total_flops / total
        print(f"{level.level_name:10s} {level.total_flops_m:8.2f}M  ({pct:5.1f}%)")

    print(f"{'TOTAL':10s} {report.total_flops_m:8.2f}M  (100.0%)")
    print()

    # Summary by operation type
    print("=" * 70)
    print("Summary by Operation Type")
    print("=" * 70)

    dw_flops = 0
    pw_flops = 0
    downsample_flops = 0
    l2_flops = 0

    for level in report.levels:
        for block in level.blocks:
            for layer in block.layers:
                if "DW" in layer.name and "Downsample" not in layer.name:
                    dw_flops += layer.flops
                elif "PW" in layer.name:
                    pw_flops += layer.flops
                elif "Downsample" in layer.name:
                    downsample_flops += layer.flops
                elif "L2" in layer.name:
                    l2_flops += layer.flops

    for name, flops in [
        ("DW 3×3", dw_flops),
        ("PW 1×1", pw_flops),
        ("Downsample", downsample_flops),
        ("L2 norm", l2_flops),
    ]:
        pct = 100 * flops / total
        print(f"{name:15s} {flops / 1_000_000:8.2f}M  ({pct:5.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    from barevision.embeddings.model import make_default_model_config

    # Default model with 167×167 input
    config = make_default_model_config()
    report = compute_flops(config, input_h=167, input_w=167)
    print_flops_report(report)
