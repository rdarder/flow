# Changes: Embedding Model Simplification

## Goal
A simplified embedding model with clearer roles for each component, within 5-10% of baseline validation loss.

## Core Tension
**Simplicity vs. Training Stability:** We're removing normalization layers (GroupNorm, mean subtraction) that may have been compensating for architectural weaknesses. The new design relies on structural regularization (depthwise multiplier, grouped projection) instead. We won't know if this trade-off works until we train.

## The Bet
The network is shallow (3 blocks, ~9 conv layers). AdamW weight decay provides sufficient regularization, making GroupNorm unnecessary. The depthwise multiplier (8 filters per channel) and grouped projection provide enough inductive bias for stable training.

## Constraints
- **L2 normalization on output (REQUIRED)** — Without L2, embeddings grow to magnitude ~30, causing softmax to collapse to one-hot (single spot wins for all queries). This is immediate, not gradual.
- **VALID padding** — Resolution math: `(H - 3) // 2 + 1` per stride=2 DW layer
- **NPU-friendly ops** — Contiguous grouping (not interleaved), standard depthwise conv
- **~$10 NPU, 0.5 TOPS budget** — Drives compact ratio, grouped projection, depthwise multiplier choices
- **Within 5-10% of baseline** — Acceptable performance hit for simplicity gains

## Decisions
| Idea | Status | Why |
|------|--------|-----|
| L2 norm on output | ✅ Adopted | Prevents magnitude exploitation of spatial variance loss; collapse mode appears immediately without it |
| GroupNorm | ❌ Rejected (for now) | Shallow network; AdamW weight decay sufficient. Add back after `pw_compact` only if training diverges |
| Mean subtraction | ❌ Rejected | Was tied to `mean_conv` hack for downsampling; not needed with explicit DW stride=2 |
| Identical blocks | ✅ Adopted | No preprocessor, no special first block, no `mean_conv`. All blocks share same structure |
| Depthwise multiplier = 8× | ✅ Adopted | 8 spatial filters per compacted channel (32 total) — good balance between diversity and FLOPs |
| Grouped projection (4 groups) | ✅ Adopted | 4× FLOP savings vs. dense; each group is a "bucket" preserving combinatorial uniqueness |
| PW Compact (dense, 3→4) | ✅ Adopted | Full channel mixing before spatial filtering; ensures each DW channel sees all input information |
| Temperature = 0.25 | ✅ Adopted | Derived from first principles: `1/√16` for healthy softmax gradient scale |
| Inverted bottleneck | ❌ Rejected | Too expensive (~712HW vs. ~168HW); expand-at-full-res PW dominates FLOPs |
| Residual connections | ❌ Rejected | Don't make sense with changing resolutions; network is shallow (3 blocks) |

## Unknowns / Ablation Questions
- Does training converge without GroupNorm? (First 100-200 steps will show divergence/NaN)
- Optimal compact ratio: ablate 16→2, 16→4, 16→8 before DW
- Optimal DW multiplier: ablate 4×, 8×, 16× filters per channel
- Optimal project groups: ablate 1 (dense), 2, 4, 8
- Is first block compact 3→4 sufficient, or need 3→8?
- Does grouped projection hurt embedding uniqueness within/between frames?
- Can direction encode "uncertainty"? (Do textureless regions cluster in embedding space?)

## Key Insights
- **L2 is a constraint, not normalization:** The spatial variance loss can be cheated by growing embedding magnitude. Without L2, dot products reach ±14,400, softmax saturates, and a single spot wins for all queries → low variance → loss is happy. L2 closes this loophole.
- **L2 doesn't waste capacity:** Convolutions learn to map patches to *directions* in 16D space. Direction on unit sphere has 15 degrees of freedom. Magnitude is a nuisance variable for cosine similarity; constraining it focuses learning on what matters.
- **Bloom filter mental model:** Different regions activate different combinations of spatial filters. Combinatorial uniqueness: region A activates {filters 1, 5, 12}, region B activates {3, 7, 12}. Grouped projection preserves this "bucket" structure.
- **Temperature derivation:** For L2-normalized vectors in R^d, `Var[u·v] = 1/d`. For healthy softmax gradients (StdDev ≈ 1), temperature = `1/√d`. For d=16: temperature = 0.25.
- **Quantization-friendly:** L2 gives fixed bounds [-1, 1] for INT8 mapping. Variable magnitude would require dynamic scaling and outlier handling.

## Numbers to Preserve
- **16D embeddings** — Output dimension for all blocks
- **3 pyramid levels** — Block0, Block1, Block2 at successive resolutions
- **Temperature 0.25** — For 16D embeddings (1/√16)
- **L2 bounds [-1, 1]** — For quantization compatibility
- **Depthwise multiplier 8×** — 8 spatial filters per compacted channel
- **Project groups 4** — 4 parallel 8→4 convolutions

## References
- `./brainstorm.md` — Full mental model, detailed spec, false starts, and temperature derivation
