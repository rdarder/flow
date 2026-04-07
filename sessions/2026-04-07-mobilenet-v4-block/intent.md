# Changes: Establish a uniform, explorable architecture

## Goal
Recover the 30% loss gap from the recent simplification by adopting a MobileNet V4-inspired Universal Inverted Block (UIB) architecture, while maintaining a simpler, more uniform model structure that enables systematic exploration.

## Core Tension
**Expressiveness vs. Simplicity**: The previous model performed better but had confusing asymmetry (StemBlock had 2 feature convs, StandardBlocks had 1). The current model is uniform but lost spatial context building capacity.

**Configuration Complexity vs. Explorability**: Need to explore many architectural variations without the configuration system becoming its own burden.

## The Bet
A disciplined, uniform block structure (UIB) with configurable depth and expansion will recover the lost capacity while enabling cleaner experimentation across architectural variations.

## Constraints
- No grouped convolutions (NPU memory bandwidth concerns until benchmarked)
- FLOPs as rough comparison metric — care about 2x differences, not 10%
- JAX/JIT compatibility — configs used at construction time only
- Configuration system should be pragmatic, not hyper-generic — just enough to express experiments clearly
- Inference cost should remain reasonable

## Decisions
| Idea | Status | Why |
|------|--------|-----|
| UIB structure (0-2 DW convs + expand/compress) | ✅ Adopted | Matches MobileNet V4 disciplined approach; flexible exploration space |
| Downsampling inside UIB config | ✅ Adopted | Each UIB has `downsample_after` flag; simplifies dispatcher logic |
| YAML configs over CLI args | ✅ Adopted | Better for defining many experiments upfront; hierarchical structure |
| Pydantic for schema + validation | ✅ Adopted | Type hints = schema + IDE support; validation on load |
| Base + override config pattern | ❌ Rejected | Prefer self-contained configs per experiment; clearer tracking |
| Downsampling at beginning of level | ❌ Rejected | Wastes high-resolution input information |
| Mean subtraction | ⚠️ Deferred | Only accounted for 4% of performance; revisit later with wider receptive field |

## Unknowns / Ablation Questions
- Optimal DW conv count per UIB (0 vs 1 vs 2 at matched FLOPs)
- Optimal expansion ratio (2x vs 4x vs 6x at matched DW count)
- UIBs per level (1 UIB with 2 DWs vs 2 UIBs with 1 DW each)
- Downsampling position effect (after UIB 0 vs after UIB 1)
- Does UIB beat previous StemBlock at matched FLOPs?
- Optimal number of levels (2 vs 3 vs 4)

## Key Insights
- **Sum of capacity**: Multiple UIBs per level allows regulating compute independently from downsampling decisions
- **Spatial context**: More DW convs per UIB should recover the lost spatial context from the previous StemBlock
- **Uniform iteration**: A level is a pipeline of 1+ UIB blocks; each UIB decides whether it downsamples its output
- **Exploration strategy**: Define 5-10 distinct experiments upfront (coarse-to-fine), run all, react to results

## Numbers to Preserve
- 16D embeddings (existing)
- 3 pyramid levels (existing)

## Configuration Approach (Choice)
- YAML file with multiple named variants (self-contained, no inheritance)
- CLI accepts `--config` (path) and `--variant` (name) for experiment selection
- Pydantic schemas for UIB/Level/Model configs with validation on load
- Analytical FLOPs calculation reported per embedding (for rough comparison)

## References
- `brainstorm.md` — Full mental model, detailed spec, and exploration hypotheses
