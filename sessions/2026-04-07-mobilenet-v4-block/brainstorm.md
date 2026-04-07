# Brainstorm: Embeddings Model UIB Refactor

## The Wish

Explore whether a **MobileNet V4-inspired Universal Inverted Block (UIB)** architecture can recover the 30% loss gap from the recent simplification, while maintaining a **simpler, more uniform** model structure.

**Success criteria:**
- Loss ≤ 0.8 (current is ~30% worse than previous baseline)
- Simpler model: fewer moving parts, easier to interpret, easier to change/compose
- Uniform structure across levels (no StemBlock vs StandardBlock asymmetry)

---

## Core Tension

**Expressiveness vs. Simplicity**: The previous model performed better but had confusing asymmetry (StemBlock had 2 feature convs, StandardBlocks had 1). The current model is uniform but lost spatial context building capacity.

**Configuration Complexity vs. Explorability**: Need to explore many architectural variations without the configuration system becoming its own burden.

---

## Explored Options

| Idea | Status | Why |
|------|--------|-----|
| **UIB structure** (0-2 DW convs + expand/compress) | ✅ Adopted | Matches MobileNet V4 disciplined approach; flexible exploration space |
| **Downsampling inside UIB config** | ✅ Adopted | Each UIB has `downsample_after` flag; simplifies dispatcher logic |
| **YAML configs over CLI args** | ✅ Adopted | Better for defining many experiments upfront; hierarchical structure |
| **Pydantic for schema + validation** | ✅ Adopted | Type hints = schema + IDE support; validation on load |
| **Grouped convolutions** | ❌ Rejected | MobileNet V4 warns about memory bandwidth issues on NPU |
| **Mean subtraction** | ⚠️ Deferred | Only accounted for 4% of performance; revisit later with wider receptive field |
| **Downsampling at beginning of level** | ❌ Rejected | Wastes high-resolution input information |
| **Base + override config pattern** | ❌ Rejected | Prefer self-contained configs per experiment; clearer tracking |

---

## Mental Models

### The UIB Block
```
Input: H×W×C
  ↓
[Optional DW 3×3]
  ↓
PW Expand: C → E×C (expansion_ratio)
  ↓
[Optional DW 3×3]
  ↓
PW Compress: E×C → C
  ↓
[Optional Downsample: 3×3 stride=2]
  ↓
Output: H'×W'×C
```

**Configurable per UIB:**
- `num_dw_convs`: 0, 1, or 2
- `expansion_ratio`: e.g., 2, 3, 4, 6
- `downsample_after`: bool

### The Level Block
A level is a **pipeline of 1+ UIB blocks**. Each UIB decides whether it downsamples its output. This allows:
- Multiple UIBs per level (regulate compute per level)
- Downsampling after any UIB (not just the last)
- Uniform iteration: "for each UIB, run it, optionally downsample"

### The Exploration Loop
1. Define 5-10 distinct experiments upfront (coarse-to-fine strategy)
2. Each experiment is a full YAML config (no overrides)
3. Run all experiments, let the machine work
4. React to results, iterate or simplify

---

## Constraints

- **Avoid grouped convolutions** until NPU benchmarking is possible (memory bandwidth concerns)
- **FLOPs as rough comparison metric** — care about 2x differences, not 10%
- **JAX/JIT compatibility** — configs used at construction time only, not inside JIT'd functions
- **Inference cost** — should be cheap to run at inference time
- **Timeline** — exploration happens today; results inform next steps

---

## Unknowns / Questions for Experimentation

| Question | How to Test |
|----------|-------------|
| **Optimal DW conv count per UIB?** | Ablate: 0 vs 1 vs 2 DW convs at matched FLOPs |
| **Optimal expansion ratio?** | Ablate: 2x vs 4x vs 6x at matched DW count |
| **UIBs per level?** | Compare: 1 UIB with 2 DWs vs 2 UIBs with 1 DW each |
| **Downsampling position effect?** | Compare: downsample after UIB 0 vs after UIB 1 |
| **Does UIB beat previous StemBlock at matched FLOPs?** | Tune UIB config to match previous model's FLOPs |
| **Optimal number of levels?** | Compare: 2 vs 3 vs 4 levels |

---

## Detailed Spec

### Config Structure (YAML)
```yaml
variants:
  experiment_name:
    name: "..."
    levels:
      - name: "level0"
        input_channels: 3
        output_channels: 32
        uib_blocks:
          - num_dw_convs: 2
            expansion_ratio: 4
            downsample_after: true
```

### CLI Interface (Tyro)
- `--config`: Path to YAML config file (default: `configs/embeddings.yaml`)
- `--variant`: Variant name to run (default: `default` or first variant)
- Ephemeral args only: `--resume`, `--debug`

### FLOPs Tracking
- Analytical calculation from config
- Report as **FLOPs per embedding** (normalized for comparison)
- Simple formula, not device-specific modeling

---

## False Starts

- **Initial concern about downsampling location** — resolved: downsampling is configured on UIB (`downsample_after` flag), dispatched by level builder
- **Concern about JAX + mutable pydantic** — resolved: configs used at construction time only, NNX modules are traceable
- **Considered base + override config pattern** — rejected: prefer self-contained configs for clarity

---

## Next Steps

1. **Define pydantic schemas** for UIB/Level/Model configs
2. **Create YAML config file** with 5-10 distinct experiment variants
3. **Add analytical FLOPs calculation** (FLOPs per embedding)
4. **Migrate settings** from CLI/dataclasses to YAML/pydantic
5. **Update model builder** to construct UIB blocks from config
6. **Run experiments**, analyze results, iterate

---

## Key Hypotheses to Test

1. **"Recover StemBlock capacity"**: 2 UIBs at Level 0 recovers most of the loss gap
2. **"More DW convs = more spatial context"**: 2 DW convs per UIB beats 1 DW conv at matched FLOPs
3. **"Expansion ratio matters"**: Higher expansion (6x) with fewer DWs beats lower expansion (2x) with more DWs
4. **"Later downsampling wins"**: Downsampling after 2 UIBs beats after 1 UIB (more high-res processing)
5. **"UIB beats handcrafted at matched FLOPs"**: Disciplined block design outperforms ad-hoc StemBlock
