# Changes: Embedding Model Simplification

## Goal

Simplify the embedding model codebase and training setup to reduce cognitive load during flow estimation development, while preserving baseline performance (within 20% loss tolerance).

## Core Tension

**Simplicity now vs. Understanding later:** We want cleaner code and fewer configuration knobs immediately, but don't fully understand what each component (normalization layers, stem architecture) contributes to performance. This change separates refactoring from experimentation — simplify the structure first, then ablate components systematically.

## The Bet

- **Model unification:** Moving the stem's dense convolution to a preprocessor layer will reduce code duplication without changing computation, making future experiments easier to implement.
- **Epoch-based checkpointing:** Aligning checkpointing with validation on an epoch schedule will simplify the mental model without significantly increasing crash risk (crashes are rare, epochs are ~5min on GPU).
- **Normalization ablation:** At least one of the three normalization layers (GroupNorm, mean subtraction, L2 norm) is redundant and can be removed, freeing budget for more channels or layers.

## Constraints

- **Hard constraint:** Do not worsen validation loss by more than 20% from baseline
- **Hard constraint:** Do not end up with anything more complex than current state (fewer classes, fewer config options, fewer normalization layers)
- **Hard constraint:** Preserve exact computation during model unification (refactor first, ablate later)
- **Hard constraint:** Must work with existing Orbax CheckpointManager infrastructure
- **Hard constraint:** Preserve `keep_best_n` functionality (uses training loss for best checkpoint selection)

## Decisions

| Idea | Status | Why |
|------|--------|-----|
| **Preprocessor layer for stem convolution** | ✅ Adopted | Move dense 3→hidden_dim conv to HierarchicalEmbeddingModel as preprocessor. All EmbeddingBlocks become identical. Clean separation, easy to ablate later. |
| **Unify StemBlock and StandardBlock into single EmbeddingBlock** | ✅ Adopted | Reduces code duplication from 2 classes to 1. Preprocessor handles the "first block is special" logic. |
| **Epoch-based checkpointing** | ✅ Adopted | Replace `every_steps` with `every_epochs` in CheckpointSettings. Aligns with validation schedule, simpler mental model. |
| **Remove step-based checkpointing option** | ✅ Adopted | User wants simple now. Can re-add later if needed for long training runs. |
| **Keep both epoch and step-based checkpointing** | ❌ Rejected | More configuration surface, defeats the simplicity goal. |
| **`is_first_block` flag inside block class** | ❌ Rejected | Still has branching logic inside the block. User wants differences pushed to container. |
| **Normalization ablation (5 configs)** | ✅ Adopted | Test GroupNorm, mean subtraction, and L2 norm in structured ablation. Phase 1: 5 configs (baseline, remove mean sub, remove L2, remove both, remove all). |
| **Switch from Adam to AdamW** | ✅ Adopted | Better regularization baseline for normalization ablation. Need new baseline run with AdamW before comparing. |
| **Remove dense conv1 entirely (before unification)** | ❌ Rejected | Behavioral change. Want to preserve baseline first, ablate in separate phase. |
| **Preprocessor ablation (disable entirely)** | ⏸️ Deferred | Structural change (first block needs different input channels). Keep flag for future experiments, but not part of Phase 1 ablations. |
| **Mean conv for downsampling ablation** | ✅ Adopted | Add `use_mean_conv_for_downsampling` flag. When False, downsample via strided slice of rich_features. Tests if learned low-pass filter adds value. |
| **Unified interval with unit (interval + interval_unit)** | ❌ Rejected | Still two knobs to think about. |

## Unknowns / Ablation Questions

- **Model unification:**
  - Does the dense conv1 in StemBlock contribute meaningfully to performance? → will ablate after unification is complete
  - Does splitting "first block" across preprocessor + EmbeddingBlock hurt conceptual clarity? → will assess during implementation
  - Are there any places in the codebase that reference StemBlock/StandardBlock directly (outside HierarchicalEmbeddingModel)? → need to grep

- **Mean conv downsampling:**
  - Does the learned Gaussian-init conv provide better anti-aliasing than direct subsampling? → config 9 tests this
  - If mean conv is removed for downsampling, should mean subtraction also be disabled? → default: independent flags, user decides
  - Does removing mean conv save meaningful compute at inference? → yes, one less conv per level

- **Checkpointing:**
  - How much overhead does `save_step()` add when policy says "don't save"? → may measure during implementation
  - Are there any tests that depend on step-based checkpointing behavior? → need to check
  - Should checkpoint happen before or after validation at epoch end? → design decision during implementation

- **Normalization ablation:**
  - Do embeddings grow in magnitude without L2 norm? → will measure avg/max norm per epoch
  - Does mean subtraction actually help, or is it vestigial from entropy loss era? → ablation will tell
  - Does removing normalizations require LR tuning? → may need separate LR sweep
  - Does AdamW (with weight decay) make L2 norm more or less important? → switching to AdamW for all configs
  - How do these changes affect downstream flow estimation? → embeddings are means to an end
  - Optimal hyperparameters (channel depth, group count, number of levels) → grid search for later phase

## Key Insights

- **Normalization layers may work as a system:** Mean subtraction reduces magnitudes uniformly, then L2 norm amplifies residuals that weren't in the direction of the mean. Removing one but not the other could give misleading results.
- **Spatial variance loss operates on attention, not embeddings directly:** Embeddings could still grow in magnitude to make logits larger (sharper softmax → lower loss). L2 norm prevents this degenerate solution.
- **The stem difference is a dense convolution, not depthwise:** Initially misidentified as "extra depthwise conv" — actually StemBlock has an extra dense 3→hidden_dim conv that StandardBlock lacks.

## Numbers to Preserve

- **Baseline validation loss:** [to be measured with AdamW before ablation] — all configs compared against this
- **20% loss tolerance:** Maximum acceptable degradation from any simplification
- **Typical epoch:** ~8k steps (~30min local, ~5min GPU) — informs checkpointing frequency decision
- **3 pyramid levels** — unchanged
- **16 embedding dimensions** — unchanged
- **32 hidden dimensions** — unchanged

## Normalization Ablation Matrix (Phase 1)

| Config ID | GroupNorm | Mean Sub | Mean Conv DS | L2 Norm | What it tests |
|-----------|-----------|----------|--------------|---------|---------------|
| 0 | ✅ | ✅ | ✅ | ✅ | Baseline (new AdamW baseline) | 
| 1 | ✅ | ❌ | ✅ | ✅ | Mean sub only removal |
| 2 | ✅ | ✅ | ✅ | ❌ | L2 only removal |
| 3 | ✅ | ❌ | ✅ | ❌ | No contrast normalization stack |
| 7 | ❌ | ❌ | ✅ | ❌ | No normalization at all |
| 9 | ✅ | ✅ | ❌ | ✅ | Direct strided slice downsampling (no mean conv) |

**Success criteria per config:**
- **Keep** if: validation loss within 5% of baseline AND no training instability
- **Reject** if: validation loss >10% worse OR diverges
- **Investigate further** if: validation loss similar but norm behavior is unusual

## Running Ablations

Each config is a training run with specific flags. Tyro uses hyphens for word separators and `--no-` prefix for disabling boolean flags.

```bash
# Config 0: Baseline (all defaults)
python -m barevision.embeddings.training \
  --run-name-prefix=ablation_0_baseline

# Config 1: No mean subtraction
python -m barevision.embeddings.training \
  --model.no-use-mean-subtraction \
  --run-name-prefix=ablation_1_no_mean_sub

# Config 2: No L2 norm
python -m barevision.embeddings.training \
  --model.no-use-l2-norm \
  --run-name-prefix=ablation_2_no_l2

# Config 3: No contrast normalization (mean sub + L2 off)
python -m barevision.embeddings.training \
  --model.no-use-mean-subtraction \
  --model.no-use-l2-norm \
  --run-name-prefix=ablation_3_no_contrast

# Config 7: No normalization at all
python -m barevision.embeddings.training \
  --model.no-use-group-norm \
  --model.no-use-mean-subtraction \
  --model.no-use-l2-norm \
  --run-name-prefix=ablation_7_no_norm

# Config 9: No mean conv for downsampling (direct strided slice)
python -m barevision.embeddings.training \
  --model.no-use-mean-conv-for-downsampling \
  --run-name-prefix=ablation_9_no_mean_conv_ds
```

**Notes:**
- All configs use AdamW optimizer (switched from Adam for better regularization baseline)
- Compare validation loss against Config 0 baseline
- TensorBoard logs are in `runs/` directory, organized by run name
- Use same dataset, epochs, and hyperparameters across all configs for fair comparison
- Boolean flags: use `--model.use-X` to enable (default for most), `--model.no-use-X` to disable

## References

- `sessions/2026-04-05-simplify/brainstorm.md` — Full mental model, detailed discussions, and false starts
