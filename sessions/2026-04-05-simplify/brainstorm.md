# Brainstorm: Embedding Model Simplification

## The Wish

**Primary goal:** Minimize code differences between StemBlock and StandardBlock to reduce maintenance burden during flow estimation development.

**Constraints:**
- Preserve baseline performance (no behavioral changes yet)
- Push "first block is special" logic to the container (HierarchicalEmbeddingModel)
- Keep experimentation (ablations, hyperparameter search) for a later phase
- One block class to maintain — don't want to change two places when iterating

---

## Core Tension

StemBlock and StandardBlock are ~90% identical, but differ in how they handle the first convolution:

| Component | StemBlock | StandardBlock |
|-----------|-----------|---------------|
| First conv | Dense 3→hidden_dim (feature_group_count=1) | Grouped hidden_dim→hidden_dim |
| Second conv | Grouped hidden_dim→hidden_dim | N/A |
| Mean conv | Depthwise (identical) | Depthwise (identical) |
| Embed branch | 1×1 projection (identical) | 1×1 projection (identical) |

**Why the difference exists:** The dense conv1 in StemBlock was added under the assumption that raw RGB (3 channels) needs more abstraction before grouped convolutions can work effectively. But this is untested — may be unnecessary complexity.

**The tension:** We want code simplicity now, but don't know if the architectural difference matters for performance.

---

## Explored Options

| Idea | Status | Why |
|------|--------|-----|
| **Option A: `is_first_block` flag inside block** | ❌ Rejected | Still has branching logic inside the block. User wants differences pushed to container. |
| **Option B: Preprocessor layer in HierarchicalEmbeddingModel** | ✅ Adopted | Clean separation. Preprocessor handles 3→hidden_dim dense conv. All EmbeddingBlocks are truly identical. Easy to ablate later. |
| **Option C: Hierarchical model handles extra conv inline** | ❌ Rejected | Moves duplication but creates "half a block" logic inline in the container — its own kind of mess. |
| **Remove dense conv1 entirely** | ⚠️ Deferred | Behavioral change. Want to preserve baseline first, ablate in separate phase. |

---

## Mental Models

**Before:**
```
StemBlock:     conv1(dense) → conv2(grouped) → mean_conv → embed
StandardBlock: conv1(grouped) → mean_conv → embed
```

**After (Option B):**
```
HierarchicalEmbeddingModel
├── preprocessor: Conv(3→hidden_dim, dense) + Norm + GELU
└── blocks: List[EmbeddingBlock]  # all identical
    └── each: conv(grouped) → mean_conv → embed
```

**Key insight:** The "first block" computation is split across two classes (preprocessor + EmbeddingBlock), but this is acceptable because:
- Preprocessor is explicit and named
- EmbeddingBlock has no special cases
- Removing preprocessor later is a one-line ablation

---

## Constraints

- **Hard constraint:** Must preserve exact computation for now (no performance regression while simplifying)
- **Soft constraint:** Keep changes localized to embeddings/model.py if possible
- **Future constraint:** Will need to verify no other code instantiates StemBlock/StandardBlock directly

---

## Unknowns

- **Unknown:** Does the dense conv1 in StemBlock contribute meaningfully to performance? → will ablate after unification
- **Unknown:** Are there any places in the codebase that reference StemBlock/StandardBlock directly (outside HierarchicalEmbeddingModel)? → need to grep
- **Unknown:** Will visualization/logging code need updates if block names change? → need to check
- **Unknown:** Does splitting "first block" across preprocessor+EmbeddingBlock hurt conceptual clarity? → will assess during implementation
- **Unknown:** Optimal hyperparameters (channel depth, group count, etc.) → grid search for later phase

---

## False Starts

- Initially thought the difference was an "extra depthwise convolution" — actually it's an extra **dense** convolution. The depthwise mean_conv is identical in both blocks.
- Initially considered removing the extra conv entirely — but that's a behavioral change, not a refactor. Separating refactor from ablation is cleaner.

---

## Next Phase (Not Now)

After unification is complete and verified:
1. Ablate preprocessor — remove it and measure performance impact
2. Hyperparameter search — channel depth, group count, number of levels
3. Explore whether the two-conv stem (dense→grouped) is necessary vs. single conv

---

*Session in progress — more topics to explore*

---

# Brainstorm: Checkpointing Simplification

## The Wish

**Primary goal:** Align checkpointing with validation on an epoch-based schedule for simpler experimentation.

**Current state:**
- Validation: every N epochs ✅
- Checkpointing: every N steps ❌

**Motivation:**
- Simpler mental model (one schedule to think about)
- Less configuration complexity during experimentation
- Reduce overhead from frequent checkpointing calls

---

## Core Tension

**Step-based (current) vs. Epoch-based (proposed):**

| Aspect | Step-based | Epoch-based |
|--------|------------|-------------|
| Granularity | Fine (mid-epoch recovery) | Coarse (epoch boundaries only) |
| Mental model | "How many batches?" | "How many passes through data?" |
| Alignment with validation | ❌ Misaligned | ✅ Aligned |
| Crash recovery | Better | Worse (could lose epoch) |
| Configuration simplicity | ❌ Two schedules | ✅ One schedule |
| Overhead | Called every step | Called once per epoch |

**Context:**
- Typical epoch: 8k steps (~30min local, ~5min GPU)
- Training crashes: rare
- Priority: simplicity now, flexibility later

---

## Explored Options

| Idea | Status | Why |
|------|--------|-----|
| **Pure epoch-based** | ✅ Adopted | User wants simple now. Can add step-based back later if needed. |
| **Keep both (every_epochs + every_steps)** | ❌ Rejected | More configuration surface, defeats the simplicity goal. |
| **Unified interval with unit (interval + interval_unit)** | ❌ Rejected | Still two knobs to think about. |
| **Call save_step() every step with policy filter** | ❌ Current (rejected) | Overhead from calling every step even when not saving. |

---

## Mental Models

**Before:**
```
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        train_step()
        checkpointer.save_step()  # ← every step, policy decides
    maybe_validate()
```

**After:**
```
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        train_step()
    maybe_checkpoint()  # ← once per epoch
    maybe_validate()    # ← once per epoch, aligned
```

---

## Constraints

- **Hard constraint:** Must work with existing Orbax CheckpointManager infrastructure
- **Hard constraint:** Preserve `keep_best_n` functionality (uses training loss)
- **Soft constraint:** Keep changes localized to checkpointer.py and training.py
- **Future flexibility:** User is comfortable re-adding step-based later if needed

---

## Unknowns

- **Unknown:** How much overhead does `save_step()` add when policy says "don't save"? → may measure during implementation
- **Unknown:** Are there any tests that depend on step-based checkpointing behavior? → need to check
- **Unknown:** Should checkpoint happen before or after validation at epoch end? → design decision during implementation

---

## Design Decision

**CheckpointSettings changes:**
```python
# Current
CheckpointSettings:
    every_steps: int = 100
    location: str
    keep_best_n: int = 3
    resume_from: Optional[Path]

# Proposed
CheckpointSettings:
    every_epochs: int = 1  # 0 to disable
    location: str
    keep_best_n: int = 3
    resume_from: Optional[Path]
```

**Training loop changes:**
- Remove `checkpointer.save_step()` call from inner loop
- Add `_maybe_checkpoint(epoch)` call at epoch end (alongside `_maybe_run_validation()`)
- Checkpointer wrapper simplified: no need for `FixedIntervalPolicy`, just check epoch divisibility

---

## False Starts

- Initially thought checkpointing was only at epoch boundaries — actually it's called every step with policy-based filtering
- Initially considered keeping both options for flexibility — but user explicitly wants simple now

---

# Brainstorm: Normalization Ablation

## The Wish

**Primary goal:** Understand what each normalization layer contributes (or doesn't) and potentially simplify the model.

**Current normalization stack (per block):**
1. **GroupNorm** — after each convolution (1-2 per block)
2. **Mean subtraction** (local contrast norm) — `rich_features - local_mean`
3. **L2 norm** — on final embeddings (projects to unit sphere)

**Motivation:**
- Don't fully understand why all three are needed
- Suspect some are redundant or work together in unclear ways
- Willing to trade ~10% performance for simpler model (budget can go to more channels/layers)
- Spatial variance loss may provide enough regularization

---

## Core Tension

**Understanding vs. Simplifying:**

| Normalization | Purpose | Interaction | Uncertainty |
|---------------|---------|-------------|-------------|
| **GroupNorm** | Stabilizes training, allows higher LR | May make other norms less necessary? | "Don't understand why we need two per block" |
| **Mean subtraction** | Removes common signal, emphasizes texture | Reduces magnitudes uniformly; may need L2 to amplify residuals | "Can't do the math — need to try" |
| **L2 norm** | Projects to unit sphere, cosine similarity | Prevents magnitude drift; may be redundant with spatial variance loss | "Added for entropy loss, may not be needed now" |

**Key insight:** These may work as a **system**, not independently. Removing one but not others could give misleading results.

---

## Explored Options

### Full Ablation Matrix (2³ = 8 configs)

| ID | GroupNorm | Mean Sub | L2 Norm | What it tests |
|----|-----------|----------|---------|---------------|
| 0 | ✅ | ✅ | ✅ | **Baseline** (current) |
| 1 | ✅ | ❌ | ✅ | Mean sub only removal (isolated) |
| 2 | ✅ | ✅ | ❌ | L2 only removal (isolated) |
| 3 | ✅ | ❌ | ❌ | No contrast normalization stack |
| 4 | ❌ | ✅ | ✅ | No GN, keep contrast norm stack |
| 5 | ❌ | ✅ | ❌ | Mean sub only |
| 6 | ❌ | ❌ | ✅ | L2 only (minimal normalization) |
| 7 | ❌ | ❌ | ❌ | **No normalization** (pure convolutions) |

### Phase 1 Plan (5 configs)

| ID | GroupNorm | Mean Sub | L2 Norm | Rationale |
|----|-----------|----------|---------|-----------|
| 0 | ✅ | ✅ | ✅ | Baseline |
| 1 | ✅ | ❌ | ✅ | Mean sub only removal |
| 2 | ✅ | ✅ | ❌ | L2 only removal |
| 3 | ✅ | ❌ | ❌ | No contrast normalization stack |
| 7 | ❌ | ❌ | ❌ | No normalization at all |

**Why these 5:**
- Config 1, 2: Test individual components
- Config 3: Test hypothesis that mean sub + L2 work together
- Config 7: Extreme case — does anything work without normalization?

**Phase 2:** Add configs 4, 5, 6 if Phase 1 is inconclusive.

---

## Mental Models

**Your hypothesis on mean subtraction + L2:**
> "Mean subtraction removes the average embedding, so most embeddings have lower norm. Without L2, attention lowers mostly uniformly. With L2, we amplify signals that weren't in the direction of the mean."

**Implication:** Removing mean sub but keeping L2 might work fine. Removing L2 but keeping mean sub might hurt (uniform reduction without amplification). Removing both might be neutral (they cancel out).

**Your hypothesis on spatial variance loss:**
> "Spatial variance loss doesn't encourage random sharpness — it penalizes sharpness in the wrong place. So it may provide enough regularization without L2 norm."

**Counterpoint:** Spatial variance loss operates on *attention*, not embeddings directly. Embeddings could still grow in magnitude to make logits larger (sharper softmax → lower loss). L2 prevents this.

---

## Constraints

- **Hard constraint:** Must be able to run all 5 configs sequentially
- **Hard constraint:** Need clean way to switch between configs (branches or flags, will clean up after)
- **Soft constraint:** Tolerate up to ~10% worse performance if model is simpler
- **Soft constraint:** Simpler model can reallocate budget to more channels/layers

---

## Unknowns

- **Unknown:** Do embeddings grow in magnitude without L2 norm? → will measure avg/max norm per epoch
- **Unknown:** Does mean subtraction actually help, or is it vestigial from entropy loss era? → ablation will tell
- **Unknown:** Does removing normalizations require LR tuning? → may need separate LR sweep
- **Unknown:** Does AdamW (with weight decay) make L2 norm more or less important? → switching to AdamW for all configs
- **Unknown:** How do these changes affect downstream flow estimation? → embeddings are means to an end

---

## Experimental Protocol

**For each config:**
1. Run 10 epochs (or until convergence signal)
2. Track:
   - Training loss curve
   - Validation loss
   - Embedding norm statistics (mean, max, std over batches)
   - Training speed (steps/sec)
3. Flag if:
   - Loss diverges
   - Norm explodes (>10x baseline)
   - Needs LR adjustment

**Success criteria:**
- **Keep config** if: val loss within 5% of baseline AND no instability
- **Reject config** if: val loss >10% worse OR diverges
- **Investigate further** if: val loss similar but norm behavior is weird

**Optimizer change:** Switch from Adam to AdamW for all configs (better regularization baseline). Need new baseline run (config 0) with AdamW before comparing.

---

## Implementation Approach

**Option A: Feature flags**
```python
NormalizationSettings:
    use_group_norm: bool = True
    use_mean_subtraction: bool = True
    use_l2_norm: bool = True
```
- ✅ Easy to script runs
- ✅ All code in one place
- ❌ Temporary complexity in model code

**Option B: Separate branches**
- ✅ Clean code per config
- ❌ More git management overhead

**Option C: Model variants as classes**
```python
class EmbeddingBlock_Baseline(nnx.Module): ...
class EmbeddingBlock_NoMeanSub(nnx.Module): ...
class EmbeddingBlock_NoNormalization(nnx.Module): ...
```
- ✅ Clear separation
- ✅ Easy to compare
- ❌ Some code duplication

**User preference:** "Switches also work as long as we then clean everything up when we chose our direction."

---

## False Starts

- Initially framed as "L2 vs. GroupNorm" — but mean subtraction is also normalization
- Initially worried about magnitude explosion — but user correctly notes spatial variance loss penalizes *wrong* sharpness, not sharpness itself
- Initially thought L2 was only for entropy loss — but it also prevents magnitude drift in spatial variance loss

---

*Session in progress — more topics to explore*
