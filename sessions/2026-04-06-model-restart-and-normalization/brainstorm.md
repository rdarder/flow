# Brainstorm: Embedding Model Simplification — Phase 2

## The Wish

**Primary goal:** Produce the simplest embedding model that works similarly well (within 5-10% validation loss) to the current baseline.

**Motivation:**
- Current model has decisions that weren't thought through (mean subtraction, normalization stack)
- Want to reallocate CPU budget to more dimensions/channel capacity
- Save complexity budget for flow estimation (the real challenge)
- Code simplicity: fewer special cases, identical blocks

**Context:** Ablations are running on the current model (no mean subtraction, no L2, no normalization). Early signal: no_mean_sub performs ~4% worse after 10 epochs, suspect long-term convergence may match baseline.

---

## Core Tension

**Simplicity vs. Training Stability:**

| Aspect | Current Model | Proposed Model |
|--------|--------------|----------------|
| **Normalization stack** | GroupNorm + Mean Subtraction + L2 Norm | **L2 Norm on output only** |
| **Block structure** | Preprocessor + 3 identical blocks with mean_conv | All blocks identical, no preprocessor, no mean_conv |
| **Downsampling** | Strided slice of mean_conv output (hacky) | Explicit DW stride=2 (clean) |
| **Spatial processing** | 1 DW filter per channel | 8 DW filters per compacted channel |
| **Channel mixing** | Dense PW throughout | Dense compact + Grouped project |

**The real tension:** We're removing normalization layers that may have been compensating for architectural weaknesses. The new design (depthwise multiplier, grouped projection) may provide enough regularization through structure, but we won't know until we train.

**Empirical finding:** Collapse mode (single-spot attention for all queries) appears **only when L2 is removed**. With L2, training is stable.

---

## Explored Options

### Normalization Stack — Final Decision

| Config | GroupNorm | Mean Sub | L2 Norm | Status | Why |
|--------|-----------|----------|---------|--------|-----|
| Baseline | ✅ | ✅ | ✅ | Current | Works, but complex |
| No mean sub | ✅ | ❌ | ✅ | Running ablation | ~4% worse at 10 epochs |
| No L2 | ✅ | ✅ | ❌ | Running ablation | ❌ **Collapse mode** — magnitude explosion |
| No contrast | ✅ | ❌ | ❌ | Running ablation | Collapse mode |
| No norm | ❌ | ❌ | ❌ | Running ablation | Collapse mode |
| **Adopted** | ❌ | ❌ | ✅ | **Final** | L2 required; GN unnecessary for shallow network |

**Key insights:**

1. **L2 norm is not optional** — it's a **constraint that prevents the spatial variance loss from being cheated**. Without L2, embeddings grow to magnitude ~30, causing softmax to collapse to one-hot (single spot wins for all queries).

2. **GroupNorm is unnecessary** — the network is shallow (3 blocks, ~9 conv layers). AdamW weight decay provides sufficient regularization. If training diverges, add GN after `pw_compact` only.

3. **L2 doesn't waste learning capacity** — convolutions learn to map patches to **directions in 16D space**, which has 15 degrees of freedom. Magnitude is a nuisance variable for cosine similarity; constraining it focuses learning on what matters.

4. **Quantization-friendly** — L2 gives fixed bounds [-1, 1] for INT8 mapping. Variable magnitude would require dynamic scaling and outlier handling.

### Block Architecture

| Design | Structure | FLOPs | Status |
|--------|-----------|-------|--------|
| Current | PW → DW → mean_sub → PW_embed | ~300HW | Baseline |
| Inverted bottleneck | PW_expand → DW → PW_project | ~712HW | ❌ Too expensive |
| **Proposed** | PW_compact → DW×multiplier → PW_project_grouped | ~168HW | ✅ Adopted |
| Direct DW→PW | DW → PW | ~100HW | ❌ Too little spatial diversity |

### Depthwise Multiplier

| Multiplier | Spatial filters per channel | Total filters | Status |
|------------|----------------------------|---------------|--------|
| 1× (standard DW) | 1 | 4 | ❌ Too few spatial patterns |
| 4× | 4 | 16 | ⚠️ Maybe enough |
| **8×** | **8** | **32** | ✅ **Adopted** — good balance |
| 16× | 16 | 64 | ⚠️ Overkill for first pass |

### Grouped Projection

| Groups | FLOPs | Cross-group mixing | Status |
|--------|-------|-------------------|--------|
| 1 (dense) | 128HW | Full | ❌ More expensive |
| **4** | **32HW** | Partial (bucketed) | ✅ **Adopted** — 4× savings |
| 8 | 16HW | Very limited | ⚠️ May hurt uniqueness |

**Mental model:** Each group is a "bucket" in the bloom filter. Output channels 0-3 encode spatial filter responses 0-7, channels 4-7 encode responses 8-15, etc. Different regions activate different combinations → unique embedding.

---

## Mental Models

### Bloom Filter of Spatial Features

> "I'm not interested in the network learning abstract stuff like 'a dog' or 'a plant' but rather a bloom filter of features that uniquely identify a region but are also not so brittle to pose changes."

**How the architecture implements this:**
1. **PW Compact (dense):** Mixes input channels into 4 "base" channels (reduces redundancy)
2. **DW ×8:** Each base channel gets 8 different 3×3 spatial filters → 32 spatial feature maps
3. **PW Project (grouped):** Each group of 8 spatial responses compacted to 4 embedding channels
4. **Result:** 16-channel embedding where each channel encodes a specific subset of spatial patterns

**Why this works for matching:**
- Different regions activate different combinations of spatial filters
- Combinatorial uniqueness: region A activates {filters 1, 5, 12}, region B activates {3, 7, 12}
- Grouped projection preserves this "bucket" structure while saving FLOPs

### Downsampling Hierarchy

**Before:**
```
RGB → Preprocessor (3→32, full res)
  → Block0: embed@full_res + downsample(mean)
  → Block1: embed@half_res + downsample(mean)
  → Block2: embed@quarter_res
```

**After:**
```
RGB → Block0: 3→4→32→16, stride=2 → (H//2, W//2, 16)
  → Block1: 16→4→32→16, stride=2 → (H//4, W//4, 16)
  → Block2: 16→4→32→16, stride=2 → (H//8, W//8, 16)
```

**Key difference:** Each level operates at its native resolution. No "downsample just to pass data to next layer" — downsampling is explicit and happens as part of spatial filtering.

### Normalization: What We Changed Our Mind On

**Initial thinking:** "Start without L2, monitor norms, add back only if they explode."

**Updated understanding:**

1. **L2 is required from the start** — empirical evidence shows collapse mode (single-spot attention for all queries) appears immediately without L2. This is not a gradual degradation; it's a fundamental loophole.

2. **Why collapse happens:**
   - Without L2, embeddings grow to magnitude ~30
   - Dot products range: ±30×30×16 = ±14,400
   - After temperature (0.25): ±57,600
   - Softmax(57600) → numerical collapse to one-hot
   - One "lucky" alignment wins for all queries → low variance → loss is happy!

3. **L2 doesn't waste capacity:**
   - Convolutions learn: `f(x) → direction in 16D`
   - Direction on unit sphere has **15 degrees of freedom**
   - Magnitude is a nuisance variable for cosine similarity
   - Constraining it focuses learning on what matters

4. **GroupNorm is optional:**
   - Network is shallow (9 conv layers total)
   - AdamW weight decay provides regularization
   - If training diverges: add GN after `pw_compact` only

5. **Quantization implications:**
   - L2: fixed bounds [-1, 1] → simple INT8 mapping
   - No L2: unbounded range → dynamic scaling, outlier clipping
   - L2 is deployment-friendly

---

## Constraints

| Constraint | Type | Impact |
|------------|------|--------|
| **VALID padding** | Hard | Resolution math: (H-3)//2 + 1 per stride=2 DW |
| **NPU-friendly ops** | Hard | Contiguous grouping (not interleaved), standard DW |
| **~$10 NPU, 0.5 TOPS** | Hard | FLOP budget drives grouped projection, compact ratio |
| **Identical blocks** | Soft | No special first block, no preprocessor |
| **Within 5-10% of baseline** | Soft | Tolerable performance hit for simplicity |
| **L2 normalization on output** | Hard | Prevents collapse; required for stable training |

---

## Unknowns

| Unknown | How we'll discover |
|---------|-------------------|
| **Does training converge without GroupNorm?** | First 100-200 steps will show divergence/NaN |
| **Optimal compact ratio** | Ablate: 16→2, 16→4, 16→8 before DW |
| **Optimal DW multiplier** | Ablate: 4×, 8×, 16× filters per channel |
| **Optimal project groups** | Ablate: groups=1 (dense), 2, 4, 8 |
| **First block compact: 3→4 sufficient?** | May need 3→8 if RGB has less redundancy to compact |
| **Temperature needs retuning?** | Current 0.25 derived for 16D; may need empirical adjustment |
| **Does grouped projection hurt uniqueness?** | Measure: embedding distinctiveness within/between frames |
| **Can direction encode "uncertainty"?** | Visualize: do textureless regions cluster in embedding space? |

---

## Detailed Spec

### EmbeddingBlock

```python
EmbeddingBlock(in_ch, out_ch=16):
    """
    Input: (B, H, W, in_ch)  # 3 for first block, 16 for others
    Output: (B, H//2, W//2, out_ch)
    
    Design: Compact → Spatial (multi-filter) → Project
    """
    # Compact: reduce channel redundancy, full mixing
    pw_compact = Conv1x1(in_ch, 4)  # dense, always compact to 4 channels
    
    # Spatial: multiple filters per channel, downsamples
    dw = DepthwiseConv3x3(
        channels=4,
        depth_multiplier=8,  # 4 × 8 = 32 output channels
        stride=2,
        padding="VALID"
    )
    
    # Project: grouped mixing to embedding dim
    pw_project = Conv1x1(
        32, out_ch,
        feature_group_count=4  # 4 parallel 8→4 convs
    )
    
    # Forward:
    x = gelu(pw_compact(x))
    x = gelu(dw(x))
    x = pw_project(x)
    x = L2_normalize(x, axis=-1, epsilon=1e-8)  # REQUIRED
    return x
```

### HierarchicalEmbeddingModel

```python
HierarchicalEmbeddingModel:
    Input: (B, H, W, 3)
      ↓
    Block0: EmbeddingBlock(in_ch=3, out_ch=16)  → (H//2-1, W//2-1, 16)
      ↓
    Block1: EmbeddingBlock(in_ch=16, out_ch=16) → (H//4-2, W//4-2, 16)
      ↓
    Block2: EmbeddingBlock(in_ch=16, out_ch=16) → (H//8-3, W//8-3, 16)
```

**Resolution formula (VALID padding, stride=2, 3×3 kernel):**
```
output_size = (input_size - 3) // 2 + 1
```

### Normalization Strategy

| Layer | Normalization | Rationale |
|-------|---------------|-----------|
| **pw_compact output** | None (add GroupNorm only if training diverges) | Shallow network; AdamW sufficient |
| **dw output** | None | GELU provides nonlinearity |
| **pw_project output** | **L2 Norm (REQUIRED)** | Prevents magnitude exploitation of spatial variance loss |

**If training fails:**
1. Reduce learning rate (1e-3 → 5e-4)
2. Add GroupNorm after pw_compact only (before GELU)
3. Don't add GN to every layer — overkill for 9 conv layers

---

## Temperature Derivation (From First Principles)

**Goal:** Choose softmax temperature so logits have healthy gradient scale (StdDev ≈ 1).

### Step 1: Expected Dot Product of Random Unit Vectors

For L2-normalized vectors in R^d:
```
u, v ∈ R^d,  ||u|| = ||v|| = 1

E[u·v] = 0  (symmetry — random vectors are orthogonal on average)
Var[u·v] = 1/d
StdDev[u·v] = 1/√d
```

For d=16: StdDev[u·v] = 1/4 = 0.25

### Step 2: Softmax Behavior by Logit Magnitude

| Logit range | Softmax behavior | Gradient quality |
|-------------|------------------|------------------|
| [-∞, -10] to [10, ∞] | One-hot (saturated) | Vanishing |
| [-2, 2] | Smooth, differentiated | Healthy ✅ |
| [-0.1, 0.1] | Nearly uniform | Weak signal |

**Target:** Logits should have StdDev ≈ 1-3 for healthy gradients.

### Step 3: Temperature Scaling

```
logits = (u·v) / temperature

StdDev[logits] = StdDev[u·v] / temperature
               = 1 / (√d × temperature)

For StdDev[logits] ≈ 1:
  temperature ≈ 1/√d
```

### Step 4: Your Case (d=16)

```
d = 16
√d = 4

Optimal temperature = 1/4 = 0.25 ✅
```

**Your current setting (0.25) is exactly right for 16D embeddings!**

### Step 5: Comparison to Transformer's √d_k

Transformers use: `attention = softmax(Q @ K^T / √d_k)`

This is equivalent to our formulation:
```
Transformers: divide by √d → logits = (u·v) / √d
Our formula:  divide by temperature → logits = (u·v) / temperature

Setting temperature = 1/√d gives the same scaling.
```

### Step 6: Temperature for Other Embedding Dimensions

| Embed Dim | √d | Temperature (1/√d) |
|-----------|-----|-------------------|
| 8 | 2.83 | 0.35 |
| **16** | **4.0** | **0.25** |
| 32 | 5.66 | 0.18 |
| 64 | 8.0 | 0.125 |

**Rule of thumb:** `temperature ≈ 1/√embed_dim`

### Step 7: What Happens at Different Temperatures?

For d=16 (StdDev[u·v] = 0.25):

| Temperature | StdDev[logits] | Softmax behavior |
|-------------|----------------|------------------|
| 0.05 | 0.25/0.05 = 5 | Saturated (one-hot) ❌ |
| **0.25** | **0.25/0.25 = 1** | **Healthy gradients** ✅ |
| 1.0 | 0.25/1.0 = 0.25 | Too uniform ⚠️ |
| 2.0 | 0.25/2.0 = 0.125 | Nearly uniform ❌ |

---

## False Starts

- **Inverted bottleneck (PW→DW→PW):** Too expensive (~712HW vs. ~168HW proposed). The expand-at-full-res PW dominates FLOPs.
- **Residual connections:** Don't make sense with changing resolutions. Network is shallow (3 blocks) — residuals matter for 50+ layers.
- **Grouped compact layer:** Would reinforce bucket structure but limits cross-bucket communication. Dense compact ensures each DW channel sees all input information.
- **Interleaved grouping:** NPU may not support efficiently. Contiguous grouping is standard and simpler.
- **Removing L2 norm:** ❌ **Collapse mode** — spatial variance loss incentivizes magnitude growth, which causes softmax to saturate. L2 is a constraint, not just normalization.
- **Starting without L2 and monitoring:** Wrong approach — collapse happens immediately, not gradually. L2 must be present from step 1.
- **Mean subtraction for downsampling:** Was a hack to reuse mean_conv computation. Explicit DW stride=2 is cleaner.
- **"Magnitude expresses uncertainty":** Red herring — spatial variance loss wants concentrated attention; low magnitude makes attention more uniform (hurts loss). Direction encodes all needed information.

---

## Implementation Notes

**Keep unchanged:**
- Spatial variance loss (temperature 0.25 is correct for 16D)
- Dataset loading (handles custom resolutions)
- Training script, logging, visualization
- Checkpointing (epoch-based)

**Change:**
- Model architecture (new block structure)
- Remove mean_conv entirely
- Remove GroupNorm (add back only if training diverges)
- **L2 norm on output (REQUIRED)**
- All blocks identical (no preprocessor, no special first block)

**Ablation strategy (after implementation):**
1. Verify baseline works (compact=4, multiplier=8, groups=4, L2, temp=0.25)
2. Ablate compact ratio: 2, 4, 8
3. Ablate DW multiplier: 4, 8, 16
4. Ablate project groups: 1, 2, 4, 8
5. Ablate GroupNorm: with/without (only if baseline is stable)
6. Ablate temperature: 0.18, 0.25, 0.35 (if changing embed_dim)

---

## Summary: Normalization + Temperature

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **L2 norm on output** | ✅ Required | Prevents magnitude explosion → softmax collapse |
| **GroupNorm** | ❌ Start without | Shallow network; AdamW sufficient |
| **Mean subtraction** | ❌ Removed | Was tied to mean_conv hack; not needed |
| **Temperature** | 0.25 | 1/√16 — derived from first principles |
| **Embedding expressiveness** | 15 DOF | Direction in 16D; magnitude is nuisance variable |
| **Quantization** | INT8-friendly | L2 gives fixed [-1, 1] bounds |

---

*Session in progress — implementation pending*
