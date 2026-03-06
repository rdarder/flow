# Embeddings: Experiments & Ideas

Active tracking document for embedding training experiments. Keep this updated as we learn.

---

## Architecture Ideas

### v0: Baseline (Single-Level, Full Resolution)

**Goal**: Validate loss functions with minimal architecture complexity.

```
Input: (B, H, W, 3) RGB
  ↓
3x3 depthwise conv: 3 in → 12 out (4 spatial filters per channel)
  ↓
ReLU
  ↓
1x1 conv: 12 in → 16 out
  ↓
Output: (B, H, W, 16) embeddings
```

**Parameters**: ~326 (108 for depthwise + 208 for 1x1 + biases)

**Notes**:
- No padding (valid convolutions only)
- Single level—no pyramid yet
- Grayscale variant: 1 in → 8 out depthwise, then 8→16 (216 params)

---

### v1: Multiple Fixed Filter Banks

**Goal**: Test if having multiple filter banks helps before adding gating.

**Idea**: N parallel filter banks (e.g., 4 banks), each producing 16-dim embeddings. Combine via:
- Concatenation → 1x1 proj back to 16-dim, or
- Averaging, or
- Learned weighted sum

---

### v2: Gating Network + Linear Mixing

**Goal**: Adapt filter weights to scene content.

**Idea**:
```
Frame → Simple MLP → mixing weights (N banks)
  ↓
Linear combination of filter bank outputs
```

**Gating input options**:
- Current frame only (assumes frame pairs are similar)
- Frame statistics (variance, gradient magnitudes)
- Coarse embedding (if pyramid exists)

---

### v3: Top-k Filter Selection

**Goal**: Cleaner specialization than linear mixing.

**Idea**: Gating network selects 1-2 filters per frame (or per patch). Harder to train, but may learn more distinct filter types.

---

### Future Variants

- **Temporal residuals**: Gating outputs Δ weights, accumulated via EMA across frames
- **Frame asymmetry**: Handle (f0,f1) vs (f1,f2) using different filter weights
- **Hybrid mixing**: Fixed base filters + gated residual filters

---

## Hierarchy Ideas (Deferred)

### Problem: Valid Convolutions Change Resolution

3x3 conv without padding: `N → N-2`

This breaks clean 2x upsampling between pyramid levels.

### Options to Explore

1. **Accept cropping**: Each level crops borders, flow priors need careful alignment
2. **Stride-2 convs**: `N → N/2` directly, but loses information
3. **Mixed approach**: Some levels stride, some don't
4. **Focus on subset**: Process only central region where valid embeddings exist
5. **Learned upsampling**: Replace simple interpolation with small network (costly)

### Key Constraint

Upsampling must be cheap. Willing to:
- Sacrifice border regions
- Reduce effective resolution
- Use tricks that preserve inference speed

Not willing to:
- Add heavy upsampling networks
- Pay significant compute for perfect alignment

---

## Training Experiments

### Priority 1: Loss Function Validation

**Goal**: Verify that self/cross-attention entropy produces meaningful embeddings.

**Setup**:
- Single-level architecture (v0)
- Synthetic data (ChairsSDHom or similar)
- Monitor: entropy values over time, visual inspection of attention maps

**Success criteria**:
- Attention maps become peaked (not uniform)
- Embeddings generalize to unseen frames
- Flow model can use these embeddings (even if not optimal)

---

### Priority 2: Architecture Ablation

**Goal**: Understand impact of design choices.

**Comparisons**:
- Depthwise-only vs depthwise + 1x1
- Number of spatial filters per channel (4 vs 8 vs 12)
- Embedding dimension (8 vs 16 vs 32)
- ReLU vs other activations

---

### Priority 3: Gating Experiments

**Goal**: Validate adaptive filter mixing.

**Comparisons**:
- No gating (fixed) vs linear mixing vs top-k
- Gating input: frame stats vs raw pixels vs coarse embedding
- Number of filter banks (2, 4, 8, 16)

---

### Priority 4: Dataset Choices

**Goal**: Find data that trains fast and generalizes.

**Options**:
- ChairsSDHom (synthetic, controlled motion)
- Real video with cut detection
- Self-collected footage

---

## Open Questions

1. **Temporal residuals**: Can we train a gating network to output weight adjustments instead of absolute weights? Requires longer sequences.

2. **Frame asymmetry**: Each frame gets processed twice (once as frame 1, once as frame 2) with different filter weights during a sequence. Is this acceptable overhead? Should we force symmetry?

3. **Gating granularity**: Global per-frame vs regional (e.g., quadrant-level) vs per-patch. Start global, but may need regional for complex scenes.

4. **Collapse prevention**: How to prevent gating from collapsing to uniform weights? Entropy regularization? Diverse initialization?

5. **Inference budget**: What's the actual compute budget for embeddings on target hardware? This constrains architecture choices.

6. **Freeze vs fine-tune**: Once trained, should embeddings be frozen in flow model or continue learning? May depend on whether they're scene-adaptive.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-05 | Start with single-level, full-resolution embeddings | Defer hierarchy complexity until loss functions are validated |
| 2026-03-05 | Baseline: depthwise(3→12) + 1x1(12→16) | Good balance of expressiveness and parameter efficiency |
| 2026-03-05 | No padding in convolutions | Avoid border pollution that would corrupt matching |
| 2026-03-05 | Gating uses absolute weights (not residuals) for v1 | Simpler to train; temporal smoothing can be added later |

---

## Notes

- Keep experiment notes close to code (comments, commit messages)
- Update this doc when conclusions are reached
- Remove or archive completed experiments to keep doc focused
