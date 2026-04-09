# Brainstorm: Training Throughput Optimization

## The Wish

Faster training iterations for the embeddings model. Currently getting ~0.7 batches/sec (~6 samples/sec) on CPU. Want to understand what's limiting throughput before adding more training stages.

## Initial Hypothesis (Rejected)

**Hypothesis:** Gradient flow through the hierarchical pyramid is causing slow training.

**Reasoning:** Each level's output feeds the next level, and all levels contribute to the loss. This means early levels receive gradients from multiple loss terms (L0, L1→L0, L2→L1→L0). Thought this might:
- Create conflicting gradient signals
- Make the optimizer's job harder
- Require more memory for autodiff (storing derivatives per-operation)

**What we learned:** This was the wrong mental model.

**Why rejected:** 
- JAX stores activations, not "derivatives per operation"
- Backward pass cost is proportional to forward pass FLOPs, not parameter count
- Profiling showed backward pass is only 88% overhead over forward pass (normal)
- Model + loss + backward = ~22ms, which is fine

**Key insight:** We were solving the wrong problem. Gradient flow affects *convergence* (how many steps to converge), not *throughput* (steps per second).

---

## Core Tension

**Throughput vs. Observability**

We want:
1. Fast training iterations (high samples/sec)
2. Rich diagnostics (embedding stats, attention maps, gradient norms)

These conflict because diagnostics require extra computation that doesn't contribute to learning.

---

## Profiling Results

### Measurement Approach

1. Profiled data loading separately: `create_dataloader()` iteration timing
2. Profiled model computation: forward-only vs forward+backward
3. Added timing instrumentation to training loop
4. Compared with diagnostics enabled vs disabled

### Numbers

| Component | Isolated | In Training | Notes |
|-----------|----------|-------------|-------|
| Data loading | 72ms/batch | 754ms/batch | Disk I/O contention |
| Model (F+B) | 22ms/batch | 12-2100ms/batch | Wild variance |
| **Total expected** | 94ms/batch | — | ~85 samples/sec theoretical |
| **Actual** | — | 1428ms/batch | ~0.7 batches/sec actual |

**Gap:** 15x slower than profiling suggested.

### Root Cause Discovery

The wild variance in compute time (12ms to 18,000ms) pointed to intermittent blocking. Traced to `log_diagnostics()` in `logging_utils.py`:

**What log_diagnostics does every logged step:**
1. `log_gradient_statistics()` — traverses optimizer state
2. **3 extra forward passes** — one per pyramid level for embedding stats
3. **6 attention variance computations** — self + cross attention for each level

**Why this is expensive:**
- Forward passes are NOT jit'd in diagnostics
- Attention variance is O(N²) per window (256×256 attention matrix per 16×16 window)
- Runs every 10 steps by default

### Comparison

| Configuration | Compute (avg) | Data (avg) | Total/batch | Samples/sec |
|---------------|---------------|------------|-------------|-------------|
| Diagnostics every 10 steps (default) | 2103ms | 718ms | 2821ms | **2.8** |
| Diagnostics every 100 steps | 231ms | 754ms | 985ms | **8.1** |
| No diagnostics (training only) | 22ms | 72ms | 94ms | **85** (theoretical) |

**Diagnostics cause 9x slowdown. Data loading causes 10x slowdown.**

---

## Constraints Discovered

1. **Attention variance is O(N²)** — window_size=16 means 256×256 attention matrices per window. Computing this 6 times per logged step is expensive.

2. **Diagnostic forward passes are un-jitted** — Training forward pass is jit'd (~12ms), but diagnostic passes run unoptimized.

3. **JPEG decoding is slow** — 754ms/batch reading 8 images from disk. PIL Image.open() + resize + numpy conversion adds up.

4. **CPU core contention** — num_workers=4 for data loading competes with JAX CPU parallelism. Didn't help when set to 0, suggesting deeper I/O bottleneck.

---

## Explored Options

| Idea | Status | Why |
|------|--------|-----|
| Stop gradients between levels | ❌ Rejected | Wrong problem — gradient flow isn't the bottleneck |
| Reduce diagnostic frequency | ✅ Adopted | every_steps: 10 → 100 gives 3x speedup |
| Remove attention stats from diagnostics | ⚠️ Deferred | Keeps embedding stats, removes O(N²) attention computation |
| Cache diagnostic embeddings | ⚠️ Deferred | Reuse training forward pass instead of 3 extra passes |
| JIT diagnostic computations | ⚠️ Deferred | Would require refactoring log_diagnostics |
| Pre-process dataset to TFRecord/numpy | ⚠️ Deferred | Avoid per-step JPEG decoding |
| Use num_workers=0 | ❌ Rejected | No improvement — disk I/O is the limit, not multiprocessing |

---

## Mental Models

**Before:** "Gradient flow through the pyramid is inefficient"

**After:** "Diagnostics are doing real work that doesn't contribute to learning"

The training step itself is fast (~22ms). The slowdown comes from:
1. **Observability tax** — logging that runs extra computation
2. **Data pipeline** — synchronous disk I/O blocking training

---

## False Starts

1. **Gradient memory hypothesis** — Assumed JAX stores derivatives per-operation. Actually stores activations. Backward pass cost is proportional to forward FLOPs.

2. **Model complexity** — 30M FLOPs, 5k parameters should be fast. It is! The model isn't the problem.

3. **Multiprocessing for data** — Tried num_workers=0 vs 4. No difference because disk I/O is the bottleneck, not CPU for decoding.

---

## Unknowns / Experiments

1. **What's the minimum useful diagnostic set?** — Embedding histograms alone? Skip attention variance entirely?

2. **Would TFRecord/numpy preprocessing help?** — Expect 5-10x data loading speedup, but need to test.

3. **How often do we actually need diagnostics?** — Every 100 steps? Every epoch? Could we log less during early training?

4. **GPU vs CPU profile** — On GPU, data loading would be even more dominant (faster compute, same disk speed).

---

## Detailed Spec: The Bottleneck

**File:** `src/barevision/embeddings/logging_utils.py`

**Function:** `log_diagnostics()`

**Called:** Every `logging.every_steps` steps (default: 10)

**Operations per call:**
```
1. log_gradient_statistics() — traverses model state
2. model(img1) — forward pass, level 0
3. model(img1) — forward pass, level 1  
4. model(img1) — forward pass, level 2
5. self_attention_variance() — level 0, O(N²)
6. cross_attention_variance() — level 0, O(N²)
7. self_attention_variance() — level 1, O(N²)
8. cross_attention_variance() — level 1, O(N²)
9. self_attention_variance() — level 2, O(N²)
10. cross_attention_variance() — level 2, O(N²)
```

**None of these are jit'd.** Each forward pass is ~12ms jit'd, probably 50-100ms un-jitted. Each attention variance is ~100-500ms for typical feature map sizes.

**Total per diagnostic call:** Estimated 1500-3000ms

---

## Implementation Notes (for later)

**Quick wins (config changes only):**
- `logging.every_steps: 100` — 3x speedup
- `logging.visualizations_every_steps: 1000` — reduce visualization overhead

**Medium effort (code changes):**
- Cache embeddings from training forward pass for diagnostics
- Remove attention variance from `log_diagnostics()` or make it less frequent
- Add `need_attention_stats: bool` flag to control O(N²) computation

**Higher effort:**
- Pre-process dataset to binary format (TFRecord, numpy, LMDB)
- JIT the diagnostic computations (may require refactoring)
- Async data loading with prefetching

---

## Session Notes

**Date:** 2026-04-08

**Initial question:** "Why is training slow? Is it gradient flow through the hierarchical pyramid?"

**Answer:** No. Gradient flow is fine. The slowdown is from diagnostic logging (9x) and data loading (10x).

**Key profiling commands:**
```bash
# Profile data loading
python profile_data_loading.py

# Profile training with timing
python -m barevision.embeddings.training --path /tmp/quick_test.yaml
```

**Files modified for profiling (to be reverted):**
- `src/barevision/embeddings/training.py` — added timing instrumentation
- `profile_data_loading.py` — standalone data loading profiler
