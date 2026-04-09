# Changes: Training Throughput Optimization

## Goal
Achieve 10x faster training iterations while maintaining essential diagnostic capabilities (less frequent logging is acceptable).

## Current Direction

**Logging strategy**:
- Loss: logged to TensorBoard every step (single scalar, already computed)
- Diagnostics (attention variance, embedding stats): every N steps (configurable, ~100 steps)
- Visualizations (attention maps, variance heatmaps): every M steps (configurable, M = 10× N)

**Key changes**:
1. Decouple loss logging from diagnostics (loss is cheap, always available)
2. Remove variance map visualizations (currently broken, not essential)
3. Conditional aux computation: fast path (no aux) for 99% of steps, slow path (with aux) only when needed
4. JIT benefits from conditional: different traces for fast/slow paths

**What we're NOT doing**:
- Fix variance map visualizations
- Add validation diagnostics (validation loss only)
- Change what diagnostics are computed (yet) — just log them less often

## Core Tension
**Throughput vs. Observability** — Diagnostics provide essential visibility into training dynamics but add computation that doesn't contribute to learning.

## The Bet
The bottlenecks are in two areas: (1) diagnostic logging doing redundant/unoptimized work, and (2) data loading pipeline. Profiling suggests 9x slowdown from diagnostics and 10x from data loading, but actual numbers need verification.

## Constraints
- CPU-only training baseline
- Must preserve all current diagnostic capabilities (can log less frequently, but capabilities must exist)
- No architectural changes to the model itself

## Decisions
| Idea | Status | Why |
|------|--------|-----|
| Reduce diagnostic frequency alone | ❌ Rejected | User doesn't trust profiling numbers; wants to understand actual bottlenecks first |
| Stop gradients between levels | ❌ Rejected | Wrong problem — gradient flow affects convergence, not throughput |
| Cache training forward pass results for diagnostics | ⚠️ Open | Shouldn't need 3 extra forward passes if we capture intermediates from training pass |
| Split expensive diagnostics by frequency | ⚠️ Open | Some stats (embedding histograms) may be cheap; others (attention variance) are O(N²) |
| Optimize attention variance computation | ⚠️ Open | Currently O(N²) per window, un-jitted |
| Pre-process dataset to binary format | ⚠️ Open | Avoid per-step JPEG decoding (currently ~754ms/batch) |
| JIT diagnostic computations | ⚠️ Open | Training forward pass is ~12ms jit'd vs 50-100ms+ un-jitted |

## Unknowns / Ablation Questions
- **What are the actual bottlenecks?** User reports poor performance even with diagnostics every 1k steps — profiling numbers may be incomplete
- **Data loading vs. diagnostics — which is easier to fix first?** Data loading suspected to be more straightforward (TFRecord, numpy, LMDB)
- **What's the true cost breakdown of log_diagnostics()?** Need to measure each component independently
- **Can we cache intermediate activations from training forward pass?** Would eliminate 3 redundant forward passes in diagnostics
- **Optimal diagnostic frequency per metric type?** Embedding stats vs. attention variance vs. gradient stats may need different frequencies

## Key Insights
- Backward pass cost is proportional to forward pass FLOPs, not parameter count — gradient flow through pyramid is not the bottleneck
- Attention variance is O(N²) per window — with window_size=16, this means 256×256 attention matrices computed 6 times per diagnostic call
- Diagnostic forward passes are un-jitted while training pass is jit'd (~12ms)
- JAX stores activations, not derivatives per operation

## Numbers to Preserve
- Baseline: ~0.7 batches/sec (~6 samples/sec) with default diagnostics (every 10 steps)
- Model forward+backward: ~22ms (jit'd, isolated)
- Data loading: ~754ms/batch (8 images from disk, PIL decode + resize + numpy)
- Attention variance: 6 computations per diagnostic (self + cross for each of 3 pyramid levels)
- Diagnostic forward passes: 3 extra passes per logged step (one per pyramid level)

## References
- `src/barevision/embeddings/logging_utils.py` — `log_diagnostics()` implementation

## Detailed Spec: log_diagnostics() Operations

**Called:** Every `logging.every_steps` steps

**Operations per call:**
```
1. log_gradient_statistics() — traverses optimizer state
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

**Key issues:**
- None of these operations are jit'd
- Items 2-4 are redundant if training already computed forward pass
- Items 5-10 are O(N²) per window — most expensive component
- Total estimated cost: 1500-3000ms per diagnostic call

## Profiling Commands (for investigation)

```bash
# Profile data loading
python profile_data_loading.py

# Profile training with timing
python -m barevision.embeddings.training --path /tmp/quick_test.yaml
```
