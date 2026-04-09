# Progress: Training Throughput Optimization

## Current State

### Variance Map Visualizations: Removed

Variance map heatmap visualizations have been removed from TensorBoard logging:

1. **What was removed**:
   - `_variance_map_to_heatmap()` function
   - Variance map logging calls in `log_visualizations()`

2. **What was kept**:
   - Variance maps are still computed and stored in `aux` data (for debugging, potential future use)
   - All other visualizations (frame grid, attention maps) remain unchanged

3. **Rationale**:
   - Variance map visualizations were broken/incorrect
   - Not essential for monitoring training health
   - Attention variance scalar statistics (histograms, mean/std) still logged via `log_diagnostics()`

### Data Loading: Pre-loaded Frame Cache

Training uses **in-memory pre-loaded frames** to eliminate per-step JPEG decoding overhead:

1. **Pre-loading phase**: At dataloader creation, all unique frames needed for the epoch are:
   - Decoded from JPEG once
   - Resized to target dimensions
   - Stored as a single JAX array on CPU

2. **Batch generation**: Frames are sliced from the pre-loaded array (instant, no decoding)

3. **Memory limit**: Configurable via `frame_cache_max_mb` (default 500MB, -1 for unlimited)
   - Fails fast if dataset exceeds limit with clear error message
   - Typical usage: 9k frames at 81×81 fits in ~700MB

4. **Performance**: 
   - First epoch: ~57 samples/sec (includes pre-loading overhead)
   - Subsequent epochs: ~90-95 samples/sec
   - **~15x improvement** over original PIL-based loading (~6 samples/sec)

5. **Trade-offs**:
   - Pre-loading happens every epoch (acceptable overhead: ~1-2 seconds)
   - No disk persistence (simpler, no cache invalidation)
   - Works for both CPU and GPU training (JAX arrays on CPU, auto-transferred when needed)

### Configuration

```yaml
dataset:
  frame_cache_max_mb: 500  # Memory limit for pre-loaded frames (-1 for unlimited)
```

### Testing

- Test fixtures: Small synthetic dataset (2 videos, 9 frames total) in `src/barevision/dataset/test_fixtures/`
- Tests verify: pre-loading, memory limits, frame reuse, batch generation
- Dataset directory override: `set_datasets_dir_override()` for test isolation
