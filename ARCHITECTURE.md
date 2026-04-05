# Barevision: Architecture

## Problem Statement

Barevision provides **non-semantic perception** for low-cost robots: understanding space well enough to move safely, localize, and navigate. Not object recognition or semantic labeling.

**Target**: Single camera, cheap hardware (~$10 NPU, 0.5 TOPS), limited power. Robot moves slowly, builds understanding iteratively.

## Constraints

These constraints shaped every design decision:

**Hardware**: $10 NPU, 0.5 TOPS INT8, limited ONNX operators.

**Critical limitation**: No `GatherND` at inference. Cannot do variable-offset slicing (per-region frame warping). Workaround: shift entire frames based on coarse flow predictions.

**Budget**: ~$25 for compute + camera in a $100 robot.

## Approach

**Hierarchical embeddings for optical flow**:

1. Project each frame into dense 16-dim feature vectors (unique within frame, matchable across frames)
2. Process at multiple scales: coarse levels see large motions as small shifts
3. Self-supervised: train on raw video, no labeled flow data

**Why hierarchical**: Matching each pixel requires searching an area growing with square of displacement. Hierarchical processing makes this tractable—coarse estimate first, then refine in small windows.

**Why self-supervised**: No labeling budget, academic datasets may not match reality, video has enough temporal consistency to learn.

## Design Decisions

### VALID Padding (no zero-padding)

Convolutions use VALID padding:
- Avoids wasting compute on artifact-laden border data
- Enables frame shifting: when coarse flow re-aligns frames, valid regions nest hierarchically (shifted coarse frame is a crop of fine frame)

### Mean Subtraction

Each block has a depthwise convolution extracting the DC component. Output is:
1. Passed to next level (downsampled)
2. Subtracted from embeddings before normalization

**Why**: DC component contributes little to uniqueness. Removing it boosts the unique signal. Improved loss by 10-15%.

**Trade-off**: Only 3×3 receptive field.

### Spatial Variance Loss

Early attempts used entropy minimization for "sharp" attention. Model learned scattered sharp peaks (entropy doesn't care about position).

**Solution**: Position-aware variance:
- Compute attention scores over lookup window
- Find center of mass of attention-weighted positions
- Minimize variance of positions weighted by attention

Encourages attention to concentrate around a single peak.

Two terms: self-attention variance (within frame) + cross-attention variance (frame 1 → frame 2).

### L2 Normalization

Embeddings are L2-normalized. Added early to prevent extreme filter weights. May no longer be necessary with spatial variance loss.

**Open question**: Candidate for removal if experiments show it's unnecessary.

### JAX/Flax Over PyTorch

Chosen to avoid "batteries included" complexity and build custom components from scratch.

**Trade-off**: Pure functional style can be awkward; ecosystem less mature.

## Entry Points

### Training

```bash
python -m barevision.embeddings.training [OPTIONS]
```

Run `python -m barevision.embeddings.training -h` for all options.

**Key hyperparameters**:
- Softmax temperature (~0.3): Lower = sharper attention peaks
- Level weight decay (~1.0): Coarser levels get weight = decay^level
- Self/cross loss balance (~0.5): Weight between self and cross attention loss
- Embedding dimension (16), Hidden dimension (32)
- Lookup window size (16)
- Number of levels (3)
- Frame distance (1-5): Temporal distance between frame pairs

### Smoke Test

```bash
python -m barevision.embeddings.smoke_test
```

Small model, minimal data, few steps. Validates checkpointing, logging, visualization, loss, gradients. Catches integration errors unit tests miss.

### Tests

```bash
pytest src/barevision
```

### Checkpoints

Configurable directory (default: `checkpoints/`). Orbax CheckpointManager with loss-based preservation (keeps best N).

## Known Limitations

1. **Frame shifting at fine levels**: Frames may have moved significantly, making cross-attention unreliable. Current level decay is global; needs per-pixel weighting.

2. **Textureless areas**: Uniform regions produce similar embeddings, diluting training signal.

3. **Mean subtraction receptive field**: Only 3×3, too local especially at fine levels.

4. **GatherND limitation**: Forces whole-frame shifting. Cannot handle multi-modal flow (independent object motion vs. camera motion) without redesign.

5. **JAX ecosystem**: Orbax and Flax have poor documentation, frequent breaking changes. Workaround: write standalone test scripts to probe behavior empirically.

## Notes for Implementors

- **Get embeddings working first**, then add flow. Embeddings are the foundation.
- **Test JAX libraries empirically**. Don't trust documentation.
- **Keep training stages isolated**. Don't combine embeddings and flow training until each works independently.
- **Texture helps**. Static, textured scenery is ideal for early training.
- **L2 normalization may be removable**. Test ablation if simplifying.
- **Whole-frame operations** (where per-region would make sense) are due to GatherND limitation.
- **Don't confuse training time vs inference time**. The hardware limits apply only to inference time, not training.
