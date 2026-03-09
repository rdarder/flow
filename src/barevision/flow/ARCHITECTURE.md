# Flow: Self-Supervised Embedding Learning for Patch Matching

This package trains embedding representations optimized for attention-based matching in optical flow estimation.

## Core Idea

What makes a good embedding for attention-based matching?

Given that flow estimation relies on cross-frame attention, **sharper attention → less ambiguous matching → more confident flow**. The goal is to learn embeddings that produce peaked attention distributions rather than diffuse ones.

This is essentially learning a **matching cost** without explicit supervision. Instead of optimizing through the flow loss, we use proxy objectives:

1. **Within-frame**: Each patch should have a clear identity relative to neighbors (peaked self-attention)
2. **Cross-frame**: Each patch should find corresponding patches in the other frame (peaked cross-attention)

Minimizing entropy favors the fewest attention peaks—each peak forms a "signature" linking the source embedding to specific target locations.

## Loss Functions

Both losses minimize entropy of attention distributions—pushing the model to commit to specific matches rather than hedging.

### Self-Attention Entropy

For a window of patches, each patch's embedding should stand out among its neighbors. Self-match is trivial (embedding always peaks with itself due to dot product properties).

**Approach**: Compute attention over the entire window via softmax over dot products, then minimize entropy. Low entropy means "only self should dominate, no other pixel competes"—encouraging unique embeddings.

### Cross-Attention Entropy

For patches in frame t and a search window in frame t+1:
- Compute cross-frame attention weights
- Minimize entropy to encourage peaked distributions

**Limitation**: Unmatchable regions (occlusions, textureless areas, motion boundaries) contribute to loss like any other pixel. We accept this noise rather than adding complexity for confidence weighting or masking.

### Combined Loss

```
loss = α * self_entropy + β * cross_entropy
```

Default weights: α=1.0, β=0.1. Cross-attention receives less weight since not all patches have reliable matches.

**Future**: A reconstructive loss will be added to the composite objective.

## Training Stabilizers: L2 Normalization + Temperature

Two mechanisms prevent attention collapse and ensure stable training:

### L2 Normalization

All embeddings are L2-normalized to unit norm before computing attention. Without normalization, embeddings grow unbounded to exaggerate dot products—a failure mode where a single high-norm embedding captures all attention regardless of content. Normalization constrains all embeddings to the unit sphere surface, eliminating norm-based competition.

### Temperature Scaling

Attention logits are divided by temperature τ=0.05 before softmax. With normalized embeddings, dot products range in [-1, 1] and produce similar values. Low temperature sharpens the softmax distribution, amplifying small differences to select clear winners. This combination (L2 norm + low temperature) produces stable, discriminative attention without collapse.

## Training Data

Video frame pairs from single continuous takes (no cuts). Temporal continuity is guaranteed by dataset structure—frames are loaded from video directories in sequence without cross-video mixing.

## Configuration

Window size is configurable (default 16×16). Input image dimensions must be divisible by window size after accounting for valid convolutions (output is 4 pixels smaller than input).

## Usage

```bash
# Training
python -m barevision.flow.train --model.window-size 16 --dataset.img-size 200 200

# Smoke test
python -m barevision.flow.train --smoke-test
```
