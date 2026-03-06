# Embedding Training: Self-Supervised Learning for Patch Matching

This package trains embedding representations optimized for attention-based matching in optical flow estimation.

## Motivation

Training embeddings jointly with the flow model presents several bottlenecks:

- **Training speed**: The computational graph is deeply connected—attention layers, hierarchical processing, and spatial operations create complex gradient paths. This makes training slow despite few parameters.

- **Gradient noise**: Flow estimation requires large batch sizes for stable gradients (flow direction tends to be consistent across image pairs). Embeddings could learn from individual image pairs since each pixel provides an independent signal.

- **Weak supervisory signal**: Embedding updates get diluted through multiple attention and blending operations before affecting the flow loss. The signal-to-noise ratio is poor.

**Strategy**: Isolate embedding learning with simpler, focused objectives that directly optimize for patch matching. This serves two purposes:
1. Enable faster experimentation by decoupling from flow training complexity
2. Understand whether embeddings are a bottleneck or if the issue lies elsewhere in the flow model

Outcome is intentionally open: embeddings may be pre-trained and frozen, fine-tuned within flow, or abandoned if flow training improves through other means.

## Core Idea

What makes a good embedding for attention-based matching?

Given that flow estimation relies on cross-frame attention, **sharper attention → less ambiguous matching → more confident flow**. The goal is to learn embeddings that produce peaked attention distributions rather than diffuse ones.

This is essentially learning a **matching cost** without explicit supervision. Instead of optimizing through the flow loss, we use proxy objectives:

1. **Within-frame**: Each patch should have a clear identity relative to neighbors (peaked self-attention)
2. **Cross-frame**: Each patch should find at most 1-2 corresponding patches in the other frame (peaked cross-attention)

Note: "Sharp" doesn't mean a single pixel match. Small clusters around the true match are acceptable and expected. The objective is to penalize diffuse attention, not to enforce unrealistic precision.

## Loss Functions

Both losses minimize entropy of attention distributions—pushing the model to commit to specific matches rather than hedging.

### Self-Attention Entropy

For a window of patches, each patch's embedding should stand out among its neighbors. However:
- Self-match is trivial (embedding always peaks with itself)
- Nearby pixels are expected to be similar

**Approach**: Compute attention over the entire window, then apply soft spatial weighting before entropy calculation:
1. Compute raw attention weights via softmax over dot products
2. Multiply element-wise by distance-based weights (lower weight for nearby positions, higher for distant)
3. Renormalize to get adjusted distribution
4. Compute entropy on adjusted distribution

This uses a Gaussian kernel (similar to `barevision.flow.token_attention.SpatialScore`) to softly downweight nearby positions without hard masking.

### Cross-Attention Entropy

For patches in frame t and a search window in frame t+1:
- Compute cross-frame attention weights
- Minimize entropy to encourage peaked distributions

**Acknowledgment**: Not all patches are matchable. Textureless regions, occlusions, and ambiguous areas will naturally produce diffuse attention. The loss should tolerate this rather than force false confidence. Future work may include confidence-aware weighting or explicit unmatchable region detection.

### Combined Loss

```
loss = α * self_entropy + β * cross_entropy
```

No regularization term is planned initially—let the sharpness objectives drive the learning. We may add terms later if collapse modes emerge.

## Integration with Flow

Embeddings must support the hierarchical coarse-to-fine estimation used in `barevision.flow`. Key constraints:

- **Multi-scale**: Coarse levels inform finer levels, enabling large flow capture and hierarchical refinement
- **Interface flexibility**: The current `EmbeddingPyramid` interface is a starting point, not a constraint. We may redefine how embeddings are structured across scales
- **Inference performance**: Resolution mismatches between embedding levels won't be solved with interpolation if it hurts inference speed. We're willing to compromise resolution or use tricks that preserve performance

The integration point is intentionally open for experimentation.
