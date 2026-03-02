# Embedding Training: Self-Supervised Learning for Patch Matching

This package trains embedding representations optimized for attention-based matching in optical flow estimation.

> **Note**: This document describes the embedding training component of the Barevision project. For high-level project overview, see [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

## Motivation

Training embeddings alongside the full flow model is difficult because:
1. **Gradient complexity**: Attention layers, hierarchical model, and spatial processing create deeply interconnected gradients
2. **Weak signal**: Embedding updates get diluted through multiple attention and blending operations
3. **Collapsing risk**: Without proper constraints, embeddings can collapse to similar representations

**Solution**: Train embeddings separately with simpler, focused objectives that directly optimize for patch matching.

## Core Idea

What makes a good embedding for an image patch?

1. **Uniqueness within window**: The embedding should identify its patch uniquely within a local window (sharp self-attention)
2. **Precise cross-frame matching**: The embedding should find at most 1-2 corresponding patches in the next frame (sharp cross-attention)
3. **Robustness to appearance changes**: Small temporal changes should not break matching

## Loss Function Design

### Self-Attention Sharpness
For a window of patches `{p₁, p₂, ..., pₙ}` with embeddings `{e₁, e₂, ..., eₙ}`:
```
self_attention_ij = softmax(dot(eᵢ, eⱼ) / √d)
self_entropyᵢ = -Σⱼ self_attention_ij * log(self_attention_ij)
```
**Goal**: Minimize self-entropy (sharp attention diagonals)

### Cross-Attention Sharpness
For patches in frame `t` and corresponding window in frame `t+1`:
```
cross_attention_ij = softmax(dot(eᵢ⁽ᵗ⁾, eⱼ⁽ᵗ⁺¹⁾) / √d)
cross_entropyᵢ = -Σⱼ cross_attention_ij * log(cross_attention_ij)
```
**Goal**: Minimize cross-entropy (few strong matches, not many weak ones)

### Combined Loss
```
loss = α * self_entropy + β * cross_entropy + γ * regularization
```
Where regularization prevents collapse (e.g., uniform distribution penalty).

## Training Data Requirements

1. **"Single-take" video**: Continuous shots without cuts
2. **Temporal consistency**: Objects move smoothly between frames
3. **Diverse motion**: Translation, rotation, scaling
4. **Texture variety**: Both textured and textureless regions

**Dataset options**:
- ChairsSDHom (synthetic, controlled)
- Real video with cut detection
- Self-collected robot footage

## Architecture Interface

### Compatibility with `barevision.flow`
Embedding models must implement the same interface as `EmbeddingPyramid`:

```python
class EmbeddingModel(nnx.Module):
    def __init__(self, num_levels: int, embed_dim: int, in_channels: int, *, rngs: nnx.Rngs):
        ...
    
    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        # Returns [coarsest, ..., finest] embeddings
        ...
```

### Design Space for Experimentation
1. **Simple CNN**: Depthwise separable convolutions (current approach)
2. **Vision Transformer**: Patch embedding + transformer blocks
3. **ConvNeXt**: Modern convolutional architecture
4. **Hybrid**: CNN stem + transformer head
5. **Multi-scale**: Separate networks per pyramid level vs shared weights

## Training Pipeline

### Phase 1: Baseline
- Start with current `EmbeddingPyramid` architecture
- Train with sharpness loss on ChairsSDHom
- Evaluate: self/cross-attention entropy metrics

### Phase 2: Architecture Search
- Experiment with different embedding architectures
- Compare: training stability, final entropy, inference speed
- Select best candidate for integration

### Phase 3: Integration Test
- Replace `EmbeddingPyramid` in flow model with trained embeddings
- Fine-tune flow model (optional)
- Compare EPE (Endpoint Error) with original

## Key Challenges

### 1. Avoiding Collapse
- Regularization: Uniform distribution penalty, diversity loss
- Negative sampling: Hard negatives within window
- Gradient clipping: Prevent extreme updates

### 2. Scale Invariance
Embeddings should work across pyramid levels:
- Option A: Separate networks per level
- Option B: Shared network with level conditioning
- Option C: Learnable positional encodings for scale

### 3. Efficiency
- Target NPU constraints: No `gatherND`, limited memory
- Attention windows fixed at 16×16 (matches flow model)
- Embedding dimension ≤ 32 (memory/bandwidth trade-off)

## Evaluation Metrics

### Primary Metrics
1. **Self-attention entropy**: Lower is better (sharper diagonals)
2. **Cross-attention entropy**: Lower is better (fewer strong matches)
3. **Ranking metrics**: Precision@k for patch matching

### Secondary Metrics
1. **Integration EPE**: Endpoint error when used in flow model
2. **Inference latency**: Forward pass time on target hardware
3. **Memory footprint**: Parameters + activations

## Configuration

```python
@dataclass
class EmbeddingTrainingSettings:
    # Architecture
    model_type: str = "cnn"  # "cnn", "vit", "convnext", "hybrid"
    embed_dim: int = 16
    num_levels: int = 2
    
    # Loss
    self_entropy_weight: float = 1.0
    cross_entropy_weight: float = 1.0
    regularization_weight: float = 0.01
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
```

## Development Plan

### Week 1: Foundation
- Implement sharpness loss functions
- Create training loop for current `EmbeddingPyramid`
- Establish baseline metrics

### Week 2: Experimentation
- Implement alternative architectures
- Run ablation studies
- Identify promising candidates

### Week 3: Integration
- Create drop-in replacement interface
- Test with flow model
- Fine-tune if needed

### Week 4: Optimization
- NPU-specific optimizations
- Quantization experiments
- Documentation and release

## Integration with Larger Project

Embeddings are a foundational component for all perception tasks:

1. **Flow**: Patch matching for optical flow
2. **Depth**: Feature matching for stereo/monocular depth
3. **Pose**: Feature tracking for camera motion
4. **SLAM**: Long-term feature consistency

A well-trained embedding model should improve performance across all downstream tasks while maintaining NPU compatibility.