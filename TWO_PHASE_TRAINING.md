# Two-Phase Training for Optical Flow

This guide explains how to train the optical flow model in two phases to avoid conservative flow predictions.

## The Problem

Training with only small motions (e.g., `max_frame_distance=2`) causes the model to learn a conservative prior - it predicts near-zero flow even for large displacements. This happens because:
1. The model rarely sees large motion during training
2. Zero flow is a safe default when uncertain
3. The FlowEstimator output is initialized with small weights

## Solution: Two-Phase Training

### Phase 1: Learn Distinctive Embeddings (10 epochs)

Train with very low reconstruction weight so embeddings learn to be distinctive without interference from noisy flow predictions:

```bash
python -m barevision.flow.training \
    --dataset.min_frame_distance=1 \
    --dataset.max_frame_distance=5 \
    --model.recon_weight=0.01 \
    --model.flow_temperature=0.2 \
    --training.epochs=10 \
    --checkpoint.save_final \
    --checkpoint.every_steps=100
```

**What happens:**
- Entropy loss (99%) dominates → embeddings become sharp and distinctive
- Tiny reconstruction signal (1%) → slight temporal consistency
- Flow Estimator learns basic tracking

### Phase 2: Learn Large Motion Tracking (20+ epochs)

Resume from Phase 1's best checkpoint and train with larger motions and higher reconstruction weight:

```bash
python -m barevision.flow.training \
    --dataset.min_frame_distance=5 \
    --dataset.max_frame_distance=15 \
    --model.recon_weight=0.5 \
    --model.flow_temperature=0.2 \
    --training.epochs=20 \
    --checkpoint.resume_from=checkpoints/run_NAME/best
```

**What happens:**
- Embeddings are already distinctive (from Phase 1)
- Now optimized for trackability across large displacements
- Flow Estimator learns to handle large motions
- Continues from where Phase 1 left off (step counter preserved)

## Key Parameters

| Parameter | Phase 1 | Phase 2 | Rationale |
|-----------|---------|---------|-----------|
| `min_frame_distance` | 1 | 5 | Phase 2 ensures minimum motion |
| `max_frame_distance` | 5 | 15 | Phase 2 covers inference range |
| `recon_weight` | 0.01 | 0.5 | Start conservative, then emphasize flow |
| `flow_temperature` | 0.2 | 0.2 | Sharp attention for precise centroids |
| `epochs` | 10 | 20 | Phase 2 needs more time to learn |

## Monitoring Progress

Watch TensorBoard for:
- **Entropy loss**: Should decrease steadily in Phase 1, stabilize in Phase 2
- **Reconstruction loss**: May be high initially in Phase 2, should decrease as Flow Estimator learns
- **Flow magnitude**: Should increase in Phase 2 as model learns to predict larger motions

## Tips

1. **Don't skip Phase 1**: Jumping straight to high `recon_weight` can cause embeddings to collapse to trivial solutions
2. **Adjust based on results**: If flow is still too conservative after Phase 2, increase `recon_weight` to 1.0
3. **Save best model**: Both phases use `--checkpoint.save_final` and automatic best model saving
4. **Continue from best**: Always resume from `best/` not `final/` to get the best checkpoint from Phase 1

## Example: Complete Training Session

```bash
# Phase 1: Embedding pre-training
python -m barevision.flow.training \
    --dataset.max_samples=1000 \
    --dataset.min_frame_distance=1 \
    --dataset.max_frame_distance=5 \
    --model.recon_weight=0.01 \
    --training.epochs=10 \
    --checkpoint.save_final \
    --checkpoint.every_steps=500

# Check TensorBoard, verify entropy loss is decreasing...

# Phase 2: Large motion training
python -m barevision.flow.training \
    --dataset.max_samples=1000 \
    --dataset.min_frame_distance=5 \
    --dataset.max_frame_distance=15 \
    --model.recon_weight=0.5 \
    --training.epochs=20 \
    --checkpoint.resume_from=checkpoints/embeddings_TIMESTAMP/best
```

## Resume Feature Details

The `--checkpoint.resume_from` option:
- Loads model weights from the specified checkpoint
- Continues training from the saved step number
- Creates a new run in TensorBoard (to separate phases)
- Does NOT restore optimizer state (starts with fresh optimizer)

For most use cases, this simplified resume is sufficient. The model weights are what matter most.
