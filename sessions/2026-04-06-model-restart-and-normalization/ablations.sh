#!/bin/bash
# Ablation study for embedding model architecture
# Runs sequential training experiments to find optimal hyperparameters

set -e

echo "============================================================"
echo "EMBEDDING MODEL ABLATION STUDY"
echo "============================================================"
echo ""
echo "Each ablation trains for 10 epochs with full dataset."
echo "Results saved to checkpoints/ with run-specific prefixes."
echo ""
echo "Ablations:"
echo "  1. baseline       - compact=4, multiplier=8, embed_dim=16"
echo "  2. channel_div    - compact=8, multiplier=4, embed_dim=16"
echo "  3. more_spatial   - compact=4, multiplier=12, embed_dim=16"
echo "  4. larger_embed   - compact=4, multiplier=8, embed_dim=24"
echo "  5. dense_proj     - compact=4, multiplier=8, project_groups=1"
echo ""
echo "Starting ablations..."
echo ""

# Common settings for all ablations
COMMON="--training.epochs=10 --dataset.num-workers=4"

# 1. Baseline: compact=4, multiplier=8, embed_dim=16
echo "============================================================"
echo "ABLATION 1/5: baseline"
echo "  compact=4, multiplier=8, embed_dim=16, project_groups=4"
echo "============================================================"
python -m barevision.embeddings.training \
    $COMMON \
    --run-name-prefix="ablation_1_baseline" \
    --model.compact-channels=4 \
    --model.depthwise-multiplier=8 \
    --model.embed-dim=16 \
    --model.project-groups=4

# 2. Channel diversity: compact=8, multiplier=4, embed_dim=16
echo ""
echo "============================================================"
echo "ABLATION 2/5: channel_div"
echo "  compact=8, multiplier=4, embed_dim=16, project_groups=4"
echo "============================================================"
python -m barevision.embeddings.training \
    $COMMON \
    --run-name-prefix="ablation_2_channel_div" \
    --model.compact-channels=8 \
    --model.depthwise-multiplier=4 \
    --model.embed-dim=16 \
    --model.project-groups=4

# 3. More spatial filters: compact=4, multiplier=12, embed_dim=16
echo ""
echo "============================================================"
echo "ABLATION 3/5: more_spatial"
echo "  compact=4, multiplier=12, embed_dim=16, project_groups=4"
echo "============================================================"
python -m barevision.embeddings.training \
    $COMMON \
    --run-name-prefix="ablation_3_more_spatial" \
    --model.compact-channels=4 \
    --model.depthwise-multiplier=12 \
    --model.embed-dim=16 \
    --model.project-groups=4

# 4. Larger embeddings: compact=4, multiplier=8, embed_dim=24
echo ""
echo "============================================================"
echo "ABLATION 4/5: larger_embed"
echo "  compact=4, multiplier=8, embed_dim=24, project_groups=4"
echo "============================================================"
python -m barevision.embeddings.training \
    $COMMON \
    --run-name-prefix="ablation_4_larger_embed" \
    --model.compact-channels=4 \
    --model.depthwise-multiplier=8 \
    --model.embed-dim=24 \
    --model.project-groups=4

# 5. Dense projection: compact=4, multiplier=8, embed_dim=16, project_groups=1
echo ""
echo "============================================================"
echo "ABLATION 5/5: dense_proj"
echo "  compact=4, multiplier=8, embed_dim=16, project_groups=1"
echo "============================================================"
python -m barevision.embeddings.training \
    $COMMON \
    --run-name-prefix="ablation_5_dense_proj" \
    --model.compact-channels=4 \
    --model.depthwise-multiplier=8 \
    --model.embed-dim=16 \
    --model.project-groups=1

echo ""
echo "============================================================"
echo "ALL ABLATIONS COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to checkpoints/"
echo "Run TensorBoard to compare:"
echo "  tensorboard --logdir checkpoints/"
echo ""
