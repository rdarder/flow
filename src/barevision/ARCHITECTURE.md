# Code Map: Barevision.

This document maps concepts to source files. It tells you **what is where**, not how things work. Read the code for details.

---

## Quick Navigation

| Component | Package | Key Files |
|-----------|---------|-----------|
| Embedding Pyramid | `embeddings/` | `model.py`, `losses.py` |
| Training | `embeddings/` | `training.py` |
| Data Loading | `dataset/` | `video.py` |
| Configuration | `embeddings/`| `settings.py` |

---

## Package Structure

```
barevision/
├── embeddings/              # Standalone embeddings training (PRIMARY)
│   ├── model.py            # EmbeddingBlock, HierarchicalEmbeddingModel, BlendMLP
│   ├── moe.py              # Rank1ExpertGroup, GatedPointwiseMoE, BlendMLP
│   ├── variance.py         # compute_per_channel_variance utility
│   ├── losses.py           # CombinedEmbeddingLoss (spatial variance + diversity)
│   ├── training.py         # EmbeddingsTrainer (Phase A), ConvergenceTrainer (Phase B)
│   ├── visualization.py    # Diagnostic visualizations
│   └── test_*.py           # Unit tests
│
│
├── dataset/video.py         # Data loading
├── settings.py              # Hyperparameters (tyro CLI)
├── checkpointer.py          # Checkpoint management
├── logging_utils.py         # Logging utilities
└── mean_conv_analysis.py    # Mean conv diagnostics
```

---

## Entry Points

```bash
# Phase A: Generalist-only training (base convolutions without experts)
python -m barevision.embeddings.training \
    --model.moe.use_expert=false \
    --training.epochs=10

pytest src/barevision
```
---

## Common Workflows

| Task | Files to Modify |
|------|-----------------|
| Change pyramid architecture | `embeddings/model.py`, `settings.py` |
| Modify MoE architecture | `embeddings/moe.py`, `settings.py` |
| Change convergence training | `embeddings/training.py` (ConvergenceTrainer) |
| Adjust loss functions | `embeddings/losses.py` |
| Adjust hyperparameters | `settings.py` |
| Add visualization | `embeddings/visualization.py` or `matching/visualization.py` |

---

## Related Documentation

- **Conceptual Overview**: [`/ARCHITECTURE.md`](../../ARCHITECTURE.md) — System design, algorithm rationale, training strategy
- **MoE Specification**: [`/moe-spec.md`](../../moe-spec.md) — Detailed MoE design, convergence training rationale
- **Implementation Tracker**: [`/moe-implementation-tracker.md`](../../moe-implementation-tracker.md) — Current implementation status
