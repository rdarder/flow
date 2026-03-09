# Implementation Plan: Flow Module (Self-Supervised Embedding Training)

## Overview

This module implements self-supervised embedding training for patch matching in optical flow estimation. The module trains embeddings that produce sharp cross-frame attention (for precise matching) while tolerating diffuse or spatially-localized self-attention (avoiding false uniqueness).

**Reference Documents:**
- `ARCHITECTURE.md` - Design rationale and loss function specifications

**Key Design Decisions:**
- **Self-attention entropy**: MAXIMIZE after spatial penalty (nearby peaks get flattened, far peaks remain sharp → penalized)
- **Cross-attention entropy**: MINIMIZE (encourage 1-2 sharp matches across frames)
- **Architecture v0**: Depthwise(3→48) + ReLU + 1x1(48→16), valid convolutions only
- **Window size**: 16×16 non-overlapping
- **Dataset**: Video frames with sparse pairing (t, t+1) through (t, t+k)

---

## Current Status

**Completed:**
- ✅ Embedding model architecture (`model.py`)
- ✅ Loss functions (`loss.py`) - self-attention and cross-attention entropy
- ✅ Training loop (`train.py`)
- ✅ Video dataset loader (`video_dataset.py`)
- ✅ Visualization utilities (`visualization.py`)
- ✅ Logging utilities (`logging_utils.py`)
- ✅ Unit tests for all components

**Usage:**
```bash
# Training
python -m barevision.flow.train --model.window-size 16 --dataset.img-size 200 200

# Smoke test
python -m barevision.flow.train --smoke-test

# Run tests
pytest src/barevision/flow/
```

---

## Future Improvements

### Loss Function Enhancements
1. **Add reconstructive loss** to the composite objective
2. **Spatial weighting** for self-attention - forgive nearby matches, penalize distant peaks
3. **Occlusion handling** - optional masking for unmatchable regions

### Architecture Improvements
1. **Overlapping windows** for better peer propagation
2. **Multi-scale embeddings** - pyramid of embeddings at different resolutions
3. **Confidence calibration** - learn temperature per level

### Training Improvements
1. **Larger datasets** - expand beyond current video collection
2. **Curriculum learning** - start with easy pairs, progress to harder cases
3. **Mixed precision** - faster training on supported hardware

---

## Module Structure

```
barevision/flow/
├── __init__.py           # Package initialization
├── ARCHITECTURE.md       # Design documentation
├── PLAN.md              # This file
├── model.py             # SimpleEmbeddingModel
├── loss.py              # Loss functions
├── train.py             # Training loop
├── video_dataset.py     # Dataset loading
├── visualization.py     # Diagnostic visualizations
├── logging_utils.py     # Logging utilities
├── settings.py          # Configuration with tyro CLI
├── test_*.py            # Unit tests
```

---

## Verification

**Quick smoke test:**
```bash
python -m barevision.flow.train --smoke-test
```

**Full test suite:**
```bash
pytest src/barevision/flow/
```

**Import validation:**
```python
from barevision.flow import SimpleEmbeddingModel
from barevision.flow.loss import compute_embedding_losses
```
