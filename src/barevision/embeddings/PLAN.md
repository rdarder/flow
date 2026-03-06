# Implementation Plan: Barevision Embeddings Training Module

## Overview

This plan implements self-supervised embedding training for patch matching in optical flow estimation. The module trains embeddings that produce sharp cross-frame attention (for precise matching) while tolerating diffuse or spatially-localized self-attention (avoiding false uniqueness).

**Reference Documents:**
- `src/barevision/embeddings/ARCHITECTURE.md` - Design rationale and loss function specifications
- `src/barevision/embeddings/EXPERIMENTS.md` - Architecture variants and experiment tracking

**Key Design Decisions:**
- **Self-attention entropy**: MAXIMIZE after spatial penalty (nearby peaks get flattened, far peaks remain sharp → penalized)
- **Cross-attention entropy**: MINIMIZE (encourage 1-2 sharp matches across frames)
- **Architecture v0**: Depthwise(3→12) + ReLU + 1x1(12→16), valid convolutions only
- **Window size**: 16×16 non-overlapping (matches `barevision.flow`)
- **Dataset**: Video frames with sparse pairing (t, t+1) through (t, t+k)
- **Train/val split**: 13 videos for training, 2 for validation (contiguous frames per video)

---

## Phase 1: Extract Grid Utilities

**Goal**: Create `barevision.utils.grid` module with window operations used by both flow and embeddings.

**Files Created:**
- `src/barevision/utils/__init__.py` - Package initialization
- `src/barevision/utils/grid.py` - Extracted from `src/barevision/flow/window_grid.py`

**Functions to Extract:**
- `compute_valid_resolution()` / `validate_resolution()` - Resolution validation
- `WindowGrid` class - Split/stitch operations for 16×16 windows
- `create_coordinate_grid()` - Normalized position grids
- `grid_to_tokens()` / `tokens_to_grid()` - Flattening utilities

**Verification:**
```bash
# Verify imports work from both packages
python -c "from barevision.utils.grid import WindowGrid; print('OK')"
python -c "from barevision.flow.window_grid import WindowGrid; print('OK')"

# Run flow smoke test to ensure no regression
python -m barevision.flow.train --smoke-test
```

**Acceptance Criteria:**
- ✅ `barevision.utils.grid` imports successfully
- ✅ `barevision.flow.train --smoke-test` passes (no breaking changes)

---

## Phase 2: Embedding Model Architecture

**Goal**: Implement v0 baseline embedding model.

**Files Created:**
- `src/barevision/embeddings/model.py` - `SimpleEmbeddingModel` class

**Architecture:**
```
Input: (B, H, W, 3) RGB
  ↓
3×3 depthwise conv: 3 in → 12 out (4 filters per channel)
  ↓
ReLU
  ↓
1×1 conv: 12 in → 16 out
  ↓
Output: (B, H-2, W-2, 16) embeddings
```

**Key Details:**
- No padding (valid convolutions only)
- Output is 2 pixels smaller than input on each dimension
- Grayscale variant supported via `in_channels` parameter

**Verification:**
```bash
# Model instantiation test
python -c "
from barevision.embeddings.model import SimpleEmbeddingModel
from flax import nnx
import jax.numpy as jnp
import jax.random as jr

model = SimpleEmbeddingModel(embed_dim=16, in_channels=3, rngs=nnx.Rngs(jr.PRNGKey(0)))
x = jnp.ones((1, 32, 32, 3))
y = model(x)
print(f'Input: {x.shape} → Output: {y.shape}')
assert y.shape == (1, 30, 30, 16), f'Unexpected shape: {y.shape}'
print('OK')
"
```

**Acceptance Criteria:**
- ✅ Model instantiates without errors
- ✅ Forward pass produces expected output shape `(B, H-2, W-2, 16)`
- ✅ Parameter count ~326 (documented in code comments)

---

## Phase 3: Loss Functions

**Goal**: Implement self-attention and cross-attention entropy losses.

**Files Created:**
- `src/barevision/embeddings/loss.py` - Loss function implementations

**Functions:**
- `self_attention_entropy_loss(embeddings, window_size=16)` - Returns per-pixel loss
- `cross_attention_entropy_loss(emb1, emb2, window_size=16)` - Returns per-pixel loss
- `combined_loss(self_loss, cross_loss, alpha=1.0, beta=1.0)` - Weighted combination

**Implementation Details:**

**Self-Attention (MAXIMIZE entropy after spatial penalty):**
```python
# For each pixel in 16×16 window:
# 1. Compute dot products with all pixels in window
# 2. Mask exact source pixel (logit = -1e9)
# 3. Subtract spatial penalty: logit_j -= scale * distance²(source, j)
# 4. Softmax → attention weights
# 5. Compute entropy
# 6. Return -entropy (maximize → minimize negative)
```

**Cross-Attention (MINIMIZE entropy):**
```python
# For each pixel in frame1, attend to frame2:
# 1. Compute dot products across frames
# 2. Softmax (no spatial penalty)
# 3. Compute entropy
# 4. Return entropy (minimize)
```

**Verification:**
```bash
# Loss function sanity checks
python -c "
from barevision.embeddings.loss import self_attention_entropy_loss, cross_attention_entropy_loss
import jax.numpy as jnp
import jax.random as jr

# Create dummy embeddings (batch=1, 32×32 spatial, 16-dim)
emb = jr.normal(jr.PRNGKey(0), (1, 32, 32, 16))
emb2 = jr.normal(jr.PRNGKey(1), (1, 32, 32, 16))

# Self-attention loss (should be positive, finite)
self_loss = self_attention_entropy_loss(emb)
assert self_loss.shape == (1, 32, 32), f'Unexpected shape: {self_loss.shape}'
assert jnp.isfinite(self_loss).all(), 'Self loss contains NaN/Inf'
print(f'Self-attention loss range: [{self_loss.min():.3f}, {self_loss.max():.3f}]')

# Cross-attention loss (should be positive, finite)
cross_loss = cross_attention_entropy_loss(emb, emb2)
assert cross_loss.shape == (1, 32, 32), f'Unexpected shape: {cross_loss.shape}'
assert jnp.isfinite(cross_loss).all(), 'Cross loss contains NaN/Inf'
print(f'Cross-attention loss range: [{cross_loss.min():.3f}, {cross_loss.max():.3f}]')

print('OK')
"
```

**Acceptance Criteria:**
- ✅ Both losses return per-pixel values with correct shape
- ✅ All values are finite (no NaN/Inf)
- ✅ Self-attention loss can be maximized (negative entropy)
- ✅ Cross-attention loss can be minimized (positive entropy)
- ✅ Gradients flow correctly (verified via `jax.grad`)

---

## Phase 4: Video Dataset Loader

**Goal**: Load video frames with sparse pairing for self-supervised training.

**Files Created:**
- `src/barevision/embeddings/video_dataset.py` - `VideoFrameDataset` class

**Dataset Structure:**
```
datasets/frames/
├── backyard/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── bookshelf/
└── ...
```

**Configuration:**
- **Training videos**: backyard through table (13 videos, alphabetically)
- **Validation videos**: toys, workshop (2 videos)
- **Frame pairing**: (t, t+1) through (t, t+k) for all valid t, where k ≤ max_distance
- **Default max_distance**: 5
- **Output format**: `(img1, img2, video_name, frame_t, frame_t+k, distance)`

**Verification:**
```bash
# Dataset loader test
python -c "
from barevision.embeddings.video_dataset import VideoFrameDataset
from torch.utils.data import DataLoader

# Training dataset
train_dataset = VideoFrameDataset(
    data_root='datasets/frames',
    split='train',
    max_frame_distance=5,
    img_size=(190, 190)
)
print(f'Training samples: {len(train_dataset)}')

# Validation dataset
val_dataset = VideoFrameDataset(
    data_root='datasets/frames',
    split='val',
    max_frame_distance=5,
    img_size=(190, 190)
)
print(f'Validation samples: {len(val_dataset)}')

# Test data loader
loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))
img1, img2, video_name, frame_t, frame_tk, distance = batch
print(f'Batch img1 shape: {img1.shape}')
print(f'Batch img2 shape: {img2.shape}')
print(f'Video names: {video_name}')
print(f'Frame pairs: {frame_t[0].item()} → {frame_tk[0].item()}, distance={distance[0].item()}')

# Verify shapes and dtypes
assert img1.shape == (4, 190, 190, 3), f'Unexpected img1 shape: {img1.shape}'
assert img2.shape == (4, 190, 190, 3), f'Unexpected img2 shape: {img2.shape}'
assert img1.dtype == jnp.float32, f'Unexpected dtype: {img1.dtype}'
assert img1.min() >= 0 and img1.max() <= 1, 'Image values should be in [0, 1]'

print('OK')
"
```

**Acceptance Criteria:**
- ✅ Training and validation splits load correctly
- ✅ Frame pairs span distances 1 through max_distance
- ✅ Images are RGB, normalized to [0, 1], shape (B, H, W, 3)
- ✅ Metadata includes video name and frame indices
- ✅ No frame leakage between train/val (contiguous splits per video)

---

## Phase 5: Settings and Configuration

**Goal**: Define CLI configuration using tyro dataclasses.

**Files Created:**
- `src/barevision/embeddings/settings.py` - Configuration dataclasses

**Settings Classes:**
```python
@dataclass
class ModelSettings:
    embed_dim: int = 16
    in_channels: int = 3

@dataclass
class DatasetSettings:
    data_root: str = "datasets/frames"
    max_frame_distance: int = 5
    img_size: Tuple[int, int] = (190, 190)
    batch_size: int = 16
    num_workers: int = 4

@dataclass
class LossSettings:
    self_entropy_weight: float = 1.0
    cross_entropy_weight: float = 1.0
    window_size: int = 16

@dataclass
class TrainingSettings:
    learning_rate: float = 1e-4
    epochs: int = 50
    steps_per_epoch: int = -1  # -1 = full dataset
    log_every_steps: int = 50
    checkpoint_freq: int = 500
    checkpoint_dir: str = "checkpoints/embeddings"
    keep_last_n_checkpoints: int = 3
    grad_clip_norm: float = 0.0
    seed: int = 42
    resume: bool = False

@dataclass
class LoggingSettings:
    log_dir: str = "runs"
    run_name_prefix: str = "embeddings"
    num_visualization_samples: int = 4

@dataclass
class Settings:
    model: ModelSettings
    dataset: DatasetSettings
    loss: LossSettings
    training: TrainingSettings
    logging: LoggingSettings
```

**Smoke Test Settings:**
```python
def create_smoke_test_settings() -> Settings:
    """Minimal settings for quick validation."""
    return Settings(
        model=ModelSettings(embed_dim=16, in_channels=3),
        dataset=DatasetSettings(
            img_size=(128, 128),  # Small for speed
            max_frame_distance=2,
            batch_size=4,
            num_workers=0,
        ),
        loss=LossSettings(self_entropy_weight=1.0, cross_entropy_weight=1.0),
        training=TrainingSettings(
            epochs=2,
            steps_per_epoch=10,
            log_every_steps=5,
            checkpoint_freq=5,
            checkpoint_dir="test_checkpoints/embeddings",
        ),
        logging=LoggingSettings(run_name_prefix="smoke_test"),
    )
```

**Verification:**
```bash
# Test CLI parsing
python -c "
from barevision.embeddings.settings import Settings, create_smoke_test_settings
import tyro

# Test smoke test settings
settings = create_smoke_test_settings()
print(f'Smoke test: {settings.dataset.img_size}, {settings.training.epochs} epochs')

# Test CLI override (simulated)
settings = tyro.cli(Settings, default=create_smoke_test_settings(), args=[
    '--dataset.batch-size', '8',
    '--training.epochs', '5'
])
print(f'Override test: batch_size={settings.dataset.batch_size}, epochs={settings.training.epochs}')

print('OK')
"
```

**Acceptance Criteria:**
- ✅ All settings classes validate correctly
- ✅ Tyro CLI parsing works with nested dataclasses
- ✅ Smoke test settings produce minimal viable config
- ✅ CLI overrides work as expected

---

## Phase 6: Logging and Checkpointing Utilities

**Goal**: Implement TensorBoard logging and Orbax checkpoint management.

**Files Created:**
- `src/barevision/embeddings/logging_utils.py` - `JaxLogger` class
- `src/barevision/embeddings/checkpoint_manager.py` - Checkpoint manager (copy flow pattern)

**JaxLogger Features:**
- `log_scalar(tag, value, step)` - Loss, learning rate, etc.
- `log_image(tag, image, step)` - Visualizations
- `log_histogram(tag, values, step)` - Parameter/gradient distributions
- Automatic run naming with timestamps

**CheckpointManager Features:**
- `should_save(step)` - Interval-based checkpoint decisions
- `save(step, model, optimizer, epoch)` - Save model state
- `restore(model, optimizer)` - Resume from checkpoint
- `latest_step()` - Get most recent checkpoint step

**Verification:**
```bash
# Logger test
python -c "
from barevision.embeddings.logging_utils import JaxLogger
import numpy as np

logger = JaxLogger(log_dir='test_runs', run_name_prefix='test')
logger.log_scalar('Loss/train', 0.5, 0)
logger.log_scalar('Loss/train', 0.4, 1)
logger.log_image('Test/image', np.random.rand(64, 64, 3), 0)
logger.log_histogram('Test/params', np.random.randn(100), 0)
logger.close()
print('Logger OK')
"

# Checkpoint manager test
python -c "
from barevision.embeddings.checkpoint_manager import create_checkpoint_manager
from barevision.embeddings.model import SimpleEmbeddingModel
from flax import nnx
import jax.random as jr

# Create model and optimizer
model = SimpleEmbeddingModel(rngs=nnx.Rngs(jr.PRNGKey(0)))
optimizer = nnx.Optimizer(model, nnx.optim.Adam(1e-4))

# Create checkpoint manager
manager = create_checkpoint_manager(
    checkpoint_dir='test_checkpoints/test',
    save_interval_steps=5,
    max_to_keep=2,
    enabled=True
)

# Test should_save
assert not manager.should_save(0)
assert not manager.should_save(1)
assert manager.should_save(5)
assert not manager.should_save(6)

# Test save/restore
manager.save(step=5, model=model, optimizer=optimizer, epoch=0)
assert manager.latest_step() == 5

manager.close()
print('Checkpoint manager OK')
"
```

**Acceptance Criteria:**
- ✅ Logger writes to TensorBoard successfully
- ✅ Checkpoint save/restore works
- ✅ `should_save()` returns correct intervals
- ✅ Old checkpoints are pruned according to `keep_last_n_checkpoints`

---

## Phase 7: Visualization Functions

**Goal**: Implement diagnostic visualizations for training monitoring.

**Files Created:**
- `src/barevision/embeddings/visualization.py` - Visualization functions

**Visualizations:**

**1. Loss Heatmap (`create_loss_heatmap_figure`)**
- Per-pixel loss overlaid on input frame
- Separate figures for self-entropy and cross-entropy
- Color scale: green (low) → red (high)

**2. Attention Map Sampling (`create_attention_maps_figure`)**
- Select 4-8 "interesting" pixels per batch:
  - 2 with highest self-attention entropy (most ambiguous)
  - 2 with lowest self-attention entropy (sharpest)
  - 2 with highest cross-attention entropy
  - 2 with lowest cross-attention entropy
- Display 16×16 attention map for each as heatmap

**3. Embedding Similarity Matrix (`create_similarity_matrix_figure`)**
- Select one 16×16 window
- Compute patch-to-patch similarity matrix
- Display as heatmap (diagonal should be sharp for self-attention)

**Verification:**
```bash
# Visualization test
python -c "
from barevision.embeddings.visualization import (
    create_loss_heatmap_figure,
    create_attention_maps_figure,
    create_similarity_matrix_figure,
)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Create dummy data
img1 = jr.uniform(jr.PRNGKey(0), (190, 190, 3))
img2 = jr.uniform(jr.PRNGKey(1), (190, 190, 3))
self_loss = jr.uniform(jr.PRNGKey(2), (190, 190))
cross_loss = jr.uniform(jr.PRNGKey(3), (190, 190))

# Test loss heatmap
fig = create_loss_heatmap_figure(img1, self_loss, cross_loss)
assert isinstance(fig, np.ndarray)
assert fig.dtype == np.uint8
print(f'Loss heatmap shape: {fig.shape}')

# Test attention maps (simplified - needs actual embeddings)
from barevision.embeddings.model import SimpleEmbeddingModel
from barevision.embeddings.loss import self_attention_entropy_loss
from flax import nnx

model = SimpleEmbeddingModel(rngs=nnx.Rngs(jr.PRNGKey(0)))
emb = model(img1[None, ...])

fig = create_attention_maps_figure(img1, emb[0], window_size=16)
assert isinstance(fig, np.ndarray)
print(f'Attention maps shape: {fig.shape}')

print('Visualizations OK')
"
```

**Acceptance Criteria:**
- ✅ All visualization functions return valid RGB numpy arrays
- ✅ Loss heatmaps show spatial variation
- ✅ Attention maps are selectable by entropy value
- ✅ Figures are TensorBoard-compatible (uint8, HWC format)

---

## Phase 8: Training Script

**Goal**: Implement main training loop with all components integrated.

**Files Created:**
- `src/barevision/embeddings/train.py` - Main training entry point

**Training Loop Structure:**
```python
class Trainer:
    def __init__(self, model, settings, checkpoint_manager, logger):
        # Initialize optimizer, dataloader, etc.
        
    def train_epoch(self, epoch):
        # Iterate over batches
        # Compute combined loss
        # Backpropagate
        # Log metrics
        # Save checkpoints
        
    def train(self):
        # Main training loop
        # Log visualizations per epoch
```

**Key Components:**
- Gradient clipping (configurable via `grad_clip_norm`)
- Learning rate logging
- Steps-per-second tracking
- Epoch-level visualization logging
- Checkpoint saving at intervals
- Resume from checkpoint support

**Verification (Smoke Test):**
```bash
# Run smoke test
python -m barevision.embeddings.train --smoke-test

# Expected output:
# - "Logging to runs/smoke_test_YYYYMMDD_HHMMSS"
# - "Step 0/10 | X.XX steps/sec"
# - "Checkpoint saved at step 5"
# - "Epoch 0: Loss = X.XXXX"
# - "Epoch 1: Loss = X.XXXX"
# - "Training complete!"
```

**Verification (Full Run - 1 Epoch):**
```bash
# Run 1 epoch for validation
python -m barevision.embeddings.train \
    --dataset.batch-size 8 \
    --training.epochs 1 \
    --training.steps-per_epoch 100 \
    --training.checkpoint-freq 50

# Check TensorBoard logs
tensorboard --logdir runs/

# Verify:
# - Loss decreases over steps
# - Self and cross entropy components are logged separately
# - Visualizations appear at epoch boundaries
# - Checkpoints saved to checkpoints/embeddings/
```

**Acceptance Criteria:**
- ✅ Smoke test completes in <2 minutes
- ✅ Loss decreases (or at least doesn't explode)
- ✅ TensorBoard logs show all metrics
- ✅ Visualizations logged at epoch boundaries
- ✅ Checkpoints saved and restorable
- ✅ `--resume` flag resumes from latest checkpoint

---

## Phase 9: Integration Testing

**Goal**: End-to-end validation of the complete training pipeline.

**Tests to Run:**

**1. Full Smoke Test with Validation:**
```bash
python -m barevision.embeddings.train --smoke-test

# Verify TensorBoard contains:
# - Loss/train_step (decreasing trend)
# - Loss/self_entropy_component
# - Loss/cross_entropy_component
# - Visualization/Loss_Heatmap (at epoch 0, 1)
# - Visualization/Attention_Maps (at epoch 0, 1)
```

**2. Checkpoint Resume Test:**
```bash
# Run 2 epochs
python -m barevision.embeddings.train \
    --smoke-test \
    --training.epochs 2 \
    --training.checkpoint-freq 5

# Resume from checkpoint
python -m barevision.embeddings.train \
    --smoke-test \
    --training.epochs 4 \
    --training.resume

# Verify:
# - Training resumes from epoch 2, not 0
# - Loss continues from previous value
# - No duplicate steps logged
```

**3. Gradient Flow Test:**
```bash
python -c "
from barevision.embeddings.train import Trainer, create_smoke_test_settings
from barevision.embeddings.settings import Settings
import jax

settings = create_smoke_test_settings()
settings.training.epochs = 1
settings.training.steps_per_epoch = 5

trainer = Trainer(settings)

# Check gradients are non-zero
img1, img2, metadata = next(iter(trainer.train_loader))
loss, grads = jax.value_and_grad(trainer.loss_fn)(trainer.model, img1, img2)

for name, grad in grads.items():
    assert jnp.abs(grad).max() > 0, f'Gradient for {name} is zero!'
    assert jnp.isfinite(grad).all(), f'Gradient for {name} contains NaN/Inf!'
    print(f'{name}: max_grad={jnp.abs(grad).max():.6f}')

print('Gradient flow OK')
"
```

**Acceptance Criteria:**
- ✅ Smoke test passes end-to-end
- ✅ Checkpoint resume works correctly
- ✅ All parameters receive non-zero gradients
- ✅ No NaN/Inf in loss or gradients
- ✅ TensorBoard shows expected visualizations

---

## Summary

| Phase | Files Created | Key Verification |
|-------|---------------|------------------|
| 1. Grid Utilities | `utils/grid.py`, `utils/__init__.py` | Flow smoke test passes |
| 2. Model | `embeddings/model.py` | Forward pass shape correct |
| 3. Loss Functions | `embeddings/loss.py` | Finite gradients, correct shapes |
| 4. Dataset | `embeddings/video_dataset.py` | Frame pairs span distances 1-k |
| 5. Settings | `embeddings/settings.py` | Tyro CLI parsing works |
| 6. Logging/Checkpoint | `logging_utils.py`, `checkpoint_manager.py` | Save/restore works |
| 7. Visualizations | `visualization.py` | Valid RGB arrays returned |
| 8. Training Script | `train.py` | Smoke test completes |
| 9. Integration | N/A | Full pipeline validated |

**Estimated Timeline**: 1-2 hours per phase with review between each.

**Dependencies** (verify in `pyproject.toml`):
- `jax`, `flax` (already present)
- `torch`, `torchvision` (already present for data loading)
- `tensorboard` (already present)
- `orbax-checkpoint` (already present)
- `tyro` (already present)
- `Pillow` (already present)
- `scikit-learn` - **NOT required** (t-SNE deferred)


