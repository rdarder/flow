# Progress: Embedding Model Simplification

## Checkpointing

Training now checkpoints at epoch boundaries instead of step intervals. `CheckpointSettings.every_epochs` controls checkpoint frequency (0 disables periodic checkpointing). Checkpointing mandates validation—if both are due on the same epoch, validation runs once and the result is used for checkpoint preservation. Validation can occur independently (controlled by `ValidationSettings.every_epochs`). Checkpoints are preserved based on validation loss: the N best checkpoints (lowest val_loss) are kept. Training always starts from epoch 1; checkpoint restoration loads model weights only, no step/epoch resumption.

## Logging

Step timing now reports accurate steps/sec within each epoch using 1-indexed step counts (`step_in_epoch` passed through the logging chain).

## Model Architecture

The embedding pyramid uses a unified `EmbeddingBlock` for all levels, preceded by a `Preprocessor` that expands RGB input to hidden dimension. This structure eliminates the previous StemBlock/StandardBlock split—all blocks are now identical. The preprocessor is a separate module, enabling independent ablation of the initial dense convolution without modifying block logic. Computation and parameter counts are preserved from the previous architecture.

## Normalization Ablation Infrastructure

Five model components are now configurable via `ModelSettings` flags:
- `use_preprocessor`: Controls Preprocessor layer (RGB→hidden_dim expansion). When False, first EmbeddingBlock takes 3-channel RGB directly with dense conv (num_groups=1)
- `use_group_norm`: Controls GroupNorm after convolutions in Preprocessor and EmbeddingBlock
- `use_mean_subtraction`: Controls local contrast normalization (subtracting local mean from features)
- `use_mean_conv_for_downsampling`: Controls whether downsampling uses mean_conv output or strided slice of rich_features
- `use_l2_norm`: Controls L2 normalization of output embeddings

All flags default to `True`, preserving baseline behavior. When disabled, corresponding parameters are not created (reducing parameter count). Training header logs active configuration.

## Optimizer

Training uses AdamW optimizer with gradient clipping (max norm 1.0). Switched from Adam to AdamW to provide better regularization baseline for normalization ablation studies.
