#!/bin/bash

#python -m barevision.embeddings.training \
#  --run-name-prefix=ablation_0_baseline

## Config 1: No mean subtraction
#python -m barevision.embeddings.training \
#  --model.no-use-mean-subtraction \
#  --run-name-prefix=ablation_1_no_mean_sub

# Config 2: No L2 norm
python -m barevision.embeddings.training \
  --model.no-use-l2-norm \
  --run-name-prefix=ablation_2_no_l2

# Config 3: No contrast normalization (mean sub + L2 off)
python -m barevision.embeddings.training \
  --model.no-use-mean-subtraction \
  --model.no-use-l2-norm \
  --run-name-prefix=ablation_3_no_contrast

# Config 7: No normalization at all
python -m barevision.embeddings.training \
  --model.no-use-group-norm \
  --model.no-use-mean-subtraction \
  --model.no-use-l2-norm \
  --run-name-prefix=ablation_7_no_norm

## Config 8: No preprocessor (first block takes RGB directly)
#python -m barevision.embeddings.training \
#  --model.no-use-preprocessor \
#  --model.no-use-group-norm \
#  --model.no-use-mean-subtraction \
#  --model.no-use-l2-norm \
#  --run-name-prefix=ablation_8_no_preproc

# Config 9: No mean conv for downsampling (direct strided slice)
python -m barevision.embeddings.training \
  --model.no-use-mean-conv-for-downsampling \
  --run-name-prefix=ablation_9_no_mean_conv_ds
