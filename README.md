# Reconstructing 1D, 2D signals with different SSM models

This repository provides ways to reconstruct 1D, 2D signals with different SSM models: S4, S4D, S4ND, S5, Mamba

## commandline examples:

### 1D

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --model s4d --variant a --outf 0509_a_s4d_1d --lr 0.08 --dataset_length 1000
```

### 2D

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_variants_2d.py --model s4d --variant a --outf 0509_a_s4d_2d --lr 0.08 --dataset_length 1000
```
