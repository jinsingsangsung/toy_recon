# CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant d --model base --outf d_base
CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant d --model transformer --outf d_transformer
# CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant d --model s4 --outf d_s4: running
CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant d --model s4d --outf d_s4d
# CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant d --model s4nd --outf d_s4nd: running