CUDA_VISIBLE_DEVICES=3 python3 train_variants.py --variant a --model base --outf a_base
CUDA_VISIBLE_DEVICES=3 python3 train_variants.py --variant a --model transformer --outf a_transformer
CUDA_VISIBLE_DEVICES=3 python3 train_variants.py --variant a --model s4 --outf a_s4
CUDA_VISIBLE_DEVICES=3 python3 train_variants.py --variant a --model s4d --outf a_s4d
CUDA_VISIBLE_DEVICES=3 python3 train_variants.py --variant a --model s4nd --outf a_s4nd
# CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant a --model mamba --outf a_mamba
# CUDA_VISIBLE_DEVICES=0 python3 train_variants.py --variant a --model hippo --outf a_hippo
