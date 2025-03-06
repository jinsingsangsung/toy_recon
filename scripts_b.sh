CUDA_VISIBLE_DEVICES=2 python3 train_variants.py --variant b --model base --outf b_base
CUDA_VISIBLE_DEVICES=2 python3 train_variants.py --variant b --model transformer --outf b_transformer
CUDA_VISIBLE_DEVICES=2 python3 train_variants.py --variant b --model s4 --outf b_s4
CUDA_VISIBLE_DEVICES=2 python3 train_variants.py --variant b --model s4d --outf b_s4d
CUDA_VISIBLE_DEVICES=2 python3 train_variants.py --variant b --model s4nd --outf b_s4nd
