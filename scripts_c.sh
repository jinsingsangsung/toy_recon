CUDA_VISIBLE_DEVICES=1 python3 train_variants.py --variant c --model base --outf c_base
CUDA_VISIBLE_DEVICES=1 python3 train_variants.py --variant c --model transformer --outf c_transformer
CUDA_VISIBLE_DEVICES=1 python3 train_variants.py --variant c --model s4 --outf c_s4
CUDA_VISIBLE_DEVICES=1 python3 train_variants.py --variant c --model s4d --outf c_s4d
CUDA_VISIBLE_DEVICES=1 python3 train_variants.py --variant c --model s4nd --outf c_s4nd