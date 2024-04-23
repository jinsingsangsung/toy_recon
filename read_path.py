import torch
import argparse
from model_all import HNeRV
import numpy as np

parser = argparse.ArgumentParser(description="")
parser.add_argument("--file", default="")
parser.add_argument('--embed', type=str, default='', help='empty string for HNeRV, and base value/embed_length for NeRV position encoding')
parser.add_argument('--ks', type=str, default='0_3_3', help='kernel size for encoder and decoder')
parser.add_argument('--enc_strds', type=int, nargs='+', default=[5,4,4,2], help='stride list for encoder')
parser.add_argument('--enc_dim', type=str, default='72_16', help='enc latent dim and embedding ratio')
parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')

# Decoding parameters: FC + Conv
parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
parser.add_argument('--lower_width', type=int, default=32, help='lowest channel width for output feature maps')
parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list for decoder')
parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
    help='conv type for encoder/decoder', choices=['pshuffel', 'conv', 'convnext', 'interpolate'])
parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
    choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')

args = parser.parse_args()

# Compute the parameter number
if 'pe' in args.embed or 'le' in args.embed:
    embed_param = 0
    embed_dim = int(args.embed.split('_')[-1]) * 2
    fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
else:
    args.final_size = 6553600
    args.full_data_length = 132
    total_enc_strds = np.prod(args.enc_strds)
    embed_hw = args.final_size / total_enc_strds**2
    enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
    embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
    embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
    args.enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
    fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9

decoder_size = args.modelsize * 1e6 - embed_param
ch_reduce = 1. / args.reduce
dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
b =  embed_dim * fc_param 
c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
args.fc_dim = int(np.roots([a,b,c - decoder_size]).max())






pretrain_dir = args.file
checkpoint = torch.load(pretrain_dir, map_location='cpu')
ckpt = checkpoint["state_dict"]
model = HNeRV(args)

import pdb; pdb.set_trace()
print(ckpt.keys())

