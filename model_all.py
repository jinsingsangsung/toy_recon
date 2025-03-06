import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from matplotlib.path import Path
from timm.models.layers import trunc_normal_, DropPath
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
import decord
decord.bridge.set_bridge('torch')
import glob
# from mambaconv import MambaConv2d, MambaGlobalConv2d, MambaUpConv2d, MambaConv2dVariant, S4NDConv2d
from einops import rearrange
import pickle
from einops import rearrange, repeat
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.modules.conv import _ConvNd
from torch import Tensor
from parallel_scan import EfficientParallelScanFunction, ParallelScanFunction
from s4 import S4Block
from s4d import S4D
from torch.nn.init import kaiming_normal_, normal_
from scipy.linalg import block_diag
from functools import partial
from ssmconv import SSMConv2d, SSMConv2dv8, SSMConv2dv4, SSMConv2dv9, SSMConv2dv10

_c2r = torch.view_as_real
contract = torch.einsum
from scipy import special as ss

def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures """
    if measure == 'lagt':
        # A_l = (1 - dt / 4) * np.eye(N) + dt / 2 * np.tril(np.ones((N, N)))
        # A_r = (1 + dt / 4) * np.eye(N) - dt / 2 * np.tril(np.ones((N, N)))
        # alpha = dt / 2 / (1 - dt / 4)
        # col = -alpha / (1 + alpha) ** np.arange(1, N + 1)
        # col[0] += 1
        # A_l_inv = la.toeplitz(col / (1 - dt / 4), np.zeros(N))
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        # b = measure_args['beta']
        b = measure_args.get('beta', 1.0)
        A = (1.-b)/2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1)[:, None] # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        B = (-1.)**Q[:, None] * R

    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
    return A, B

def batch_solve_triangular(A, b, upper=False):
    """
    A: shape (batch_size, n, n)
    b: shape (batch_size, n) or (batch_size, n, m)
    """
    return torch.triangular_solve(b.unsqueeze(-1) if b.dim() == 2 else b, A, upper=upper)[0]

def construct_A_B_stacked(A, B, T, discretization='bilinear'):
    """
    A: shape (N, N)
    B: shape (N)
    """
    N, _ = A.shape
    device = A.device
    dtype = A.dtype

    t_range = torch.arange(1, T + 1, device=device, dtype=dtype).view(T, 1, 1)
    
    At = A.unsqueeze(0).expand(T, N, N) / t_range
    Bt = B.unsqueeze(0).expand(T, N) / t_range.squeeze(-1)
    
    eye_N = torch.eye(N, device=device, dtype=dtype)
    eye_N_stacked = eye_N.unsqueeze(0).expand(T, N, N)
    
    if discretization == 'forward':
        A_stacked = eye_N_stacked + At
        B_stacked = Bt
    elif discretization == 'backward':
        A_stacked = torch.triangular_solve(eye_N_stacked, eye_N_stacked - At, upper=False)[0]
        B_stacked = torch.triangular_solve(Bt.unsqueeze(-1), eye_N_stacked - At, upper=False)[0].squeeze(-1)
    elif discretization == 'bilinear':
        A_stacked = torch.triangular_solve(eye_N_stacked + At / 2, eye_N_stacked - At / 2, upper=False)[0]
        B_stacked = torch.triangular_solve(Bt.unsqueeze(-1), eye_N_stacked - At / 2, upper=False)[0].squeeze(-1)
    elif discretization == 'zoh':
        log_t = torch.log(t_range + 1) - torch.log(t_range)
        A_stacked = torch.matrix_exp(A.unsqueeze(0) * log_t)
        B_stacked = torch.triangular_solve(A_stacked @ B.unsqueeze(0).unsqueeze(-1) - B.unsqueeze(0).unsqueeze(-1), A.unsqueeze(0).expand(T, N, N), upper=False)[0].squeeze(-1)
    else:
        raise ValueError(f"Unknown discretization method: {discretization}")
    
    return A_stacked, B_stacked

class HippoConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, # means nothing, just for compatibility
        d_state: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,  # means nothing, just for compatibility
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        measure: str = "legs",
        discretization: str = "bilinear",
        dt: float = 0.27,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode,)
        if stride != kernel_size:
            stride = kernel_size
        self.N = d_state
        self.measure = measure
        assert discretization in ["zoh", "bilinear", "forward", "backward"], \
            "discretization must be one of 'zoh', 'bilinear', 'forward' or 'backward'"
        self.discretization = discretization
        self.dt = dt
        N = d_state
        A, B = transition(self.measure, N)
        B = B.squeeze(-1)
        A = torch.tensor(A, **factory_kwargs).float()
        B = torch.tensor(B, **factory_kwargs).float()
        T = kernel_size**2
        A_stacked, B_stacked = construct_A_B_stacked(A, B, T, discretization=discretization)
        self.A_stacked = A_stacked.requires_grad_(False).contiguous()
        self.B_stacked = B_stacked.requires_grad_(False).contiguous()
        vals = np.linspace(0.0, 1.0, T)
        B_cpu = B.cpu().numpy()
        self.eval_matrix = torch.Tensor((B_cpu[:, None] * ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)).T).float().contiguous()
        self.linear = nn.Linear(N, 1)

    def zigzag_flatten(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C, H, W]
        output: torch.Tensor of size [B, C, H*W],
        where spatial tokens are aligned in zigzag manner
        '''
        B, C, H, W = input.shape
        # flattened zigzag indices
        zigzag = torch.arange(H*W).view(H, W)
        zigzag[1::2, :] = zigzag[1::2, :].flip(dims=[1])
        zigzag = zigzag.view(-1)
        # Rearrange the input tensor
        input = input.view(B, C, -1)
        input = input[:, :, zigzag]
        return input

    def forward(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C, H, W]
        output: torch.Tensor of size [B, C, N, H/K, W/K]
        where N: d_state, K: kernel_size
        '''
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            input = F.pad(input, self.padding*2)

        # patchify input sequences
        input = input.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        input = input.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        batch_size, _, H2, W2, _, _ = input.shape

        input = rearrange(input, "B C H2 W2 K1 K2 -> (B H2 W2) C K1 K2")
        input = self.zigzag_flatten(input) # B*H2*W2, C, K1*K2
        N = self.N
        if self.kernel_size[0]**2 > 32:
            scan_function = EfficientParallelScanFunction.apply
        else:
            scan_function = ParallelScanFunction.apply
        # if input.size(0) > 32:
        if False:
            # to regard GPU memory
            outputs = []
            num_batches = input.size(0) // 512 + 1
            for i in range(num_batches):
                start_idx = i * 512
                end_idx = min((i + 1) * 512, input.size(0))
                batch_input = input[start_idx:end_idx]
                batch_A = repeat(self.A_stacked, "T N1 N2 -> B C T N1 N2", B=batch_input.size(0), C=input.size(1), N1=N, N2=N).cuda()
                batch_B = repeat(self.B_stacked, "T N -> B C T N", B=batch_input.size(0), C=input.size(1), N=N).cuda()
                batch_output = scan_function(batch_input, batch_A, batch_B)
                outputs.append(batch_output.cpu())
                del batch_input, batch_A, batch_B, batch_output
                torch.cuda.empty_cache()
            output = torch.cat(outputs, dim=0).cuda()
        else:
            A = repeat(self.A_stacked, "T N1 N2 -> B C T N1 N2", B=input.size(0), C=input.size(1), N1=N, N2=N).contiguous().cuda()
            B = repeat(self.B_stacked, "T N -> B C T N", B=input.size(0), C=input.size(1), N=N).contiguous().cuda()
            output = scan_function(input, A, B)
            self.output = output

        output = rearrange(output[:,:,:,:], "(B H2 W2) C (K1 K2) N -> B C (H2 K1) (W2 K2) N", B=batch_size, H2=H2, W2=W2, K1=self.kernel_size[0], K2=self.kernel_size[1])
        output = self.linear(output)[..., 0]

        return output

    def reconstruct(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C*N, H/K, W/K]
        output: torch.Tensor of size [B, C, H, W]
        '''
        B, C, H, W = input.shape
        output = rearrange(input, "B (C N) H W -> B C N H W", B=B, C=C//self.N, N=self.N, H=H, W=W)
        output = torch.einsum("bcnhw,kn -> bckhw", output, self.eval_matrix.cuda())
        output = self.zigzag_flatten(rearrange(output, "B C (K1 K2) H W -> (B H W) C K1 K2", K1=self.kernel_size[0], K2=self.kernel_size[1]))
        output = rearrange(output, "(B H W) C (K1 K2) -> B C (H K1) (W K2)", B=B, H=H, W=W, K1=self.kernel_size[0], K2=self.kernel_size[1])
        return output


# Video dataset
class Cifar(Dataset):
    def __init__(self, args):
        # self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
        data_path = os.path.join(args.data_path, "test")
        with open(data_path, "rb") as fo:
            self.cifar_images = torch.from_numpy(pickle.load(fo, encoding = "bytes")[b"data"])
        if args.dataset_length < 10000:
            self.cifar_images = self.cifar_images[:args.dataset_length]

        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list

        first_frame = self.img_load(0)
        self.h, self.w = first_frame.size(-2), first_frame.size(-1)
        self.final_size = self.h * self.w
        self.variant = args.variant
        self.enc_strds = [2,2,2]

    def img_load(self, idx):
        img = self.cifar_images[idx].reshape(3, 32, 32) # c h w
        return img / 255.

    def __len__(self):
        return len(self.cifar_images)

    def __getitem__(self, idx):
        gt_img = self.img_load(idx)
        sample = {"gt_img": [gt_img]}
        if self.variant in ["b", "c"]:
            imgs = [gt_img] # graudally downscaled images
            for i, strd in enumerate(self.enc_strds):
                if i == 0:
                    imgs.append(F.avg_pool2d(gt_img, kernel_size=strd, stride=strd))
                else:
                    imgs.append(F.avg_pool2d(imgs[-1], kernel_size=strd, stride=strd))
            sample["downscaled"] = imgs
        if self.variant == "d":
            imgs = []
            for i, strd in enumerate(self.enc_strds):
                if i == 0:
                    imgs.append(F.avg_pool2d(gt_img, kernel_size=strd, stride=strd))
                else:
                    imgs.append(F.avg_pool2d(imgs[-1], kernel_size=strd, stride=strd))
            imgs_up = []
            for i, img in enumerate(imgs):
                if i == 0:
                    imgs_up.append(F.interpolate(imgs[i][None], size=gt_img.shape[-2:], mode='bilinear', align_corners=False)[0])
                else:
                    imgs_up.append(F.interpolate(imgs[i][None], size=imgs[i-1].shape[-2:], mode='bilinear', align_corners=False)[0])

            laplacians = [gt_img - imgs_up[0]]
            imgs_up.append(imgs_up[-1]) # dummy
            for i, (img, img_up) in enumerate(zip(imgs, imgs_up[1:])):
                if i < len(imgs) - 1:
                    laplacians.append(img - img_up)
                else:
                    laplacians.append(img)
            sample["laplacians"] = laplacians
        return sample

class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        if isinstance(conv, UpConv):
            print(kargs["ks"])
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'], 
            conv_type=kargs['conv_type'], bias=kargs['bias'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def Quantize_tensor(img_embed, quant_bit):
    out_min = img_embed.min(dim=1, keepdim=True)[0]
    out_max = img_embed.max(dim=1, keepdim=True)[0]
    scale = (out_max - out_min) / 2 ** quant_bit
    img_embed = ((img_embed - out_min) / scale).round()
    img_embed = out_min + scale * img_embed  
    return img_embed


def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


class SimpleConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 8, 8) # output: 8,4,4 (#param: 768)
        # self.act = nn.GELU()
        # self.norm = LayerNorm(8, eps=1e-6, data_format="channels_first")
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias

    def forward(self, input):
        img_embed = self.encoder(input)
        output = self.decoder(img_embed)
        img_out = OutImg(output, self.out_bias)
        return  img_out

class SSMConvVariant(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = MambaConv2dVariant(3, 8, 8, 8) # output: 8,4,4 (#param: 768)
        # self.act = nn.SiLU()
        # self.norm = LayerNorm(8, eps=1e-6, data_format="channels_first")
        self.decoder = nn.Sequential(
            nn.Conv2d(12, 3, 1, 1),
        )
        self.out_bias = args.out_bias

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = self.encoder(input[None])
        # expand: b c h w -> b c h' w' using the matrix C
        b, c, h, w = img_embed.shape
        img_embed = rearrange(img_embed, 'b c h w -> (b h w) 1 c')
        decode_mat = self.encoder.mamba_kernels[0].mixer.C
        output = torch.einsum("blc,blL->bLc", img_embed, decode_mat)
        output = self.encoder.mamba_kernels[0].mixer.out_proj(output)
        output = self.encoder.fused_add_norm(output, None)
        output = output.permute(0,2,1)
        # output = rearrange(output, "(b h w) c (k1 k2) -> b (c k1 k2) (h w)", b=b, c=c, h=h, w=w, k1=4, k2=4)
        # output = F.fold(output, (H, W), (4, 4), stride=(4, 4))
        output = rearrange(output, "(b h w) c (k1 k2) -> b c (h k1) (w k2)", b=b, c=c, h=h, w=w, k1=4, k2=4)
        output = self.decoder(output)
        img_out = OutImg(output, self.out_bias)[0]
        return  img_out

# from mamba_ssm import Mamba

class SSMConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.encoder = nn.ModuleList([
            # MambaConv2d(3, 3, 32, 32), # output: 12,32,32 (#param: 768)
            Mamba(d_model=3, d_state=16, d_conv=4, expand=2),
            nn.Conv2d(3, 8, 8, 8), # 8, 4, 4
        ])
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias
        self.ln = nn.LayerNorm(3)
        h, w = 32, 32
        indices = torch.zeros(h * w, dtype=torch.long)
        # Fill indices in zig-zag pattern
        idx = 0
        for i in range(h):
            if i % 2 == 0:  # Even rows go left to right
                for j in range(w):
                    indices[i*w + j] = idx
                    idx += 1
            else:  # Odd rows go right to left
                for j in range(w-1, -1, -1):
                    indices[i*w + j] = idx
                    idx += 1
        self.indices = indices

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = rearrange(input[None], "b c h w -> b (h w) c")
        img_embed = img_embed[:, self.indices]
        # from thop import profile
        # total_params = 0
        # for name, param in self.encoder[0].named_parameters():
        #     total_params += param.numel()
        # print(f"Total parameters in self.encoder[0]: {total_params:,}")
        img_embed = self.encoder[0](img_embed)
        img_embed = rearrange(img_embed[:, self.indices], "b (h w) c -> b c h w", h=H, w=W) + self.pos_embed
        img_embed = self.encoder[1](img_embed)
        output = self.decoder(img_embed)
        img_out = OutImg(output, self.out_bias)[0]
        return  img_out

class MySSMConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        # pos_embed = torch.zeros(1, 3, 32, 32)
        # nn.init.kaiming_normal_(pos_embed)
        # self.pos_embed = nn.Parameter(pos_embed)
        self.encoder = nn.ModuleList([
            # SSMConv2dv10(3, 8, (16, 16), (8, 8), (4, 4), bidirectional=True, state_size=16, adapt_C=True),
            SSMConv2dv10(3, 8, (8, 8), (8, 8), (0, 0), bidirectional=True, state_size=16, adapt_C=True),
            # nn.Conv2d(16, 8, 1, 1)
        ])
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias
        self.dct = args.dct

    def forward(self, input):
        # H, W = input.shape[-2:]
        # from thop import profile
        # macs, params = profile(self.encoder[0], inputs=(input[None],))
        # print(f"macs: {macs}, Params: {params}")
        # import pdb; pdb.set_trace()
        img_embed = self.encoder[0](input[None])
        # img_embed = self.encoder[1](img_embed.real)
        # output = self.encoder[0].reconstruct(img_embed)
        output = self.decoder(img_embed)
        if self.dct:
            img_out = F.sigmoid(output[0])*8
        else:
            img_out = OutImg(output, self.out_bias)[0]
        return  img_out
    
class PureTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.transformer = nn.Sequential( # 28999
            nn.Conv2d(3, 48, 1, 1),
            nn.MultiheadAttention(
                embed_dim=48,
                num_heads=8,
                batch_first=True
            ),
            # nn.LayerNorm([48, 48, 48]),
            nn.LayerNorm(48),
            nn.Sequential(
                nn.Linear(48, 196),
                nn.GELU(),
                nn.Linear(196, 48)
            ),
            nn.LayerNorm(48),
            nn.Conv2d(48, 3, 1, 1)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 8, 8), # 8, 4, 4
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        
        self.dct = args.dct
        if args.dct:
            self.proj = nn.Conv2d(3, 6, 1, 1)
        else:
            self.out_bias = args.out_bias


    def forward(self, input):
        H, W = input.shape[-2:]
        total_params = 0
        # for name, param in self.transformer.named_parameters():
        #     total_params += param.numel()
        # print(f"Total parameters in transformer: {total_params:,}")
        # for name, param in self.encoder.named_parameters():
        #     total_params += param.numel()
        # total_params += self.pos_embed.numel()
        # print(f"Total parameters in encoder + transformer: {total_params:,}")
        # import pdb; pdb.set_trace()
        img_embed = self.transformer[0](input[None] + self.pos_embed)
        img_embed = rearrange(img_embed, "B C H W -> B (H W) C")
        img_embed_trans = self.transformer[1](img_embed, img_embed, img_embed)[0]
        
        
        # img_embed = self.transformer[2](img_embed_trans + img_embed)
        # img_embed_ffn = self.transformer[3](img_embed)
        # img_embed = self.transformer[4](img_embed_ffn + img_embed)
        img_embed = img_embed_trans + img_embed
        img_embed = rearrange(img_embed, "B (H W) C -> B C H W", H=H, W=W)
        img_embed = self.transformer[5](img_embed)
        img_embed = self.encoder(img_embed)
        output = self.decoder(img_embed)
        if self.dct:
            img_out = F.sigmoid(output[0])*8
        else:
            img_out = OutImg(output, self.out_bias)[0]
        return  img_out        

class S4NDConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Sequential(
            # S4NDConv2d(3, 12, 4, 4), # output: 3,4,8,8 (#param: 512)
            nn.Conv2d(12, 12, 4, 4),
            # nn.Conv2d(4, 4, 4, 4), # down scale
            # nn.Conv2d(4, 4, 4, 4), # down scale
            # nn.Conv2d(4, 4, 4, 4), # down scale per channel
        )
        # self.act = nn.SiLU()
        # self.norm = LayerNorm(12, eps=1e-6, data_format="channels_first")
        self.decoder = nn.Sequential(
            nn.Conv2d(12, 48, 1, 1),
            # nn.Conv2d(4, 16, 1, 1),
            # nn.Conv2d(4, 16, 1, 1),
            # nn.Conv2d(4, 16, 1, 1),
            nn.PixelShuffle(4),
        ) 
        self.c_proj = nn.ModuleList([nn.Conv2d(4,16,1,1) for _ in range(3)])
        self.out_bias = args.out_bias
        self.A_decode = nn.ModuleList([
            nn.ModuleList([nn.Linear(2,64), nn.SiLU(), nn.Linear(64,4)]) for _ in range(2)
        ])
        self.B_decode = nn.ModuleList([
            nn.ModuleList([nn.Linear(2,64), nn.SiLU(), nn.Linear(64,4)]) for _ in range(2)
        ])
        # self.agg = nn.ModuleList([
        #     nn.Linear(4,1) for _ in range(2)
        # ])

    def generate_decode_tensor(self, N, A_list, B_list, img_embed):
        tgt_list = []
        for j, (A, B) in enumerate(zip(A_list, B_list)):
            vals = np.linspace(0.0, 1.0, 4)
            tgt = torch.linspace(0,1, 4, device=A.device)
            # tgt = ss.eval_legendre(np.arange(N)[:, None], 2 * vals - 1)
            # import pdb; pdb.set_trace()
            deg = torch.arange(0,N, device=tgt.device)
            A_ = self.A_decode[j][2](self.A_decode[j][1](self.A_decode[j][0](_c2r(A))))
            A_ = self.A_decode[j][2](self.A_decode[j][1](self.A_decode[j][0](A_.transpose(-1,-2))))
            B_ = self.B_decode[j][2](self.B_decode[j][1](self.B_decode[j][0](_c2r(B))))
            B_ = self.B_decode[j][2](self.B_decode[j][1](self.B_decode[j][0](B_.transpose(-1,-2))))
            tgt = repeat(tgt.pow(deg), "n -> b c n", b=B_.size(0), c=B_.size(1))
            # tgt = repeat(torch.tensor(tgt, device=B_.device), "m n -> b c m n", b=B_.size(0), c=B_.size(1))
            tgt_list.append(tgt[..., None] * (A_ + B_))
            # tgt_list.append(tgt)
        tgt_x, tgt_y = tgt_list[0], tgt_list[1]
        tgt_map = contract("bcmn,bcon->bcnmo", tgt_x, tgt_y)
        output = contract("cnhw,bcnkl->bchkwl", img_embed, tgt_map)
        output = rearrange(output, "b c h k1 w k2 -> b c (h k1) (w k2)")
        return output

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = self.encoder(input[None])
        # output = []
        # for j, c_embed in enumerate(img_embed.transpose(0,1)):
        #     output.append(self.encoder[j+1](c_embed))
        # img_embed = torch.stack(output, dim=1)
        # extract A
        A_list = [self.encoder[0].ssm_kernel.kernel[i]._get_params()[1] for i in range(2)]
        B_list = [self.encoder[0].ssm_kernel.kernel[i]._get_params()[2] for i in range(2)]
        # C_output = []
        # for i, c_ in enumerate(img_embed):
        #     C_output.append(self.c_proj[i](c_[None]))
        # output = self.generate_decode_tensor(4, A_list, B_list, img_embed)
        # C_output = torch.cat(C_output, dim=1)
        # C_output = self.decoder(C_output)
        # output = C_output + output
        # output = []
        # for i, c_output in enumerate(img_embed.transpose(0,1)):
        #     output.append(self.decoder[i](c_output))
        # output = torch.cat(output, dim=1)
        # output = self.decoder[-1](output)
        output = self.decoder(img_embed)
        # import pdb; pdb.set_trace()
        # output = output.transpose(0,1)
        img_out = OutImg(output, self.out_bias)[0]
        return  img_out, img_embed

# class S4NDConv(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             S4NDConv2d(3, 24, 32, 32) # output: 8,8,8 (#param: 512)
#         )
#         # self.act = nn.SiLU()
#         # self.norm = LayerNorm(12, eps=1e-6, data_format="channels_first")
#         self.decoder = nn.Sequential(
#             nn.Conv2d(8, 48, 1, 1),
#             nn.PixelShuffle(4)
#         )
#         self.out_bias = args.out_bias

#     def forward(self, input):
#         H, W = input.shape[-2:]
#         img_embed = self.encoder(input[None])
#         # output = self.decoder(img_embed)
#         # img_out = OutImg(output, self.out_bias)[0]
#         img_out = OutImg(img_embed, self.out_bias)[0]
#         return  img_out

class S4NDPure(nn.Module):
    def __init__(self, args):
        super().__init__()
        from src.models.sequence.modules.s4nd import S4ND
        self.encoder = nn.ModuleList([
            S4ND(3, 8),
            nn.Conv2d(3, 8, 8, 8),
            # S4ND(8, 4),
        ])
        # self.act = nn.SiLU()
        # self.norm = LayerNorm(12, eps=1e-6, data_format="channels_first")
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.out_bias = args.out_bias

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed, _ = self.encoder[0](input[None] + self.pos_embed)
        img_embed = self.encoder[1](img_embed)
        output = self.decoder(img_embed)
        img_out = OutImg(output, self.out_bias)[0]
        # img_out = OutImg(img_embed, self.out_bias)[0]
        return  img_out

class HiPPOConvPure(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Sequential(
            HippoConv2d(3, 3, 16, 32, 32), # output: 12,32,32 (#param: 768)
            nn.Conv2d(3, 8, 8, 8), # 8, 4, 4
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.hippo_feat = None
        self.dct = args.dct

    def forward(self, input):
        H, W = input.shape[-2:]
        if self.hippo_feat == None:
            hippo_feat = self.encoder[0](input[None])
            self.hippo_feat = hippo_feat.detach().clone()
            img_embed = self.encoder[1](hippo_feat  + self.pos_embed)
            # img_embed = self.encoder[1](hippo_feat)
        else:
            img_embed = self.encoder[1](self.hippo_feat + self.pos_embed)
            # img_embed = self.encoder[1](self.hippo_feat)
        output = self.decoder(img_embed)
        if self.dct:
            img_out = F.sigmoid(output[0])*8
        else:
            img_out = OutImg(output, self.out_bias)[0]
        return  img_out

class S4Pure(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.s4block = S4Block(d_model=3, d_state=16, transposed=True)
        self.ln = nn.LayerNorm(3)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 8, 8), # 8, 4, 4
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias
        self.dct = args.dct
        h, w = 32, 32
        indices = torch.zeros(h * w, dtype=torch.long)
        # Fill indices in zig-zag pattern
        idx = 0
        for i in range(h):
            if i % 2 == 0:  # Even rows go left to right
                for j in range(w):
                    indices[i*w + j] = idx
                    idx += 1
            else:  # Odd rows go right to left
                for j in range(w-1, -1, -1):
                    indices[i*w + j] = idx
                    idx += 1
        self.indices = indices
    
    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = rearrange(input[None], "b c h w -> b c (h w)")
        img_embed = img_embed[:, :, self.indices] + self.pos_embed.flatten(2)
        # import thop 
        # macs, params = thop.profile(self.s4block, inputs=(img_embed,))
        # print(f"macs: {macs}, Params: {params}")
        # for name, param in self.s4block.named_parameters():
        #     print(f"{name}: {param.shape}")
        # params += self.pos_embed.numel()
        # params += self.encoder[0].weight.numel() + self.encoder[0].bias.numel()
        # print(f"Total parameters in encoder + s4block: {params:,}")
        # import pdb; pdb.set_trace()
        img_embed = self.s4block(img_embed)[0]
        img_embed = img_embed[:, :, self.indices]
        # img_embed = self.ln(img_embed.transpose(1,2)).transpose(1,2)
        img_embed = rearrange(img_embed, "b c (h w) -> b c h w", h=H, w=W)
        img_embed = self.encoder(img_embed)
        output = self.decoder(img_embed)
        if self.dct:
            img_out = F.sigmoid(output[0])*8
        else:
            img_out = OutImg(output, self.out_bias)[0]
        return  img_out

class S4DPure(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.s4dblock = S4D(d_model=3, d_state=16, transposed=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 8, 8), # 8, 4, 4
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias
        h, w = 32, 32
        indices = torch.zeros(h * w, dtype=torch.long)
        # Fill indices in zig-zag pattern
        idx = 0
        for i in range(h):
            if i % 2 == 0:  # Even rows go left to right
                for j in range(w):
                    indices[i*w + j] = idx
                    idx += 1
            else:  # Odd rows go right to left
                for j in range(w-1, -1, -1):
                    indices[i*w + j] = idx
                    idx += 1
        self.indices = indices
        self.dct = args.dct

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = input[None]
        img_embed = rearrange(img_embed, "b c h w -> b c (h w)")
        img_embed = img_embed[:, :, self.indices] # + self.pos_embed.flatten(2)
        # from thop import profile
        # macs, params = profile(self.s4dblock, inputs=(img_embed,))
        # print(f"macs: {macs}, Params: {params}")
        # import pdb; pdb.set_trace()
        img_embed = self.s4dblock(img_embed)[0]
        img_embed = img_embed[:, :, self.indices]
        img_embed = rearrange(img_embed, "b c (h w) -> b c h w", h=H, w=W)
        # from thop import profile
        # macs, params = profile(self.encoder, inputs=(img_embed,))
        # print(f"macs: {macs}, Params: {params}")
        # import pdb; pdb.set_trace()
        img_embed = self.encoder(img_embed)
        output = self.decoder(img_embed)
        if self.dct:
            img_out = F.sigmoid(output[0])*8
        else:
            img_out = OutImg(output, self.out_bias)[0]
        return  img_out

class S5Pure(nn.Module):
    def __init__(self, args):
        super().__init__()
        from s5 import make_DPLR_HiPPO, SequenceLayer, S5SSM
        blocks = 4
        ssm_size = 16
        block_size = int(ssm_size / blocks)
        self.block_size = block_size
        
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks)) #.astype(np.complex64)
        Vinv = block_diag(*([Vc] * blocks)) #.astype(np.complex64)
        ssm =S5SSM(Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            H=3,
            P=16,
            C_init="trunc_standard_normal",
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
            conj_sym=False,
            clip_eigs=False,
            bidirectional=False)

        self.pos_embed = nn.Parameter(torch.zeros(1, 3, 32, 32))
        self.s5block = SequenceLayer(
            ssm=ssm, 
            dropout=0.0, 
            d_model=3, 
            activation="gelu", 
            training=True, 
            prenorm=False, 
            batchnorm=False, 
            bn_momentum=0.9, 
            step_rescale=1.0
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 8, 8), # 8, 4, 4
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 3*64, 1, 1),
            nn.PixelShuffle(8)
        )
        self.out_bias = args.out_bias
    
    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = rearrange(input, "c h w -> (h w) c")
        img_embed = self.s5block(img_embed)
        img_embed = rearrange(img_embed, "(h w) c -> c h w", h=H, w=W)
        img_embed = self.encoder(img_embed)
        output = self.decoder(img_embed)
        img_out = OutImg(output, self.out_bias)
        return  img_out


class HNeRVDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        for layer in self.decoder[1:]:
            output = layer(output) 
        output = self.head_layer(output)

        return  OutImg(output, self.out_bias)


###################################  Basic layers like position encoding/ downsample layers/ upscale blocks   ###################################
class PositionEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionEncoding, self).__init__()
        self.pe_embed = pe_embed
        if 'pe' in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split('_')[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        if 'pe' in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin()
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd !=1 else nn.Identity(),
                nn.Conv2d(ngf * strd**2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'])
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = nn.Conv2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2), bias=kargs['bias'])
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, ks+strd, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )
        
    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if  kargs['conv_type']  == 'pshuffel':
            self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        elif  kargs['conv_type']  == 'mamba':
            self.upconv = nn.Sequential(
                MambaUpConv2d(ngf, new_ngf, strd, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
            )            
        elif  kargs['conv_type']  == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2))
        elif  kargs['conv_type']  == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )

    def forward(self, x):
        return self.upconv(x)


class ModConv(nn.Module):
    def __init__(self, **kargs):
        super(ModConv, self).__init__()
        mod_ks, mod_groups, ngf = kargs['mod_ks'], kargs['mod_groups'], kargs['ngf']
        self.mod_conv_multi = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1)//2, groups=(ngf if mod_groups==-1 else mod_groups))
        self.mod_conv_sum = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1)//2, groups=(ngf if mod_groups==-1 else mod_groups))

    def forward(self, x):
        sum_att = self.mod_conv_sum(x)
        multi_att = self.mod_conv_multi(x)
        return torch.sigmoid(multi_att) * x + sum_att


###################################  Tranform input for denoising or inpainting   ###################################
def RandomMask(height, width, points_num, scale=(0, 1)):
    polygon = [(x, y) for x,y in zip(np.random.randint(height * scale[0], height * scale[1], size=points_num), 
                             np.random.randint(width * scale[0], width * scale[1], size=points_num))]
    poly_path=Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
    mask = poly_path.contains_points(coors).reshape(height, width)
    return 1 - torch.from_numpy(mask).float()


class TransformInput(nn.Module):
    def __init__(self, args):
        super(TransformInput, self).__init__()
        self.vid = args.vid
        if 'inpaint' in self.vid:
            self.inpaint_size = int(self.vid.split('_')[-1]) // 2

    def forward(self, img):
        inpaint_mask = torch.ones_like(img)
        if 'inpaint' in self.vid:
            gt = img.clone()
            h,w = img.shape[-2:]
            inpaint_mask = torch.ones((h,w)).to(img.device)
            for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                inpaint_mask[ctr_x - self.inpaint_size: ctr_x + self.inpaint_size, ctr_y - self.inpaint_size: ctr_y + self.inpaint_size] = 0
            input = (img * inpaint_mask).clamp(min=0,max=1)
        else:
            input, gt = img, img

        return input, gt, inpaint_mask.detach()


###################################  Code for ConvNeXt   ###################################
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]

class MambaConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6, multiscale_mamba=True,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 2:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        MambaConv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            elif i in [1,2]:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )                
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.multiscale_mamba = multiscale_mamba
        if multiscale_mamba:
            from mambaconv import create_block
            from mamba_ssm.ops.triton.layernorm import RMSNorm
            factory_kwargs = {'device': None, 'dtype': None}
            d_state = 64
            drop_path = 0.
            rms_norm = True
            mamba_channels = 4*64
            self.mamba_block = create_block(
                d_model=mamba_channels,
                d_state=d_state,
                rms_norm = rms_norm,
                bimamba_type = "v2",
                if_devide_out=True,
                **factory_kwargs,
            )
            self.dim_matcher = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=1,stride=1)
            self.dim_matcher2 = nn.Conv2d(in_channels=4*64,out_channels=16,kernel_size=1,stride=1)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
               mamba_channels, eps=1e-5, **factory_kwargs
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def fused_add_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))   

    def multi_scale_mamba(self, multi_scale_list):
        segmented_list = []
        stride = [32,16,4,2,1]
        b, c, h, w = multi_scale_list[-1].shape
        for i, feature_map in enumerate(multi_scale_list):
            # og_shape = feature_map.shape
            if i == 0:
                continue
            else:
                if i == len(multi_scale_list)-1:
                    feature_map = self.dim_matcher(feature_map)
                feature_map = feature_map.unfold(dimension=2, size=stride[i], step=stride[i])
                feature_map = feature_map.unfold(dimension=3, size=stride[i], step=stride[i]) # b, c, h, w, k, k
                feature_maps = []
                for i, dir in enumerate([(),(4),(5),(4,5)]):
                    dir_input = feature_map.flip(dir)
                    feature_maps.append(dir_input)
                feature_map = torch.cat(feature_maps, dim=1)
                feature_map = rearrange(feature_map, 'b c h w k1 k2 -> (b h w) (k1 k2) c')
            segmented_list.append(feature_map)
        multi_scale_maps = torch.cat(segmented_list, dim=1)
        multi_agg_feature_map = rearrange(self.fused_add_norm(*self.mamba_block(multi_scale_maps))[:, -1, :], "(b h w) c -> b c h w", h=h, w=w)
        return self.dim_matcher2(multi_agg_feature_map)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        if self.multiscale_mamba:
            out_list.append(self.multi_scale_mamba(out_list))
        return out_list[-1]


class MambaDecoder(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=12, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i == 0:
                downsample_layer = nn.Sequential(
                        MambaUpConv2d(in_chans, dims[i], kernel_size=strds[i], stride=strds[i], dim_change=True),
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                )                
            elif i in [1,2]:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        MambaUpConv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i], dim_change=i<3),
                )
            else:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        UpConv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i], dim_change=i<3),
                )        
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def initialize_upscale(self, weight: list):
        for i in range(self.stage_num):
            self.downsample_layers[i][int(i!=0)].initialize_upscale(weight[i])

    def initialize_maps(self, weight: list):
        for i in range(self.stage_num):
            self.downsample_layers[i][int(i!=0)].initialize_maps(weight[i])

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        # print([a.shape for a in out_list])
        return out_list[-1]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
