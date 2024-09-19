import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
from matplotlib.path import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
import decord
decord.bridge.set_bridge('torch')
import glob
from mambaconv import MambaConv2d, MambaGlobalConv2d, MambaUpConv2d, MambaConv2dVariant, S4NDConv2d
from einops import rearrange
import pickle

# Video dataset
class Cifar(Dataset):
    def __init__(self, args):
        # self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
        data_path = os.path.join(args.data_path, "test_batch")
        with open(data_path, "rb") as fo:
            self.cifar_images = torch.from_numpy(pickle.load(fo, encoding = "bytes")[b"data"])
        if args.dataset_length < 10000:
            self.cifar_images = self.cifar_images[:args.dataset_length]

        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list

        first_frame = self.img_load(0)
        self.h, self.w = first_frame.size(-2), first_frame.size(-1)
        self.final_size = self.h * self.w

    def img_load(self, idx):
        img = self.cifar_images[idx].reshape(3, 32, 32) # c h w
        return img / 255.

    def __len__(self):
        return len(self.cifar_images)

    def __getitem__(self, idx):
        sample = self.img_load(idx)
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
        self.encoder = nn.Conv2d(3, 8, 4, 4) # output: 8,8,8 (#param: 512)
        # self.act = nn.GELU()
        # self.norm = LayerNorm(8, eps=1e-6, data_format="channels_first")
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 48, 1, 1),
            nn.PixelShuffle(4)
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
        self.encoder = MambaConv2dVariant(3, 8, 4, 4) # output: 12,8,8 (#param: 768)
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

class SSMConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = MambaConv2d(3, 12, 4, 4) # output: 12,8,8 (#param: 768)
        self.decoder = nn.Sequential(
            nn.Conv2d(12, 48, 1, 1),
            nn.PixelShuffle(4)
        )
        self.out_bias = args.out_bias

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = self.encoder(input[None])
        output = self.decoder(img_embed)
        img_out = OutImg(output, self.out_bias)[0]
        return  img_out

class S4NDConv(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 4, 4),
            S4NDConv2d(8, 64, 8, 8) # output: 8,8,8 (#param: 512)
        )
        # self.act = nn.SiLU()
        # self.norm = LayerNorm(12, eps=1e-6, data_format="channels_first")
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 48, 1, 1),
            nn.PixelShuffle(4)
        )
        self.out_bias = args.out_bias

    def forward(self, input):
        H, W = input.shape[-2:]
        img_embed = self.encoder(input[None])
        output = self.decoder(img_embed)
        img_out = OutImg(output, self.out_bias)[0]
        return  img_out

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
        act_layer = Sin
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
