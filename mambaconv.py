import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functools import partial

from einops import rearrange, repeat
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.modules.conv import _ConvNd
from typing import Optional, List, Tuple, Union
from mamba_ssm.modules.mamba_simple_og import Mamba, Block
from pos_embedding import build_position_encoding
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class MambaConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        rms_norm: bool = False,
        drop_path: float = 0.,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        # assert out_channels % in_channels == 0, \
        #     f"output channel size {out_channels} must be divisible by input channel size {in_channels}"
        d_state = kernel_size**2 // 2
        num_kernels = 1
        # self.mamba_kernels = nn.ModuleList([
        #     partial(Mamba, d_state=d_state, layer_idx=layer_idx, **factory_kwargs)(in_channels)
        #     for layer_idx in range(num_kernels)
        # ])
        self.pos_embed = build_position_encoding(N_steps=in_channels//2)
        self.agg_token_pe = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.agg_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.mamba_kernels = nn.ModuleList([
            create_block(
                d_model=in_channels,
                d_state=d_state,
                layer_idx = i,
                rms_norm = rms_norm,
                **factory_kwargs,
            )
            for i in range(num_kernels)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            in_channels, eps=1e-5, **factory_kwargs
        )        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)

    def fused_add_norm(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))   

    def forward(self, input: Tensor) -> Tensor:
        '''
        input: torch.Tensor of size [B, C, H, W]
        '''
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            input = F.pad(input, self.padding*2)

        # patchify input sequences
        input = input.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        input = input.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        B, C, H2, W2, K1, K2 = input.shape
        pos_embedding = self.pos_embed(input)
        pos_embedding = repeat(pos_embedding, "B C K1 K2 -> B C H2 W2 K1 K2", H2=H2, W2=W2)
        pos_embedding = rearrange(pos_embedding, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        agg_token_pe = self.agg_token_pe.expand(B*H2*W2, -1, -1)
        agg_token = self.agg_token.expand(B*H2*W2, -1, -1)
        input = rearrange(input, "B C H2 W2 K1 K2 -> (B H2 W2) (K1 K2) C")
        
        agg_token_position = K1*K2 // 2
        input = torch.cat((input[:, :agg_token_position, :], agg_token, input[:, agg_token_position:, :]), dim=1)
        pos = torch.cat((pos_embedding[:, :agg_token_position, :], agg_token_pe, pos_embedding[:, agg_token_position:, :]), dim=1)
        input = input + pos
        # import pdb; pdb.set_trace()
        output = torch.cat([rearrange(self.fused_add_norm(*kernel(input))[:, -1, :], "(B H2 W2) C -> B C H2 W2", H2=H2, W2=W2)
                  for kernel in self.mamba_kernels], dim=1)
        output = self.conv(output)
        return output

def create_block(
    d_model,
    d_state,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state=d_state, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

# def main():
#     B, C, H, W = 2, 3, 150, 320
#     input = torch.rand(B, C, H, W).cuda()
#     mamba_convolution = MambaConv2D(in_channels=3,
#                                     out_channels=12,
#                                     kernel_size=5,
#                                     stride=1,
#                                     padding=1,
#                                     device=input.device)
#     output = mamba_convolution(input)
    

# if __name__ == '__main__':
#     main()
        