import torch.nn as nn
import numpy as np
from ssmconv import SSMConv2dv9
from model_all import HippoConv2d
from s4 import S4Block
from s4d import S4D
from einops import rearrange
from src.models.sequence.modules.s4nd import S4ND
# from mamba_ssm import Mamba
from functools import partial
import torch


def build_2d_sincos_pos_embed(channels, h, w):
    """
    Creates positional embeddings per channel
    Args:
        channels: number of channels (e.g., 3 for RGB)
        h, w: height and width of the tensor
    Returns:
        pos_embed of shape (channels, h, w)
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    
    # Create different frequency bands for each channel
    pos_embeds = []
    for c in range(channels):
        # Use different frequency bases for each channel
        omega_h = 1. / (10000 ** (torch.arange(h) / h))
        omega_w = 1. / (10000 ** (torch.arange(w) / w))
        
        # Compute embeddings for this channel
        emb_h = grid_h.unsqueeze(-1) * omega_h.unsqueeze(0)
        emb_w = grid_w.unsqueeze(-1) * omega_w.unsqueeze(0)
        
        # Combine using average of sin components
        pos_emb = (torch.sin(emb_h).mean(-1) + torch.sin(emb_w).mean(-1)) / 2
        pos_embeds.append(pos_emb)
    
    pos_embed = torch.stack(pos_embeds, dim=0)  # (channels, h, w)
    return pos_embed


class DummyIdentity(nn.Module):
    def __init__(self, in_channels):
        super(DummyIdentity, self).__init__()
        self.in_channels = in_channels
    
    def forward(self, x):
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, in_channels, dropout=0.0, h=32, w=32, kernel_size=8):
        super(TransformerBlock, self).__init__()
        self.inflation = nn.Conv2d(in_channels, d_model, 1, 1)
        self.transformer = nn.MultiheadAttention(d_model, 1, dropout=dropout)
        self.deflation = nn.Conv2d(d_model, in_channels, 1, 1)
        # Use PyTorch's built-in PositionalEncoding
        self.pos_encoder = nn.Parameter(build_2d_sincos_pos_embed(d_model, h, w), requires_grad=False)
    
    def forward(self, x):
        x = self.inflation(x)
        x = x + self.pos_encoder
        H, W = x.shape[-2:]
        x = rearrange(x, "B C H W -> B (H W) C")
        x = self.transformer(x, x, x)[0] # takes b (h w) c
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        x = self.deflation(x)
        return x
    
class SSMBlock(nn.Module):
    def __init__(self, model, d_state, in_channels, h, w, kernel_size=8):
        super(SSMBlock, self).__init__()
        self.model = model
        if model == "hippo":
            self.block = HippoConv2d(in_channels=in_channels, out_channels=in_channels, d_state=d_state, kernel_size=kernel_size, stride=kernel_size) # takes b c h w
        elif model == "s4":
            self.block = S4Block(d_state=d_state, d_model=in_channels, transposed=False) # takes b l c
        elif model == "s4d":
            self.block = S4D(d_state=d_state, transposed=False, d_model=in_channels) # takes b l c
        elif model == "s4nd":
            self.block = S4ND(d_state=d_state//2, d_model=in_channels) # takes b c h w
        elif model == "mamba":
            self.block = Mamba(d_state=d_state, d_model=in_channels) # takes b l c
        # elif model == "s5":
        #     self.block = partial(S5, d_state=d_state)
        else:
            self.block = DummyIdentity()
        
        if model in ["s4", "s4d", "mamba"]:
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
            self.need_flatten = True
            self.pos_encoder = nn.Parameter(build_2d_sincos_pos_embed(in_channels, h, w), requires_grad=False)
        else:
            self.need_flatten = False
    
    def forward(self, x):
        H, W = x.shape[-2:]
        if self.need_flatten:
            x = x + self.pos_encoder
            x = rearrange(x, "B C H W -> B (H W) C")
            if self.indices is not None:
                x = x[:, self.indices]
        x = self.block(x) # takes b (h w) c
        if isinstance(x, tuple): # some models return a tuple
            x = x[0]
        if self.need_flatten:
            x = x[:, self.indices]
            x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        return x

class Variants(nn.Module):
    def __init__(self,
                 model,
                 in_channels=3,
                 out_channels=8,
                 kernel_size=8,
                 d_state=16,
                 ):
        super(Variants, self).__init__()
        if model == "base":
            self.block = nn.Identity()
        elif model == "transformer":
            self.block = partial(TransformerBlock, d_model=d_state)
        else:
            self.block = partial(SSMBlock, model=model, d_state=d_state)

        self.decoder = nn.Sequential(
            nn.Conv2d(out_channels, in_channels*kernel_size**2, 1, 1),
            nn.PixelShuffle(kernel_size) if kernel_size !=1 else nn.Identity(),
        )

        self.encoder = nn.Conv2d(in_channels, out_channels, kernel_size, kernel_size)

    def forward(self, x):
        raise NotImplementedError

class Vanilla(Variants):
    def __init__(self, model, in_channels, out_channels, kernel_size=8):
        super(Vanilla, self).__init__(model, in_channels, out_channels, kernel_size)

        self.encoder = nn.Sequential(
            self.block(in_channels = in_channels, h=32, w=32),
            nn.Conv2d(in_channels, out_channels, kernel_size, kernel_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VariantsA(Variants):
    """
    stacked blocks
    """
    def __init__(self, model, in_channels, out_channels, kernel_size=8):
        super(VariantsA, self).__init__(model, in_channels, out_channels, kernel_size)

        self.encoder = nn.Sequential(
            self.block(in_channels = in_channels, h=32, w=32, kernel_size=kernel_size),
            nn.Conv2d(in_channels, out_channels, kernel_size//2, kernel_size//2),
            self.block(in_channels = out_channels, h=8, w=8, kernel_size=kernel_size),
            nn.Conv2d(out_channels, out_channels, kernel_size//4, kernel_size//4)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VariantsB(Variants):
    """
    takes multi-scale inputs
    """
    def __init__(self, model, in_channels, out_channels, kernel_size=8):
        super(VariantsB, self).__init__(model, in_channels, out_channels)
        self.encoder = nn.ModuleList([
            self.block(in_channels=in_channels, h=32, w=32),
            nn.Sequential(
                self.block(in_channels=in_channels, h=16, w=16),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, h=8, w=8),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, h=4, w=4, kernel_size=4),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
            # nn.Sequential(nn.Upsample(scale_factor=1/2, mode='bicubic', align_corners=False),nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
            # nn.Sequential(nn.Upsample(scale_factor=1/2, mode='bicubic', align_corners=False),nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)),
            # nn.Sequential(nn.Upsample(scale_factor=1/2, mode='bicubic', align_corners=False),nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)),
            nn.Identity()
        ])
    
    def forward(self, x_list):
        if not isinstance(x_list, list):
            raise ValueError("Input must be a list of multi-scale inputs")

        embeds = []
        for x, block in zip(x_list, self.encoder):
            x = block(x)
            embeds.append(x)

        for i, (x, conv) in enumerate(zip(embeds, self.downsample)):
            if i == 0:
                out = conv(x)
            else:
                out = conv(out + x)
        
        out = self.decoder(out)
        return out

class VariantsC(Variants):
    """
    takes multi-scale inputs
    """
    def __init__(self, model, in_channels, out_channels, kernel_size=8):
        super(VariantsC, self).__init__(model, in_channels, out_channels)

        self.encoder = nn.ModuleList([
            self.block(in_channels=in_channels, h=32, w=32),
            nn.Sequential(
                self.block(in_channels=in_channels, h=16, w=16),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, h=8, w=8),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, h=4, w=4, kernel_size=4),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
        ])
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=1/2, mode='bicubic', align_corners=False),nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
            nn.Sequential(nn.Upsample(scale_factor=1/2, mode='bicubic', align_corners=False),nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)),
            nn.Sequential(nn.Upsample(scale_factor=1/2, mode='bicubic', align_corners=False),nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)),
            nn.Identity()
        ])
    
    def forward(self, x_list):
        if not isinstance(x_list, list):
            raise ValueError("Input must be a list of multi-scale inputs")

        embeds = []
        for x, block in zip(x_list, self.encoder):
            x = block(x)
            embeds.append(x)

        for i, (x, conv) in enumerate(zip(embeds, self.downsample)):
            if i == 0:
                out = conv(x)
            else:
                out = conv(out + x)
        
        out = self.decoder(out)
        return out