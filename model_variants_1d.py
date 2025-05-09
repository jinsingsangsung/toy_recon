import torch.nn as nn
import numpy as np
from ssmconv import SSMConv2dv9
from model_all import HippoConv2d, HippoConv1d
from s3kconv import S3KConv1dv2
from s4 import S4Block
from s4d import S4D
from s5 import S5SSM, SequenceLayer
from einops import rearrange
from src.models.sequence.modules.s4nd import S4ND
from mamba_ssm import Mamba
from functools import partial
import torch


def build_1d_sincos_pos_embed(channels, length):
    """
    Creates positional embeddings per channel for 1D sequence
    Args:
        channels: number of channels
        length: length of the sequence
    Returns:
        pos_embed of shape (channels, length)
    """
    grid = torch.arange(length, dtype=torch.float32)
    
    # Create different frequency bands for each channel
    pos_embeds = []
    for c in range(channels):
        # Use different frequency bases for each channel
        omega = 1. / (10000 ** (torch.arange(length) / length))
        
        # Compute embeddings for this channel
        emb = grid.unsqueeze(-1) * omega.unsqueeze(0)
        
        # Use sin component
        pos_emb = torch.sin(emb).mean(-1)
        pos_embeds.append(pos_emb)
    
    pos_embed = torch.stack(pos_embeds, dim=0)  # (channels, length)
    return pos_embed

def init_s5(d_state, blocks=8):
    from scipy.linalg import block_diag
    from s5 import make_DPLR_HiPPO
    ssm_size = d_state
    block_size = int(ssm_size / blocks)

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
    return Lambda, V, Vinv

class DummyIdentity(nn.Module):
    def __init__(self, in_channels, L=1024):
        super(DummyIdentity, self).__init__()
        self.in_channels = in_channels
        self.L = L
    
    def forward(self, x, in_channels=None, h=None, w=None, kernel_size=None, L=None):
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, in_channels, dropout=0.0, L=1024, kernel_size=8):
        super(TransformerBlock, self).__init__()
        self.inflation = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.transformer = nn.MultiheadAttention(in_channels, 1, dropout=dropout)
        self.deflation = nn.Conv1d(in_channels, in_channels, 1, 1)
        # Use PyTorch's built-in PositionalEncoding
        # self.pos_encoder = nn.Parameter(build_1d_sincos_pos_embed(d_model, L), requires_grad=False)
    
    def forward(self, x):
        x = self.inflation(x)
        # x = x + self.pos_encoder
        # L = x.shape[-1]
        x = rearrange(x, "B C L -> B L C")
        x = self.transformer(x, x, x)[0] # takes b (h w) c
        x = rearrange(x, "B L C -> B C L")
        x = self.deflation(x)
        return x
    
class SSMBlock(nn.Module):
    def __init__(self, model, d_state, in_channels, L, kernel_size=8):
        super(SSMBlock, self).__init__()
        self.model = model
        if model == "hippo":
            self.block = HippoConv1d(in_channels=in_channels, out_channels=in_channels, d_state=d_state, kernel_size=kernel_size, stride=kernel_size) # takes b c h w
        elif model == "s4":
            self.block = S4Block(d_state=d_state, d_model=in_channels, transposed=False, final_act=None, activation="id") # takes b l c
        elif model == "s4d":
            self.block = S4D(d_state=d_state, transposed=False, d_model=in_channels) # takes b l c
        elif model == "s4nd":
            self.block = S4ND(d_state=d_state//2, d_model=in_channels, dim=1, transposed=False, contract_version=1) # takes b c h w
        elif model == "mamba":
            self.block = Mamba(d_state=d_state, d_model=in_channels, use_fast_path=True) # takes b l c
        elif model == "s3k":
            self.block = S3KConv1dv2(in_channels=in_channels, out_channels=8, kernel_size=kernel_size, stride=kernel_size, state_size=d_state//4, bidirectional=False, d_in=3) # takes b c l
        elif model == "s5":
            Lambda, V, Vinv = init_s5(d_state=d_state)
            ssm = S5SSM(Lambda_re_init=Lambda.real,
                                Lambda_im_init=Lambda.imag,
                                V=V,
                                Vinv=Vinv,
                                H=in_channels,
                                P=d_state,
                                C_init="trunc_standard_normal",
                                discretization="zoh",
                                dt_min=0.001,
                                dt_max=0.1,
                                conj_sym=False,
                                clip_eigs=False,
                                bidirectional=False)
            self.block = SequenceLayer(ssm=ssm, dropout=0.0,
                                       d_model=in_channels, activation="full_glu",
                                       training=True, prenorm=False, batchnorm=False,
                                       bn_momentum=0.9, step_rescale=1.0)
        else:
            self.block = DummyIdentity(in_channels=in_channels, L=1024)
    
    def forward(self, x):
        # L = x.shape[-2]
        x = rearrange(x, "B C L -> B L C")
        x = self.block(x) # takes b (h w) c
        if isinstance(x, tuple): # some models return a tuple
            x = x[0]
        x = rearrange(x, "B L C -> B C L")
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
        if model == "transformer":
            self.block = partial(TransformerBlock, d_model=d_state)
        else:
            self.block = partial(SSMBlock, model=model, d_state=d_state)

        self.decoder = nn.Sequential(
            # nn.Conv1d(out_channels, in_channels*kernel_size, 1, 1),
            # nn.PixelShuffle(kernel_size) if kernel_size !=1 else nn.Identity(),
            nn.ConvTranspose1d(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.encoder = nn.Conv1d(in_channels, out_channels, kernel_size, kernel_size)

    def forward(self, x):
        raise NotImplementedError

class Vanilla(Variants):
    def __init__(self, model, in_channels, out_channels, kernel_size=8):
        super(Vanilla, self).__init__(model, in_channels, out_channels, kernel_size)

        self.encoder = self.block(in_channels = in_channels, L=1024, kernel_size=kernel_size) if model == "s3k" \
            else nn.Sequential(
                self.block(in_channels = in_channels, L=1024),
                nn.Conv1d(in_channels, out_channels, kernel_size, kernel_size),
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
            self.block(in_channels = in_channels, L=1024, kernel_size=kernel_size),
            nn.Conv1d(in_channels, out_channels, kernel_size//2, kernel_size//2),
            self.block(in_channels = out_channels, L=256, kernel_size=kernel_size),
            nn.Conv1d(out_channels, out_channels, kernel_size//4, kernel_size//4)
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
            self.block(in_channels=in_channels, L=1024),
            nn.Sequential(
                self.block(in_channels=in_channels, L=512),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, L=256),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, L=128, kernel_size=4),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
        ])
        self.downsample = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=2),
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
            self.block(in_channels=in_channels, L=1024),
            nn.Sequential(
                self.block(in_channels=in_channels, L=512),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, L=256),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                self.block(in_channels=in_channels, L=128, kernel_size=4),
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
            ),
        ])
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=1/2, mode='linear', align_corners=False),nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)),
            nn.Sequential(nn.Upsample(scale_factor=1/2, mode='linear', align_corners=False),nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)),
            nn.Sequential(nn.Upsample(scale_factor=1/2, mode='linear', align_corners=False),nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)),
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