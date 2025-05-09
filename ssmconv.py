import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
from einops import rearrange, repeat
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.modules.conv import _ConvNd
from torch import Tensor
from torch.nn.init import kaiming_normal_, normal_
from functools import partial
from ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, make_DPLR_HiPPO, init_VinvB_takeB,\
      init_CV_takeC, init_CV_real, make_DPLR_HiPPO_real
from torch.nn.init import kaiming_normal_, normal_
from scipy import special as ss
# import unfoldNd
import math

_c2r = torch.view_as_real
contract = torch.einsum
from scipy import special as ss

# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar

def discretize_bilinear_v2(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (B, C, H2, W2, K, P)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (B, C, H2, W2, K, P)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[None, None, None, None, None, :] * B_tilde
    return Lambda_bar, B_bar

def discretize_zoh_v3(A, B, step):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            A : means diagonal state matrix                        (N, )
            B : input projection matrix                            (B, C, H2, W2, K, N)
            step : discretization step sizes                       (B, C, H2, W2, K, N)
        Returns:
            discretized A, B                (B, C, H2, W2, K, N), (B, C, H2, W2, K, P)
    """
    
    Identity = torch.ones(A.shape[0], device=A.device)
    power_A_bar = torch.exp(A * torch.cumsum(step, dim=-2).flip(-2))
    Lambda_bar = torch.exp(A * step)
    B_bar = (1/A * (Lambda_bar-Identity)) * B
    # A = repeat(A, "N -> C N", C=B.size(1))
    # power_A_bar = torch.exp(contract("b c h w k n, c n -> b c h w k n", torch.cumsum(step, dim=-2), A))
    # B_bar = B * step
    return power_A_bar, B_bar

def discretize_zoh_v4(A, B, step):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            A : means diagonal state matrix                        (N, )
            B : input projection matrix                            (B, H2, W2, K, N)
            step : discretization step sizes                       (B, H2, W2, K, C)
        Returns:
            discretized A, B                (B, C, H2, W2, K, N), (B, C, H2, W2, K, N)
    """
    
    # Identity = torch.ones(A.shape[0], device=A.device)
    # power_A_bar = torch.exp(A * torch.cumsum(step, dim=-2).flip(-2))
    # Lambda_bar = torch.exp(A * step)
    # B_bar = (1/A * (Lambda_bar-Identity)) * B
    A = repeat(A, "N -> C N", C=step.size(-1))
    power_A_bar = torch.exp(contract("b h w k c, c n -> b c h w k n", torch.cumsum(step, dim=-2), A))
    B_bar = contract("b h w k n, b h w k c -> b c h w k n", B, step)
    return power_A_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar

def discretize_zoh_v2(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (B, C, H2, W2, K, P)
            Delta (float32): discretization step sizes             (B, C, H2, W2, K, P)
            # Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (B, C, H2, W2, K, P)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    
    power_lambda_bar = torch.exp(Lambda * torch.cumsum(Delta, dim=-2).flip(-2))
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity)) * B_tilde
    return power_lambda_bar, B_bar


def eval_legendre(x):
    """Evaluate Legendre polynomials from 0 to N-1 at points x using recurrence relation
    Args:
        x (Tensor): Points at which to evaluate, shape (L, N)
    Returns:
        Tensor: Legendre polynomial values of shape (L, N)
    """
    L, N = x.shape
    out = torch.zeros((L, N), dtype=x.dtype, device=x.device)
    # P_0(x) = 1
    out = out.clone()  # Create a new tensor to avoid in-place operations
    out[:, 0] = torch.ones(L, dtype=x.dtype, device=x.device)
    
    if N > 1:
        # P_1(x) = x
        out[:, 1] = x[:, 0].clone()  # Clone to avoid in-place modification
        
        # Use recurrence relation for higher order terms:
        # (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
        for n in range(1, N-1):
            term1 = (2*n + 1) * x[:, 0]
            term2 = out[:, n].clone()  # Clone to avoid in-place modification
            term3 = n * out[:, n-1].clone()  # Clone to avoid in-place modification
            
            # Compute next term without in-place operations
            next_term = (term1 * term2 - term3) / (n + 1)
            out[:, n+1] = next_term
            
    return out

class SSMConv2DNaive(_ConvNd):
    def __init__(
        self,
        ssm_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        dim_preserve: bool = False,
        drop_path: float = 0.,
        flip: bool = True,
        d_state=None,
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
        self.dim_preserve = dim_preserve
        if d_state is None:
            d_state = max(kernel_size_[0]*kernel_size_[1] // 4, kernel_size_[0]*kernel_size_[1])
        ks = kernel_size_[0]*kernel_size_[1]   

        self.flip = flip
        if flip:
            ssm_channels = 4*in_channels
        else:
            ssm_channels = in_channels

        self.pos_embed = nn.Parameter(torch.zeros(1, ks, ssm_channels))
        trunc_normal_(self.pos_embed, std=.02)   

        self.ssm_type = ssm_type
        if ssm_type == 'mamba':
            from mamba_ssm import Mamba
            self.ssm_kernel = Mamba(d_model=ssm_channels, d_state=d_state)
        elif ssm_type == 's4':
            from s4 import S4Block
            self.ssm_kernel = S4Block(d_model=ssm_channels, d_state=d_state)
        elif ssm_type == 's4nd':
            from src.models.sequence.modules.s4nd import S4ND
            self.ssm_kernel = S4ND(d_model=in_channels, d_state=d_state*2, dim=2, transposed=False)
        elif ssm_type == 's4d':
            from s4d import S4D
            self.ssm_kernel = S4D(d_model=ssm_channels, d_state=d_state, transposed=False)
        elif ssm_type == 'hippo':
            from hippoconv import HippoConv2d
            self.ssm_kernel = HippoConv2d(in_channels=in_channels, d_state=d_state, kernel_size=1, stride=1, padding=0, bias=True)
        
        h, w = kernel_size_[0], kernel_size_[1]
        
        if ssm_type in ["mamba", "s4", "s4d"]:
            indices = torch.zeros(h * w, dtype=torch.long)
            # Fill indices in zig-zag pattern
            idx = 0
            for i in range(h):
                if i % 2 == 0: 
                    for j in range(w):
                        indices[i*w + j] = idx
                        idx += 1
                else: 
                    for j in range(w-1, -1, -1):
                        indices[i*w + j] = idx
                        idx += 1
            self.register_buffer('indices', indices)
        self.conv = nn.Conv2d(ssm_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)

    # def fused_add_norm(self, hidden_states, residual):
    #     if residual is None:
    #         residual = hidden_states
    #     else:
    #         residual = residual + self.drop_path(hidden_states)
    #     return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))   

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: torch.Tensor of size [B, C, H, W]
        '''
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            x = F.pad(x, self.padding*2)

        # patchify input sequences
        x = x.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        x = x.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        B, C, H2, W2, K1, K2 = x.shape

        pos = self.pos_embed.expand(B*H2*W2, -1, -1)

        x = rearrange(x, "B C H2 W2 K1 K2 -> (B H2 W2) K1 K2 C")

        if self.ssm_type == 's4nd':
            x = self.ssm_kernel(x)[0]
            x = rearrange(x, "(B H2 W2) K1 K2 C -> B C (H2 K1) (W2 K2)", H2=H2, W2=W2, K1=K1, K2=K2)
            x = self.conv(x)
            return x

        if self.flip:
            x_stacked = []
            for i, dir in enumerate([(),(1),(2),(1,2)]):
                x_dir = x.flip(dir).flatten(1,2)
                x_dir = x_dir.cpu()[:, self.indices.cpu()].to(x.device)
                x_stacked.append(x_dir)
            x = torch.cat(x_stacked, dim=-1)
        else:
            x = x.flatten(1,2)
            x = x.cpu()[:, self.indices.cpu()].to(x.device)

        x = x + pos

        x = self.ssm_kernel(x)
        if isinstance(x, tuple):
            x = x[0]
        x = x.cpu()[:, self.indices.cpu()].to(x.device)
        x = rearrange(x, "(B H2 W2) (K1 K2) C -> B C (H2 K1) (W2 K2)", H2=H2, W2=W2, K1=K1, K2=K2)
        x = self.conv(x)
        return x
    

# class SSMConv2d(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: _size_2_t,
#                  stride: _size_2_t = 1,
#                  padding: Union[str, _size_2_t] = 0,
#                  dilation: _size_2_t = 1,
#                  groups: int = 1,
#                  blocks: int = 1,
#                  C_init_method: str = "trunc_standard_normal",
#                  discretization: str = "zoh",
#                  dt_min: float = 0.001,
#                  dt_max: float = 0.1,
#                  conj_sym: bool = False,
#                  clip_eigs: bool = False,
#                  bidirectional: bool = False,
#                  step_rescale: float = 1.0,
#                  dim_preserve: bool = False,
#                  realize: bool = True, # whether to realize the kernel
#     ):
#         super().__init__()

#         # convolution arguments
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.kernel_size = kernel_size

#         H = in_channels
#         P = out_channels
#         if bidirectional:
#             P = P // 2
#         block_size = P // blocks
#         self.H = H
#         self.P = P
#         self.conj_sym = conj_sym
#         self.clip_eigs = clip_eigs
#         self.bidirectional = bidirectional
#         self.discretization = discretization
#         try:
#             k1, k2 = kernel_size[0], kernel_size[1]
#         except:
#             k1, k2 = kernel_size, kernel_size
#         self.k1 = k1
#         self.k2 = k2
        
#         local_P = block_size
#         if conj_sym:
#             block_size = block_size // 2

#         self.B_tildes = nn.ParameterList()
#         for dim, k in enumerate([k1, k2]):
#             # Initialize state matrix A using approximation to HiPPO-LegS matrix
#             Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            
#             Lambda = Lambda[:block_size]
#             V = V[:, :block_size]
#             Vc = V.conj().T

#             Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
#             V = torch.block_diag(*([V] * blocks))
#             Vinv = torch.block_diag(*([Vc] * blocks))

#             # Register V and Vinv as buffers
#             self.register_buffer(f'V_{dim}', V)
#             self.register_buffer(f'Vinv_{dim}', Vinv)

#             self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
#             self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

#             # Initialize B
#             B_shape = (local_P, H) if conj_sym else (block_size, H)
#             B_init = kaiming_normal_
#             B = init_VinvB(B_init, B_shape, Vinv)
#             self.register_parameter(f'B_{dim}', nn.Parameter(B))

#             # Initialize learnable discretization timescale value
#             log_step = init_log_steps((block_size, dt_min, dt_max))
#             self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

#         # Initialize state to output (C) matrix
#         bottleneck_size = local_P*3//4
#         # bias_size = bottleneck_size*2 if bidirectional else bottleneck_size
#         bias_size = H

#         if C_init_method in ["trunc_standard_normal"]:
#             C_shape = (bottleneck_size*2, local_P, 2) if bidirectional else (bottleneck_size, local_P, 2)
#             C_init = trunc_standard_normal
#         elif C_init_method in ["lecun_normal"]:
#             C_init = kaiming_normal_
#             C_shape = (bottleneck_size*2, local_P, 2) if bidirectional else (bottleneck_size, local_P, 2)
#         elif C_init_method in ["complex_normal"]:
#             C_init = partial(normal_, std=0.5 ** 0.5)
#         else:
#             raise NotImplementedError(
#                 "C_init method {} not implemented".format(C_init))
        
#         if bidirectional:
#             C1 = init_CV(C_init, C_shape, V)
#             C2 = init_CV(C_init, C_shape, V)
#             self.register_parameter('C1', nn.Parameter(C1))
#             self.register_parameter('C2', nn.Parameter(C2))
#             C1 = C1[..., 0] + 1j * C1[..., 1]
#             C2 = C2[..., 0] + 1j * C2[..., 1]
#         else:
#             C = init_CV(C_init, C_shape, V)
#             self.register_parameter('C', nn.Parameter(C))

#         # Initialize feedthrough (D) matrix
#         D = normal_(torch.empty(1, H*k1*k2, 1, 1), std=1.0)
#         self.register_parameter('D', nn.Parameter(D))
    
#         self.blocks = blocks
#         self.block_size = block_size
#         self.local_P = local_P
#         self.step_rescale = step_rescale
#         self.dim_preserve = dim_preserve
        
#         self.realize = realize
#         if realize:
#             self.act = nn.GELU()
#             c_s = bottleneck_size*2 if bidirectional else bottleneck_size
#             self.C_second = nn.Conv2d(c_s, H*k1*k2, 1, 1)
#             self.bias1 = nn.Parameter(torch.zeros(1, bias_size, 1, 1, dtype=torch.float))
#         else:
#             self.act = ComplexActivations(act="gelu", activation_type="amp_phase")
#             c_s = bottleneck_size*2 if bidirectional else bottleneck_size
#             self.C_second = ComplexConv2d(c_s, H*k1*k2, 1, 1)
#             self.bias1 = nn.Parameter(torch.zeros(1, bias_size, 1, 1, dtype=torch.complex64))

#     def print_model_parameters(self,):
#         print("\nModel Parameters:")
#         print("-" * 80)
#         total_params = 0
#         for name, param in self.named_parameters():
#             num_params = param.numel()
#             total_params += num_params
#             print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
#         print("-" * 80)
#         print(f"Total parameters: {total_params:,}")

#     def get_Lambda(self, dim: int):
#         """Get complex Lambda for given dimension, applying clipping if needed."""
#         Lambda_re = getattr(self, f'Lambda_re_{dim}')
#         Lambda_im = getattr(self, f'Lambda_im_{dim}')
#         if self.clip_eigs:
#             Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
#         return Lambda_re + 1j * Lambda_im
    
#     def get_B_tilde(self, dim: int):
#         """Get complex B_tilde for given dimension."""
#         return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

#     def get_C_tilde(self):
#         """Get complex C_tilde, handling bidirectional case."""
#         if self.bidirectional:
#             C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
#             C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
#             if self.realize:
#                 C1 = C1.real
#                 C2 = C2.real
#             return torch.cat([C1, C2], axis=-1)
#         else:
#             if self.realize:
#                 return self.C[..., 0]
#             else:
#                 return self.C[..., 0] + 1j * self.C[..., 1]

#     def compute_kernel(self):
        
#         kernels = []
#         self.B_bars = []
#         for dim, k in enumerate([self.k1, self.k2]):
#             step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
#             Lambda = self.get_Lambda(dim)
#             B_tilde = self.get_B_tilde(dim)
            
#             if self.discretization == "zoh":
#                 Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
#             elif self.discretization == "bilinear":
#                 Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

#             # Construct conv kernel for this dimension
#             powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
#             kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
#             kernels.append(kernel)

#         kernel = contract("khp,lhp -> phkl", kernels[0], kernels[1])
        
#         # def visualize_complex_kernel(kernel, save_path='kernel.png', expand_channels = False):
#         #     import matplotlib.pyplot as plt
#         #     import numpy as np
#         #     """
#         #     Visualize complex kernel characteristics.
#         #     kernel shape: (out_ch, in_ch, k[1], k[2])
#         #     """
#         #     # Reshape kernel to combine out_ch and in_ch dimensions
#         #     out_ch, in_ch, h, w = kernel.shape
#         #     kernel_reshaped = kernel.transpose(0,1).reshape(out_ch * in_ch, h, w)
            
#         #     # Calculate amplitude and phase maps
#         #     amplitudes = np.abs(kernel_reshaped.cpu().detach().numpy())  # Get magnitudes
#         #     phases = np.angle(kernel_reshaped.cpu().detach().numpy())    # Get phases
            
#         #     if not expand_channels: 
#         #         # Calculate norm of amplitudes across channels for each spatial position
#         #         amplitude_map = np.linalg.norm(amplitudes, axis=0)
                
#         #         # Calculate average phase rotation across channels
#         #         # We use the mean of absolute phases to capture overall rotation intensity
#         #         phase_map = np.mean(np.abs(phases), axis=0)
                
#         #         # Create figure with two subplots
#         #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
#         #         # Plot amplitude map
#         #         im1 = ax1.imshow(amplitude_map, cmap='viridis')
#         #         ax1.set_title('Amplitude Norm')
#         #         plt.colorbar(im1, ax=ax1)
                
#         #         # Plot phase map
#         #         im2 = ax2.imshow(phase_map, cmap='twilight', vmin=-np.pi, vmax=np.pi)
#         #         ax2.set_title('Phase Rotation')
#         #         plt.colorbar(im2, ax=ax2)
                
#         #         plt.tight_layout()
#         #         plt.savefig(save_path)
#         #         plt.close()
#         #     else:
#         #         # Create gif showing amplitude and phase changes across channels
#         #         import io
#         #         from PIL import Image

#         #         # Create frames for the gif
#         #         frames = []
                
#         #         # Calculate global min/max for consistent colorbar scaling
#         #         amp_vmin, amp_vmax = amplitudes.min(), amplitudes.max()
                
#         #         for i in range(amplitudes.shape[0]):
#         #             # Create figure for this channel
#         #             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    
#         #             # Plot amplitude
#         #             im1 = ax1.imshow(amplitudes[i], cmap='viridis', 
#         #                            vmin=amp_vmin, vmax=amp_vmax)
#         #             ax1.set_title(f'Channel {i//8, i%8} - Amplitude')
#         #             plt.colorbar(im1, ax=ax1)
                    
#         #             # Plot phase
#         #             im2 = ax2.imshow(phases[i], cmap='twilight', 
#         #                            vmin=-np.pi, vmax=np.pi)
#         #             ax2.set_title(f'Channel {i//8, i%8} - Phase')
#         #             plt.colorbar(im2, ax=ax2)
                    
#         #             plt.tight_layout()
                    
#         #             # Convert plot to image
#         #             buf = io.BytesIO()
#         #             plt.savefig(buf, format='png')
#         #             buf.seek(0)
#         #             frames.append(Image.open(buf))
#         #             plt.close()
                    
#         #         # Save as gif
#         #         frames[0].save(
#         #             save_path.replace('.png', '.gif'),
#         #             save_all=True,
#         #             append_images=frames[1:],
#         #             duration=600,  # 200ms per frame
#         #             loop=0
#         #         )
#         # visualize_complex_kernel(kernel, expand_channels=True)
#         return kernel

#     def dim_preserve_forward(self, x, kernel):
#         x = x.unfold(dimension=2, size=self.k1, step=self.stride[0])
#         x = x.unfold(dimension=3, size=self.k2, step=self.stride[1])
#         # x: B, C, H2, W2, K1, K2
#         # kernel: P, H, K1, K2
#         if not x.is_complex():
#             real_out = contract("pckl,bchwkl->bphwkl", kernel.real, x)
#             imag_out = contract("pckl,bchwkl->bphwkl", kernel.imag, x)
#             real_out = rearrange(real_out, "b p h w k l -> b p (h k) (w l)")
#             imag_out = rearrange(imag_out, "b p h w k l -> b p (h k) (w l)")
#             out = real_out + 1j*imag_out
#         else:
#             a = contract("pckl,bchwkl->bphwkl", kernel.real, x.real)
#             b = contract("pckl,bchwkl->bphwkl", kernel.imag, x.imag)
#             c = contract("pckl,bchwkl->bphwkl", kernel.real + kernel.imag, x.real + x.imag)
#             real_out = a - b
#             imag_out = c - a - b
#             real_out = rearrange(real_out, "b p h w k l -> b p (h k) (w l)")
#             imag_out = rearrange(imag_out, "b p h w k l -> b p (h k) (w l)")
#             out = real_out + 1j*imag_out
#         return out

#     def realized_forward(self, x, kernel, conv):
        
#         kernel = kernel.real
#         if not x.is_complex():
#             real_out = conv(x, kernel)
#             if self.bidirectional:
#                 real_out_f = conv(x, kernel.real.flip(-1,-2))
#                 real_out = torch.cat([real_out, real_out_f], dim=1)
#         else:
#             a = conv(x.real, kernel)
#             b = conv(x.imag, kernel)
#             if self.bidirectional:
#                 a = torch.cat([a, conv(x.real, kernel.flip(-1,-2))], dim=1)
#                 b = torch.cat([b, conv(x.imag, kernel.flip(-1,-2))], dim=1)
#             real_out = a + 1j*b

#         return real_out

#     def forward(self, x):
#         """
#         since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
#         conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
#                     + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
#         instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
#         a = conv(W.real, x.real)
#         b = conv(W.imag, x.imag)
#         c = conv(W.real + W.imag, x.real + x.imag)
#         conv(W, x) = a - b + i(c - a - b)
#         """
#         conv = partial(F.conv2d,
#                        stride=self.stride, 
#                        padding=self.padding, 
#                        dilation=self.dilation, 
#                        groups=self.groups)
#         kernel = self.compute_kernel()

#         if self.dim_preserve:
#             return self.dim_preserve_forward(x, kernel)

#         if self.realize:
#             return self.realized_forward(x, kernel, conv)
        
#         if not x.is_complex():
#             real_out = conv(x, kernel.real)
#             imag_out = conv(x, kernel.imag)
#             if self.bidirectional:
#                 real_out_f = conv(x, kernel.real.flip(-1,-2))
#                 imag_out_f = conv(x, kernel.imag.flip(-1,-2))
#                 real_out = torch.cat([real_out, real_out_f], dim=1)
#                 imag_out = torch.cat([imag_out, imag_out_f], dim=1)
#         else:
#             a = conv(x.real, kernel.real)
#             b = conv(x.imag, kernel.imag)
#             c = conv(x.real + x.imag, kernel.real + kernel.imag)
#             real_out = a - b
#             imag_out = c - a - b
#             if self.bidirectional:
#                 a = conv(x.real, kernel.flip(-1,-2).real)
#                 b = conv(x.imag, kernel.flip(-1,-2).imag)
#                 c = conv(x.real + x.imag, kernel.flip(-1,-2).real + kernel.flip(-1,-2).imag)
#                 real_out = torch.cat([real_out, a - b], dim=1)
#                 imag_out = torch.cat([imag_out, c - a - b], dim=1)
#         return real_out + 1j*imag_out
    
#     def reconstruct(self, x):
#         # bidirectional not regarded yet
#         # eval_matrices = []
#         # for dim, k in enumerate([self.k1, self.k2]):
#         #     step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
#         #     Lambda = self.get_Lambda(dim)
#         #     B_tilde = self.get_B_tilde(dim)

#         #     # Get V and Vinv from buffers
#         #     V = getattr(self, f'V_{dim}')
#         #     Vinv = getattr(self, f'Vinv_{dim}')
#         #     B = V @ B_tilde
#         #     Lambda_diag = torch.diag(Lambda)
#         #     A = V @ Lambda_diag @ Vinv

#         #     log_step = getattr(self, f'log_step_{dim}')
#         #     step = torch.exp(log_step)[None]
#         #     x_pts = torch.linspace(k-1, 0, k, device=log_step.device)[:, None]
#         #     grid = torch.exp(contract("kc,cn -> kn", x_pts, step)).to(torch.complex64)
#         #     gridA = contract("kn,nm->nkm", grid, A)
#         #     eval_matrix = contract("nkm,md->nkd", gridA, B)
#         #     # eval_matrix = (grid @ A) @ B
#         #     eval_matrices.append(eval_matrix)
#         # eval_matrix = contract("nkd,nld->nkld", eval_matrices[0], eval_matrices[1])
#         # out = torch.einsum("bnhw ,nkld->bdhkwl", x, eval_matrix)
#         # out = rearrange(out, "b d h k w l -> b d (h k) (w l)") + self.bias1
        
#         C = self.get_C_tilde()[None].flatten(0,1)
#         # C = (self.get_C_tilde()[None] * eval_matrix).flatten(0,1)
#         out2 = torch.einsum("bchw, dc -> bdhw", x, C)
#         out2 = self.C_second(out2)
#         # out = torch.einsum("bchw, dc -> bdhw", x, C_2) + self.bias2
#         # out = torch.einsum("bchw, dc -> bdhw", x, C) + self.D
#         out = nn.PixelShuffle(self.kernel_size[0])(out2)
#         out = self.act(out).real
#         return out

class PairedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1):
        super().__init__()
        self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.upsample_conv = nn.Sequential(nn.Conv2d(out_channels, in_channels*kernel_size**2, 1, 1),
                                            nn.PixelShuffle(kernel_size))

    def reconstruct(self, x):
        if not x.is_complex():
            x = self.upsample_conv(x)
        else:
            x = self.upsample_conv(x.real) + 1j * self.upsample_conv(x.imag)
        return x
    
    def forward(self, x):
        if not x.is_complex():
            x = self.downsample_conv(x)
        else:
            x = self.downsample_conv(x.real) + 1j * self.downsample_conv(x.imag)
        return x


class ComplexActivations(nn.Module):
    def __init__(self, act="gelu", activation_type="split"):
        super().__init__()
        self.activation_type = activation_type
        self.act = nn.GELU() if act == "gelu" else nn.SiLU()
        
    def component_wise(self, x):
        """Split approach - apply activation separately to real and imaginary parts"""
        real = x.real
        imag = x.imag
        
        # Apply activation separately
        real_activated = self.act(real)
        imag_activated = self.act(imag)
        
        return torch.complex(real_activated, imag_activated)
    
    def complex_relu(self, x):
        """ModReLU - maintains phase but modifies magnitude"""
        magnitude = torch.abs(x)
        phase = x / (magnitude + 1e-8)  # avoid division by zero
        return phase * self.act(magnitude)
    
    def amplitude_phase(self, x):
        """Amplitude-Phase activation"""
        magnitude = torch.abs(x)
        phase = torch.angle(x)
        
        # Activate magnitude only, preserve phase
        activated_magnitude = self.act(magnitude)
        
        # Convert back to complex
        return activated_magnitude * torch.exp(1j * phase)
    
    def forward(self, x):
        if self.activation_type == "split":
            return self.component_wise(x)
        elif self.activation_type == "modrelu":
            return self.complex_relu(x)
        elif self.activation_type == "amp_phase":
            return self.amplitude_phase(x)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, padding_mode="zero"):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode=padding_mode)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode=padding_mode)

    def forward(self, x):
        if not x.is_complex():
            return self.real_conv(x) + 1j * self.imag_conv(x)
        else:
            a = self.real_conv(x.real)  
            b = self.imag_conv(x.imag)
            c = self.real_conv(x.real + x.imag) + self.imag_conv(x.real + x.imag)
            real_out = a - b
            imag_out = c - a - b
            return real_out + 1j * imag_out
        
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        if not x.is_complex():
            return self.linear_real(x) + 1j * self.linear_imag(x)
        else:
            a = self.linear_real(x.real)
            b = self.linear_imag(x.imag)
            c = self.linear_real(x.real + x.imag) + self.linear_imag(x.real + x.imag)
            return a - b + 1j * (c - a - b)

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

    def forward(self, x):
        if not x.is_complex():
            return self.conv(x)
        else:
            real_out = self.conv(x.real)
            imag_out = self.conv(x.imag)
            return real_out + 1j * imag_out

class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            # Complex parameters for affine transformation
            self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.complex64))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.complex64))

    def complex_layernorm(self, x):
        """
        Normalize both real and imaginary parts jointly
        This preserves the complex structure better
        """
        # Calculate mean of real and imaginary parts separately
        mean_r = x.real.mean(dim=-1, keepdim=True)
        mean_i = x.imag.mean(dim=-1, keepdim=True)
        
        # Center both real and imaginary parts
        x_centered = torch.complex(x.real - mean_r, x.imag - mean_i)
        
        # Calculate variance using the centered complex values
        # var = E(|x - μ|²) for complex x
        variance = torch.abs(x_centered).pow(2).mean(dim=-1, keepdim=True)
        
        # Normalize
        x_normalized = x_centered / torch.sqrt(variance + self.eps)
        
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
            
        return x_normalized

    def component_layernorm(self, x):
        """
        Normalize real and imaginary parts independently
        Simpler approach but might not preserve complex relationships as well
        """
        # Split into real and imaginary parts
        real = x.real
        imag = x.imag
        
        # Apply layer norm to each part separately
        real_normalized = F.layer_norm(
            real, 
            self.normalized_shape,
            weight=self.weight.real if self.elementwise_affine else None,
            bias=self.bias.real if self.elementwise_affine else None,
            eps=self.eps
        )
        
        imag_normalized = F.layer_norm(
            imag, 
            self.normalized_shape,
            weight=self.weight.imag if self.elementwise_affine else None,
            bias=self.bias.imag if self.elementwise_affine else None,
            eps=self.eps
        )
        
        return torch.complex(real_normalized, imag_normalized)

    def forward(self, x, mode='complex'):
        if mode == 'complex':
            return self.complex_layernorm(x)
        elif mode == 'component':
            return self.component_layernorm(x)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

class SSMConvBlock(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 padding, 
                 dilation=1, 
                 groups=1,
                 bidirectional=True,
                 act="silu",
                 act_type="full",
                 prenorm=False,
                 dropout=0.0):
        super().__init__()
        self.conv = SSMConv2dv2(in_channels=in_channels,
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding,
                              bidirectional=bidirectional)
        self.act = ComplexActivations(act="relu", activation_type="amp_phase")
        self.drop = nn.Dropout(dropout)
        self.prenorm = prenorm
        self.act_type = act_type
        if prenorm:
            self.norm = ComplexLayerNorm(in_channels)
        else:
            self.norm = ComplexLayerNorm(out_channels)
        
        if act_type == "full":
            self.out1 = ComplexConv2d(out_channels, out_channels, 1, 1)
            self.out2 = ComplexConv2d(out_channels, out_channels, 1, 1)
        if act_type == "half":
            self.out2 = ComplexConv2d(out_channels, out_channels, 1, 1)

        self.skip_downsample = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

    def forward(self, x):
        skip = self.skip_downsample(x)
        if self.prenorm:
            x = self.norm(x.transpose(1,-1)).transpose(1,-1)
        x = self.conv(x)
        if self.act_type == "full":
            x = self.drop(self.act(x))
            x = self.out1(x) * F.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.act_type == "half":
            x = self.drop(self.act(x))
            x = x * F.sigmoid(self.out2(x))
            x = self.drop(x)
        else:
            x = self.act(x)
            x = self.drop(x)

        x = x + skip
        if not self.prenorm:
            x = self.norm(x.transpose(1,-1)).transpose(1,-1)
        return x


class SSMConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
                 dim_preserve: bool = False,
                 realize: bool = False, # whether to realize the kernel
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        H = in_channels
        P = out_channels
        if bidirectional:
            P = P // 2
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.k = kernel_size
        
        local_P = block_size
        if conj_sym:
            block_size = block_size // 2

        self.B_tildes = nn.ParameterList()
        
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
        
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
        V = torch.block_diag(*([V] * blocks))
        Vinv = torch.block_diag(*([Vc] * blocks))

        # Register V and Vinv as buffers
        self.register_buffer(f'V', V)
        self.register_buffer(f'Vinv', Vinv)

        self.register_parameter(f'Lambda_re', nn.Parameter(Lambda.real))
        self.register_parameter(f'Lambda_im', nn.Parameter(Lambda.imag))

        # Initialize B
        B_shape = (local_P, H) if conj_sym else (block_size, H)
        B_init = kaiming_normal_
        B = init_VinvB(B_init, B_shape, Vinv)
        self.register_parameter(f'B', nn.Parameter(B))

        # Initialize learnable discretization timescale value
        log_step = init_log_steps((block_size, dt_min, dt_max))
        self.register_parameter(f'log_step', nn.Parameter(log_step))

        # Initialize state to output (C) matrix

        if C_init_method in ["trunc_standard_normal"]:
            # C_shape = (self.k, local_P, 2) # if reconstruction_2
            C_shape = (local_P, local_P, 2) # if reconstruction
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_init = kaiming_normal_
            C_shape = (local_P, local_P, 2)
        elif C_init_method in ["complex_normal"]:
            C_init = partial(normal_, std=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))
        
        if bidirectional:
            C1 = init_CV(C_init, C_shape, V)
            C2 = init_CV(C_init, C_shape, V)
            self.register_parameter('C1', nn.Parameter(C1))
            self.register_parameter('C2', nn.Parameter(C2))
            C1 = C1[..., 0] + 1j * C1[..., 1]
            C2 = C2[..., 0] + 1j * C2[..., 1]
        else:
            C = init_CV(C_init, C_shape, V)
            self.register_parameter('C', nn.Parameter(C))

        # Initialize feedthrough (D) matrix
        D = normal_(torch.empty(1, H*self.k, 1, 1), std=1.0)
        self.register_parameter('D', nn.Parameter(D))
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale
        self.dim_preserve = dim_preserve
        
        self.realize = realize
        if realize:
            self.act = nn.GELU()
            self.C_second = nn.Conv1d(P, H, 1, 1)
        else:
            self.act = ComplexActivations(act="gelu", activation_type="amp_phase")
            self.C_second = nn.Conv1d(P, H, 1, 1)
        self.norm = nn.InstanceNorm1d(P)

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re')
        Lambda_im = getattr(self, f'Lambda_im')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B')[..., 0] + 1j * getattr(self, f'B')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        if self.bidirectional:
            C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
            C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
            if self.realize:
                C1 = C1.real
                C2 = C2.real
            return torch.cat([C1, C2], axis=-1)
        else:
            if self.realize:
                return self.C[..., 0]
            else:
                return self.C[..., 0] + 1j * self.C[..., 1]

    def compute_kernel(self):
        
        self.B_bars = []
        k = self.k
        step = self.step_rescale * torch.exp(getattr(self, f'log_step'))
        
        Lambda = self.get_Lambda()
        B_tilde = self.get_B_tilde()
        
        if self.discretization == "zoh":
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretization == "bilinear":
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        
        # Construct conv kernel for this dimension
        powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
        kernel = rearrange(powers_lambda[:, None] * B_bar.transpose(0,1), "k h p -> p h k")
        
        self.register_buffer('Lambda_bar', Lambda_bar) # save for reconstruction
        self.register_buffer('powers_lambda', powers_lambda) # save for reconstruction
        
        return kernel

    def realized_forward(self, x, kernel, conv):
        
        kernel = kernel.real
        if not x.is_complex():
            real_out = conv(x, kernel)
            if self.bidirectional:
                real_out_f = conv(x, kernel.real.flip(-1,-2))
                real_out = torch.cat([real_out, real_out_f], dim=1)
        else:
            a = conv(x.real, kernel)
            b = conv(x.imag, kernel)
            if self.bidirectional:
                a = torch.cat([a, conv(x.real, kernel.flip(-1,-2))], dim=1)
                b = torch.cat([b, conv(x.imag, kernel.flip(-1,-2))], dim=1)
            real_out = a + 1j*b

        return real_out

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv1d,
                       stride=self.stride, 
                       padding=self.padding, 
                       dilation=self.dilation, 
                       groups=self.groups)
        kernel = self.compute_kernel()

        if self.dim_preserve:
            return self.dim_preserve_forward(x, kernel)

        if self.realize:
            return self.realized_forward(x, kernel, conv)
        
        if not x.is_complex():
            real_out = conv(x, kernel.real)
            imag_out = conv(x, kernel.imag)
            if self.bidirectional:
                real_out_f = conv(x, kernel.real.flip(-1))
                imag_out_f = conv(x, kernel.imag.flip(-1))
                real_out = torch.cat([real_out, real_out_f], dim=1)
                imag_out = torch.cat([imag_out, imag_out_f], dim=1)
        else:
            a = conv(x.real, kernel.real)
            b = conv(x.imag, kernel.imag)
            c = conv(x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
            if self.bidirectional:
                a = conv(x.real, kernel.flip(-1,-2).real)
                b = conv(x.imag, kernel.flip(-1,-2).imag)
                c = conv(x.real + x.imag, kernel.flip(-1,-2).real + kernel.flip(-1,-2).imag)
                real_out = torch.cat([real_out, a - b], dim=1)
                imag_out = torch.cat([imag_out, c - a - b], dim=1)
        return real_out + 1j*imag_out
    
    def reconstruct(self, x):
        # bidirectional not regarded yet
        # x : b, P, L
        Lambda_L = getattr(self, 'powers_lambda')[0]
        """
        V = torch.linalg.vander(Lambda).T
        V = V * Lambda_L[None]
        Vinv = torch.linalg.inv(V)
        g = self.powers_lambda @ Vinv # k, P
        """
        V = self.get_C_tilde() * Lambda_L
        g = self.powers_lambda @ V # k, P
        out = contract("bcL, dc -> bcLd", x, g)
        out = rearrange(out, "b c L d -> b c (L d)").real
        out = F.gelu(out)
        out = self.C_second(self.norm(out)) # do not apply norm
        return out
    
    def reconstruct_2(self, x):
        # bidirectional not regarded yet
        # x : b, P, L
        g = self.get_C_tilde()
        out = contract("bcL, dc -> bcLd", x, g)
        out = rearrange(out, "b c L d -> b c (L d)").real
        out = F.gelu(out)
        out = self.C_second(self.norm(out))
        return out


class SSMConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
                 dim_preserve: bool = False,
                 realize: bool = False, # whether to realize the kernel
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        H = in_channels
        P = out_channels
        if bidirectional:
            P = P // 2
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2
        
        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_shape = (local_P, H)
            B_init = kaiming_normal_
            B = init_VinvB(B_init, B_shape, Vinv)
            self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            # Initialize state to output (C) matrix
            if C_init_method in ["trunc_standard_normal"]:
                C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
                C_init = trunc_standard_normal
            elif C_init_method in ["lecun_normal"]:
                C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
                C_init = kaiming_normal_
            elif C_init_method in ["complex_normal"]:
                C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
                C_init = partial(normal_, std=0.5 ** 0.5)
            else:
                raise NotImplementedError(
                    "C_init method {} not implemented".format(C_init))
        
            if bidirectional:
            # if False:   
                C1 = init_CV(C_init, C_shape, V)
                C2 = init_CV(C_init, C_shape, V)
                C1 = C1[..., 0] + 1j * C1[..., 1]
                C2 = C2[..., 0] + 1j * C2[..., 1]
                self.register_parameter(f'C1_{dim}', nn.Parameter(C1))
                self.register_parameter(f'C2_{dim}', nn.Parameter(C2))
            else:
                C = init_CV(C_init, C_shape, V)
                self.register_parameter(f'C_{dim}', nn.Parameter(C))


        c_s = P*2 if bidirectional else P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
        # setattr(self, f'C_second_{dim}', nn.Conv2d(c_s, H, 1, 1))
        # else:
        #     self.act = ComplexActivations(act="gelu", activation_type="amp_phase")
        #     self.C_second = ComplexConv2d(c_s, H, 1, 1)
        #     # setattr(self, f'C_second_{dim}', ComplexConv2d(c_s, H, 1, 1))
        self.norm = nn.InstanceNorm2d(P)

        # Initialize feedthrough (D) matrix
        D = normal_(torch.empty(1, H*k1*k2, 1, 1), std=1.0)
        self.register_parameter('D', nn.Parameter(D))
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale
        self.dim_preserve = dim_preserve
        # self.channel_recover = nn.Conv2d(P, P*P*k1*k2, 1, 1)
        
        self.realize = realize

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self, dim: int):
        """Get complex C_tilde, handling bidirectional case."""
        if self.bidirectional:
        # if False:
            C1 = getattr(self, f'C1_{dim}')
            C2 = getattr(self, f'C2_{dim}')
            if self.realize:
                C1 = C1.real
                C2 = C2.real
            return torch.cat([C1, C2], axis=-1)
        else:
            if self.realize:
                return getattr(self, f'C_{dim}')[..., 0]
            else:
                return getattr(self, f'C_{dim}')[..., 0] + 1j * getattr(self, f'C_{dim}')[..., 1]

    def compute_kernel(self):
        
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.get_B_tilde(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("khp,lhp -> phkl", kernels[0], kernels[1])
        
        return kernel

    def dim_preserve_forward(self, x, kernel):
        x = x.unfold(dimension=2, size=self.k1, step=self.stride[0])
        x = x.unfold(dimension=3, size=self.k2, step=self.stride[1])
        # x: B, C, H2, W2, K1, K2
        # kernel: P, H, K1, K2
        if not x.is_complex():
            real_out = contract("pckl,bchwkl->bphwkl", kernel.real, x)
            imag_out = contract("pckl,bchwkl->bphwkl", kernel.imag, x)
            real_out = rearrange(real_out, "b p h w k l -> b p (h k) (w l)")
            imag_out = rearrange(imag_out, "b p h w k l -> b p (h k) (w l)")
            out = real_out + 1j*imag_out
        else:
            a = contract("pckl,bchwkl->bphwkl", kernel.real, x.real)
            b = contract("pckl,bchwkl->bphwkl", kernel.imag, x.imag)
            c = contract("pckl,bchwkl->bphwkl", kernel.real + kernel.imag, x.real + x.imag)
            real_out = a - b
            imag_out = c - a - b
            real_out = rearrange(real_out, "b p h w k l -> b p (h k) (w l)")
            imag_out = rearrange(imag_out, "b p h w k l -> b p (h k) (w l)")
            out = real_out + 1j*imag_out
        return out

    def realized_forward(self, x, kernel, conv):
        """
        Realized forward pass for SSMConv2d.
        Only uses real part of the kernel
        """
        kernel = kernel.real
        if not x.is_complex():
            real_out = conv(x, kernel)
            if self.bidirectional:
                real_out_f = conv(x, kernel.real.flip(-1,-2))
                real_out = torch.cat([real_out, real_out_f], dim=1)
        else:
            a = conv(x.real, kernel)
            b = conv(x.imag, kernel)
            if self.bidirectional:
                a = torch.cat([a, conv(x.real, kernel.flip(-1,-2))], dim=1)
                b = torch.cat([b, conv(x.imag, kernel.flip(-1,-2))], dim=1)
            real_out = a + 1j*b

        return real_out

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv2d,
                       stride=self.stride,  
                       dilation=self.dilation, 
                       groups=self.groups)
        x = F.pad(x, 2*self.padding, mode="replicate")
        kernel = self.compute_kernel()

        if self.dim_preserve:
            return self.dim_preserve_forward(x, kernel)

        if self.realize:
            return self.realized_forward(x, kernel, conv)
        
        if not x.is_complex():
            real_out = conv(x, kernel.real)
            imag_out = conv(x, kernel.imag)
            if self.bidirectional:
                real_out_f = conv(x, kernel.real.flip(-1,-2))
                imag_out_f = conv(x, kernel.imag.flip(-1,-2))
                real_out = torch.cat([real_out, real_out_f], dim=1)
                imag_out = torch.cat([imag_out, imag_out_f], dim=1)
        else:
            a = conv(x.real, kernel.real)
            b = conv(x.imag, kernel.imag)
            c = conv(x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
            if self.bidirectional:
                a = conv(x.real, kernel.flip(-1,-2).real)
                b = conv(x.imag, kernel.flip(-1,-2).imag)
                c = conv(x.real + x.imag, kernel.flip(-1,-2).real + kernel.flip(-1,-2).imag)
                real_out = torch.cat([real_out, a - b], dim=1)
                imag_out = torch.cat([imag_out, c - a - b], dim=1)
        return (real_out + 1j*imag_out).real
    
    def reconstruct(self, x):
        g_list = []
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = self.get_Lambda(dim)
            # G = (self.get_C_tilde(dim).T * lambda_L) + (self.get_C_tilde(dim) * lambdas)
            # G = (self.get_C_tilde(dim).T * (lambda_L[:, None] @ lambda_L[None]) / lambdas) # P x P
            # G = (self.get_C_tilde(dim).T * lambda_L / lambdas) # P x P
            G = (self.get_C_tilde(dim).T * lambda_L * lambdas) # P x P
            # local_P = len(self.get_C_tilde(dim)) // 2
            # C_1 = self.get_C_tilde(dim)[:local_P]
            # C_2 = self.get_C_tilde(dim)[local_P:]
            # G = (C_1 * lambdas * lambda_L)# P x P
            # G = (self.get_C_tilde(dim).T * lambdas * lambda_L) # P x P
            # G = (lambda_L[:, None] @ lambda_L[None]) * (self.get_C_tilde(dim) @ lambdas)
            g = powers_lambda @ G # k, P
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1])
        if self.bidirectional:
            # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
            g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0],
                           (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            div = div[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out

    def reconstruct_pixelshuffle(self, x):
        """
        Construct PixelShuffle weight (PxPxKxK) using
        powers_lambda_{dim}: K, P
        lambdas_{dim}: P
        learnable weight: C_tilde_{dim}: PxP
        currently only applicable if k1 = k2
        """
        px_out = F.pixel_shuffle(self.channel_recover(x), self.k1)
        g_list = []
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = self.get_Lambda(dim)
            G = (self.get_C_tilde(dim).T * lambda_L * lambdas) # P x P
            g = powers_lambda @ G # k, P
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1])
        if self.bidirectional:
            # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
            g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            div = div[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        out += px_out
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        return out

class SSMConv2dv2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
                 dim_preserve: bool = False,
                 realize: bool = True, # whether to realize the kernel
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        H = in_channels
        P = out_channels
        if bidirectional:
            P = P // 2
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2
        
        local_P = block_size
        if conj_sym:
            block_size = block_size // 2

        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_shape = (local_P, H) if conj_sym else (block_size, H)
            B_init = kaiming_normal_
            B = init_VinvB(B_init, B_shape, Vinv)
            self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((block_size, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            # Initialize state to output (C) matrix
            if C_init_method in ["trunc_standard_normal"]:
                C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
                C_init = trunc_standard_normal
            elif C_init_method in ["lecun_normal"]:
                C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
                C_init = kaiming_normal_
            elif C_init_method in ["complex_normal"]:
                C_init = partial(normal_, std=0.5 ** 0.5)
            else:
                raise NotImplementedError(
                    "C_init method {} not implemented".format(C_init))
        
            # if bidirectional:
            if False:   
                C1 = init_CV(C_init, C_shape, V)
                C2 = init_CV(C_init, C_shape, V)
                C1 = C1[..., 0] + 1j * C1[..., 1]
                C2 = C2[..., 0] + 1j * C2[..., 1]
                self.register_parameter(f'C1_{dim}', nn.Parameter(C1))
                self.register_parameter(f'C2_{dim}', nn.Parameter(C2))
            else:
                C = init_CV(C_init, C_shape, V)
                self.register_parameter(f'C_{dim}', nn.Parameter(C))


        if realize:
            self.act = nn.GELU()
            # c_s = P*2 if bidirectional else P
            self.C_second = nn.Conv2d(H, H, 1, 1)
            # setattr(self, f'Hecond_{dim}', nn.Conv2d(c_s, H, 1, 1))
        else:
            self.act = ComplexActivations(act="gelu", activation_type="amp_phase")
            # c_s = P*2 if bidirectional else P
            self.C_second = ComplexConv2d(c_s, H, 1, 1)
            # setattr(self, f'C_second_{dim}', ComplexConv2d(c_s, H, 1, 1))
        self.norm = nn.InstanceNorm2d(P)

        # Initialize feedthrough (D) matrix
        D = normal_(torch.empty(1, H*k1*k2, 1, 1), std=1.0)
        self.register_parameter('D', nn.Parameter(D))
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale
        self.dim_preserve = dim_preserve
        
        self.realize = realize

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self, dim: int):
        """Get complex C_tilde, handling bidirectional case."""
        # if self.bidirectional:
        if False:
            C1 = getattr(self, f'C1_{dim}')
            C2 = getattr(self, f'C2_{dim}')
            if self.realize:
                C1 = C1.real
                C2 = C2.real
            return torch.cat([C1, C2], axis=-1)
        else:
            if self.realize:
                return getattr(self, f'C_{dim}')[..., 0]
            else:
                return getattr(self, f'C_{dim}')[..., 0] + 1j * getattr(self, f'C_{dim}')[..., 1]

    def compute_kernel(self):
        
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.get_B_tilde(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
            kernels.append(kernel)

            self.register_buffer(f'Lambda_bar_{dim}', Lambda_bar)
            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)
        
        return kernels

    def realized_forward(self, x, kernel, conv):
        
        kernel = kernel.real
        if not x.is_complex():
            real_out = conv(x, kernel)
            if self.bidirectional:
                real_out_f = conv(x, kernel.real.flip(-1))
                real_out = torch.cat([real_out, real_out_f], dim=1)
        else:
            a = conv(x.real, kernel)
            b = conv(x.imag, kernel)
            if self.bidirectional:
                a = torch.cat([a, conv(x.real, kernel.flip(-1))], dim=1)
                b = torch.cat([b, conv(x.imag, kernel.flip(-1))], dim=1)
            real_out = a + 1j*b

        return real_out

    def apply_kernel(self, x, kernel):
        # we assume the kernel of shape: l, h, p
        # x: b, h, L
        if not x.is_complex():
            real_out = contract("bhl, lhp -> bhp", x, kernel.real)
            imag_out = contract("bhl, lhp -> bhp", x, kernel.imag)
            if self.bidirectional:
                real_out_f = contract("bhl, lhp -> bhp", x, kernel.flip(-1))
                imag_out_f = contract("bhl, lhp -> bhp", x, kernel.imag.flip(-1))
                real_out = torch.cat([real_out, real_out_f], dim=1)
                imag_out = torch.cat([imag_out, imag_out_f], dim=1)
        else:
            a = contract("bhl, lhp -> bhp", x.real, kernel.real)
            b = contract("bhl, lhp -> bhp", x.imag, kernel.imag)
            c = contract("bhl, lhp -> bhp", x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
            if self.bidirectional:
                a = contract("bhl, lhp -> bhp", x.real, kernel.flip(-1).real)
                b = contract("bhl, lhp -> bhp", x.imag, kernel.flip(-1).imag)
                c = contract("bhl, lhp -> bhp", x.real + x.imag, kernel.flip(-1,-2).real + kernel.flip(-1).imag)
                real_out = torch.cat([real_out, a - b], dim=1)
                imag_out = torch.cat([imag_out, c - a - b], dim=1)
        return real_out + 1j*imag_out

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv1d,
                       stride=self.stride, 
                       padding=self.padding, 
                       dilation=self.dilation, 
                       groups=self.groups)
        kernels = self.compute_kernel()
        kernel_h = kernels[0]
        kernel_w = kernels[1]

        b, c, h, w = x.shape

        # apply kernel to each dimension
        x = rearrange(x, "b c h w -> (b w) c h")
        x = self.apply_kernel(x, kernel_h)
        x = rearrange(x, "(b w) c p -> (b p) c w", w=w)
        x = self.apply_kernel(x, kernel_w)
        x = rearrange(x, "(b p) c w -> b c p w", p=self.P) # x: b, c, p, p
        return x
    
    def reconstruct(self, x):
        g_list = []
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            Lambda_L = powers_lambda[0]
            V = self.get_C_tilde(dim).T * Lambda_L
            g = powers_lambda @ V # k, P
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        x = rearrange(x, "b c h w -> (b h) c w")
        x = contract("bcw, Ww -> bcW", x, g_list[1])
        x = rearrange(x, "(b h) c W -> (b W) c h", h=self.P)
        x = contract("bch, Hh -> bcH", x, g_list[0])
        # if self.bidirectional:
        #     # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
        #     g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        out = rearrange(x, "(b W) c H -> b c H W", W=self.k2).real
        out = F.gelu(out)
        out = self.C_second(out)
        return out

class SSMConv2dv3(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        H = in_channels
        P = out_channels
        # if bidirectional:
        #     P = P // 2
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2
        
        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_shape = (local_P, H)
            B_init = kaiming_normal_
            B = init_VinvB(B_init, B_shape, Vinv)
            self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

        # Initialize state to output (C) matrix
        if C_init_method in ["trunc_standard_normal"]:
            C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
            C_init = kaiming_normal_
        elif C_init_method in ["complex_normal"]:
            C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
            C_init = partial(normal_, std=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))
    
        if bidirectional:
            C1 = init_CV(C_init, C_shape, V[:,:self.P//4])
            C2 = init_CV(C_init, C_shape, V[:,self.P//4:self.P//2])
            C3 = init_CV(C_init, C_shape, V[:,self.P//2:3*self.P//4])
            C4 = init_CV(C_init, C_shape, V[:,3*self.P//4:])
            C1 = C1[..., 0] + 1j * C1[..., 1]
            C2 = C2[..., 0] + 1j * C2[..., 1]
            C3 = C3[..., 0] + 1j * C3[..., 1]
            C4 = C4[..., 0] + 1j * C4[..., 1]
            self.register_parameter(f'C1', nn.Parameter(C1))
            self.register_parameter(f'C2', nn.Parameter(C2))
            self.register_parameter(f'C3', nn.Parameter(C3))
            self.register_parameter(f'C4', nn.Parameter(C4))
        else:
            C = init_CV(C_init, C_shape, V)
            self.register_parameter(f'C', nn.Parameter(C))


        c_s = P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
        # setattr(self, f'C_second_{dim}', nn.Conv2d(c_s, H, 1, 1))
        # else:
        #     self.act = ComplexActivations(act="gelu", activation_type="amp_phase")
        #     self.C_second = ComplexConv2d(c_s, H, 1, 1)
        #     # setattr(self, f'C_second_{dim}', ComplexConv2d(c_s, H, 1, 1))
        self.norm = nn.InstanceNorm2d(P)

        # Initialize feedthrough (D) matrix
        D = normal_(torch.empty(1, H*k1*k2, 1, 1), std=1.0)
        self.register_parameter('D', nn.Parameter(D))
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale
        # self.channel_recover = nn.Conv2d(P, P*P*k1*k2, 1, 1)

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        if self.bidirectional:
            C1 = getattr(self, f'C1')
            C2 = getattr(self, f'C2')
            C3 = getattr(self, f'C3')
            C4 = getattr(self, f'C4')
            return torch.cat([C1, C2, C3, C4], axis=-1)
        else:
            return getattr(self, f'C')[..., 0] + 1j * getattr(self, f'C')[..., 1]

    def compute_kernel(self):
        
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.get_B_tilde(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("khp,lhp -> phkl", kernels[0], kernels[1])
        if self.bidirectional:
            kernel = torch.cat([
                kernel[:self.P//4],
                kernel[self.P//4:self.P//2].flip(-1),
                kernel[self.P//2:3*self.P//4].flip(-2),
                kernel[3*self.P//4:].flip(-1,-2)
            ], dim=0)
        
        return kernel

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv2d,
                       stride=self.stride,  
                       dilation=self.dilation, 
                       groups=self.groups)
        x = F.pad(x, 2*self.padding, mode="replicate")
        kernel = self.compute_kernel()
        
        if not x.is_complex():
            real_out = conv(x, kernel.real)
            imag_out = conv(x, kernel.imag)
            # if self.bidirectional:
            #     real_out_f = conv(x, kernel.real.flip(-1,-2))
            #     imag_out_f = conv(x, kernel.imag.flip(-1,-2))
            #     real_out = torch.cat([real_out, real_out_f], dim=1)
            #     imag_out = torch.cat([imag_out, imag_out_f], dim=1)
        else:
            a = conv(x.real, kernel.real)
            b = conv(x.imag, kernel.imag)
            c = conv(x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
            # if self.bidirectional:
            #     a = conv(x.real, kernel.flip(-1,-2).real)
            #     b = conv(x.imag, kernel.flip(-1,-2).imag)
            #     c = conv(x.real + x.imag, kernel.flip(-1,-2).real + kernel.flip(-1,-2).imag)
            #     real_out = torch.cat([real_out, a - b], dim=1)
            #     imag_out = torch.cat([imag_out, c - a - b], dim=1)
        out = real_out + 1j*imag_out

        # C = self.get_C_tilde()
        # out = contract("bphw, pc -> bchw", out, C).real
        return out
    
    def reconstruct(self, x):
        g_list = []
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = self.get_Lambda(dim)
            # G = (self.get_C_tilde(dim).T * lambda_L) + (self.get_C_tilde(dim) * lambdas)
            # G = (self.get_C_tilde(dim).T * (lambda_L[:, None] @ lambda_L[None]) / lambdas) # P x P
            # G = (self.get_C_tilde(dim).T * lambda_L / lambdas) # P x P
            G = (self.get_C_tilde().T * lambda_L * lambdas) # P x P
            # local_P = len(self.get_C_tilde(dim)) // 2
            # C_1 = self.get_C_tilde(dim)[:local_P]
            # C_2 = self.get_C_tilde(dim)[local_P:]
            # G = (C_1 * lambdas * lambda_L)# P x P
            # G = (self.get_C_tilde(dim).T * lambdas * lambda_L) # P x P
            # G = (lambda_L[:, None] @ lambda_L[None]) * (self.get_C_tilde(dim) @ lambdas)
            g = powers_lambda @ G # k, P
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 
        if self.bidirectional:
            # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
            # g = torch.cat([g, g.flip(-1,-2)], dim=-1)
            g = torch.cat([
                g[..., :self.P//4],
                g[..., self.P//4:self.P//2].flip(-1),
                g[..., self.P//2:3*self.P//4].flip(-2),
                g[..., 3*self.P//4:].flip(-1,-2)
            ], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            div = div[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        return out

    def reconstruct_pixelshuffle(self, x):
        """
        Construct PixelShuffle weight (PxPxKxK) using
        powers_lambda_{dim}: K, P
        lambdas_{dim}: P
        learnable weight: C_tilde_{dim}: PxP
        currently only applicable if k1 = k2
        """
        px_out = F.pixel_shuffle(self.channel_recover(x), self.k1)
        g_list = []
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = self.get_Lambda(dim)
            G = (self.get_C_tilde(dim).T * lambda_L * lambdas) # P x P
            g = powers_lambda @ G # k, P
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1])
        if self.bidirectional:
            # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
            g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            div = div[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        out += px_out
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        return out

class SSMConv2dv4(nn.Module):
    """
    different way to reconstruct the original input sequence
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        H = in_channels
        P = out_channels
        # if bidirectional:
        #     P = P // 2
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2
        
        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_shape = (local_P, H)
            B_init = kaiming_normal_
            B = init_VinvB(B_init, B_shape, Vinv)
            self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            G_inv_real = torch.empty(k, local_P)
            G_inv_imag = torch.empty(k, local_P)
            nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))
        
        pos_embed = torch.empty(1, H, k1, k2)
        nn.init.kaiming_normal_(pos_embed, mode='fan_out', nonlinearity='linear')
        self.register_parameter(f'pos_embed', nn.Parameter(pos_embed))

        c_s = P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    # def get_C_tilde(self):
    #     """Get complex C_tilde, handling bidirectional case."""
    #     if self.bidirectional:
    #         C1 = getattr(self, f'C1')
    #         C2 = getattr(self, f'C2')
    #         C3 = getattr(self, f'C3')
    #         C4 = getattr(self, f'C4')
    #         return torch.cat([C1, C2, C3, C4], axis=-1)
    #     else:
    #         return getattr(self, f'C')[..., 0] + 1j * getattr(self, f'C')[..., 1]
        
    def get_reconstruct_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_layer_{dim}')

    def get_reconstruct_second_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_second_layer_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_kernel(self):
        
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.get_B_tilde(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("khp,lhp -> phkl", kernels[0], kernels[1])
        if self.bidirectional:
            kernel = torch.cat([
                kernel[:self.P//4],
                kernel[self.P//4:self.P//2].flip(-1),
                kernel[self.P//2:3*self.P//4].flip(-2),
                kernel[3*self.P//4:].flip(-1,-2)
            ], dim=0)
        
        return kernel

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv2d,
                       stride=self.stride,  
                       dilation=self.dilation, 
                       groups=self.groups)
        padding_doubled = tuple(x for p in self.padding[::-1] for x in (p, p))
        # Get input size and pos_embed size
        _, _, H, W = x.shape
        h, w = self.pos_embed.shape[-2:]
        
        # Calculate repeat factors needed
        repeat_h = H // h
        repeat_w = W // w

        # Repeat each element the required number of times
        pos_embed_expanded = self.pos_embed.repeat_interleave(repeat_h, dim=-2).repeat_interleave(repeat_w, dim=-1)
        x = x + pos_embed_expanded
        x = F.pad(x, padding_doubled, mode="replicate")
        kernel = self.compute_kernel()
        
        if not x.is_complex():
            real_out = conv(x, kernel.real)
            imag_out = conv(x, kernel.imag)
        else:
            a = conv(x.real, kernel.real)
            b = conv(x.imag, kernel.imag)
            c = conv(x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
        out = real_out + 1j*imag_out

        return out
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        # if self.bidirectional:
        #     # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
        #     # g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        #     g = torch.cat([
        #         g[..., :self.P//4],
        #         g[..., self.P//4:self.P//2].flip(-1),
        #         g[..., self.P//2:3*self.P//4].flip(-2),
        #         g[..., 3*self.P//4:].flip(-1,-2)
        #     ], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out

    def reconstruct_pixelshuffle(self, x):
        """
        Construct PixelShuffle weight (PxPxKxK) using
        powers_lambda_{dim}: K, P
        lambdas_{dim}: P
        learnable weight: C_tilde_{dim}: PxP
        currently only applicable if k1 = k2
        """
        px_out = F.pixel_shuffle(self.channel_recover(x), self.k1)
        g_list = []
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = self.get_Lambda(dim)
            G = (self.get_C_tilde(dim).T * lambda_L * lambdas) # P x P
            g = powers_lambda @ G # k, P
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1])
        if self.bidirectional:
            # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
            g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            div = div[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        out += px_out
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        return out

class SSMConv2dv5(nn.Module):
    """
    channel inflated and reverted to out_channels to match the num of parameters
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2

        H = in_channels
        Q = out_channels

        # conv2d_params = H*Q*k1*k2 + Q
        # # ssmconv2d_params = 4*Q + 4*H*Q + 2*Q
        # # scaled_ssmconv2d_params = 4*Q + 4*H*Q + 2*Q*P + 2*P

        # for p in range(16, 1024):
        #     scaled_ssmconv2d_params = 4*p + 4*H*p + 2*p*Q + 2*Q
        #     if scaled_ssmconv2d_params > conv2d_params:
        #         print(f"P: {p}, scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")
        #         if p % blocks == 0:
        #             break
        # if scaled_ssmconv2d_params < conv2d_params:
        #     raise ValueError("kernel size is too big")
        P = Q
        
        block_size = P // blocks
        self.H = H
        self.P = P
        self.Q = Q
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2
        
        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_shape = (local_P, H)
            B_init = kaiming_normal_
            B = init_VinvB(B_init, B_shape, Vinv)
            self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            G_inv_real = torch.empty(k, Q)
            G_inv_imag = torch.empty(k, Q)
            nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))

        # Initialize state to output (C) matrix
        if C_init_method in ["trunc_standard_normal"]:
            C_shape = (Q, local_P, 2)
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_shape = (Q, local_P, 2)
            C_init = kaiming_normal_
        elif C_init_method in ["complex_normal"]:
            C_shape = (Q, local_P, 2)
            C_init = partial(normal_, std=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))

        # if bidirectional:
        #     C1 = init_CV(C_init, C_shape, V)
        #     C2 = init_CV(C_init, C_shape, V)
        #     C3 = init_CV(C_init, C_shape, V)
        #     C4 = init_CV(C_init, C_shape, V)
        #     C1 = C1[..., 0] + 1j * C1[..., 1]
        #     C2 = C2[..., 0] + 1j * C2[..., 1]
        #     C3 = C3[..., 0] + 1j * C3[..., 1]
        #     C4 = C4[..., 0] + 1j * C4[..., 1]
        #     self.register_parameter(f'C1', nn.Parameter(C1))
        #     self.register_parameter(f'C2', nn.Parameter(C2))
        #     self.register_parameter(f'C3', nn.Parameter(C3))
        #     self.register_parameter(f'C4', nn.Parameter(C4))
        C = init_CV(C_init, C_shape, V)
        self.register_parameter(f'C', nn.Parameter(C))

        C_bias = torch.zeros(Q, 2)
        self.register_parameter(f'C_bias', nn.Parameter(C_bias))

        c_s = Q
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        return getattr(self, f'C')[..., 0] + 1j * getattr(self, f'C')[..., 1]
    
    def get_C_bias(self):    
        C_bias_real = getattr(self, f'C_bias')[..., 0]
        C_bias_imag = getattr(self, f'C_bias')[..., 1]
        return (C_bias_real + 1j * C_bias_imag)[None, :, None, None]
        
    def get_reconstruct_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_layer_{dim}')

    def get_reconstruct_second_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_second_layer_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_kernel(self):
        
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.get_B_tilde(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("khp,lhp -> phkl", kernels[0], kernels[1])
        # kernel += self.pos_embed

        if self.bidirectional:
            kernel = torch.cat([
                kernel[:self.P//4],
                kernel[self.P//4:self.P//2].flip(-1),
                kernel[self.P//2:3*self.P//4].flip(-2),
                kernel[3*self.P//4:].flip(-1,-2)
            ], dim=0)
        
        return kernel

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv2d,
                       stride=self.stride,  
                       dilation=self.dilation, 
                       groups=self.groups)
        padding_doubled = tuple(x for p in self.padding[::-1] for x in (p, p))

        x = F.pad(x, padding_doubled, mode="replicate")
        kernel = self.compute_kernel()
        
        if not x.is_complex():
            real_out = conv(x, kernel.real)
            imag_out = conv(x, kernel.imag)
        else:
            a = conv(x.real, kernel.real)
            b = conv(x.imag, kernel.imag)
            c = conv(x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
        out = real_out + 1j*imag_out

        C_tilde = self.get_C_tilde()
        C_bias = self.get_C_bias()
        out = contract("bphw, cp -> bchw", out, C_tilde)
        out = out + C_bias

        return out.real
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        # if self.bidirectional:
        #     # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
        #     # g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        #     g = torch.cat([
        #         g[..., :self.P//4],
        #         g[..., self.P//4:self.P//2].flip(-1),
        #         g[..., self.P//2:3*self.P//4].flip(-2),
        #         g[..., 3*self.P//4:].flip(-1,-2)
        #     ], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)

        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out

class SSMConv2dv6(nn.Module):
    """
    following Mamba, make input-dependent \delta, B, and C.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = True,
                 step_rescale: float = 1.0,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2

        H = in_channels
        P = out_channels

        # conv2d_params = H*Q*k1*k2 + Q
        # ssmconv2d_params = 4*Q + 4*H*Q + 2*Q
        # scaled_ssmconv2d_params = 4*Q + 4*H*Q + 2*Q*P + 2*P

        # for p in range(Q, 1024):
        #     scaled_ssmconv2d_params = 4*p + 4*H*p + 2*p*Q + 2*Q
        #     if scaled_ssmconv2d_params > conv2d_params:
        #         print(f"P: {p}, scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")
        #         if p % blocks == 0:
        #             break
        # if scaled_ssmconv2d_params < conv2d_params:
        #     raise ValueError("kernel size is too big")
        # P = p
        
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2

        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks)) # P x P

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_proj = nn.Linear([k2, k1][dim], local_P)
            setattr(self, f'B_proj_{dim}', B_proj)
            # B_shape = (local_P, H)
            # B_init = kaiming_normal_
            # B = init_VinvB_takeB(Vinv, B_init(B_shape))
            # self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            G_inv_real = torch.empty(k, P)
            G_inv_imag = torch.empty(k, P)
            nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))

        # Initialize state to output (C) matrix
        if C_init_method in ["trunc_standard_normal"]:
            C_shape = (P, local_P, 2)
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_shape = (P, local_P, 2)
            C_init = kaiming_normal_
        elif C_init_method in ["complex_normal"]:
            C_shape = (P, local_P, 2)
            C_init = partial(normal_, std=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))

        # if bidirectional:
        #     C1 = init_CV(C_init, C_shape, V)
        #     C2 = init_CV(C_init, C_shape, V)
        #     C1 = C1[..., 0] + 1j * C1[..., 1]
        #     C2 = C2[..., 0] + 1j * C2[..., 1]
        #     self.register_parameter(f'C1', nn.Parameter(C1))
        #     self.register_parameter(f'C2', nn.Parameter(C2))
        # else:
        C = init_CV(C_init, C_shape, V)
        self.register_parameter(f'C', nn.Parameter(C))

        C_bias = torch.zeros(P, 2)
        self.register_parameter(f'C_bias', nn.Parameter(C_bias))

        c_s = P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    # def get_B_tilde(self, dim: int):
    #     """Get complex B_tilde for given dimension."""
    #     return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        # if self.bidirectional:
        #     C1 = getattr(self, f'C1')
        #     C2 = getattr(self, f'C2')
        #     C3 = getattr(self, f'C3')
        #     C4 = getattr(self, f'C4')
        #     return torch.cat([C1, C2, C3, C4], axis=-1)
        # else:
        return getattr(self, f'C')[..., 0] + 1j * getattr(self, f'C')[..., 1]
    
    def get_C_bias(self):
        C_bias_real = getattr(self, f'C_bias')[..., 0]
        C_bias_imag = getattr(self, f'C_bias')[..., 1]
        return (C_bias_real + 1j * C_bias_imag)[None, :, None, None]
        
    def get_reconstruct_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_layer_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_B_tilde(self, x, dim: int):
        """
        compute B_tilde based on x
        input x: B, C, H2, W2, K1, K2
        output B_tilde: complex tensor of shape;s
        (B, C, H2, W2, K2, local_P) or (B, C, H2, W2, K1, local_P) depending on dim
        """
        B_proj = getattr(self, f'B_proj_{dim}')
        if dim == 0:
            B = B_proj(x)
        elif dim == 1:
            B = B_proj(x.transpose(-1,-2))

        V_inv = getattr(self, f'Vinv_{dim}')
        B_tilde = init_VinvB_takeB(V_inv, B)
        B_tilde = B_tilde[..., 0] + 1j * B_tilde[..., 1]

        return B_tilde

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, H2, W2, K1, K2
        output kernel: B, C, H2, W2, K1, K2
        """
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.compute_B_tilde(x, dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh_v2(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear_v2(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[None, None, None, None] * B_bar # B, C, H2, W2, k, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("bchwkp,bchwlp -> bchwklp", kernels[0], kernels[1])

        # if self.bidirectional:
        #     kernel = torch.cat([
        #         kernel[:self.P//4],
        #         kernel[self.P//4:self.P//2].flip(-1),
        #         kernel[self.P//2:3*self.P//4].flip(-2),
        #         kernel[3*self.P//4:].flip(-1,-2)
        #     ], dim=0)
        
        return kernel

    def forward(self, x):
        """
        we compute and input-adaptive kernel (as well as B) and then compute the output
        """
        padding_doubled = tuple(a for p in self.padding[::-1] for a in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")

        x = x.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        x = x.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        # B, C, H2, W2, K1, K2 = x.shape

        kernel = self.compute_kernel(x)
        # B, C, H2, W2, K1, K2, P = kernel.shape

        # we always assume real input x
        real_out = contract("bchwkl, bchwklp -> bphw", x, kernel.real)
        imag_out = contract("bchwkl, bchwklp -> bphw", x, kernel.imag)
        out = real_out + 1j*imag_out

        C_tilde = self.get_C_tilde()
        C_bias = self.get_C_bias()
        out = contract("bphw, cp -> bchw", out, C_tilde)
        out = out + C_bias

        return out.real
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        # if self.bidirectional:
        #     # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
        #     # g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        #     g = torch.cat([
        #         g[..., :self.P//4],
        #         g[..., self.P//4:self.P//2].flip(-1),
        #         g[..., self.P//2:3*self.P//4].flip(-2),
        #         g[..., 3*self.P//4:].flip(-1,-2)
        #     ], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out

class SSMConv2dv7(nn.Module):
    """
    following Mamba, make input-dependent B and \delta.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = True,
                 step_rescale: float = 1.0,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size

        try:
            s1, s2 = stride[0], stride[1]
        except:
            s1, s2 = stride, stride

        self.k1 = k1
        self.k2 = k2
        self.s1 = s1
        self.s2 = s2

        H = in_channels
        P = out_channels

        # conv2d_params = H*Q*k1*k2 + Q
        # ssmconv2d_params = 4*Q + 4*H*Q + 2*Q
        # scaled_ssmconv2d_params = 4*Q + 4*H*Q + 2*Q*P + 2*P

        # for p in range(Q, 1024):
        #     scaled_ssmconv2d_params = 4*p + 4*H*p + 2*p*Q + 2*Q
        #     if scaled_ssmconv2d_params > conv2d_params:
        #         print(f"P: {p}, scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")
        #         if p % blocks == 0:
        #             break
        # if scaled_ssmconv2d_params < conv2d_params:
        #     raise ValueError("kernel size is too big")
        # P = p
        
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2

        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks)) # P x P

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            x_proj = nn.Linear([k2, k1][dim], 2*local_P)
            setattr(self, f'x_proj_{dim}', x_proj)
            # B_shape = (local_P, H)
            # B_init = kaiming_normal_
            # B = init_VinvB_takeB(Vinv, B_init(B_shape))
            # self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            # log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            # self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            G_inv_real = torch.empty(k, P)
            G_inv_imag = torch.empty(k, P)
            nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))

        # Initialize state to output (C) matrix
        if C_init_method in ["trunc_standard_normal"]:
            C_shape = (P, local_P, 2)
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_shape = (P, local_P, 2)
            C_init = kaiming_normal_
        elif C_init_method in ["complex_normal"]:
            C_shape = (P, local_P, 2)
            C_init = partial(normal_, std=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))

        # if bidirectional:
        #     C1 = init_CV(C_init, C_shape, V)
        #     C2 = init_CV(C_init, C_shape, V)
        #     C1 = C1[..., 0] + 1j * C1[..., 1]
        #     C2 = C2[..., 0] + 1j * C2[..., 1]
        #     self.register_parameter(f'C1', nn.Parameter(C1))
        #     self.register_parameter(f'C2', nn.Parameter(C2))
        # else:
        C = init_CV(C_init, C_shape, V)
        self.register_parameter(f'C', nn.Parameter(C))

        C_bias = torch.zeros(P, 2)
        self.register_parameter(f'C_bias', nn.Parameter(C_bias))

        c_s = P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    # def get_B_tilde(self, dim: int):
    #     """Get complex B_tilde for given dimension."""
    #     return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        # if self.bidirectional:
        #     C1 = getattr(self, f'C1')
        #     C2 = getattr(self, f'C2')
        #     C3 = getattr(self, f'C3')
        #     C4 = getattr(self, f'C4')
        #     return torch.cat([C1, C2, C3, C4], axis=-1)
        # else:
        return getattr(self, f'C')[..., 0] + 1j * getattr(self, f'C')[..., 1]
    
    def get_C_bias(self):
        C_bias_real = getattr(self, f'C_bias')[..., 0]
        C_bias_imag = getattr(self, f'C_bias')[..., 1]
        return (C_bias_real + 1j * C_bias_imag)[None, :, None, None]
        
    def get_reconstruct_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_layer_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_B_tilde_and_log_step(self, x, dim: int):
        """
        compute B_tilde based on x
        input x: B, C, H2, W2, K1, K2
        output B_tilde: complex tensor of shape;
        (B, C, H2, W2, K2, local_P) or (B, C, H2, W2, K1, local_P) depending on dim
        """
        x_proj = getattr(self, f'x_proj_{dim}')
        if dim == 0:
            B, log_step = x_proj(x).split(self.local_P, dim=-1)
        elif dim == 1:
            B, log_step = x_proj(x.transpose(-1,-2)).split(self.local_P, dim=-1)
        
        V_inv = getattr(self, f'Vinv_{dim}')
        B_tilde = init_VinvB_takeB(V_inv, B)
        B_tilde = B_tilde[..., 0] + 1j * B_tilde[..., 1]
        log_step = log_step * (
            torch.log(torch.tensor(self.dt_max)) - torch.log(torch.tensor(self.dt_min))
        ) + torch.log(torch.tensor(self.dt_min))

        return B_tilde, log_step

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, H2, W2, K1, K2
        output kernel: B, C, H2, W2, K1, K2
        """
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            B_tilde, log_step = self.compute_B_tilde_and_log_step(x, dim)
            step = self.step_rescale * torch.exp(log_step)
            
            Lambda = self.get_Lambda(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh_v2(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear_v2(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[None, None, None, None] * B_bar # B, C, H2, W2, k, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("bchwkp,bchwlp -> bchwklp", kernels[0], kernels[1])

        if self.bidirectional:
            kernel = torch.cat([
                kernel[..., :self.P//4],
                kernel[..., self.P//4:self.P//2].flip(-1),
                kernel[..., self.P//2:3*self.P//4].flip(-2),
                kernel[..., 3*self.P//4:].flip(-1,-2)
            ], dim=-1)
        
        return kernel

    def forward(self, x):
        """
        we compute and input-adaptive kernel (as well as B) and then compute the output
        """
        padding_doubled = tuple(a for p in self.padding[::-1] for a in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")

        x = x.unfold(dimension=2, size=self.k1, step=self.s1)
        x = x.unfold(dimension=3, size=self.k2, step=self.s2)
        # B, C, H2, W2, K1, K2 = x.shape

        kernel = self.compute_kernel(x)
        # B, C, H2, W2, K1, K2, P = kernel.shape

        # we always assume real input x
        real_out = contract("bchwkl, bchwklp -> bphw", x, kernel.real)
        imag_out = contract("bchwkl, bchwklp -> bphw", x, kernel.imag)
        out = real_out + 1j*imag_out

        C_tilde = self.get_C_tilde()
        C_bias = self.get_C_bias()
        out = contract("bphw, cp -> bchw", out, C_tilde)

        out = out + C_bias

        return out.real
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        # if self.bidirectional:
        #     # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
        #     # g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        #     g = torch.cat([
        #         g[..., :self.P//4],
        #         g[..., self.P//4:self.P//2].flip(-1),
        #         g[..., self.P//2:3*self.P//4].flip(-2),
        #         g[..., 3*self.P//4:].flip(-1,-2)
        #     ], dim=-1)
        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out


class SSMConv2dv8(nn.Module):
    """
    following Mamba, make input-dependent B and \delta, and real A.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = True,
                 step_rescale: float = 1.0,
                 add_conv_kernel: bool = False,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size

        try:
            s1, s2 = stride[0], stride[1]
        except:
            s1, s2 = stride, stride

        self.k1 = k1
        self.k2 = k2
        self.s1 = s1
        self.s2 = s2

        H = in_channels
        P = out_channels

        # conv2d_params = H*Q*k1*k2 + Q
        # ssmconv2d_params = 4*Q + 4*H*Q + 2*Q
        # scaled_ssmconv2d_params = 4*Q + 4*H*Q + 2*Q*P + 2*P

        # for p in range(Q, 1024):
        #     scaled_ssmconv2d_params = 4*p + 4*H*p + 2*p*Q + 2*Q
        #     if scaled_ssmconv2d_params > conv2d_params:
        #         print(f"P: {p}, scaled_ssmconv2d_params: {scaled_ssmconv2d_params}")
        #         if p % blocks == 0:
        #             break
        # if scaled_ssmconv2d_params < conv2d_params:
        #     raise ValueError("kernel size is too big")
        # P = p
        
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2

        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO_real(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks)) # P x P

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_{dim}', nn.Parameter(Lambda))
            # self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            x_proj = nn.Linear([k2, k1][dim], 2*local_P)
            setattr(self, f'x_proj_{dim}', x_proj)
            # B_shape = (local_P, H)
            # B_init = kaiming_normal_
            # B = init_VinvB_takeB(Vinv, B_init(B_shape))
            # self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            # log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            # self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            G_inv_real = torch.empty(k, P)
            G_inv_imag = torch.empty(k, P)
            nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))

        # Initialize state to output (C) matrix
        if C_init_method in ["trunc_standard_normal"]:
            C_shape = (P//2, local_P)
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_shape = (P, local_P)
            C_init = kaiming_normal_
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))

        C = init_CV_real(C_init, C_shape, V)
        self.register_parameter(f'C', nn.Parameter(C))

        C_bias = torch.zeros(P//2)
        self.register_parameter(f'C_bias', nn.Parameter(C_bias))

        c_s = P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv2d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale
        self.add_conv_kernel = add_conv_kernel

        if add_conv_kernel:
            conv_kernel = nn.Conv2d(H, P, kernel_size, stride, padding, dilation, groups)
            self.register_parameter(f'conv_kernel_weight', nn.Parameter(conv_kernel.weight))
            self.register_parameter(f'conv_kernel_bias', nn.Parameter(conv_kernel.bias))

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda = getattr(self, f'Lambda_{dim}')
        if self.clip_eigs:
            Lambda = torch.clamp(Lambda, max=-1e-4)
        return Lambda
    
    # def get_B_tilde(self, dim: int):
    #     """Get complex B_tilde for given dimension."""
    #     return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        # if self.bidirectional:
        #     C1 = getattr(self, f'C1')
        #     C2 = getattr(self, f'C2')
        #     C3 = getattr(self, f'C3')
        #     C4 = getattr(self, f'C4')
        #     return torch.cat([C1, C2, C3, C4], axis=-1)
        # else:
        return getattr(self, f'C')
    
    def get_C_bias(self):
        return getattr(self, f'C_bias')
        
    def get_reconstruct_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_layer_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_B_tilde_and_log_step(self, x, dim: int):
        """
        compute B_tilde based on x
        input x: B, C, H2, W2, K1, K2
        output B_tilde: complex tensor of shape;
        (B, C, H2, W2, K2, local_P) or (B, C, H2, W2, K1, local_P) depending on dim
        """
        x_proj = getattr(self, f'x_proj_{dim}')
        if dim == 0:
            B, log_step = x_proj(x).split(self.local_P, dim=-1)
        elif dim == 1:
            B, log_step = x_proj(x.transpose(-1,-2)).split(self.local_P, dim=-1)
        
        V_inv = getattr(self, f'Vinv_{dim}')
        B_tilde = init_VinvB_takeB(V_inv, B, real=True)

        log_step = log_step * (
            torch.log(torch.tensor(self.dt_max)) - torch.log(torch.tensor(self.dt_min))
        ) + torch.log(torch.tensor(self.dt_min))

        return B_tilde, log_step

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, H2, W2, K1, K2
        output kernel: B, C, H2, W2, K1, K2
        """
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            B_tilde, log_step = self.compute_B_tilde_and_log_step(x, dim)
            step = self.step_rescale * torch.exp(log_step)
            
            Lambda = self.get_Lambda(dim)
            
            if self.discretization == "zoh":
                power_lambda_bar, B_bar = discretize_zoh_v2(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                raise NotImplementedError("Bilinear discretization not implemented yet")
                # Lambda_bar, B_bar = discretize_bilinear_v2(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            # powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P

            kernel = power_lambda_bar * B_bar # B, C, H2, W2, k, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', power_lambda_bar)

        kernel = contract("bchwkp,bchwlp -> bchwklp", kernels[0], kernels[1])
        kernel = kernel.real

        if self.bidirectional:
            kernel = torch.cat([
                kernel[..., :self.P//4],
                kernel[..., self.P//4:self.P//2].flip(-1),
                kernel[..., self.P//2:3*self.P//4].flip(-2),
                kernel[..., 3*self.P//4:].flip(-1,-2)
            ], dim=-1)
        
        return kernel

    def forward(self, x):
        """
        we compute and input-adaptive kernel (as well as B) and then compute the output
        """
        padding_doubled = tuple(a for p in self.padding[::-1] for a in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")

        x = x.unfold(dimension=2, size=self.k1, step=self.s1)
        x = x.unfold(dimension=3, size=self.k2, step=self.s2)
        # B, C, H2, W2, K1, K2 = x.shape

        kernel = self.compute_kernel(x)
        if self.add_conv_kernel:
            conv_kernel_weight = getattr(self, f'conv_kernel_weight')
            conv_kernel_weight = rearrange(conv_kernel_weight, "p h k l -> 1 h 1 1 k l p")
            kernel = conv_kernel_weight * kernel
        # B, C, H2, W2, K1, K2, P = kernel.shape

        # we always assume real input x
        out = contract("bchwkl, bchwklp -> bphw", x, kernel.real)
        # imag_out = contract("bchwkl, bchwklp -> bphw", x, kernel.imag)
        # out = real_out + 1j*imag_out
        if self.add_conv_kernel:
            conv_kernel_bias = getattr(self, f'conv_kernel_bias')
            out = out + conv_kernel_bias[None, :, None, None]

        C_tilde = self.get_C_tilde()
        C_bias = self.get_C_bias()
        out = contract("bphw, cp -> bchw", out, C_tilde)

        out = out + C_bias[None, :, None, None]
        return out
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out


class SSMConv2dv9(nn.Module):
    """
    following Mamba, make input-dependent B, C, and \delta, and real A.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 state_size: int = 16,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_rank: int = 0,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = True,
                 step_rescale: float = 1.0,
                 add_conv_kernel: bool = False,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size

        try:
            s1, s2 = stride[0], stride[1]
        except:
            s1, s2 = stride, stride

        self.k1 = k1
        self.k2 = k2
        self.s1 = s1
        self.s2 = s2

        H = in_channels
        P = out_channels
        N = state_size

        self.H = H
        self.P = P
        self.N = N
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_rank = dt_rank
        if dt_rank == 0:
            self.dt_rank = N

        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            A = torch.arange(1, N+1)
            A_log = nn.Parameter(torch.log(A))
            self.register_parameter(f'A_log_{dim}', A_log)

            # to generate B, \delta
            x_proj = nn.Linear([k2, k1][dim], N + dt_rank)
            setattr(self, f'x_proj_{dim}', x_proj)

            dt_proj = nn.Linear(dt_rank, N)
            setattr(self, f'dt_proj_{dim}', dt_proj)

            # G_inv_real = torch.empty(k, P)
            # G_inv_imag = torch.empty(k, P)
            # nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            # nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            # self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))
        
        self.C_proj = nn.Conv2d(H, N*P, (k1,k2), (k1,k2))

        self.step_rescale = step_rescale
        self.add_conv_kernel = add_conv_kernel

        if add_conv_kernel:
            conv_kernel = nn.Conv2d(H, P, kernel_size, stride, padding, dilation, groups)
            self.register_parameter(f'conv_kernel_weight', nn.Parameter(conv_kernel.weight))
            self.register_parameter(f'conv_kernel_bias', nn.Parameter(conv_kernel.bias))


    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")
    
    
    def get_A_log(self, dim: int):
        """Get A_log for given dimension."""
        return getattr(self, f'A_log_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_B_and_step(self, x, dim: int):
        """
        compute B and step based on x
        input x: B, C, H2, W2, K1, K2
        output B: real tensor of shape;
        (B, C, H2, W2, K2, N) or (B, C, H2, W2, K1, N) depending on dim
        output step: real tensor of shape;
        (B, C, H2, W2, K2, N) or (B, C, H2, W2, K1, N) depending on dim
        """
        x_proj = getattr(self, f'x_proj_{dim}')
        dt_proj = getattr(self, f'dt_proj_{dim}')
        if dim == 0:
            B, step = x_proj(x).split(self.N, dim=-1)
        elif dim == 1:
            B, step = x_proj(x.transpose(-1,-2)).split(self.N, dim=-1)

        step = F.softplus(dt_proj(step))

        return B, step

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, H2, W2, K1, K2
        output kernel: B, C, H2, W2, K1, K2
        """
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            B, step = self.compute_B_and_step(x, dim)
            step = self.step_rescale * step
            
            A_log = self.get_A_log(dim)
            A = -torch.exp(A_log.float())
            
            if self.discretization == "zoh":
                power_lambda_bar, B_bar = discretize_zoh_v3(A, B, step)
            elif self.discretization == "bilinear":
                raise NotImplementedError("Bilinear discretization not implemented yet")
                # Lambda_bar, B_bar = discretize_bilinear_v2(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            kernel = power_lambda_bar * B_bar # B, C, H2, W2, k, P
            kernels.append(kernel)

        kernel = contract("bchwkp,bchwlp -> bchwklp", kernels[0], kernels[1])

        if self.bidirectional:
            kernel = torch.cat([
                kernel[..., :self.P//4],
                kernel[..., self.P//4:self.P//2].flip(-1),
                kernel[..., self.P//2:3*self.P//4].flip(-2),
                kernel[..., 3*self.P//4:].flip(-1,-2)
            ], dim=-1)
        
        return kernel
    
    def compute_C(self, x):
        """
        B, C, H2, W2, K1, K2 -> B, C, H2, W2, N * P
        """
        C = self.C_proj(rearrange(x, "b c h2 w2 k1 k2 -> (b h2 w2) c k1 k2"))
        C = rearrange(C, "(b h2 w2) (n p) 1 1 -> b n p h2 w2", 
                      b=x.size(0), h2=x.size(2), w2=x.size(3), n=self.N, p=self.P)
        return C

    def forward(self, x):
        """
        we compute and input-adaptive kernel (as well as B) and then compute the output
        """
        padding_doubled = tuple(a for p in self.padding[::-1] for a in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")

        x = x.unfold(dimension=2, size=self.k1, step=self.s1)
        x = x.unfold(dimension=3, size=self.k2, step=self.s2)
        # B, C, H2, W2, K1, K2 = x.shape

        kernel = self.compute_kernel(x)
        # B, C, H2, W2, K1, K2, N = kernel.shape

        C = self.compute_C(x)
        kernel = contract("bchwkln, bnphw -> bchwklp", kernel, C)

        if self.add_conv_kernel:
            conv_kernel_weight = getattr(self, f'conv_kernel_weight')
            conv_kernel_weight = rearrange(conv_kernel_weight, "n h k l -> 1 h 1 1 k l n")
            kernel = conv_kernel_weight * kernel

        out = contract("bchwkl, bchwklp -> bphw", x, kernel)

        if self.add_conv_kernel:
            conv_kernel_bias = getattr(self, f'conv_kernel_bias')
            out = out + conv_kernel_bias[None, :, None, None]

        return out
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        # out = F.tanh(out)
        # out = 0.5*F.tanh(out) + 0.5
        return out
    
class SSMConv2dv10(nn.Module):
    """
    same as v9, but with different projection scheme.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 state_size: int = 16,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = True,
                 step_rescale: float = 1.0,
                 add_conv_kernel: bool = False,
                 adapt_C: bool = False,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        try:
            k1, k2 = kernel_size[0], kernel_size[1]
        except:
            k1, k2 = kernel_size, kernel_size

        try:
            s1, s2 = stride[0], stride[1]
        except:
            s1, s2 = stride, stride

        self.k1 = k1
        self.k2 = k2
        self.s1 = s1
        self.s2 = s2

        H = in_channels
        P = out_channels
        N = state_size

        self.H = H
        self.P = P
        self.N = N
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        dt_rank = math.ceil(H / 16)
        self.dt_rank = dt_rank

        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2]):
            A = torch.arange(1, N+1)
            A_log = nn.Parameter(torch.log(A))
            self.register_parameter(f'A_log_{dim}', A_log)

            # to generate B, \delta
            x_proj = nn.Conv1d(H, N + dt_rank, [k2, k1][dim], [k2, k1][dim])
            # x_proj = nn.Linear([k2, k1][dim], N + dt_rank)
            setattr(self, f'x_proj_{dim}', x_proj)

            dt_proj = nn.Linear(dt_rank, H)
            setattr(self, f'dt_proj_{dim}', dt_proj)

            # G_inv_real = torch.empty(k, P)
            # G_inv_imag = torch.empty(k, P)
            # nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            # nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            # self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))
        
        if adapt_C:
            self.C_proj = nn.Conv2d(H, N*P, 1, 1)
        else:
            self.C_proj = nn.Conv2d(N, P, 1, 1)

        self.step_rescale = step_rescale
        self.add_conv_kernel = add_conv_kernel
        self.adapt_C = adapt_C

        if add_conv_kernel:
            conv_kernel = nn.Conv2d(H, P, kernel_size, stride, padding, dilation, groups)
            self.register_parameter(f'conv_kernel_weight', nn.Parameter(conv_kernel.weight))
            self.register_parameter(f'conv_kernel_bias', nn.Parameter(conv_kernel.bias))


    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")
    
    
    def get_A_log(self, dim: int):
        """Get A_log for given dimension."""
        return getattr(self, f'A_log_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_B_and_step(self, x, dim: int):
        """
        compute B and step based on x
        input x: B, C, H2, W2, K1, K2
        output B: real tensor of shape;
        (B, H2, W2, K2, N) or (B, H2, W2, K1, N) depending on dim
        output step: real tensor of shape;
        (B, H2, W2, K2, H) or (B, H2, W2, K1, H) depending on dim
        """
        x_proj = getattr(self, f'x_proj_{dim}')
        dt_proj = getattr(self, f'dt_proj_{dim}')
        if dim == 0:
            B_step = x_proj(rearrange(x, "b c h2 w2 k1 k2 -> (b h2 w2 k1) c k2"))
        elif dim == 1:
            B_step = x_proj(rearrange(x, "b c h2 w2 k1 k2 -> (b h2 w2 k2) c k1"))
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        B_step = rearrange(B_step, "(b h2 w2 k) n_rank 1 -> b h2 w2 k n_rank", b=x.size(0), h2=x.size(2), w2=x.size(3))
        B, step = B_step.split(self.N, dim=-1)
        step = F.softplus(dt_proj(step)) #.clamp(min=self.dt_min, max=self.dt_max)
        return B, step

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, H2, W2, K1, K2
        output kernel: B, C, H2, W2, K1, K2
        """
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2]):
            B, step = self.compute_B_and_step(x, dim)
            step = self.step_rescale * step
            
            A_log = self.get_A_log(dim)
            A = -torch.exp(A_log.float())
            
            if self.discretization == "zoh":
                power_lambda_bar, B_bar = discretize_zoh_v4(A, B, step)
            elif self.discretization == "bilinear":
                raise NotImplementedError("Bilinear discretization not implemented yet")
                # Lambda_bar, B_bar = discretize_bilinear_v2(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            kernel = power_lambda_bar * B_bar # B, C, H2, W2, k, P
            kernels.append(kernel)

        kernel = contract("bchwkp,bchwlp -> bchwklp", kernels[0], kernels[1])

        if self.bidirectional:
            kernel = torch.cat([
                kernel[..., :self.P//4],
                kernel[..., self.P//4:self.P//2].flip(-1),
                kernel[..., self.P//2:3*self.P//4].flip(-2),
                kernel[..., 3*self.P//4:].flip(-1,-2)
            ], dim=-1)
        
        return kernel
    
    def compute_C(self, x):
        """
        adapt_C: B, C, H2, W2, K1, K2 -> B, C, H2, W2, N * P
        non-adapt_C: B, C, H2, W2, K1, K2, P -> B, N, H2, W2, K1, K2, P

        """
        if self.adapt_C:
            C = self.C_proj(rearrange(x, "b c h2 w2 k1 k2 -> (b h2 w2) c k1 k2"))
            C = rearrange(C, "(b h2 w2) (n p) k1 k2 -> b n p h2 w2 k1 k2", 
                        b=x.size(0), h2=x.size(2), w2=x.size(3), n=self.N, p=self.P)
        # else:
        #     kernel = rearrange(kernel, "b c h2 w2 k1 k2 n -> (b h2 w2 n) c k1 k2")
        #     C = self.C_proj(kernel)
        #     C = rearrange(C, "(b h2 w2 n) p 1 1 -> b n p h2 w2", 
        #                 b=x.size(0), h2=x.size(2), w2=x.size(3), n=self.N, p=self.P)
        return C

    def forward(self, x):
        """
        we compute and input-adaptive kernel (as well as B) and then compute the output
        """
        padding_doubled = tuple(a for p in self.padding[::-1] for a in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")

        x = x.unfold(dimension=2, size=self.k1, step=self.s1)
        x = x.unfold(dimension=3, size=self.k2, step=self.s2)
        # B, C, H2, W2, K1, K2 = x.shape

        kernel = self.compute_kernel(x)
        # B, C, H2, W2, K1, K2, N = kernel.shape

        
        if self.adapt_C:
            C = self.compute_C(x)
            # kernel = contract("bchwkln, bnphw -> bchwklp", kernel, C)

        if self.add_conv_kernel:
            conv_kernel_weight = getattr(self, f'conv_kernel_weight')
            conv_kernel_weight = rearrange(conv_kernel_weight, "n h k l -> 1 h 1 1 k l n")
            kernel = conv_kernel_weight + kernel

        out = contract("bchwkl, bchwklp -> bphwkl", x, kernel)

        if self.adapt_C:
            out = contract("bphwkl, bpqhwkl -> bqhw", out, C)

        if not self.adapt_C:
            out = self.C_proj(out)

        if self.add_conv_kernel:
            conv_kernel_bias = getattr(self, f'conv_kernel_bias')
            out = out + conv_kernel_bias[None, :, None, None]

        return out
    
    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[0], g_list[1]) 

        if self.conj_sym:
            out = 2*contract("bphw, klp -> bphkwl", x, g).real
        else:
            out = contract("bphw, klp -> bphkwl", x, g).real

        if self.padding != (0, 0):
            # print(out.shape)
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0], (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1])
            out = rearrange(out, "b p h k w l -> b (p k l) (h w)")
            div = torch.ones_like(out, device=out.device, dtype=out.dtype)
            out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1]]
            out = (out / div)
        else:
            out = rearrange(out, "b p h k w l -> b p (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        # out = F.tanh(out)
        # out = 0.5*F.tanh(out) + 0.5
        return out 


class SSMConv3d(nn.Module):
    """
    different way to reconstruct the original input sequence
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 groups: int = 1,
                 blocks: int = 1,
                 C_init_method: str = "trunc_standard_normal",
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 conj_sym: bool = False,
                 clip_eigs: bool = False,
                 bidirectional: bool = False,
                 step_rescale: float = 1.0,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        H = in_channels
        P = out_channels
        # if bidirectional:
        #     P = P // 2
        block_size = P // blocks
        self.H = H
        self.P = P
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.discretization = discretization
        try:
            k1, k2, k3 = kernel_size[0], kernel_size[1], kernel_size[2]
        except:
            k1, k2, k3 = kernel_size, kernel_size, kernel_size
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        local_P = self.P
        
        if conj_sym:
            block_size = block_size // 2
            local_P = local_P // 2
        
        self.B_tildes = nn.ParameterList()
        for dim, k in enumerate([k1, k2, k3]):
            # Initialize state matrix A using approximation to HiPPO-LegS matrix
            Lambda, _, B, V, _ = make_DPLR_HiPPO(block_size)
            # Lambda, _, B, V, _ = dplr('legs', block_size*2, B_init="constant")
            # Lambda = Lambda[:block_size]
            # V = V[:, :block_size]
            Vc = V.conj().T

            Lambda = (Lambda * torch.ones((blocks, block_size))).flatten()
            V = torch.block_diag(*([V] * blocks))
            Vinv = torch.block_diag(*([Vc] * blocks))

            # Register V and Vinv as buffers
            self.register_buffer(f'V_{dim}', V)
            self.register_buffer(f'Vinv_{dim}', Vinv)

            self.register_parameter(f'Lambda_re_{dim}', nn.Parameter(Lambda.real))
            self.register_parameter(f'Lambda_im_{dim}', nn.Parameter(Lambda.imag))

            # Initialize B
            B_shape = (local_P, H)
            B_init = kaiming_normal_
            B = init_VinvB(B_init, B_shape, Vinv)
            self.register_parameter(f'B_{dim}', nn.Parameter(B))

            # Initialize learnable discretization timescale value
            log_step = init_log_steps((self.P // 2, dt_min, dt_max)) if conj_sym else init_log_steps((self.P, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))

            G_inv_real = torch.empty(k, local_P)
            G_inv_imag = torch.empty(k, local_P)
            nn.init.kaiming_normal_(G_inv_real, mode='fan_out', nonlinearity='linear')
            nn.init.kaiming_normal_(G_inv_imag, mode='fan_out', nonlinearity='linear')
            self.register_parameter(f'G_inv_{dim}', nn.Parameter(G_inv_real + 1j*G_inv_imag))

        if C_init_method in ["trunc_standard_normal"]:
            C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
            C_init = trunc_standard_normal
        elif C_init_method in ["lecun_normal"]:
            C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
            C_init = kaiming_normal_
        elif C_init_method in ["complex_normal"]:
            C_shape = (local_P, local_P, 2) if bidirectional else (local_P, local_P, 2)
            C_init = partial(normal_, std=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                "C_init method {} not implemented".format(C_init))
    
        if bidirectional:
            C1 = init_CV(C_init, C_shape, V[:,:self.P//4])
            C2 = init_CV(C_init, C_shape, V[:,self.P//4:self.P//2])
            C3 = init_CV(C_init, C_shape, V[:,self.P//2:3*self.P//4])
            C4 = init_CV(C_init, C_shape, V[:,3*self.P//4:])
            C1 = C1[..., 0] + 1j * C1[..., 1]
            C2 = C2[..., 0] + 1j * C2[..., 1]
            C3 = C3[..., 0] + 1j * C3[..., 1]
            C4 = C4[..., 0] + 1j * C4[..., 1]
            self.register_parameter(f'C1', nn.Parameter(C1))
            self.register_parameter(f'C2', nn.Parameter(C2))
            self.register_parameter(f'C3', nn.Parameter(C3))
            self.register_parameter(f'C4', nn.Parameter(C4))
        else:
            C = init_CV(C_init, C_shape, V)
            self.register_parameter(f'C', nn.Parameter(C))


        c_s = P
        if conj_sym:
            c_s = c_s // 2
        self.act = nn.GELU()
        self.C_second = nn.Conv3d(c_s, H, 1, 1, padding=0, padding_mode="replicate")
        
        self.norm = nn.InstanceNorm2d(P)

        # Initialize feedthrough (D) matrix
        # D = normal_(torch.empty(1, H*k1*k2, 1, 1), std=1.0)
        # self.register_parameter('D', nn.Parameter(D))
    
        self.blocks = blocks
        self.block_size = block_size
        self.local_P = local_P
        self.step_rescale = step_rescale

    def print_model_parameters(self,):
        print("\nModel Parameters:")
        print("-" * 80)
        total_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            print(f"{name:40} | Shape: {str(param.shape):20} | Parameters: {num_params:10}")
        print("-" * 80)
        print(f"Total parameters: {total_params:,}")

    def get_Lambda(self, dim: int):
        """Get complex Lambda for given dimension, applying clipping if needed."""
        Lambda_re = getattr(self, f'Lambda_re_{dim}')
        Lambda_im = getattr(self, f'Lambda_im_{dim}')
        if self.clip_eigs:
            Lambda_re = torch.clamp(Lambda_re, max=-1e-4)
        return Lambda_re + 1j * Lambda_im
    
    def get_B_tilde(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'B_{dim}')[..., 0] + 1j * getattr(self, f'B_{dim}')[..., 1]

    def get_C_tilde(self):
        """Get complex C_tilde, handling bidirectional case."""
        if self.bidirectional:
            C1 = getattr(self, f'C1')
            C2 = getattr(self, f'C2')
            C3 = getattr(self, f'C3')
            C4 = getattr(self, f'C4')
            return torch.cat([C1, C2, C3, C4], axis=-1)
        else:
            return getattr(self, f'C')[..., 0] + 1j * getattr(self, f'C')[..., 1]
        
    def get_reconstruct_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_layer_{dim}')

    def get_reconstruct_second_layer(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'reconstruct_second_layer_{dim}')
    
    def get_G_inv(self, dim: int):
        """Get complex B_tilde for given dimension."""
        return getattr(self, f'G_inv_{dim}')

    def compute_kernel(self):
        
        kernels = []
        self.B_bars = []
        for dim, k in enumerate([self.k1, self.k2, self.k3]):
            step = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            
            Lambda = self.get_Lambda(dim)
            B_tilde = self.get_B_tilde(dim)
            
            if self.discretization == "zoh":
                Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
            elif self.discretization == "bilinear":
                Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

            # Construct conv kernel for this dimension
            powers_lambda = torch.stack([Lambda_bar**(k-1-i) for i in range(k)]) # k, P
            kernel = powers_lambda[:, None] * B_bar.transpose(0,1) # k, H, P
            kernels.append(kernel)

            self.register_buffer(f'powers_lambda_{dim}', powers_lambda)

        kernel = contract("khp,lhp -> phkl", kernels[0], kernels[1])
        kernel = contract("mhp,phkl -> phklm", kernels[2], kernel)

        # if self.bidirectional:
        #     kernel = torch.cat([
        #         kernel[:self.P//4],
        #         kernel[self.P//4:self.P//2].flip(-1),
        #         kernel[self.P//2:3*self.P//4].flip(-2),
        #         kernel[3*self.P//4:].flip(-1,-2)
        #     ], dim=0)
        
        return kernel

    def forward(self, x):
        """
        since F.conv2d only supports real inputs, we split the complex input into real and imaginary parts.
        conv(W, x) = conv(W.real, x.real) - conv(W.imag, x.imag) 
                    + i(conv(W.real, x.imag)) + i(conv(W.imag, x.real)) (Note that we have no bias)
        instead of doing 4 convolutions, we reduce this to 3 convolutions by doing Gauss trick:
        a = conv(W.real, x.real)
        b = conv(W.imag, x.imag)
        c = conv(W.real + W.imag, x.real + x.imag)
        conv(W, x) = a - b + i(c - a - b)
        """
        conv = partial(F.conv3d,
                       stride=self.stride,  
                       dilation=self.dilation, 
                       groups=self.groups)
        padding_doubled = tuple(x for p in self.padding[::-1] for x in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")
        kernel = self.compute_kernel()

        if not x.is_complex():
            real_out = conv(x, kernel.real)
            imag_out = conv(x, kernel.imag)
        else:
            a = conv(x.real, kernel.real)
            b = conv(x.imag, kernel.imag)
            c = conv(x.real + x.imag, kernel.real + kernel.imag)
            real_out = a - b
            imag_out = c - a - b
        out = real_out + 1j*imag_out

        return out
    
    # def tiled_contract(self, x, g, tile_size=8):
    #     B, P, T, H, W = x.shape
    #     M, K, L, P = g.shape
    #     out = torch.zeros(B, P, T, M, H, K, W, L, device=x.device)
        
    #     for t in range(0, T, tile_size):
    #         for h in range(0, H, tile_size):
    #             for w in range(0, W, tile_size):
    #                 t_slice = slice(t, min(t + tile_size, T))
    #                 h_slice = slice(h, min(h + tile_size, H))
    #                 w_slice = slice(w, min(w + tile_size, W))
                    
    #                 x_tile = x[..., t_slice, h_slice, w_slice]
    #                 try:
    #                     out_tile = contract("bpthw,mklp->bptmhkwl", x_tile, g).real
    #                 except:
    #                     import pdb; pdb.set_trace()
    #                 out[..., t_slice, :, h_slice, :, w_slice, :] = out_tile

    #     return out

    def reconstruct(self, x):
        g_list = []
        eps = 1e-9
        for dim, k in enumerate([self.k1, self.k2, self.k3]):
            powers_lambda = getattr(self, f'powers_lambda_{dim}')
            lambda_L = powers_lambda[0]
            lambdas = torch.exp(self.get_Lambda(dim))[None].repeat((self.P, 1))    
            # ell = (lambda_L[:, None] @ lambda_L[None])
            # lambdas_ = lambdas + lambdas.T
            # ell = ell / (lambdas_)
            # Normalization
            # ell_magnitude = torch.abs(ell)
            # ell = ell / (ell_magnitude)
            # G = self.get_reconstruct_layer(dim)(ell[None])[0]
            # G = self.get_reconstruct_second_layer(dim)(G)[0]
            # G = self.get_reconstruct_layer(dim)(lambda_L * lambdas)
            # G = self.get_reconstruct_layer(dim)[0].linear_real.weight * lambda_L #(lambda_L * lambdas)
            G_inv = self.get_G_inv(dim)
            g = G_inv
            # g = powers_lambda @ G_inv # k, P
            # g_magnitude = torch.abs(g)
            # g = g / (g_magnitude)
            g_list.append(g)
            """
            out = contract("bchw, dc -> bcdhw", x, g)
            if dim == 0:
                out = rearrange(out, "b c d h w -> b c (h d) w").real
            else:
                out = rearrange(out, "b c d h w -> b c h (w d)").imag
            out = F.gelu(out)
            C_second = getattr(self, f'C_second_{dim}')
            out = C_second(getattr(self, f'norm_{dim}')(out)) # do not apply norm
            """
        g = contract("kp, lp -> klp", g_list[1], g_list[2])
        g = contract("mp, klp -> mklp", g_list[0], g)

        # if self.bidirectional:
        #     # g = torch.cat([g[..., :self.P], g[..., self.P:].flip(-1,-2)], dim=-1)
        #     # g = torch.cat([g, g.flip(-1,-2)], dim=-1)
        #     g = torch.cat([
        #         g[..., :self.P//4],
        #         g[..., self.P//4:self.P//2].flip(-1),
        #         g[..., self.P//2:3*self.P//4].flip(-2),
        #         g[..., 3*self.P//4:].flip(-1,-2)
        #     ], dim=-1)
        if self.conj_sym:
            out = 2*contract("bpthw, mklp -> bptmhkwl", x, g).real
        else:
            # # Apply gradient checkpointing to reduce memory usage during training
            # if self.training:
            #     out = torch.utils.checkpoint.checkpoint(
            #         lambda x, g: contract("bpthw, mklp -> bptmhkwl", x, g).real,
            #         x, g
            #     )
            # else:
            #     out = contract("bpthw, mklp -> bptmhkwl", x, g).real
            # x = self.tiled_contract(x, g, tile_size=16)
            x = x.to('cpu')
            g = g.to('cpu')
            out = contract("bpthw, mklp -> bptmhkwl", x, g).real

        if self.padding != (0, 0, 0):
            output_size = ((self.kernel_size[0]-2*self.padding[0])*out.size(2)+2*self.padding[0],
                           (self.kernel_size[1]-2*self.padding[1])*out.size(4)+2*self.padding[1],  
                           (self.kernel_size[2]-2*self.padding[2])*out.size(6)+2*self.padding[2])
            out = rearrange(out, "b p t m h k w l -> b (p m k l) (t h w)").to('cpu')
            div = torch.ones_like(out, device='cpu', dtype=out.dtype)
            # out = F.fold(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            # div = F.fold(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = unfoldNd.foldNd(out, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            div = unfoldNd.foldNd(div, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride)
            out = out[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1], self.padding[2]:output_size[2]-self.padding[2]]
            div = div[:, :, self.padding[0]:output_size[0]-self.padding[0], self.padding[1]:output_size[1]-self.padding[1], self.padding[2]:output_size[2]-self.padding[2]]
            out = (out / div).to('cuda')
        else:
            out = rearrange(out, "b p t m h k w l -> b p (t m) (h k) (w l)")
        # out = out / (out.abs() + eps)
        out = F.gelu(out).to(torch.float32)
        out = self.C_second(out)
        out = 0.5*F.tanh(out) + 0.5
        return out


class BaselineConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv_inv = nn.Conv1d(out_channels, in_channels*kernel_size, 1, 1)
        self.k = kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.C_second = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.stride = stride

    def forward(self, x):
        return self.conv(x)

    def reconstruct(self, x):
        # self.conv.weight: out_channels, in_channels, k
        # kernel_weight = self.conv_inv(self.conv.weight.transpose(-1,0)) # k, in_channels, out_channels
        # kernel_weight = rearrange(kernel_weight, "k i o -> (i k) o")[..., None]
        # x = F.conv1d(x, kernel_weight, stride=1)
        x = self.conv_inv(x)
        x = rearrange(x, "b (c k) l -> b c (l k)", k=self.k)
        x = self.C_second(x)
        return x

class BaselineConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_inv = nn.Conv2d(out_channels, in_channels*kernel_size[0]*kernel_size[1], 1, 1)
        self.k1 = kernel_size[0]
        self.k2 = kernel_size[1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.C_second = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.stride = stride
        self.pixel_shuffle = nn.PixelShuffle(kernel_size[0])

    def forward(self, x):
        return self.conv(x)

    def reconstruct(self, x):
        # self.conv.weight: out_channels, in_channels, k
        # kernel_weight = self.conv_inv(self.conv.weight.transpose(-1,0)) # k, in_channels, out_channels
        # kernel_weight = rearrange(kernel_weight, "k i o -> (i k) o")[..., None]
        # x = F.conv1d(x, kernel_weight, stride=1)
        x = self.conv_inv(x)
        x = self.pixel_shuffle(x)
        x = self.C_second(x)
        return x

def draw_scatter_plot_conv1d():
    import matplotlib.pyplot as plt
    import numpy as np

    # Data for construct()
    x1 = np.array([20.48, 15.06, 13.47, 11.13, 5.25, 3.24, 1.79])
    y1 = np.array([0.008794, 0.005511, 0.006022, 0.006111, 0.005564, 0.004094, 0.001090])

    # Data for construct_2()
    x2 = np.array([15.28, 12.34, 7.64, 3.82, 3.08, 2.31, 1.84])
    y2 = np.array([0.011513, 0.008953, 0.006692, 0.002003, 0.001327, 0.000673, 0.000707])

    # Data for Naiveconv1d
    x3 = np.array([14.16, 11.59, 8.90, 7.82, 4.85, 3.34, 2.38, 2.01])
    y3 = np.array([0.015756, 0.013926, 0.005038, 0.007482, 0.005339, 0.001469, 0.001565, 0.000202])

    plt.figure(figsize=(10, 6))
    
    # Create scatter plots with lines
    plt.scatter(x1, y1, color='blue', label='reconstruct()', zorder=2)
    plt.plot(x1, y1, color='blue', linestyle='-', alpha=0.5, zorder=1)
    
    plt.scatter(x2, y2, color='red', label='reconstruct_2()', zorder=2)
    plt.plot(x2, y2, color='red', linestyle='-', alpha=0.5, zorder=1)

    plt.scatter(x3, y3, color='green', label='Naiveconv1d', zorder=2)
    plt.plot(x3, y3, color='green', linestyle='-', alpha=0.5, zorder=1)

    # Set y-axis to log scale
    plt.yscale('log')

    # Labels and title
    plt.xlabel('Compression Ratio (1/r)')
    plt.ylabel('Reconstruction Loss (log scale)')
    plt.title('Reconstruction Loss vs Compression Ratio')
    
    # Add grid
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add legend
    plt.legend()

    # Save the plot
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_conv1d():
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms


    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # Load and resize image to 32x32
    img = Image.open('sample_image.png')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Shape: (1, 3, 32, 32)
    # Create zigzag indices for 32x32 image
    def create_zigzag_indices(size):
        indices = torch.zeros(size*size, dtype=torch.long)
        for i in range(size*size):
            if (i // size) % 2 == 0:
                indices[i] = i
            else:
                indices[i] = ((i // size) + 1) * size - (i % size) - 1
        return indices
    indices = create_zigzag_indices(32) 
    x = img_tensor.flatten(2,3)[..., indices]
    # optimal kernel size is sqrt(L)
    kernel_size = 64
    conv = SSMConv1d(in_channels=3,
                     out_channels=10,
                     kernel_size=kernel_size,
                     stride=kernel_size,
                     blocks=1,
                     C_init_method="trunc_standard_normal", 
                     discretization="zoh", 
                     dt_min=0.001, 
                     dt_max=0.1, 
                     conj_sym=False, # will half the channels
                     clip_eigs=False, 
                     bidirectional=False, 
                     step_rescale=1.0,
                     dim_preserve=False)
    # conv = BaselineConv1d(in_channels=3,
    #                       out_channels=1,
    #                       kernel_size=kernel_size,
    #                       stride=kernel_size)
    
    # out = conv(x)
    # out2 = conv.reconstruct_2(out)
    # Training loop
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(conv.parameters(), lr=0.04)
    epochs = 1000
    print("Starting training...")
    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        compressed = conv(x)  # Compress
        reconstructed = conv.reconstruct(compressed)  # Reconstruct
        # reconstructed = conv.reconstruct_2(compressed)  # Reconstruct
        
        # Calculate loss between original and reconstructed
        loss = criterion(reconstructed, x)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    print("Training finished!")
    

    # Print final reconstruction error
    with torch.no_grad():
        import time
        start_time = time.time()
        final_compressed = conv(x)
        final_reconstructed = conv.reconstruct(final_compressed)
        # final_reconstructed = conv.reconstruct_2(final_compressed)
        elapsed_time = time.time() - start_time
        print(f"Inference time: {elapsed_time:.4f} seconds")
        final_loss = criterion(final_reconstructed, x)
        # compression_ratio_2 = x.numel() / (final_compressed.numel() + conv.P*kernel_size + conv.P*conv.H)
        # compression_ratio = x.numel() / (final_compressed.numel() + conv.P**2 + conv.P*conv.H)
        compression_ratio_naive = x.numel() / (final_compressed.numel() + conv.in_channels**2 + conv.in_channels*conv.out_channels*conv.k)
        print(f"\nFinal reconstruction loss: {final_loss.item():.6f}")
        # print(f"Compression ratio (construct_2): {compression_ratio_2:.2f}x")
        # print(f"Compression ratio (construct): {compression_ratio:.2f}x")
        print(f"Compression ratio (naive): {compression_ratio_naive:.2f}x")
        # print(f"embedding size:{final_compressed.shape}")
        # print(f"decoder size:{conv.P**2 + conv.P*conv.H}")
        # print(f"real compress ratio:{x.numel() / }")
        draw_scatter_plot_conv1d()


def load_image(path="../test_img/kodim15.png"):
    # Load and resize image to 32x32
    # img = Image.open('sample_image.png')
    img = Image.open(path)
    # img = Image.open('./yubit.JPG')
    transform = transforms.Compose([
        # transforms.Resize((512, 768)),
        # transforms.Resize((1440, 1080)),
        transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0).cuda()  # Shape: (1, 3, 32, 32)
    b, c, h, w = x.shape
    print("input shape:", x.shape)
    return x

def load_video(path="./data/bunny/"):
    import os
    from pathlib import Path
    
    png_files = sorted([f for f in Path(path).glob('*.png')])
    
    if len(png_files) == 0:
        raise ValueError(f"No PNG files found in {path}")
        
    # Load first image to get dimensions
    transform = transforms.Compose([transforms.ToTensor()])
    first_frame = transform(Image.open(png_files[0]))
    c, h, w = first_frame.shape
    
    frames = torch.zeros(len(png_files), c, h, w)
    
    for i, png_file in enumerate(png_files):
        frame = transform(Image.open(png_file))
        frames[i] = frame
        
    # Add batch dimension and move to GPU
    frames = frames.unsqueeze(0).cuda()  # Shape: (1, T, C, H, W)
    
    # Rearrange to (B, C, T, H, W) format
    frames = frames.permute(0, 2, 1, 3, 4)
    print("input shape:", frames.shape)
    return frames

if __name__ == "__main__":
    # x = torch.randn(1, 3, 256, 256)
    # conv = SSMConv2d(in_channels=3,
    #                  out_channels=16, 
    #                  kernel_size=(8,8), 
    #                  stride=(1,1),
    #                  blocks=1,
    #                  C_init_method="trunc_standard_normal", 
    #                  discretization="zoh", 
    #                  dt_min=0.001, 
    #                  dt_max=0.1, 
    #                  conj_sym=False, # will half the channels
    #                  clip_eigs=False, 
    #                  bidirectional=False, 
    #                  step_rescale=1.0,
    #                  dim_preserve=False)
    # out = conv(x)
    # Load and preprocess image
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms


    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # x = load_image()
    x = load_video() #[:,:,:4,:,:]
    b,c,t,h,w = x.shape
    # padding = (0,20,20)
    # stride = (4,40,40)
    # kernel_size = (4,80,80)
    # state_size = 128
    
    # padding = (18,32)
    # stride = (36,64)
    # kernel_size = (72, 128) # padding*2 + stride
    # padding = (90, 160) 
    
    # padding = (36, 64) 
    # stride = (72, 128)
    # kernel_size = (144, 256) 

    padding = (90, 160) 
    stride = (180, 320)
    kernel_size = (360, 640)

    # kernel_size = (5, 5)
    state_size = 64
    # padding = (0,0)
    # stride = (5,5)
    # stride = (720,1280)
    # kernel_size = (720,1280)

    # Create random permutation indices for pixels
    # n_pixels = h * w
    # perm_indices = torch.randperm(n_pixels)
    # inv_perm_indices = torch.zeros_like(perm_indices)
    # inv_perm_indices[perm_indices] = torch.arange(n_pixels)

    # def permute_pixels(img):
    #     # Input shape: (batch, channels, height, width)
    #     b, c, h, w = img.shape
    #     # Reshape to (batch, channels, pixels)
    #     img_flat = img.view(b, c, -1)
    #     # Permute pixels
    #     img_perm = img_flat[..., perm_indices]
    #     # Reshape back
    #     return img_perm.view(b, c, h, w)

    # def inverse_permute_pixels(img):
    #     # Input shape: (batch, channels, height, width) 
    #     b, c, h, w = img.shape
    #     # Reshape to (batch, channels, pixels)
    #     img_flat = img.view(b, c, -1)
    #     # Inverse permute pixels
    #     img_unperm = img_flat[..., inv_perm_indices]
    #     # Reshape back
    #     return img_unperm.view(b, c, h, w)

    # # Apply permutation to input image
    # x_permuted = permute_pixels(x)

    # # Calculate intensity as mean across RGB channels
    # intensity = x.mean(dim=1)  # Shape: (B, H, W)
    
    # # Get sorted indices based on intensity
    # sorted_indices = torch.argsort(intensity.flatten(), dim=-1)
    
    # # Create zigzag pattern indices for target positions
    # h, w = x.shape[2:]
    # zigzag_indices = torch.zeros(h * w, dtype=torch.long)
    # idx = 0
    # for diag in range(h + w - 1):
    #     if diag % 2 == 0:  # Going up
    #         row = min(diag, h-1)
    #         while row >= 0 and diag-row < w:
    #             zigzag_indices[idx] = row * w + (diag-row)
    #             idx += 1
    #             row -= 1
    #     else:  # Going down
    #         col = max(0, diag-(h-1))
    #         while col < w and diag-col >= 0:
    #             zigzag_indices[idx] = (diag-col) * w + col
    #             idx += 1
    #             col += 1
    
    # # Create inverse mapping
    # inv_zigzag_indices = torch.zeros_like(zigzag_indices)
    # inv_zigzag_indices[zigzag_indices] = torch.arange(h * w)
    
    # def sort_and_zigzag(img):
    #     # Input shape: (batch, channels, height, width)
    #     b, c, h, w = img.shape
    #     intensity = img.mean(dim=1)
    #     sorted_indices = torch.argsort(intensity.flatten(), dim=-1)
        
    #     # Rearrange pixels according to intensity in zigzag pattern
    #     img_flat = img.view(b, c, -1)
    #     img_sorted = img_flat[..., sorted_indices]
    #     img_zigzag = torch.zeros_like(img_flat)
    #     img_zigzag[..., zigzag_indices] = img_sorted
    #     return img_zigzag.view(b, c, h, w), sorted_indices
    
    # def inverse_sort_and_zigzag(img, sorted_indices):
    #     # Input shape: (batch, channels, height, width)
    #     b, c, h, w = img.shape
    #     img_flat = img.view(b, c, -1)
        
    #     # First undo zigzag pattern
    #     img_sorted = torch.zeros_like(img_flat)
    #     img_sorted[..., :] = img_flat[..., inv_zigzag_indices]
        
    #     # Then undo intensity sorting
    #     inv_sorted_indices = torch.zeros_like(sorted_indices)
    #     inv_sorted_indices[sorted_indices] = torch.arange(h * w, device=img.device)
    #     img_original = torch.zeros_like(img_flat)
    #     img_original[..., inv_sorted_indices] = img_sorted
        
    #     return img_original.view(b, c, h, w)
    
    # # Apply the transformation
    # x, sorted_indices = sort_and_zigzag(x)
    # x_inv = inverse_sort_and_zigzag(x, sorted_indices)
    # assert torch.allclose(x_inv, x)
    
    # restorer = nn.Sequential(nn.Conv2d(128, 3*32*32, 1, 1), nn.PixelShuffle(32), nn.Sigmoid())
    conv = SSMConv2dv7(in_channels=3,
                     out_channels=state_size,
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding,
                     blocks=8,
                    #  C_init_method="complex_normal", 
                     C_init_method="trunc_standard_normal", 
                    #  C_init_method="lecun_normal", 
                     discretization="zoh", 
                     dt_min=0.001, 
                     dt_max=0.1, 
                     conj_sym=False, # will half the channels
                     clip_eigs=False, 
                     bidirectional=False, 
                     step_rescale=1.0,
                     ).cuda()
    # conv = SSMConv3d(in_channels=3,
    #                  out_channels=state_size,
    #                  kernel_size=kernel_size, 
    #                  stride=stride,
    #                  padding=padding,
    #                  blocks=8,
    #                  C_init_method="trunc_standard_normal", 
    #                  discretization="zoh", 
    #                  dt_min=0.001, 
    #                  dt_max=0.1, 
    #                  conj_sym=False, # will half the channels
    #                  clip_eigs=False, 
    #                  bidirectional=False, 
    #                  step_rescale=1.0,
    #                  ).cuda()
    naive_conv = BaselineConv2d(in_channels=3,
                          out_channels=state_size,
                          kernel_size=kernel_size, 
                          padding=padding,
                          stride=stride).cuda()
    
    # out = conv(x)
    # out2 = conv.reconstruct(out)

    # Training loop    
    criterion = nn.MSELoss()
    # Apply different learning rates to encoder parameters
    ssm_params = []
    non_ssm_params = []

    # Group encoder parameters
    num_params = 0
    for name, param in conv.named_parameters():
        if any(x in name for x in ['Lambda_re', 'Lambda_im', 'B_', 'log_step']):
            ssm_params.append(param)
        else:
            non_ssm_params.append(param)
        print(f"name: {name}, shape: {param.shape}, numel: {param.numel()}")
        num_params += param.numel()
    print(f"num_params: {num_params}")

    C = 3
    P = conv.P
    # H = 720
    # W = 1280
    print(f"Conv2d params: {C*P*conv.k1*conv.k2 + P}, in millions: {(C*P*conv.k1*conv.k2 + P) / 1e6}M")
    # print(f"SSMConv2d params: {4*P + 4*C*P + 2*P + conv.pos_embed.numel()}, in millions: {(4*P + 4*C*P + 2*P + conv.pos_embed.numel()) / 1e6}M") if conv.pos_embed is not None else \
    #     print(f"SSMConv2d params: {4*P + 4*C*P + 2*P}, in millions: {(4*P + 4*C*P + 2*P) / 1e6}M")
    print(f"SSMConv2d params: {4*P + 4*C*P + 2*P + conv.C.numel() + conv.C_bias.numel()}, in millions: {(4*P + 4*C*P + 2*P + conv.C.numel() + conv.C_bias.numel()) / 1e6}M")
    
    #  + 4*C*conv.k1*conv.k2
    decoding_params = P*(conv.k1 + conv.k2) + P*C
    print(f"decoding params: {decoding_params}")
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': ssm_params, 'weight_decay': 0.0, 'lr': 0.003},  # Lower learning rate for ssm_params
        {'params': non_ssm_params, 'weight_decay': 0.0, 'lr': 0.03}  # Default learning rate for non_ssm_params
    ]
    # from lion_pytorch import Lion
    # optimizer = Lion(param_groups)
    optimizer = torch.optim.AdamW(param_groups)
    # optimizer = torch.optim.AdamW(conv.parameters(), lr=0.01)
    epochs = 300
    print("Starting training...")
    # Create separate schedulers for each parameter group
    scheduler_ssm = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    x = rearrange(x, 'b c t h w -> (b t) c h w')[:2]
    t,c,h,w = x.shape

    # from dataset import VideoDataSet
    # train_dataset = VideoDataSet(path="./data/bunny/", laplacian=False)
    # full_dataset = VideoDataSet(path="./data/bunny/", laplacian=False)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
    #         num_workers=1, pin_memory=True, sampler=None, drop_last=False, worker_init_fn=None)
    # val_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=False,
    #         num_workers=1, pin_memory=True, sampler=None, drop_last=False, worker_init_fn=None)
    # device = next(conv.parameters()).device

    # for epoch in range(epochs):
    #     # Zero gradients
    #     optimizer.zero_grad()
    #     for i, vid in enumerate(train_dataloader):
    #         img = vid['img'].to(device)
    #         # Forward pass
    #         compressed = conv(img)
    #         # compressed = naive_conv(vid)
    #         reconstructed = conv.reconstruct(compressed)  # Reconstruct
    #         # reconstructed = conv.reconstruct_2(compressed)  # Reconstruct
    #         # reconstructed = restorer(compressed)
    #         # Calculate loss between original and reconstructed
    #         loss = criterion(reconstructed, img)
            
    #         # Backward pass and optimize
    #         loss.backward()
    #         optimizer.step()
    #         # scaler.scale(loss).backward()
    #         # scaler.step(optimizer)
    #         # scaler.update()
            
    #         # Step the scheduler
    #         scheduler_ssm.step()
            
    #     if (epoch + 1) % 1 == 0:
    #         current_lrs = scheduler_ssm.get_last_lr()  # Get all learning rates
    #         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LRs: SSM params = {current_lrs[0]:.6f}, Non-SSM params = {current_lrs[1]:.6f}')



    for epoch in range(epochs):
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        compressed = conv(x)
        # compressed = naive_conv(x)
        reconstructed = conv.reconstruct(compressed)  # Reconstruct
        # reconstructed = conv.reconstruct_2(compressed)  # Reconstruct
        # reconstructed = restorer(compressed)
        # Calculate loss between original and reconstructed
        loss = criterion(reconstructed, x)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        # Step the scheduler
        scheduler_ssm.step()
            
        if (epoch + 1) % 10 == 0:
            current_lrs = scheduler_ssm.get_last_lr()  # Get all learning rates
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, LRs: SSM params = {current_lrs[0]:.6f}, Non-SSM params = {current_lrs[1]:.6f}')

    print("Training finished!")

    # Print final reconstruction error
    with torch.no_grad():
        import time
        start_time = time.time()
        reconstructed_list = []
        final_compressed = conv(x)
        # final_compressed = naive_conv(vid)
        final_reconstructed = conv.reconstruct(final_compressed)
        # final_reconstructed = conv.reconstruct_2(final_compressed)
        # final_reconstructed = restorer(final_compressed)
        reconstructed_list.append(final_reconstructed)
        # for i, vid in enumerate(val_dataloader):
        #     img = vid['img'].to(device)
        #     final_compressed = conv(img)
        #     # final_compressed = naive_conv(vid)
        #     final_reconstructed = conv.reconstruct(final_compressed)
        #     # final_reconstructed = conv.reconstruct_2(final_compressed)
        #     # final_reconstructed = restorer(final_compressed)
        #     reconstructed_list.append(final_reconstructed)
        elapsed_time = time.time() - start_time
        reconstructed_video = torch.stack(reconstructed_list)
        print(f"Inference time: {elapsed_time:.4f} seconds")
        final_loss = criterion(reconstructed_video, x)
        # Calculate PSNR
        mse = torch.mean((reconstructed_video - x) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        print(f"PSNR: {psnr:.2f} dB")
        # compression_ratio_2 = x.numel() / (final_compressed.numel() + conv.P*kernel_size + conv.P*conv.H)
        # compression_ratio = x.numel() / (final_compressed.numel() + conv.P**2 + conv.P*conv.H)
        # compression_ratio_naive = x.numel() / (final_compressed.numel() + conv.in_channels**2 + conv.in_channels*conv.out_channels*conv.k)
        # compression_ratio_naive = x.numel() / (final_compressed.numel() + conv.in_channels**2 + conv.in_channels*conv.out_channels*conv.k)
        # compression_ratio = x.numel() / (final_compressed.numel() + conv.P**2 + conv.P*(conv.H+1))
        compression_ratio = x.numel() / (final_compressed.numel() + decoding_params)
        bpp = (final_compressed.numel() + decoding_params) * 32 / (x.numel())
        # compression_ratio_naive = x.numel() / (final_compressed.numel() + conv.conv.in_channels*conv.conv.out_channels*conv.k1*conv.k2 + 1)
        # print(f"\nFinal reconstruction loss: {final_loss.item():.6f}")
        # print(f"Compression ratio (construct_2): {compression_ratio_2:.2f}x")
        print(f"Compression ratio (construct): {compression_ratio:.2f}x")
        print(f"bpp: {bpp:.2f} bpp")
        # print(f"Compression ratio (naive): {compression_ratio_naive:.2f}x")
        print(f"embedding size:{final_compressed.shape}")
        # print(f"decoder size:{conv.P**2 + conv.P*conv.H}")
        # print(f"real compress ratio:{x.numel() / }")
        # draw_scatter_plot_conv1d() 
        # Convert reconstructed tensor back to image and save
        # Process reconstructed image
        recon_img = reconstructed_video
        recon_img = recon_img.view(t,3,h,w)  # Reshape back to image dimensions
        recon_img = torch.clamp(recon_img, 0, 1)  # Clamp values between 0 and 
        # recon_img = final_reconstructed.squeeze(0)  # Remove batch dimension
        # recon_img = recon_img.view(3, h, w)  # Reshape back to image dimensions
        # recon_img = torch.clamp(recon_img, 0, 1)  # Clamp values between 0 and 1
        
        # Process original image
        orig_img = x
        orig_img = orig_img.view(t, c, h, w)  # Reshape back to image dimensions
        # orig_img = x.squeeze(0)  # Remove batch dimension
        # orig_img = orig_img.view(c, h, w)  # Reshape back to image dimensions
        
        # Convert frames to PIL images and create GIF
        transform = transforms.ToPILImage()
        frames = []
        
        for frame_idx, (orig_frame, recon_frame) in enumerate(zip(orig_img, recon_img)):
            # Get current frame from original and reconstructed
            # orig_frame = orig_img[:, frame_idx, :, :]
            # recon_frame = recon_img[:, frame_idx, :, :]
            
            # Convert to PIL images
            orig_frame = transform(orig_frame)
            recon_frame = transform(recon_frame)
            
            # Create side by side comparison
            combined_frame = Image.new('RGB', (2*w, h))
            combined_frame.paste(orig_frame, (0, 0))  # Original on left
            combined_frame.paste(recon_frame, (w, 0))  # Reconstructed on right
            
            frames.append(combined_frame)
            
            # Also save individual frame
            combined_frame.save(f'comparison_frame_{frame_idx}.png')
        
        # Save as animated GIF
        frames[0].save(
            'comparison.gif',
            save_all=True,
            append_images=frames[1:],
            duration=100, # 100ms per frame
            loop=0
        )

        # # Convert both to PIL images
        # transform = transforms.ToPILImage()
        # recon_img = transform(recon_img)
        # orig_img = transform(orig_img)
        
        # # Create a new image combining both side by side
        # combined_img = Image.new('RGB', (2*w, h))  # Width is doubled to fit both images
        # combined_img.paste(orig_img, (0, 0))  # Original on left
        # combined_img.paste(recon_img, (w, 0))  # Reconstructed on right
        
        # # Save combined image
        # combined_img.save('comparison.png')