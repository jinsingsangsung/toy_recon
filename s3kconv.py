import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
import math
from ssm_init import init_CV, init_VinvB, init_VinvB_takeB, init_log_steps, trunc_standard_normal, make_DPLR_HiPPO, make_DPLR_HiPPO_real, init_CV_real
# from s3k_cuda.s3k_cuda import S3KCudaFunction

_r2c = torch.view_as_complex
_c2r = torch.view_as_real

contract = torch.einsum

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

class DyT(nn.Module):
    def __init__(self, C, init_alpha, data_format="channels_last"):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))
        self.data_format = data_format
    def forward(self, x):
        if self.data_format == "channels_first":
            x = x.transpose(-1, 1)
        x = F.tanh(self.alpha * x)
        x = self.gamma * x + self.beta
        if self.data_format == "channels_first":
            x = x.transpose(-1, 1)
        return x


class S3KConv2dv2(nn.Module):
    """
    S3K conv2d with adaptive B
    different projection scheme for B and delta
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 state_size: int = 16,
                 block_size: int = 1,
                 d_in = 16,
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_rank: int = 1,
                 step_rescale: float = 1.0,
                 bidirectional: bool = False,
                 add_conv_kernel: bool = False,
                 use_norm: bool = False,  # Add flag to control normalization,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        if isinstance(kernel_size, int):
            k1, k2 = kernel_size, kernel_size
        else:
            k1, k2 = kernel_size[0], kernel_size[1]

        if isinstance(stride, int):
            s1, s2 = stride, stride
        else:
            s1, s2 = stride[0], stride[1]
        
        if isinstance(padding, int):
            padding = (padding, padding)

        self.padding = padding
        self.k1 = k1
        self.k2 = k2
        self.s1 = s1
        self.s2 = s2

        H = in_channels
        P = out_channels
        N = state_size

        self.P = P

        if bidirectional:
            assert P % 4 == 0, "P must be divisible by 4 for bidirectional S3KConv2d"
            P = P // 4
        
        assert N % block_size == 0, "N must be divisible by block_size for S3KConv2d"
        blocks = N // block_size

        self.H = H
        self.N = N
        self.d_in = d_in
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        if dt_rank == 0:
            dt_rank = N
        self.dt_rank = dt_rank
        self.step_rescale = step_rescale

        self.in_proj = nn.Conv2d(H, d_in + self.N, 1, 1, bias=False)

        for dim, k in enumerate([k1, k2]):
            A = torch.arange(1, N+1)
            # A = repeat(A, "N -> C N", C=d_in)
            A_log = nn.Parameter(torch.log(A))
            self.register_parameter(f'A_log_{dim}', A_log)
            x_proj = nn.Linear([k2, k1][dim], N, bias=False)
            log_step = init_log_steps((N, dt_min, dt_max))
            self.register_parameter(f'log_step_{dim}', nn.Parameter(log_step))
            
            setattr(self, f'x_proj_{dim}', x_proj)

        self.use_norm = use_norm
        if use_norm:
            self.kernel_norm = DyT(self.P, 0.5, data_format="channels_first")
        self.C = nn.Conv2d(N, out_channels, 1, 1, bias=False)

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

    def compute_B(self, x, dim: int):
        """
        compute B and step based on x
        input x: B, C, H2, W2, K1, K2
        """
        x_proj = getattr(self, f'x_proj_{dim}')
        if dim == 1:
            x = rearrange(x, "b c h2 w2 k1 k2 -> b c h2 w2 k2 k1")
        B = x_proj(x)
        # B: b c h2 w2 k N
        return B

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, H2, W2, K1, K2
        output kernel: B, C, H2, W2, K1, K2
        """
        kernels = []
        for dim, k in enumerate([self.k1, self.k2]):
            B = self.compute_B(x, dim)
            A_log = self.get_A_log(dim)
            A = -torch.exp(A_log.float())
            # A = -torch.exp(A_log)
            b, c, h2, w2, k, _ = B.shape
            delta = self.step_rescale * torch.exp(getattr(self, f'log_step_{dim}'))
            delta = repeat(delta, "p -> b c h2 w2 k p", b=b, c=c, h2=h2, w2=w2, k=k)
            # discretization
            deltaA, deltaB = discretize_zoh_v2(A, B, delta)
            kernel = deltaA*deltaB
            kernels.append(kernel)

        kernel = contract("bchwkn,bchwln -> bchwkln", kernels[0], kernels[1])
        # kernel = self.kernel_norm(kernel)

        # if self.bidirectional:
        #     kernel = torch.cat([
        #         kernel,
        #         kernel.flip(-2),
        #         kernel.flip(-3),
        #         kernel.flip(-2,-3)
        #     ], dim=1)

        return kernel
    
        # B, C, delta = self.compute_B_C_delta(x, 0)  # First dimension
        # B2, C2, delta2 = self.compute_B_C_delta(x, 1)  # Second dimension
        
        # return S3KCudaFunction.apply(
        #     x, 
        #     self.A_log_0,
        #     self.A_log_1,
        #     torch.stack([B, B2], dim=0),
        #     torch.stack([C, C2], dim=0),
        #     torch.stack([delta, delta2], dim=0)
        # )

    def forward(self, x):
        """
        we compute an input-adaptive kernel and then compute the output
        """
        padding_doubled = tuple(a for p in self.padding[::-1] for a in (p, p))
        x = F.pad(x, padding_doubled, mode="replicate")
        x = self.in_proj(x)
        x, res = x.split([self.d_in, self.N], dim=1)

        x = x.unfold(dimension=2, size=self.k1, step=self.s1)
        x = x.unfold(dimension=3, size=self.k2, step=self.s2)
        # B, C, H2, W2, K1, K2 = x.shape
        res = res.unfold(dimension=2, size=self.k1, step=self.s1)
        res = res.unfold(dimension=3, size=self.k2, step=self.s2)
        # B, P, H2, W2, K1, K2 = res.shape

        kernel = self.compute_kernel(x)
        # B, C, H2, W2, K1, K2, N = kernel.shape

        # if self.add_conv_kernel:
        #     conv_kernel_bias = getattr(self, f'conv_kernel_bias')
        #     if self.bidirectional:
        #         conv_kernel_bias = conv_kernel_bias.repeat(4)
        #     out = contract("bchwkl, bchwklp -> bphwkl", x, kernel) + conv_kernel_bias[None, :, None, None, None, None]
        #     out = out * F.silu(res)
            # out = contract("bchwkl, bchwklp -> bphwkl", x.cfloat(), kernel).float() * F.silu(res)
        if self.bidirectional:
            x = x.repeat(1, 4, 1, 1, 1, 1)
            out = contract("bchwkl,bchwklp->bphw", x, kernel)
            out_flip2 = contract("bchwkl,bchwklp->bphw", x, kernel.flip(-2))
            out_flip3 = contract("bchwkl,bchwklp->bphw", x, kernel.flip(-3))
            out_flip23 = contract("bchwkl,bchwklp->bphw", x, kernel.flip(-2,-3))
            out = torch.cat([out, out_flip2, out_flip3, out_flip23], dim=1) * F.silu(res)
        else:
            out = contract("bchwkl, bchwklp -> bphwkl", x, kernel) * F.silu(res)
        out = reduce(out, "b p h w k l -> b p h w", "sum")
        # if self.use_norm:
        #     out = self.kernel_norm(out)
        out = self.C(out)
        return out
    
class S3KConv1dv2(nn.Module):
    """
    S3K conv1d with adaptive B
    different projection scheme for B and delta
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 state_size: int = 16,
                 block_size: int = 1,
                 d_in = 8,
                 discretization: str = "zoh",
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dt_rank: int = 1,
                 step_rescale: float = 1.0,
                 bidirectional: bool = False,
                 add_conv_kernel: bool = False,
                 use_norm: bool = False,  # Add flag to control normalization,
    ):
        super().__init__()

        # convolution arguments
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size

        k = kernel_size
        s = stride
        p = padding
        d = dilation

        self.padding = p
        self.k = k
        self.s = s

        H = in_channels
        P = out_channels
        N = state_size

        self.P = P

        if bidirectional:
            assert P % 2 == 0, "P must be divisible by 2 for bidirectional S3KConv1d"
            P = P // 2
        
        assert N % block_size == 0, "N must be divisible by block_size for S3KConv1d"
        blocks = N // block_size

        self.H = H
        self.N = N
        self.d_in = d_in
        self.bidirectional = bidirectional
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        if dt_rank == 0:
            dt_rank = N
        self.dt_rank = dt_rank
        self.step_rescale = step_rescale

        self.in_proj = nn.Conv1d(H, d_in + self.N, 1, 1, bias=False)

        A = torch.arange(1, N+1)
        # A = repeat(A, "N -> C N", C=d_in)
        A_log = nn.Parameter(torch.log(A))
        self.register_parameter(f'A_log', A_log)
        x_proj = nn.Linear(k, N, bias=False)
        log_step = init_log_steps((N, dt_min, dt_max))
        self.register_parameter(f'log_step', nn.Parameter(log_step))
        self.x_proj = x_proj

        self.use_norm = use_norm
        if use_norm:
            self.kernel_norm = DyT(self.P, 0.5, data_format="channels_first")
        if self.bidirectional:
            self.C = nn.Conv1d(N*2, out_channels, 1, 1, bias=False)
        else:
            self.C = nn.Conv1d(N, out_channels, 1, 1, bias=False)

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
    
    def get_A_log(self):
        """Get A_log for given dimension."""
        return getattr(self, f'A_log')

    def compute_B(self, x):
        """
        compute B and step based on x
        input x: B, C, L2, K
        """
        # c = x.size(1)
        x_proj = self.x_proj
        B = x_proj(x)
        B = repeat(B, "b c l2 n -> b c l2 k n", k=self.k)
        # B = rearrange(B, "b c l2 n -> b l2 k n", n=self.N)
        # B = repeat(B, "b l2 k n -> b c l2 k n", c=c)
        # B = rearrange(B, "b (c n) l2 k -> b c l2 k n", n=self.N)
        return B

    def compute_kernel(self, x):
        """
        compute dynamic kernel based on x
        input x: B, C, L2, K
        output kernel: B, C, L2, K, N
        """
        B = self.compute_B(x)
        A_log = self.get_A_log()
        A = -torch.exp(A_log.float())
        b, c, l2, k, n = B.shape
        # b, n, l2, k = B.shape
        # B = repeat(B, "b n l2 k -> b c n l2 k", c=c)
        # B = rearrange(B, "b c n l2 k -> b c l2 k n")
        delta = self.step_rescale * torch.exp(self.log_step)
        delta = repeat(delta, "n -> b c l2 k n", b=b, c=c, l2=l2, k=k)
        # discretization
        deltaA, deltaB = discretize_zoh_v2(A, B, delta)
        kernel = deltaA*deltaB
        # kernel = rearrange(kernel, "b l k c n -> b c l k n")

        return kernel
    
        # B, C, delta = self.compute_B_C_delta(x, 0)  # First dimension
        # B2, C2, delta2 = self.compute_B_C_delta(x, 1)  # Second dimension
        
        # return S3KCudaFunction.apply(
        #     x, 
        #     self.A_log_0,
        #     self.A_log_1,
        #     torch.stack([B, B2], dim=0),
        #     torch.stack([C, C2], dim=0),
        #     torch.stack([delta, delta2], dim=0)
        # )

    def forward(self, x):
        """
        we compute an input-adaptive kernel and then compute the output
        """
        x = rearrange(x, "b l c -> b c l")
        padding_doubled = (self.padding, self.padding)
        x = F.pad(x, padding_doubled, mode="replicate")
        x = self.in_proj(x)
        x, res = x.split([self.d_in, self.N], dim=1)

        x = x.unfold(dimension=2, size=self.k, step=self.s)
        # B, C, L2, K = x.shape
        res = res.unfold(dimension=2, size=self.k, step=self.s)
        # B, N, L2, K = res.shape

        kernel = self.compute_kernel(x)
        # B, C, L2, K, N = kernel.shape

        # if self.add_conv_kernel:
        #     conv_kernel_bias = getattr(self, f'conv_kernel_bias')
        #     if self.bidirectional:
        #         conv_kernel_bias = conv_kernel_bias.repeat(4)
        #     out = contract("bchwkl, bchwklp -> bphwkl", x, kernel) + conv_kernel_bias[None, :, None, None, None, None]
        #     out = out * F.silu(res)
            # out = contract("bchwkl, bchwklp -> bphwkl", x.cfloat(), kernel).float() * F.silu(res)
        if self.bidirectional:
            res = res.repeat(1, 2, 1, 1)
            out = contract("bclk,bclkp->bplk", x, kernel)
            out_flip2 = contract("bclk,bclkp->bplk", x, kernel.flip(-2))
            out = torch.cat([out, out_flip2], dim=1) # * F.silu(res)
        else:
            out = contract("bclk, bclkp -> bplk", x, kernel) #  * F.silu(res)
        out = reduce(out, "b p l k -> b p l", "sum")
        # if self.use_norm:
        #     out = self.kernel_norm(out)
        out = self.C(out)
        out = rearrange(out, "b c l -> b l c")
        return out
