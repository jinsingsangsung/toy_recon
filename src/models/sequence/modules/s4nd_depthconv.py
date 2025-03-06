"""Implementation of S4ND module (https://arxiv.org/abs/2210.06583)."""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import numpy as np
from src.models.sequence import SequenceModule
from src.models.sequence.kernels import registry as kernel_registry
from src.models.nn import LinearActivation, Activation, DropoutNd
import src.utils.train
import src.utils as utils

log = src.utils.train.get_logger(__name__)

contract = torch.einsum


def multiple_axis_slice(x, L):
    """
    x: (..., L1, L2, .., Lk)
    L: list of length k [l1, l2, .., lk]
    returns: x[..., :l1, :l2, .., :lk]
    """
    # TODO I don't see a way to do this programmatically in Pytorch without sacrificing speed so...
    assert len(L) > 0
    if len(L) == 1:
        return x[..., :L[0]]
    elif len(L) == 2:
        return x[..., :L[0], :L[1]]
    elif len(L) == 3:
        return x[..., :L[0], :L[1], :L[2]]
    elif len(L) == 4:
        return x[..., :L[0], :L[1], :L[2], :L[3]]
    else: raise NotImplementedError("lol")


class S4ND(SequenceModule):
    requires_length = True

    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=None, # Maximum length of sequence (list or tuple). None for unbounded
        dim=2, # Dimension of data, e.g. 2 for images and 3 for video
        out_channels=None, # Do depthwise-separable or not
        channels=1, # maps 1-dim to C-dim
        bidirectional=False,
        # Arguments for FF
        activation='gelu', # activation in between SS and FF
        ln=False, # Extra normalization
        final_act=None, # activation after FF
        initializer=None, # initializer on FF
        weight_norm=False, # weight normalization on FF
        hyper_act=None, # Use a "hypernetwork" multiplication
        dropout=0.0, tie_dropout=False,
        transposed=True, # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        trank=1, # tensor rank of C projection tensor
        linear=True,
        return_state=True,
        contract_version=0,
        # SSM Kernel arguments
        kernel=None,  # New option
        mode='dplr',  # Old option
        base_rank=4,  # Changed from fixed rank to base_rank
        min_rank=1,
        max_rank=None,  # Will default to base_rank
        rank_dropout=0.0,  # Randomly drop ranks during training
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4ND (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.linear = linear
        self.return_state = return_state
        self.contract_version = contract_version
        self.out_channels = out_channels
        self.verbose = verbose
        self.kernel_args = kernel_args

        self.D = nn.Parameter(torch.randn(self.channels, self.h)) # TODO if self.out_channels

        self.trank = trank

        if self.out_channels is not None:
            channels *= self.out_channels

            # # Swap channels and out_channels
            # # self.out_channels = channels
            # self.h = self.out_channels
            # # channels *= out_channels
            # self.in_channels = d_model
            # channels *= d_model
            assert self.linear # TODO change name of linear_output

        channels *= self.trank

        if self.bidirectional:
            channels *= 2

        # Check dimensions and kernel sizes
        if dim is None:
            assert utils.is_list(l_max)

        # assert l_max is not None # TODO implement auto-sizing functionality for the kernel
        if l_max is None:
            self.l_max = [None] * dim
        elif isinstance(l_max, int):
            self.l_max = [l_max] * dim
        else:
            assert l_max is None or utils.is_list(l_max)
            self.l_max = l_max

        # SSM Kernel
        if kernel is None and mode is not None: kernel = mode
        self._kernel_channels = channels
        self.kernel = nn.ModuleList([
            # SSKernel(self.h, N=self.n, L=L, channels=channels, verbose=verbose, **kernel_args)
            kernel_registry[kernel](d_model=self.h, d_state=self.n, l_max=L, channels=channels, verbose=verbose, **kernel_args)
            for L in self.l_max
        ])

        if not self.linear:

            self.activation = Activation(activation)
            dropout_fn = partial(DropoutNd, transposed=self.transposed) if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()


            # position-wise output transform to mix features
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )

        ## To handle some operations with unspecified number of dims, we're going to define the einsum/einops contractions programmatically

        # Outer product function for the convolution kernel taking arbitary number of dims
        contract_str = ', '.join([f'... {chr(i+97)}' for i in range(len(self.l_max))]) \
            + ' -> ... ' \
            + ' '.join([f'{chr(i+97)}' for i in range(len(self.l_max))])
        # self.nd_outer = oe.contract_expression(
        #     contract_str,
        #     *[(channels*self.trank, self.h, 2*l) for l in l_max]
        # )

        # Slice first half of each length dimension after the iFFT
        # e.g. in 2D the forward pass wants to call:
        #   y = rearrange(y, '... (f0 l1) (f1 l1) -> (f0 f1) ... (l0 l1)', f0=2, f1=2)
        #   y = y[0]
        # self.nd_slice = '... ' + ' '.join([f"(f{i} l{i})" for i in range(len(l_max))]) + ' -> (' + ' '.join([f"f{i}" for i in range(len(l_max))]) + ') ... (' + ' '.join([f"l{i}" for i in range(len(l_max))]) + ')'

        # unflattened L dim by removing last '()'
        # self.nd_slice = '... ' + ' '.join([f"(f{i} l{i})" for i in range(len(l_max))]) + ' -> (' + ' '.join([f"f{i}" for i in range(len(l_max))]) + ') ... ' + ' '.join([f"l{i}" for i in range(len(l_max))])
        # self.nd_slice_args = { f"f{i}": 2 for i in range(len(l_max)) }

        # Initialize projection matrices for low-rank approximation
        self.base_rank = base_rank
        self.min_rank = min_rank
        self.max_rank = max_rank or base_rank
        self.rank_dropout = rank_dropout
        
        # Compute adaptive rank based on sequence length
        max_len = max(self.l_max) if isinstance(self.l_max, list) else self.l_max
        self.adaptive_rank = min(
            self.max_rank,
            max(self.min_rank, int(np.log2(max_len)))
        )
        
        # Initialize projections with adaptive rank
        self.proj1 = nn.Parameter(
            torch.randn(channels, max_len, self.adaptive_rank) / np.sqrt(max_len)
        )
        self.proj2 = nn.Parameter(
            torch.randn(channels, max_len, self.adaptive_rank) / np.sqrt(max_len)
        )
        
        # Learnable singular values with decay structure
        self.singular_values = nn.Parameter(
            torch.exp(-torch.arange(self.adaptive_rank).float())
        )

    def _reinit(self, dt_min=None, dt_max=None, normalize=False, **kwargs):
        """ Sets time kernel to custom value """
        assert len(self.l_max) == 3
        L = self.l_max[-3]
        # init = init or 'fourier'
        dt_min = dt_min or 2./L
        dt_max = dt_max or 2./L
        print(f"S4ND reinit args: {dt_min=} {dt_max=}", kwargs)
        kernel_args = {
            **self.kernel_args, **{
                'H': self.h,
                'N': self.n,
                'L': L,
                # 'init': init,
                'dt_min': dt_min,
                'dt_max': dt_max,
                # 'deterministic': True,
                'channels': self._kernel_channels,
                **kwargs,
            }
        }
        time_kernel = SSKernel(**kernel_args)
        if normalize:
            with torch.no_grad():
                time_kernel.kernel.C /= (0.5 * time_kernel.kernel.log_dt.exp()[:, None, None])
        self.kernel[-3] = time_kernel


    def forward(self, u, rate=1.0, state=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """

        half_precision = False

        # fft can only handle float32
        if u.dtype == torch.float16:
            half_precision = True
            u = u.to(torch.float32)

        assert state is None, f"state not currently supported in S4ND"

        # ensure shape is B, C, L (L can be multi-axis)
        if not self.transposed:
            u = rearrange(u, "b ... h -> b h ...")

        L_input = u.shape[2:]

        L_kernel = [
            l_i if l_k is None else min(l_i, round(l_k / rate)) for l_i, l_k in zip(L_input, self.l_max)
        ]

        # Compute SS Kernel
        # 1 kernel for each axis in L
        k = [kernel(L=l, rate=rate)[0] for kernel, l in zip(self.kernel, L_kernel)]
        
        if self.bidirectional: # halves channels
            k = [torch.chunk(_k, 2, dim=-3) for _k in k] # (C H L)
            k = [
                F.pad(k0, (0, l)) + F.pad(k1.flip(-1), (l, 0))
                # for l, (k0, k1) in zip(L_kernel, k) # TODO bug??
                for l, (k0, k1) in zip(L_input, k)
            ]

        # fft can only handle float32
        if u.dtype == torch.float16:
            half_precision = True
            # cast to fp32
            k.dtype = torch.float32

        L_padded = [l_input + l_kernel for l_input, l_kernel in zip(L_input, L_kernel)]
        u_f = torch.fft.rfftn(u, s=tuple([l for l in L_padded])) # (B H L)
        k_f = [torch.fft.fft(_k, n=l) for _k, l in zip(k[:-1], L_padded[:-1])] + [torch.fft.rfft(k[-1], n=L_padded[-1])] # (C H L)
        import pdb; pdb.set_trace()
        # Take outer products
        ##### Original S4ND implementation #####################################################################################################################################################
        if self.contract_version == 0: # TODO set this automatically if l_max is provided
            # self.kernel[0]._check() 
            k_f = contract('... c h m, ... c h n -> ... c h m n', k_f[0], k_f[1]) # (H L1 L2) # 2D case of next line
            # k_f = self.nd_outer(*k_f)
            # sum over tensor rank
            k_f = reduce(k_f, '(r c) h ... -> c h ...', 'sum', r=self.trank) / self.trank # reduce_mean not available for complex... # TODO does it matter if (r c) or (c r)?
            y_f = contract('bh...,ch...->bch...', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)

        else:
            contract_str_l = [f'{chr(i+100)}' for i in range(len(L_input))]
            contract_str = 'b ... ' + ' '.join(contract_str_l) + ', ' \
                + ', '.join(['... ' + l for l in contract_str_l]) \
                + ' -> b ... ' \
                + ' '.join(contract_str_l)
            y_f = contract(contract_str, u_f, *k_f)
            k_f = reduce(y_f, 'b (r c) h ... -> b c h ...', 'sum', r=self.trank) / self.trank # reduce_mean not available for complex... # TODO does it matter if (r c) or (c r)?

        # # Contract over channels if not depthwise separable
        # if self.out_channels is not None:
        #     y_f = reduce(y_f, 'b (i c) h ... -> b c i ...', 'sum', i=self.out_channels) # TODO normalization might not be right

        
        # y = torch.fft.irfftn(y_f, s=tuple([l for l in L_padded]))     
        ############################################################################################################################################################

        # Get frequency representations
        u_f = torch.fft.rfft2(u, s=tuple(L_padded))
        k_f1, k_f2 = k_f[0], k_f[1]  # Kernels for each dimension
        
        import pdb; pdb.set_trace()

        # Low-rank approximation
        # Project each kernel to lower dimension
        k_f1_low = contract('...chm,cmr->...chr', k_f1, self.proj1)
        k_f2_low = contract('...chn,cnr->...chr', k_f2, self.proj2)
        
        # Scale by learnable singular values
        k_f1_low = k_f1_low * self.singular_values[None, None, None, :]
        
        # Reconstruct 2D kernel through outer product in rank space
        k_f = contract('...chr,...chr->...chmn', k_f1_low, k_f2_low)
        
        # Apply reconstructed kernel
        y_f = contract('bh...,ch...->bch...', u_f, k_f)
        
        # Transform back to time domain
        y = torch.fft.irfft2(y_f, s=tuple(L_padded))        

        # need to cast back to half if used
        if half_precision:
            y = y.to(torch.float16)

        # y = y[..., :self.l_max[0], :self.l_max[1]] # 2D case of next line
        # y = rearrange(y, self.nd_slice, **self.nd_slice_args) # programmatically using einops
        # y = y[0]

        y = multiple_axis_slice(y, L_input)

        # Compute D term in state space equation - essentially a skip connection
        # B, C, H, L (not flat)
        if not self.out_channels:
            y = y + contract('bh...,ch->bch...', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        # B, H, L (not flat)
        if not self.kernel[0].no_C:
            y = rearrange(y, 'b c h ... -> b (c h) ...')

        if not self.linear:
            y = self.dropout(self.activation(y))

        # ensure output and input shape are the same
        if not self.transposed:
            if not self.kernel[0].no_C:
                # B, H, L -> B, H, C
                y = rearrange(y, "b h ... -> b ... h")
            else:
                y = rearrange(y, "b c h ... -> b ... c h")

        # y = self.norm(y)

        if not self.linear:
            y = self.output_linear(y)

        if self.return_state:
            return y, None
        else: return y

    def default_state(self, *batch_shape, device=None):
        return self._initial_state.repeat(*batch_shape, 1, 1)

    @property
    def d_output(self):
        return self.h
        # return self.h if self.out_channels is None else self.out_channels

    @property
    def d_state(self):
        raise NotImplementedError

    @property
    def state_to_tensor(self):
        raise NotImplementedError

    def get_regularization_loss(self):
        """Compute various regularization terms"""
        reg_loss = 0.0
        
        # L1 regularization on singular values to encourage sparsity
        reg_loss += 0.01 * torch.norm(self.singular_values, p=1)
        
        # Nuclear norm regularization on projection matrices
        reg_loss += 0.001 * (
            torch.norm(self.proj1.reshape(-1, self.adaptive_rank), p='nuc') +
            torch.norm(self.proj2.reshape(-1, self.adaptive_rank), p='nuc')
        )
        
        # Encourage exponential decay in singular values
        target_decay = torch.exp(-torch.arange(self.adaptive_rank).float().to(self.singular_values.device))
        reg_loss += 0.1 * F.mse_loss(self.singular_values, target_decay)
        
        return reg_loss

    def orthogonalize_projections(self):
        """Maintain orthogonality of projection matrices"""
        with torch.no_grad():
            # Reshape to 2D for SVD
            U1, S1, V1 = torch.svd(self.proj1.reshape(-1, self.adaptive_rank))
            U2, S2, V2 = torch.svd(self.proj2.reshape(-1, self.adaptive_rank))
            
            # Reconstruct with orthogonal matrices
            self.proj1.data = (U1 @ V1.t()).reshape(self.proj1.shape)
            self.proj2.data = (U2 @ V2.t()).reshape(self.proj2.shape)

    def analyze_rank_structure(self):
        """Analyze the learned rank structure"""
        with torch.no_grad():
            # Get normalized singular values
            sv_norm = F.softmax(self.singular_values, dim=0)
            
            # Compute effective rank (where 95% of singular value mass is contained)
            cumsum = torch.cumsum(sv_norm, dim=0)
            effective_rank = torch.sum(cumsum < 0.95) + 1
            
            # Compute condition numbers of projection matrices
            _, S1, _ = torch.svd(self.proj1.reshape(-1, self.adaptive_rank))
            _, S2, _ = torch.svd(self.proj2.reshape(-1, self.adaptive_rank))
            cond1 = S1[0] / S1[-1]
            cond2 = S2[0] / S2[-1]
            
            return {
                'effective_rank': effective_rank.item(),
                'singular_values': sv_norm.cpu().numpy(),
                'condition_numbers': (cond1.item(), cond2.item())
            }

