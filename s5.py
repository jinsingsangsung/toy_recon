from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import kaiming_normal_, normal_
from scipy.linalg import block_diag
import math

def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(shape):
        """ Init function
             Args:
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return torch.empty(shape).uniform_(np.log(dt_min), np.log(dt_max))

    return init


def init_log_steps(H, dt_min, dt_max):
    """ Initialize an array of learnable timescale parameters
         Args:
             H: array shape
             dt_min, dt_max: min/max timescale values
         Returns:
             initialized array of timescales (float32): (H,)
     """
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)((1,))
        log_steps.append(log_step)

    return torch.cat(log_steps)


def init_VinvB(init_fun, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. kaiming_normal_
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = torch.empty(shape)
    init_fun(B)
    VinvB = torch.from_numpy(Vinv @ B.numpy())
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return torch.stack((VinvB_real, VinvB_imag), dim=-1)


def trunc_standard_normal(shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        C = torch.empty(1, P, 2)
        kaiming_normal_(C)
        Cs.append(C)
    return torch.cat(Cs, dim=0)


def init_CV(init_fun, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. kaiming_normal_
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C = torch.empty(*shape, 2)
    init_fun(C.shape)
    C_ = C[..., 0] + 1j * C[..., 1]
    CV = C_ @ torch.from_numpy(V).to(torch.complex64)
    CV_real = CV.real
    CV_imag = CV.imag
    return torch.stack((CV_real, CV_imag), dim=-1)


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


def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

def parallel_scan(Lambda_elements, Bu_elements):
    """
    Does the equivalent of the following:
    L = Lambda_elements.shape[0]
    P = Lambda_elements.shape[1]
    xs = []
    x = torch.zeros_like(Bu_elements[0])
    for i in range(L):
        x = Lambda_elements[i] * x + Bu_elements[i]
        xs.append(x)
    xs = torch.stack(xs)
    """
    device = Bu_elements.device
    x = torch.zeros_like(Bu_elements, device=device)
    L, N = Lambda_elements.shape
    pad_len = 2**(math.ceil(math.log2(L))) - L
    # x_padded = F.pad(x, (0, pad_len), "constant", 0)
    Lambda_padded = F.pad(Lambda_elements, (0, 0, 0, pad_len), "constant", 0)
    Bu_padded = F.pad(Bu_elements, (0, 0, 0, pad_len), "constant", 0)

    L_padded = L + pad_len
    levels = int(math.log2(L_padded))

    y = torch.zeros(L_padded, N, device=device, dtype=Bu_elements.dtype) # L, N
    z = Bu_padded # L, N
    # Up-sweep phase
    for d in range(levels):
        step = 2 ** d
        indices = torch.arange(0, L_padded, 2 * step, device=device)
        z[indices + 2*step - 1] += Lambda_padded[indices + 2*step - 1] * z[indices + step - 1]
        if len(indices) > 1:
            Lambda_padded[indices + 2*step - 1] = Lambda_padded[indices + 2*step - 1] * Lambda_padded[indices + step - 1]

    # Down-sweep phase
    y[:] = z[:]
    z[-1, :] = 0
    for d in range(levels -1, -1, -1):
        step = 2 ** d
        indices = torch.arange(0, L_padded, 2 * step, device=device)
        y[indices + step - 1] = z[indices + 2 * step - 1]
        y[indices + 2*step - 1] = z[indices + step - 1] + Lambda_padded[indices + step - 1] * z[indices + 2*step - 1]
        z[:] = y[:]

    y_L = Lambda_elements[-1] * y[-1] + Bu_elements[-1]
    y = torch.cat([z[1:], y_L[None]], dim=0)

    y = y[:L]
    return y

def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    L = input_sequence.shape[0]
    P = Lambda_bar.shape[0]
    
    Lambda_elements = Lambda_bar.unsqueeze(0).expand(L, -1)

    # Equivalent to jax.vmap(lambda u: B_bar @ u)(input_sequence)
    # B_bar shape is (P, H), input_sequence shape is (L, H)
    # Need to match dtypes and get output shape (L, P)
    Bu_elements = torch.einsum('ph,lh->lp', B_bar, input_sequence.to(B_bar.dtype))

    # Forward pass
    xs = []
    if False:
        x = torch.zeros_like(Bu_elements[0])
        for i in range(L):
            x = Lambda_elements[i] * x + Bu_elements[i]
            xs.append(x)
        xs = torch.stack(xs)
    else:
        xs = parallel_scan(Lambda_elements, Bu_elements)

    if bidirectional:
        # Backward pass
        xs2 = []
        x = torch.zeros_like(Bu_elements[0])
        for i in range(L-1, -1, -1):
            x = Lambda_elements[i] * x + Bu_elements[i]
            xs2.append(x)
        xs2 = torch.stack(xs2[::-1])
        xs = torch.cat((xs, xs2), dim=-1)

    if conj_sym:
        return 2 * torch.matmul(xs, C_tilde.t()).real
    else:
        return torch.matmul(xs.to(torch.complex64), C_tilde.t()).real


class S5SSM(nn.Module):
    def __init__(self, 
                 Lambda_re_init,
                 Lambda_im_init,
                 V,
                 Vinv,
                 H,
                 P,
                 C_init,
                 discretization,
                 dt_min,
                 dt_max,
                 conj_sym=True,
                 clip_eigs=False,
                 bidirectional=False,
                 step_rescale=1.0):
        """ The S5 SSM
            Args:
                Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
                Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
                V           (complex64): Eigenvectors used for init           (P,P)
                Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
                H           (int32):     Number of features of input seq 
                P           (int32):     state size
                C_init      (string):    Specifies How C is initialized
                             Options: [trunc_standard_normal: sample from truncated standard normal 
                                                            and then multiply by V, i.e. C_tilde=CV.
                                       kaiming_normal: sample from Kaiming_normal and then multiply by V.
                                       complex_normal: directly sample a complex valued output matrix 
                                                        from standard normal, does not multiply by V]
                conj_sym    (bool):    Whether conjugate symmetry is enforced
                clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                       constrain real part of eigenvalues to be negative. 
                                       True recommended for autoregressive task/unbounded sequence lengths
                                       Discussed in https://arxiv.org/pdf/2206.11893.pdf.
                bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
                discretization: (string) Specifies discretization method 
                                 options: [zoh: zero-order hold method,
                                           bilinear: bilinear transform]
                dt_min:      (float32): minimum value to draw timescale values from when 
                                        initializing log_step
                dt_max:      (float32): maximum value to draw timescale values from when 
                                        initializing log_step
                step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                        on a different resolution for the speech commands benchmark
        """
        super().__init__()

        self.H = H
        self.P = P
        self.C_init = C_init
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale

        if self.conj_sym:
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state matrix Lambda (eigenvalues)
        self.Lambda_re = nn.Parameter(torch.from_numpy(Lambda_re_init))
        self.Lambda_im = nn.Parameter(torch.from_numpy(Lambda_im_init))

        # Initialize input to state (B) matrix
        B_shape = (local_P, self.H)
        self.B = nn.Parameter(init_VinvB(kaiming_normal_, B_shape, Vinv))

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init_fn = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["kaiming_normal"]:
            C_init_fn = lambda shape: kaiming_normal_(torch.empty(*shape))
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init_fn = lambda shape: normal_(torch.empty(*shape), std=0.5**0.5)
        else:
            raise NotImplementedError(f"C_init method {self.C_init} not implemented")

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                self.C = nn.Parameter(C_init_fn((self.H, 2 * self.P, 2)))
            else:
                self.C = nn.Parameter(C_init_fn((self.H, self.P, 2)))
        else:
            if self.bidirectional:
                self.C1 = nn.Parameter(init_CV(C_init_fn, C_shape[:2], V))
                self.C2 = nn.Parameter(init_CV(C_init_fn, C_shape[:2], V))
            else:
                self.C = nn.Parameter(init_CV(C_init_fn, C_shape[:2], V))

        # Initialize feedthrough (D) matrix
        self.D = nn.Parameter(torch.empty(self.H).normal_(std=1.0))

        # Initialize learnable discretization timescale value
        self.log_step = nn.Parameter(init_log_steps(self.P, self.dt_min, self.dt_max))

    def forward(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        if self.clip_eigs:
            Lambda = torch.clamp(self.Lambda_re, max=-1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        step = self.step_rescale * torch.exp(self.log_step)

        # Discretize
        if self.discretization == "zoh":
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretization == "bilinear":
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f"Discretization method {self.discretization} not implemented")

        if self.bidirectional:
            if self.C_init in ["complex_normal"]:
                C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
            else:
                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                C_tilde = torch.cat((C1, C2), dim=-1)
        else:
            C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        ys = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, self.conj_sym, self.bidirectional)

        # Add feedthrough matrix output Du
        Du = input_sequence * self.D.unsqueeze(0)
        return ys + Du

class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    def __init__(self, ssm, dropout, d_model, activation="gelu", training=True,
                 prenorm=False, batchnorm=False, bn_momentum=0.90, step_rescale=1.0):
        super().__init__()
        
        self.ssm = ssm
        self.dropout = dropout
        self.d_model = d_model
        self.activation = activation
        self.training = training
        self.prenorm = prenorm
        self.batchnorm = batchnorm
        self.step_rescale = step_rescale

        if self.activation in ["full_glu"]:
            self.out1 = nn.Linear(d_model, d_model)
            self.out2 = nn.Linear(d_model, d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Linear(d_model, d_model)

        if self.batchnorm:
            self.norm = nn.BatchNorm1d(d_model, momentum=bn_momentum)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        # skip = x
        if len(x.shape) == 3:
            x = x[0]
        # if self.prenorm:
        #     x = self.norm(x)
        x = self.ssm(x)
        x = self.out2(torch.sigmoid(x))

        # if self.activation in ["full_glu"]:
        #     x = self.drop(F.gelu(x))
        #     x = self.out1(x) * torch.sigmoid(self.out2(x))
        #     x = self.drop(x)
        # elif self.activation in ["half_glu1"]:
        #     x = self.drop(F.gelu(x))
        #     x = x * torch.sigmoid(self.out2(x))
        #     x = self.drop(x)
        # elif self.activation in ["half_glu2"]:
        #     # Only apply GELU to the gate input
        #     x1 = self.drop(F.gelu(x))
        #     x = x * torch.sigmoid(self.out2(x1))
        #     x = self.drop(x)
        # elif self.activation in ["gelu"]:
        #     x = self.drop(F.gelu(x))
        # else:
        #     x = self.drop(x)
        #     # raise NotImplementedError(f"Activation: {self.activation} not implemented")

        # x = skip + x
        # if not self.prenorm:
        #     x = self.norm(x)
        if len(x.shape) == 2:
            x = x[None]
        return x

if __name__ == "__main__":
    blocks = 4
    ssm_size = 16
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

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    ssm = S5SSM(Lambda_re_init=Lambda.real,
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

    layer = SequenceLayer(ssm=ssm, dropout=0.0, d_model=3, activation="gelu", training=True, prenorm=False, batchnorm=False, bn_momentum=0.9, step_rescale=1.0)
    x = torch.randn(1024, 3)
    print(layer(x).shape)

