import torch
from torch import random
from torch.nn.init import kaiming_normal_
from torch.linalg import eigh

"""
modified from https://github.com/lindermanlab/S5/blob/main/s5/ssm_init.py
"""

def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
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
    P = torch.sqrt(torch.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = torch.sqrt(2 * torch.arange(N) + 1.0)
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

    S = A + P[:, None] * P[None, :]

    S_diag = torch.diagonal(S)
    Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)
    P = V.conj().T @ P.to(torch.complex64)
    B_orig = B
    B = V.conj().T @ B.to(torch.complex64)
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def make_DPLR_HiPPO_real(N):
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

    S = A + P[:, None] * P[None, :]

    S_diag = torch.diagonal(S)
    Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S)
    P = V.T @ P
    B_orig = B
    B = V.T @ B
    return Lambda_real, P, B, V, B_orig


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
        return torch.rand(shape) * (
            torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))
        ) + torch.log(torch.tensor(dt_min))

    return init


def init_log_steps(input):
    """ Initialize an array of learnable timescale parameters
         Args:
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
        log_steps.append(log_step)

    return torch.cat(log_steps, dim=0)


def init_VinvB(init_fun, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = torch.empty(shape)
    init_fun(B)
    VinvB = Vinv @ B.to(torch.complex64)
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return torch.cat((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)

def init_VinvB_takeB(Vinv, B, real=True):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             B: (B, C, H2, W2, K2, local_P) or (B, C, H2, W2, K1, local_P) depending on dim
             Vinv: (complex64)     the inverse eigenvectors used for initialization (P, P)
         Returns:
             B_tilde (complex64) of shape (B, C, H2, W2, K1, local_P, 2)
     """
    if real:
        VinvB = B @ Vinv
        return VinvB
    else:
        VinvB = B.to(torch.complex64) @ Vinv
        VinvB_real = VinvB.real
        VinvB_imag = VinvB.imag
        return torch.cat((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: torch tensor to initialize
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P = key.shape
    Cs = []
    for i in range(H):
        C = torch.empty(1, P)
        kaiming_normal_(C)
        Cs.append(C)
    return torch.cat(Cs, dim=0)

def init_CV(init_fun, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(torch.empty(shape))
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return torch.cat((CV_real[..., None], CV_imag[..., None]), axis=-1)

def init_CV_real(init_fun, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C = init_fun(torch.empty(shape))
    CV = C @ V
    CV_real = CV
    return CV_real

def init_CV_takeC(C, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return torch.cat((CV_real[..., None], CV_imag[..., None]), axis=-1)