import torch
import torch.nn as nn
import math

class EfficientParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B):
        """
        x: (B, C, L)
        A: (B, C, L, N, N)
        B: (B, C, L, N)
        """
        batch_size, C, L = x.shape
        N = A.shape[-1]
        device = x.device
        dtype = x.dtype

        # Pad to nearest power of 2
        pad_len = 2 ** math.ceil(math.log2(L)) - L
        x_padded = torch.cat([x, torch.zeros(batch_size, C, pad_len, device=device, dtype=dtype)], dim=2)
        A_padded = torch.cat([A, torch.eye(N, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, C, pad_len, 1, 1)], dim=2)
        B_padded = torch.cat([B, torch.zeros(batch_size, C, pad_len, N, device=device, dtype=dtype)], dim=2)

        L_padded = x_padded.shape[2]
        levels = int(math.log2(L_padded))

        # Initialize y and z
        y = torch.zeros(batch_size, C, L_padded, N, device=device, dtype=dtype)
        z = B_padded * x_padded.unsqueeze(-1)

        # Up-sweep phase
        for d in range(levels):
            step = 2 ** d
            indices = torch.arange(0, L_padded, 2 * step, device=device)
            z[:, :, indices + 2*step - 1] += torch.matmul(A_padded[:, :, indices + 2*step - 1], z[:, :, indices + step - 1].unsqueeze(-1)).squeeze(-1)
            if len(indices) > 1:
                A_padded[:, :, indices + 2*step - 1] = torch.matmul(A_padded[:, :, indices + 2*step - 1], A_padded[:, :, indices + step - 1])

        # Down-sweep phase
        y[:] = z[:]
        z[:, :, -1] = 0
        for d in range(levels - 1, -1, -1):
            step = 2 ** d
            indices = torch.arange(0, L_padded, 2 * step, device=device)
            y[:, :, indices + step - 1] = z[:, :, indices + 2 * step - 1]
            y[:, :, indices + 2*step - 1] = z[:, :, indices + step - 1] + torch.matmul(A_padded[:, :, indices + step - 1], z[:, :, indices + 2 * step - 1].unsqueeze(-1)).squeeze(-1)
            z[:] = y[:]

        y_L = torch.matmul(A[:, :, -1], y[:, :, -1].unsqueeze(-1)).squeeze(-1) + B[:, :, -1] * x[:, :, -1].unsqueeze(-1)
        y = torch.cat([z[:, :, 1:], y_L.unsqueeze(2)], dim=2)
        # Remove padding
        y = y[:, :, :L]

        ctx.save_for_backward(x, A, B, y, A_padded, B_padded)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B, y, A_padded, B_padded = ctx.saved_tensors
        batch_size, C, L = x.shape
        N = A.shape[-1]
        device = x.device
        dtype = x.dtype

        # Pad to nearest power of 2
        pad_len = 2 ** math.ceil(math.log2(L)) - L
        grad_output_padded = torch.cat([grad_output, torch.zeros(batch_size, C, pad_len, N, device=device, dtype=dtype)], dim=2)
        y_padded = torch.cat([y, torch.zeros(batch_size, C, pad_len, N, device=device, dtype=dtype)], dim=2)
        x_padded = torch.cat([x, torch.zeros(batch_size, C, pad_len, device=device, dtype=dtype)], dim=2)

        L_padded = grad_output_padded.shape[2]
        levels = int(math.log2(L_padded))

        # Initialize gradients
        grad_x = torch.zeros_like(x_padded)
        grad_A = torch.zeros_like(A_padded)
        grad_B = torch.zeros_like(B_padded)

        # Backward pass
        grad_y = grad_output_padded.clone()
        for i in range(L_padded - 1, -1, -1):
            if i < L_padded - 1:
                grad_y[:, :, i] += torch.matmul(A_padded[:, :, i + 1].transpose(-1, -2), grad_y[:, :, i + 1].unsqueeze(-1)).squeeze(-1)
            grad_x[:, :, i] = torch.sum(grad_y[:, :, i] * B_padded[:, :, i], dim=-1)
            grad_A[:, :, i] = torch.matmul(grad_y[:, :, i].unsqueeze(-1), y_padded[:, :, i - 1].unsqueeze(-2) if i > 0 else torch.zeros_like(y_padded[:, :, 0]).unsqueeze(-2))
            grad_B[:, :, i] = grad_y[:, :, i] * x_padded[:, :, i].unsqueeze(-1)

        # Remove padding
        grad_x = grad_x[:, :, :L]
        grad_A = grad_A[:, :, :L]
        grad_B = grad_B[:, :, :L]

        return grad_x, grad_A, grad_B

class EfficientParallelScan(nn.Module):
    def __init__(self):
        super(EfficientParallelScan, self).__init__()

    def forward(self, x, A, B):
        return EfficientParallelScanFunction.apply(x, A, B)
    

import torch
import torch.nn as nn

class ParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B):
        """
        x: (B, C, L)
        A: (B, C, L, N, N)
        B: (B, C, L, N)
        """
        batch_size, C, L, N = B.shape
        y = torch.zeros(batch_size, C, L, N, device=x.device, dtype=x.dtype)
        
        # Implement the forward pass
        y_prev = torch.zeros(batch_size, C, N, device=x.device, dtype=x.dtype)
        for i in range(L):
            y_new = torch.matmul(A[:, :, i], y_prev.unsqueeze(-1)).squeeze(-1) + B[:, :, i] * x[:, :, i].unsqueeze(-1)
            y[:, :, i] = y_new
            y_prev = y_new
        
        # Save tensors needed for backward pass
        ctx.save_for_backward(x, A, B, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, A, B, y = ctx.saved_tensors
        batch_size, C, L, N = B.shape
        
        grad_x = torch.zeros_like(x)
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        
        # Implement the backward pass
        grad_y_next = torch.zeros(batch_size, C, N, device=x.device, dtype=x.dtype)
        for i in range(L - 1, -1, -1):
            grad_y = grad_output[:, :, i] + torch.matmul(A[:, :, i].transpose(-1, -2), grad_y_next.unsqueeze(-1)).squeeze(-1)
            grad_x[:, :, i] = torch.sum(grad_y * B[:, :, i], dim=-1)
            grad_A[:, :, i] = torch.matmul(grad_y.unsqueeze(-1), (y[:, :, i-1] if i > 0 else torch.zeros_like(y[:, :, 0])).unsqueeze(-2))
            grad_B[:, :, i] = grad_y * x[:, :, i].unsqueeze(-1)
            grad_y_next = grad_y
        
        return grad_x, grad_A, grad_B

class ParallelScan(nn.Module):
    def __init__(self):
        super(ParallelScan, self).__init__()

    def forward(self, x, A, B):
        return ParallelScanFunction.apply(x, A, B)

if __name__ == '__main__':
    batch_size, C, L, N = 1, 1, 128, 16
    # Import necessary functions
    from mambaconv import transition

    # Define parameters for transition matrices
    measure = 'legs'  # You can change this to 'legt', 'lagt', or 'tlagt' if needed

    # Generate A and B matrices
    A_np, B_np = transition(measure, N)

    # Convert to PyTorch tensors
    A_single = torch.from_numpy(A_np).float()
    B_single = torch.from_numpy(B_np).squeeze(-1).float()

    # Import necessary functions
    from mambaconv import construct_A_B_stacked

    # Generate A_stacked and B_stacked
    A_stacked, B_stacked = construct_A_B_stacked(A_single, B_single, L, discretization='bilinear')

    # Expand A_stacked and B_stacked to match the dimensions
    A_stacked = A_stacked.unsqueeze(0).unsqueeze(0).expand(batch_size, C, L, N, N)
    B_stacked = B_stacked.unsqueeze(0).unsqueeze(0).expand(batch_size, C, L, N)

    parallel_scan = ParallelScan()
    efficient_parallel_scan = EfficientParallelScan()

    x = torch.randn(batch_size, C, L)

    # Use A_stacked and B_stacked in the existing functions
    par_output_stacked = parallel_scan(x, A_stacked, B_stacked)
    eff_output_stacked = efficient_parallel_scan(x, A_stacked, B_stacked)
    
    # Reshape outputs to match dimensions
    par_output_stacked = par_output_stacked.view(batch_size, C, L, N)
    eff_output_stacked = eff_output_stacked.view(batch_size, C, L, N)
    
    print("Parallel Scan (stacked) output shape:", par_output_stacked.shape)
    print("Efficient Parallel Scan (stacked) output shape:", eff_output_stacked.shape)

    # Calculate and print the difference between stacked versions
    diff_stacked = torch.abs(par_output_stacked - eff_output_stacked)
    print("Maximum absolute difference (stacked):", diff_stacked.max().item())
    print("Mean absolute difference (stacked):", diff_stacked.mean().item())
    print("Standard deviation of difference (stacked):", diff_stacked.std().item())