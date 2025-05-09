import torch

def arnoldi(A, b, k):
    """
    Compute k steps of the Arnoldi iteration.
    
    Args:
        A: Square matrix (n x n)
        b: Starting vector (n x 1)
        k: Number of iterations
    
    Returns:
        Q: Orthonormal basis for the Krylov subspace
        H: Upper Hessenberg matrix
    """
    n = A.size(0)
    Q = torch.zeros((n, k), dtype=A.dtype, device=A.device)
    H = torch.zeros((k, k), dtype=A.dtype, device=A.device)
    
    # Normalize the starting vector
    Q[:, 0] = b.squeeze() / torch.norm(b)
    
    for j in range(k-1):
        # Compute new vector
        v = A @ Q[:, j]
        
        # Orthogonalize against previous vectors (Modified Gram-Schmidt)
        for i in range(j + 1):
            H[i, j] = torch.dot(Q[:, i], v)
            v = v - H[i, j] * Q[:, i]
            
        # Get norm of remaining vector
        H[j+1, j] = torch.norm(v)
        
        if H[j+1, j] < 1e-12:  # Check for invariant subspace
            return Q[:, :j+1], H[:j+1, :j+1]
            
        # Add normalized vector to basis
        Q[:, j+1] = v / H[j+1, j]
    
    return Q, H

def krylov_inverse(A, b, k):
    """
    Compute inverse for a Krylov-like matrix using Hessenberg decomposition.
    
    Args:
        A: Square matrix (n x n)
        b: Starting vector (n x 1)
        k: Size of Krylov subspace
    """
    # Get Arnoldi decomposition
    Q, H = arnoldi(A, b, k)
    print([H[a,a] for a in range(20)])
    import pdb; pdb.set_trace()
    
    # Compute inverse of H (it's upper Hessenberg, so more efficient)
    H_inv = torch.linalg.inv(H)  # Could be optimized further for Hessenberg structure
    
    # Reconstruct approximate inverse
    return Q @ H_inv @ Q.T

# Example usage
def test_krylov():
    n = 100
    k = 20
    
    # Create a test matrix and vector
    A = torch.randn(n, n)
    b = torch.randn(n, 1)
    
    # Create Krylov matrix
    K = torch.zeros(n, k)
    v = b
    for i in range(k):
        K[:, i] = v.squeeze()
        v = A @ v
        
    # Compute inverse using Hessenberg decomposition
    K_inv = krylov_inverse(A, b, k)
    
    # Test reconstruction
    error = torch.norm(K @ K_inv @ K - K) / torch.norm(K)
    print(f"Relative error: {error.item()}")

test_krylov()