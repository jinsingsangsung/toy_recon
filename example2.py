import numpy as np
from scipy.linalg import svd
from numpy.linalg import pinv

# Parameters
L = 128  # Input spatial size (L x L)
C = 3  # Input channels
N = 256 # Output dimension

# Step 1: Define well-conditioned Ax, Ay in log-space
lambda_x = np.random.uniform(-1, 1, N)  # Log-space eigenvalues
lambda_y = np.random.uniform(-1, 1, N)

Ax = np.diag(np.exp(lambda_x))  # Exponential to prevent degeneracy
Ay = np.diag(np.exp(lambda_y))

# Step 2: Define factorized Bx, By
Cx = 5  # Rank of Bx
Cy = 5  # Rank of By

Ux, _ = np.linalg.qr(np.random.randn(N, Cx))  # Orthonormal
Uy, _ = np.linalg.qr(np.random.randn(N, Cy))  # Orthonormal

Vx = np.random.randn(Cx, C)
Vy = np.random.randn(Cy, C)

Bx = Ux @ Vx
By = Uy @ Vy

# Step 3: Construct 2D Kernel Matrix M
M_list = []
for i in range(L):
    for j in range(L):
        # Mi = np.kron(Ax**i @ Bx, Ay**j @ By)  # Kronecker product
        Mi = (Ax**i @ Bx) * (Ay**j @ By)  # Kronecker product
        M_list.append(Mi)

M = np.hstack(M_list)  # Shape (N, L^2 * C)

# Step 4: Generate input u (L x L x C)
u = np.random.randn(L, L, C)
u_flat = u.reshape(-1, C)  # Shape (L^2, C)
u_flat = u_flat.flatten()   # Shape (L^2 * C, )

# Step 5: Compute output s
s = M @ u_flat  # Shape (N, )

# Step 6: Recover u using Moore-Penrose Pseudoinverse
M_pinv = pinv(M)  # Compute pseudoinverse
u_recovered = M_pinv @ s  # Shape (L^2 * C, )
u_recovered = u_recovered.reshape(L, L, C)

# Check reconstruction error
error = np.linalg.norm(u - u_recovered) / np.linalg.norm(u)
print(f"Reconstruction Error: {error:.6e}")

# Generate random matrix with same shape as u
u_random = np.random.randn(L, L, C)

# Check error between random matrix and original u
random_error = np.linalg.norm(u - u_random) / np.linalg.norm(u)
print(f"Random Matrix Error: {random_error:.6e}")
