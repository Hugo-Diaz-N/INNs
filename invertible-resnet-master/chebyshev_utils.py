import torch
import numpy as np
import math
torch.set_default_dtype(torch.float64)

from linear_operator.operators import LinearOperator

class MatvecOperator(LinearOperator):
    """
    Create a LinearOperator representing M @ v = (I + A^T)(I + A) @ v = v + A@v +A^T @ v + A^T(A@v)
    """
    def __init__(self, A):
        super().__init__(dtype=A.dtype, device=A.device)  # or use .device of A_mv
        self.A = A
        self._shape = A.shape[-2:] # A.shape = (B, d,d)

    def __matmul__(self, x):
        # Compute (I + Aᵗ)(I + A)x = x + A @ x + Aᵗ @ x + Aᵗ @ (A @ x)
        xplusAx = x + self.A @ x
        z = xplusAx + self.A.T@(xplusAx) # .transpose(1, 2) 
        return z

    def _size(self):
        return self._shape

class ChebMatvecOperator(LinearOperator):
    """
    Create a LinearOperator representing:
    M @ v = (2/(b-a)) * ((I + J^T)(I + J) @ v) - ((b+a)/(b-a)) * v
    """
    def __init__(self, A, a=-1.0, b=1.0):
        dtype = A.dtype
        device = A.device
        shape = A.shape[-2:] 
        self._args = (A,)
        super().__init__(dtype=A.dtype, device=A.device)
        self.A = A
        self.a = a
        self.b = b
        self._shape = shape 

    def __matmul__(self, x):
        xplusAx = x + self.A @ x
        z = xplusAx + self.A.T @ xplusAx
        return (2/(self.b - self.a)) * z - ((self.b + self.a)/(self.b - self.a)) * x

    def _size(self):
        return self._shape
    
    @property
    def dtype(self):
        return self.A.dtype
    
    @property
    def device(self):
        return self.A.device
   
    

def chebyshev_coeffs(f, degree=3, dt=0.0001): # original code: https://github.com/erhallma/erhallma.github.io
    """
        Compute the first (k+1) Chebyshev coefficients for a function f(x)
        on the interval [-1, 1] 
        Args:
            f (callable): Function to approximate on [-1, 1].
            k (int): Highest degree Chebyshev coefficient to compute.

        Returns:
            coeffs (np.ndarray): Array of shape (k+1,) with Chebyshev coefficients.
    """
    # ts = torch.arange(0, torch.pi, dt)
    # x = torch.cos(ts)
    # print("x:",x)
    # print("x+1:",x+1)
    # fx = f(x)
    
    # Approximate coefficients using Clenshaw-Curtis (Fejer) quadrature
    n_quad = 1000 
    theta = np.pi * (np.arange(n_quad) + 0.5) / n_quad
    x = np.cos(theta)
    fx =f(x)
    k = degree
    coeffs = np.zeros(k + 1)
    for j in range(k + 1): # Fejér quadrature
        Tj = np.cos(j * theta)
        coeffs[j] = (2 / n_quad) * np.sum(fx * Tj)  # a_k \approx \frac{2}{N} \sum_{n=0}^{N-1} f(\cos[(n+0.5)\pi/N]) \cos[(n+0.5) k \pi/N]

    coeffs[0] /= 2
    return coeffs


def chebyshev_series_action_from_function(A: torch.Tensor, f, k: int, Omega: torch.Tensor) -> torch.Tensor:
    """
    Computes p(A) @ Omega where p(x) is the Chebyshev approximation of f(x)
    of degree k on [-1, 1].

    Args:
        A (torch.Tensor): Square matrix of shape (n, n).
        f (Callable): Function to approximate.
        k (int): Degree of the Chebyshev polynomial.
        Omega (torch.Tensor): Matrix of shape (n, m).

    Returns:
        torch.Tensor: Result of applying the Chebyshev approximation p(A) to Omega.
    """
    coeffs = chebyshev_coeffs(f, k)  # This should return a 1D torch.Tensor or list of floats
    K = len(coeffs)
    assert A.shape[0] == A.shape[1],     "Matrix A must be square"
    assert A.shape[1] == Omega.shape[0], "Matrix A and Omega must align for multiplication"

    T0 = Omega.clone()
    result = coeffs[0] * T0
    if K == 1:
        return result

    T1 = A @ Omega
    result = result + coeffs[1] * T1

    for j in range(2, K):
        Tj = 2 * (A @ T1) - T0
        result = result + coeffs[j] * Tj
        T0, T1 = T1, Tj

    return result


def conv_chebyshev_series_action_from_function(A, f, k: int, Omega: torch.Tensor) -> torch.Tensor:
    """
    Computes p(A) @ Omega where p(x) is the Chebyshev approximation of f(x)
    of degree k on [-1, 1].

    Args:
        A (torch.Tensor): Batched linear operator matrix of shape (B, n, -1).
        f (Callable): Function to approximate.
        k (int): Degree of the Chebyshev polynomial.
        Omega (torch.Tensor): Matrix of shape (n, m).

    Returns:
        torch.Tensor: Result of applying the Chebyshev approximation p(A) to Omega.
    """
    coeffs = chebyshev_coeffs(f, k)  # This should return a 1D torch.Tensor or list of floats
    K = len(coeffs)
    # B = A.shape[0]
    # n = Omega.shape[1]
    # output_shape = Omega.shape[2:] 


    T0 = Omega.clone() # (B, n, C*H*W) # I* \Omega 
    #print("Omega shape:", T0.shape)
    result = coeffs[0] * T0
    if K == 1:
        return result

    T1 = A @ Omega
    #print("A \Omega shape:", T1.shape)
    result = result + coeffs[1] * T1

    for j in range(2, K):
        Tj = 2 * (A @ T1) - T0
        result = result + coeffs[j] * Tj
        T0, T1 = T1, Tj

    return result

def chebyshev_series_bilinear_form(A: torch.Tensor, f, k: int, Omega: torch.Tensor) -> torch.Tensor:
    """
    Computes  @ Omega^T @ p(A) @ Omega where p(x) is the Chebyshev approximation of f(x)
    of degree k on [-1, 1].

    Args:
        A (torch.Tensor): Square matrix of shape (n, n). TODO allow (b,n,n)
        f (Callable): Function to approximate.
        k (int): Degree of the Chebyshev polynomial.
        Omega (torch.Tensor): Matrix of shape (n, m).

    Returns:
        torch.Tensor: Result of applying the Chebyshev approximation p(A) to Omega.

    Based on Algorithm 2 Two-sided evaluation (Chebyshev basis): "Faster stochastic trace estimation with a Chebyshev product identity" by Eric Hallman 
    """
    coeffs = chebyshev_coeffs(f, k)  # =(\alpha0,...,alpha_{n}) This should return a 1D torch.Tensor or list of floats

    K = len(coeffs) # = n+1 
   
    assert A.shape[0] == A.shape[1],     "Matrix A must be square"
    assert A.shape[1] == Omega.shape[0], "Matrix A and Omega must align for multiplication"

    halfK = int(torch.ceil(torch.tensor((K-1)/2)))

    z0 = Omega.clone()
    zeta0 = torch.matmul(z0.T, z0)
    result = coeffs[0] * zeta0   # s 

    if K == 1:  # halfK = 1
        return result
    z1 = A @ z0
    zeta1 = torch.matmul(z0.T, z1) 
    result +=  coeffs[1] * zeta1
    if K == 2: # halfK =1 
        return result
    result +=  coeffs[2] * (2*torch.matmul(z1.T, z1)  -zeta0)
    if K == 3: # halfK = 2
        return result
    
    for j in range(2, halfK + 1): # terms come in pairs. PS. It is more natural 1/2 P(A) than p(A) 
        zj = 2 * (A @ z1) - z0
        result +=  coeffs[2*j-1] * ( 2* torch.matmul(z1.T, zj)  - zeta1 )
        if K ==2*j: # K = n+1
            break
        result +=  coeffs[2*j] * ( 2* torch.matmul(zj.T, zj)  - zeta0 )
        z0, z1 = z1, zj

    return result


def chebyshev_coeffs_log1p(k, exact=False):
    """
    Compute the first (k+1) Chebyshev coefficients for f(x) = log(1 + x)
    on the interval [-1, 1].

    Args:
        k (int): Highest degree Chebyshev coefficient to compute.
        exact (bool): Whether to use exact formula (default: False).

    Returns:
        coeffs (np.ndarray): Array of shape (k+1,) with Chebyshev coefficients.
    """
    coeffs = np.zeros(k + 1)

    if exact:
        coeffs[0] = -np.log(2)
        for n in range(1, k + 1):
            coeffs[n] = 2 * (-1)**(n + 1) / n
        return coeffs

    # Approximate coefficients using Clenshaw-Curtis quadrature
    n_quad = 1000 # FEJER quad.
    theta = np.pi * (np.arange(n_quad) + 0.5) / n_quad
    x = np.cos(theta)
    fx = np.log1p(x)

    for j in range(k + 1): # Fejér quadrature
        Tj = np.cos(j * theta)
        coeffs[j] = (2 / n_quad) * np.sum(fx * Tj)  # a_k \approx \frac{2}{N} \sum_{n=0}^{N-1} f(\cos[(n+0.5)\pi/N]) \cos[(n+0.5) k \pi/N]

    coeffs[0] /= 2
    return coeffs

def Chebhutchplusplus(A: torch.Tensor, f, k: int, num_queries: int,  sketchMethod= 'signs', debugging_mode=False) -> torch.Tensor: # https://github.com/RaphaelArkadyMeyerNYU/HutchPlusPlus/blob/main/simple/simple_hutchplusplus.m
    """
    Hutch++ trace estimator for a square matrix A.

    Parameters:
    - A (torch.Tensor): Square matrix (n x n)
    - num_queries (int): Total number of matrix-vector products to use

    Returns:
    - trace_est (torch.Tensor): Estimated trace of A
    """
    if debugging_mode:
        torch.manual_seed(0)

    n = A.shape[0]
    dtype = A.dtype
    device = A.device

    assert A.shape[0] == A.shape[1], "Matrix A must be square."

    # s_cols = int(torch.ceil(torch.tensor(num_queries / 3.0)).item())
    # g_cols = int(torch.floor(torch.tensor(num_queries / 3.0)).item())
    s_cols = math.ceil(num_queries / 3.0)
    g_cols = math.floor(num_queries / 3.0)

    # Random ±1 entries
    if sketchMethod in ['rademacher', 'signs']:

        S = -1.0 + 2 * torch.randint(0, 2, (n, s_cols), dtype=torch.int8, device=device).to(dtype) 
        G = -1.0 + 2 * torch.randint(0, 2, (n, g_cols), dtype=torch.int8, device=device).to(dtype) 

    elif sketchMethod == 'gaussian':

        S = torch.randn(n, s_cols, device=device).to(dtype)
        G = torch.randn(n, g_cols, device=device).to(dtype)  

    # QR decomposition (only Q is needed)
    AS =  chebyshev_series_action_from_function(A, f, k=k, Omega=S) #p(A) @ S
    Q, _ = torch.linalg.qr(AS, mode='reduced')

    # Project G orthogonal to Q
    G_proj = G - Q @ (Q.T @ G)

    # Hutch++ estimator
    AQ = chebyshev_series_action_from_function(A, f, k=k, Omega = Q) # p(A) @ Q
    trace_est = torch.einsum('nm,nm->', Q, AQ)  #  torch.trace(Q.T @ (A @ Q))
    if g_cols > 0:
        AGproj = chebyshev_series_action_from_function(A, f, k=k, Omega = G_proj) #p(A) @ G_proj
        trace_est += (1.0 / g_cols) * torch.einsum('nm,nm->', G_proj, AGproj) # instead of torch.trace(G_proj.T @ (A @ G_proj)) # 

    return trace_est 


def trace_chebyshev_series_action(A: torch.Tensor, Omega: torch.Tensor, coeffs: list[float]) -> torch.Tensor:
    """
    Estimate trace(p(A)) where p(A) = sum_j c_j * T_j(A), using Chebyshev polynomials of the first kind
    and stochastic trace estimation with a test matrix Omega.

    This computes:
        trace(p(A)) ~ (1/m) * trace(Omega^T @ p(A) @ Omega)

    Args:
        A (torch.Tensor): Square matrix of shape (n, n).
        Omega (torch.Tensor): Test matrix of shape (n, m).
        coeffs (Sequence[float]): List of Chebyshev coefficients [c0, c1, ..., cK].

    Returns:
        float: Approximate trace of p(A).
    """
    assert A.shape[0] == A.shape[1],     "Matrix A must be square"
    assert A.shape[1] == Omega.shape[0], "Matrix A and Omega must align for multiplication"

    n, m = Omega.shape
    K = len(coeffs)

    T0 = Omega.clone()                      # T_0(A) @ Omega = I @ Omega = Omega
    result = coeffs[0] * T0

    if K > 1:
        T1 = A @ Omega                      # T_1(A) @ Omega = A @ Omega 
        result = result + coeffs[1] * T1
    else:
        T1 = None

    for k in range(2, K):
        Tk = 2 * (A @ T1) - T0                # Recurrence: T_k = 2A T_{k-1}(A) - T_{k-2}(A)
        result = result + coeffs[k] * Tk
        T0, T1 = T1, Tk

    # trace(Omega^T @ result) = torch.sum(Omega * result) # add entry-wise products. for batch use return torch.einsum('bij,bij->b', Omega, result).item() 
    trace_est = torch.trace(Omega.T@result)/m #torch.einsum('nm,nm->', Omega, result).item() / m
    return trace_est


if __name__ == "__main__":
    import torch  # type: ignore
    import numpy as np
    from scipy.linalg import logm, expm, cosm, tanhm, sqrtm, sinhm

    torch.set_default_dtype(torch.float64)

    # Problem size and parameters
    n = 1000
    degree = 6
    num_queries = int(0.12 * n)

    print("\n", "---" * 10, " LOGM TEST (Symmetric Matrix) ", "---" * 10)

    # Construct symmetric matrix A = Q D Qᵀ with eigenvalues in [-0.9,..., 0.9]
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    diag_entries = 0.9 * (2 * torch.rand(n) - 1)
    A = Q * diag_entries @ Q.T
    A2 = A + torch.eye(n)

    z = torch.randn(n, num_queries)

    # Ground truth with SciPy
    A_np, z_np = A2.numpy(), z.numpy()
    y_exact = torch.tensor(z_np.T @ (logm(A_np) @ z_np)).to(dtype=z.dtype)

    # Approximate with Chebyshev (function)
    f = np.log1p
    y_approx = z.T @ chebyshev_series_action_from_function(A, f, degree, z)
    rel_error = torch.norm(y_exact - y_approx) / torch.norm(y_exact)
    print("Relative error (function Chebyshev):         ", rel_error.item())

    # Approximate with Chebyshev (bilinear)
    y_bilinear = chebyshev_series_bilinear_form(A, f, degree, z)
    rel_error_bilinear = torch.norm(y_exact - y_bilinear) / torch.norm(y_exact)
    print("Relative error (bilinear Chebyshev):         ", rel_error_bilinear.item())

    print("||A|| =", torch.norm(A).item(), "   Tr(A) =", torch.trace(A).item())

    print("\n", "---" * 10, " HUTCHINSON TRACE ESTIMATE ", "---" * 10)

    I = torch.eye(n, dtype=A.dtype)
    Omega = torch.randn(n, num_queries)
    coeffs = chebyshev_coeffs_log1p(degree, exact=True)
    sign_, true_slogdet = np.linalg.slogdet((I + A).numpy())

    approx_trace0 = trace_chebyshev_series_action(A, Omega, coeffs)
    approx_trace = torch.trace(Omega.T @ chebyshev_series_action_from_function(A, f, degree, Omega)) / num_queries
    rel_err_trace = torch.abs((approx_trace0 - approx_trace) / true_slogdet)

    print("Relative Error trace estimate(chebyshev_action):", rel_err_trace.item())
    print("Estimated trace (log(I+A)):                  ", approx_trace.item())
    print("True logdet(I+A):                            ", true_slogdet)
    print("Chebyshev coeffs log(1 + x):                 ", chebyshev_coeffs(np.log1p, degree))
    print("Exact Chebyshev coeffs log(1 + x):           ", coeffs)
    print("Relative error logdet(I+A):                  ", np.abs((approx_trace.item() - true_slogdet) / true_slogdet))

    print("\n", "---" * 10, " B = (I + A)(I + Aᵗ) Analysis ", "---" * 10)

    sigma = 0.9
    A = torch.randn(n, n)
    A = sigma * A / torch.linalg.norm(A, ord=2)

    I = torch.eye(n, dtype=A.dtype)
    B = (I + A.T) @ (I + A)

    k = 1
    evals_small, _ = torch.lobpcg(B, k=k, niter=50, largest=False)
    evals_larg, _ = torch.lobpcg(B, k=k, niter=50, largest=True)

    print("Smallest eigenvalue of B:                    ", evals_small.item())
    print("Largest eigenvalue of B:                     ", evals_larg.item())
    print("Spectral norm of A:                          ", torch.linalg.norm(A, ord=2).item())
    print("Spectral norm of (I + A^T)(I + A):           ", torch.linalg.norm(B, ord=2).item())
    print("(1 - |A|)^2 and (1 + |A|)^2:                 ", (1 - sigma) ** 2, "and", (1 + sigma) ** 2)

    sign_, slog_B = np.linalg.slogdet(B.numpy())
    print("True (1/2) logdet((I + A^T)(I + A)):          ", 0.5 * slog_B)
    sign_, slog_IA = np.linalg.slogdet((I + A).numpy())
    print("True logdet(I + A):                           ", slog_IA)

    print("\n", "---" * 10, " MatvecOperator Test ", "---" * 10)

    v = torch.randn(n)
    M = MatvecOperator(A)
    Bv, Mv = B @ v, M @ v
    rel_err_matvec = torch.norm(Bv - Mv) / torch.norm(Bv)
    print("Relative error (matvec Bv vs Mv):            ", rel_err_matvec.item())

    print("\n", "---" * 10, " ChebMatvecOperator Test: C = (2 / (b - a)) * B - ((b + a) / (b - a)) * I ", "---" * 10)

    a = evals_small.item()
    b = evals_larg.item()

    ChebA = ChebMatvecOperator(A, a=a, b=b)
    C = (2 / (b - a)) * B - ((b + a) / (b - a)) * I
    Cv, ChebAv = C @ v, ChebA @ v
    rel_err_cheb = torch.norm(Cv - ChebAv) / torch.norm(Cv)
    print("Relative error (matvec Cv vs ChebAv):        ", rel_err_cheb.item())

    print("\n", "---" * 10, " Hutchinson Chebyshev Log-Trace Estimate ", "---" * 10)

    f = lambda x: np.log(((b - a) / 2) * x + (b + a) / 2)
    num_queries = 3* num_queries
    TraceC = 0.5 * torch.trace(Omega.T @ chebyshev_series_action_from_function(C, f, k=degree, Omega=Omega)) / num_queries
    TraceC2 = 0.5 * torch.trace(Omega.T @ chebyshev_series_action_from_function(ChebA, f, k=degree, Omega=Omega)) / num_queries
    f_Omega = chebyshev_series_action_from_function(ChebA, f, k=degree, Omega=Omega)
    TraceC3 = 0.5 * torch.einsum("bn,bn->", Omega, f_Omega) / num_queries

    print("Trace (Chebyshev, matrix C):                 ", TraceC.item())
    print("Trace (Chebyshev, ChebA class):              ", TraceC2.item())
    print("Trace (Chebyshev, einsum with ChebA):        ", TraceC3.item())

    print("\n", "---" * 10, " Hutch++ Chebyshev Log-Trace Estimate ", "---" * 10)
    print(0.5* Chebhutchplusplus(C, f,num_queries=num_queries, sketchMethod = 'gaussian',k=degree, debugging_mode=True))
    print(0.5* Chebhutchplusplus(ChebA, f,num_queries=num_queries, sketchMethod = 'gaussian', k=degree, debugging_mode=True))


