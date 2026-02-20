import torch # type: ignore
torch.set_default_dtype(torch.float64)
import time
import math
import matplotlib.pyplot as plt
from scipy.linalg import logm
import scipy.linalg as spqr 
import numpy as np



def sample_sketch_tensor(shape, method='gaussian', dtype=torch.float32, device='cpu'):
    """
    Sample a random tensor of the given shape using a specified distribution.

    Parameters:
    - shape (tuple of int): Desired tensor shape.
    - method (str): One of {'rademacher', 'signs', 'gaussian'}.
    - dtype (torch.dtype): Desired data type (default: torch.float32).
    - device (str or torch.device): Target device (default: 'cpu').

    Returns:
    - torch.Tensor of the specified shape and dtype.
    """
    if method in ['rademacher', 'signs']:
        return -1.0 + 2 * torch.randint(0, 2, shape, dtype=torch.int8, device=device).to(dtype)
    elif method == 'gaussian':
        return torch.randn(shape, device=device).to(dtype)
    else:
        raise ValueError(f"Unknown sketching method: {method}")
    

def batched_hutchplusplus(A: torch.Tensor, num_queries: int, sketchMethod= 'signs',  debugging_mode=False) -> torch.Tensor:
    """
    Batched Hutch++ trace estimator for square matrices.

    Parameters:
    - A (torch.Tensor): Batched square matrices (b x n x n)
    - num_queries (int): Total number of matrix-vector products to use

    Returns:
    - trace_est (torch.Tensor): Estimated trace for each batch (b,)
    """
    if debugging_mode:
        torch.manual_seed(0)

    assert A.dim() == 3 and A.shape[1] == A.shape[2], "A must be a batch of square matrices."

    b, n, _ = A.shape
    s_cols = math.ceil(num_queries / 3.0)
    g_cols = math.floor(num_queries / 3.0)

    dtype = A.dtype
    device = A.device

    # Random ±1 entries
    if sketchMethod in ['rademacher', 'signs']:

        S = -1.0 + 2 * torch.randint(0, 2, (b,n, s_cols), dtype=torch.int8, device=device).to(dtype) 
        G = -1.0 + 2 * torch.randint(0, 2, (b,n, g_cols), dtype=torch.int8, device=device).to(dtype) 

    elif sketchMethod == 'gaussian':

        S = torch.randn(b, n, s_cols, device=device).to(dtype)
        G = torch.randn(b, n, g_cols, device=device).to(dtype) 

    AS = A@S #torch.bmm(A, S)  # (b, n, s_cols)  # TODO change it to call A(S) A@ S #

    Q, _ = torch.linalg.qr(AS,  mode='reduced')  # (b, n, s_cols)

    QtG = torch.bmm(Q.transpose(1, 2), G)  # (b, s_cols, g_cols) # Akin to  Q^T @ G but for each slice or batch 
    G = G - torch.bmm(Q, QtG)              # (b, n, g_cols)      # Akin to (I- QQ^T)G   

    AQ = A@Q #torch.bmm(A, Q)       #A@Q        # (b, n, s_cols) # TODO change to call A(Q) 
    trace_Q = torch.einsum('bij,bij->b', Q, AQ)  #  tr(Q^T A Q)  batch-wise (b, )

    trace_est = trace_Q
    if g_cols > 0: # Einstein notation, not much faster 
        AG_proj = torch.bmm(A, G) #  A@G                          # (b, n, g_cols)
        trace_G = torch.einsum('bij,bij->b', G, AG_proj)/ g_cols  # tr( G^T(I-QQ^T) A (I-QQ^T)G)  shape: (b, )
        trace_est = trace_est + trace_G

    return trace_est  # shape (b,)



def batched_hutchplusplus_from_products(  # TODO fix 
    AS: torch.Tensor,
    AQ: torch.Tensor,
    AG_proj: torch.Tensor,
    G_proj: torch.Tensor = None
) -> torch.Tensor:
    """
    Batched Hutch++ trace estimator from precomputed A @ S, A @ Q, and A @ G_proj.

    Parameters:
    - AS (torch.Tensor): Result of A @ S, shape (b, n, s)
    - AQ (torch.Tensor): Result of A @ Q, shape (b, n, s)
    - AG_proj (torch.Tensor): Result of A @ G_proj, shape (b, n, g)
    - G_proj (torch.Tensor, optional): Projected G = (I - QQᵗ) G, shape (b, n, g)

    Returns:
    - trace_est (torch.Tensor): Estimated trace for each batch (b,)
    """
    b, n, s = AS.shape
    _, _, g = AG_proj.shape

    # QR decomposition of AS to get Q (b, n, s)
    Q, _ = torch.linalg.qr(AS, mode='reduced')

    # trace_Q = tr(Qᵗ A Q), via trace of Q * AQ
    trace_Q = torch.einsum('bij,bij->b', Q, AQ)

    trace_est = trace_Q
    if g > 0:
        assert G_proj is not None, "G_proj must be provided if g > 0"
        # trace_G = tr(G_projᵗ A G_proj) / g
        trace_G = torch.einsum('bij,bij->b', G_proj, AG_proj) / g
        trace_est = trace_est + trace_G

    return trace_est



def batched_hutchplusplus3(A: torch.Tensor, num_queries: int, sketchMethod='signs', debugging_mode=False) -> torch.Tensor:
    """
    Batched Hutch++ trace estimator for square matrices A with convolutional shape (B, C, H, W).

    Parameters:
    - A (callable): A function that applies the matrix A to a tensor of shape (B, k, C, H, W)
    - num_queries (int): Total number of matrix-vector products to use
    - sketchMethod (str): Method to generate sketch matrices ('signs', 'rademacher', 'gaussian')
    - debugging_mode (bool): If True, sets torch.manual_seed(0) for reproducibility

    Returns:
    - trace_est (torch.Tensor): Estimated trace for each batch (B,)
    """
    if debugging_mode:
        torch.manual_seed(0)

    B, C, H, W = A.shape
    s_cols = math.ceil(num_queries / 3.0)
    g_cols = math.floor(num_queries / 3.0)

    dtype = A.dtype
    device = A.device

    # Generate sketch tensors
    S = sample_sketch_tensor((B, s_cols, C, H, W), method=sketchMethod, dtype=dtype, device=device)
    G = sample_sketch_tensor((B, g_cols, C, H, W), method=sketchMethod, dtype=dtype, device=device)

    # Compute A @ S and its QR factorization
    AS = A @ S
    Q, _ = torch.linalg.qr(AS.transpose(-1, -2), mode='reduced')  # shape: (B, C*H*W, s_cols)

    # Project G orthogonal to Q
    G_flat = G.view(B, -1, g_cols)
    QtG = torch.bmm(Q.transpose(1, 2), G_flat)      # (B, s_cols, g_cols)
    G_proj = G_flat - torch.bmm(Q, QtG)             # (B, C*H*W, g_cols)

    # Apply A to Q and projected G
    AQ = A @ Q.view(B, s_cols, C, H, W)
    trace_Q = torch.einsum('bij,bij->b', Q.transpose(-1, -2), AQ)

    trace_est = trace_Q
    if g_cols > 0:
        AG_proj = A @ G_proj.view(B, g_cols, C, H, W)
        trace_G = torch.einsum('bij,bij->b', G_proj.transpose(-1, -2), AG_proj) / g_cols
        trace_est = trace_est + trace_G

    return trace_est

def simple_hutchplusplus(A: torch.Tensor, num_queries: int,  sketchMethod= 'signs') -> torch.Tensor: # https://github.com/RaphaelArkadyMeyerNYU/HutchPlusPlus/blob/main/simple/simple_hutchplusplus.m
    """
    Hutch++ trace estimator for a square matrix A.

    Parameters:
    - A (torch.Tensor): Square matrix (n x n)
    - num_queries (int): Total number of matrix-vector products to use

    Returns:
    - trace_est (torch.Tensor): Estimated trace of A
    """
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
    Q, _ = torch.linalg.qr(A @ S, mode='reduced')

    # Project G orthogonal to Q
    G_proj = G - Q @ (Q.T @ G)

    # Hutch++ estimator
    trace_est = torch.einsum('nm,nm->', Q, (A @ Q))  #  torch.trace(Q.T @ (A @ Q))
    if g_cols > 0:
        trace_est += (1.0 / g_cols) * torch.einsum('nm,nm->', G_proj, (A @ G_proj)) # instead of torch.trace(G_proj.T @ (A @ G_proj)) # 

    return trace_est

def hutch(A, num_queries: int, sketchMethod='signs', debugging_mode=False) -> torch.Tensor:
    """
    Hutchinson trace estimator for a square matrix A.

    Parameters:
    - A (torch.Tensor): Square matrix (n x n)
    - num_queries (int): Number of matrix-vector products to use
    - sketchMethod (str): Method to generate random vectors ('signs', 'rademacher', or 'gaussian')
    - debugging_mode (bool): If True, sets torch.manual_seed(0) for reproducibility

    Returns:
    - trace_est (torch.Tensor): Estimated trace of A
    """
    if debugging_mode:
        torch.manual_seed(0)

    n = A.shape[-1]
    assert A.shape[-1] == A.shape[-2], "Matrix A must be square."
    dtype = A.dtype
    device = A.device

    # Generate random sketching matrix
    if sketchMethod in ['rademacher', 'signs']:
        Omega = -1.0 + 2 * torch.randint(0, 2, (n, num_queries), dtype=torch.int8, device=device).to(dtype)
    elif sketchMethod == 'gaussian':
        Omega = torch.randn(n, num_queries, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown sketchMethod '{sketchMethod}'. Use 'signs', 'rademacher', or 'gaussian'.")

    trace_est = torch.einsum('nm,nm->', Omega, (A @ Omega)) / num_queries
    return trace_est

def batched_hutch(A, num_queries: int, sketchMethod='signs', debugging_mode=False) -> torch.Tensor:
    """
    Hutchinson trace estimator for a square matrix A.

    Parameters:
    - A (torch.Tensor): Square matrix (n x n)
    - num_queries (int): Number of matrix-vector products to use
    - sketchMethod (str): Method to generate random vectors ('signs', 'rademacher', or 'gaussian')
    - debugging_mode (bool): If True, sets torch.manual_seed(0) for reproducibility

    Returns:
    - trace_est (torch.Tensor): Estimated trace of A
    """
    if debugging_mode:
        torch.manual_seed(0)

    n = A.shape[-1]
    assert A.shape[-1] == A.shape[-2], "Matrix A must be square."
    dtype = A.dtype
    device = A.device

    # Generate random sketching matrix
    if sketchMethod in ['rademacher', 'signs']:
        Omega = -1.0 + 2 * torch.randint(0, 2, (n, num_queries), dtype=torch.int8, device=device).to(dtype)
    elif sketchMethod == 'gaussian':
        Omega = torch.randn(n, num_queries, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown sketchMethod '{sketchMethod}'. Use 'signs', 'rademacher', or 'gaussian'.")

    trace_est = torch.einsum('nm,nm->', Omega, (A @ Omega)) / num_queries
    return trace_est


def batched_hutch_from_action(Omega: torch.Tensor, AOmega: torch.Tensor) -> torch.Tensor:
    """
    Hutchinson trace estimator from a given random sketch matrix Omega and matrix-vector products A @ Omega.

    Parameters:
    - Omega (torch.Tensor): Random sketching matrix of shape (n, q)
    - AOmega (torch.Tensor): Result of applying the matrix A to Omega, also of shape (n, q)

    Returns:
    - trace_est (torch.Tensor): Estimated trace of A
    """
    return torch.einsum('ni,ni->', AOmega, Omega) / Omega.shape[1]





def batched_hutchplusplus2(A: torch.Tensor, num_queries: int, sketchMethod= 'signs',  debugging_mode=False) -> torch.Tensor:
    """
    Batched Hutch++ trace estimator for square matrices.

    Parameters:
    - A (torch.Tensor): Batched square matrices (b x n x n)
    - num_queries (int): Total number of matrix-vector products to use

    Returns:
    - trace_est (torch.Tensor): Estimated trace for each batch (b,)
    """
    if debugging_mode:
        torch.manual_seed(0)

    #assert A.dim() == 3 and A.shape[1] == A.shape[2], "A must be a batch of square matrices."
    #print("A.shape", A.shape)
    B, C, H, W = A.shape
    
    s_cols = math.ceil(num_queries / 3.0)
    g_cols = math.floor(num_queries / 3.0)

    dtype = A.dtype
    device = A.device

    # Random ±1 entries
    if sketchMethod in ['rademacher', 'signs']:
        # Generate random ±1 entries (Rademacher)
        S = -1.0 + 2.0 * torch.randint(0, 2, (B, s_cols, C,H, W), dtype=torch.int8, device=device)
        S = S.to(dtype)

        G = -1.0 + 2.0 * torch.randint(0, 2, (B, g_cols, C, H, W), dtype=torch.int8, device=device)
        G = G.to(dtype)

    elif sketchMethod == 'gaussian':
    # Gaussian random projections for conv-shaped inputs
        S = torch.randn(B, s_cols, C, H, W, device=device, dtype=dtype)
        G = torch.randn(B, g_cols, C, H, W, device=device, dtype=dtype)



    AS = A@S #torch.bmm(A, S)  # (b, n, s_cols)  # TODO change it to call A(S) A@ S #

    # print("AS.shape", AS.shape)
    #Q, _ = torch.linalg.qr(AS,  mode='reduced')  # (b, n, s_cols)
    Q, _ = torch.linalg.qr(AS.transpose(-1, -2), mode='reduced') # [B, CHW, s_cols])
    # print("Q.shape", Q.shape)
    # print("Q.shape", Q.transpose(-1, -2).shape)
    G = G.view(B,  -1, g_cols)
    # portions of G orthogonal to Q 
    QtG = torch.bmm(Q.transpose(1, 2), G)  # (b, s_cols, g_cols) # Akin to  Q^T @ G but for each slice or batch 
    G = G - torch.bmm(Q, QtG)              # (b, n, g_cols)      # Akin to (I- QQ^T)G   
         # (b, n, g_cols)      # Akin to (I- QQ^T)G
    AQ = A@(Q.reshape(B,s_cols, C, H, W)) #torch.bmm(A, Q)       #A@Q        # (b, n, s_cols) # TODO change to call A(Q) 
    #print("AQ.shape", AQ.shape)
    trace_Q = torch.einsum('bij,bij->b', Q.transpose(-1, -2), AQ)  #  tr(Q^T A Q)  batch-wise (b, )

    trace_est = trace_Q
    if g_cols > 0: # Einstein notation, not much faster 
        AG_proj = A@(G.view(B, g_cols, C, H, W))   
        # print("shape AG", AG_proj.shape)
        # print("shape G", G.shape)                       # (b, n, g_cols)
        trace_G = torch.einsum( 'bij,bij->b', G.transpose(-1, -2), AG_proj)/ g_cols  # tr( G^T(I-QQ^T) A (I-QQ^T)G)  shape: (b, )
        trace_est = trace_est + trace_G

    return trace_est  # shape (b,)



def test_batched_hutchplusplus():
    print("\n--- Testing Hutch++: Regular vs Batched Version ---\n\n")
    
    # Settings
    b, n, num_queries = 30, 1000, 150
    dtype = torch.float64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Batch size (b):         {b}")
    print(f"Matrix size (n x n):    {n} x {n}")
    print(f"# Hutch++ queries:      {num_queries}")
    print(f"Device:                 {device}")
    print(f"Dtype:                  {dtype}")

    # Generate batch of SPD matrices
    A_batch = torch.randn(b, n, n, dtype=dtype, device=device)
    A_batch = torch.bmm(A_batch, A_batch.transpose(1, 2))  # Make SPD

    # Compute reference trace estimates (non-batched)
    start = time.time()
    ref = torch.stack([simple_hutchplusplus(A_batch[i], num_queries) for i in range(b)])
    t_ref = time.time() - start

    # Compute batched trace estimates
    start = time.time()
    batched = batched_hutchplusplus(A_batch, num_queries)
    t_batched = time.time() - start

    # Ground truth traces
    true_trace = torch.tensor([torch.trace(A_batch[i]) for i in range(b)], dtype=dtype, device=device)

    # Errors
    rel_error_ref = torch.norm(ref - true_trace) / torch.norm(true_trace)
    rel_error_batched = torch.norm(batched - true_trace) / torch.norm(true_trace)
    speedup = t_ref / t_batched if t_batched > 0 else float('inf')

    # Results
    print(f"\n--- Timing ---")
    print(f"Runtime (Reference)        : {t_ref:.6f} s")
    print(f"Runtime (Batched)          : {t_batched:.6f} s")
    print(f"Speedup (Ref / Batched)    : {speedup:.2f}x")

    print(f"\n--- Accuracy ---")
    print(f"Relative Error (Reference) : {rel_error_ref:.2e}")
    print(f"Relative Error (Batched)   : {rel_error_batched:.2e}")
    
    print("\n--- End of Hutch++ Test ---\n")




#------------------- Xtrace ---------------------# https://github.com/eepperly/XTrace/tree/main

def diag_prod(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute diag(A^T @ B) efficiently."""
    return torch.sum(A * B, dim=0) # TODO check dim=0

def cnormc(A: torch.Tensor, dim =0) -> torch.Tensor:
    """Column normalize a matrix."""
    return A / torch.norm(A, dim=dim, keepdim=True)

def rand_with_evals(evals)-> torch.Tensor:
    """
    Generates a random symmetric matrix with the given eigenvalues.
    Args:
        evals (torch.Tensor): A 1D tensor of real eigenvalues.
    Returns:
        torch.Tensor: A symmetric matrix with the specified eigenvalues.
    """
    n = evals.size(0)
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    A = Q @ torch.diag(evals) @ Q.T
    return (A + A.T) / 2

def xtrace_helper(Om: torch.Tensor, Z: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, improved: bool):
    """
    Translation of the MATLAB function xtrace_helper into PyTorch.

    Args:
        Om (torch.Tensor): Omega matrix (n x m)
        Z (torch.Tensor) : Matrix Z (n x p)
        Q (torch.Tensor) : Orthonormal basis (n x m)
        R (torch.Tensor) : Matrix (m x m)
        improved (bool)  : Whether to use the improved estimator

    Returns:
        (float, float): Estimated trace and standard error
    """
    n, m = Om.shape
    W = Q.T @ Om
    I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    R_inv_T = torch.linalg.lstsq(R,I, driver='gels').solution.T #  TODO try using a SVD on A@Q  driver ‘gesvd’
    #R_inv_T = torch.linalg.pinv(R).T  # pseudo-inverse for numerical stability. Issues for float32 and fast exponential decay. 
    S = cnormc(R_inv_T)

    if improved:
        scale = (n - m + 1) / (n - torch.norm(W, dim=0) ** 2 + (diag_prod(S, W) * torch.norm(S, dim=0)) ** 2) # torch.abs
    else:
        scale = torch.ones(m, device=Om.device, dtype=Om.dtype)

    # Quantities needed for trace estimation

    H  = Q.T @ Z
    HW =   H @ W
    T  = Z.T @ Om

    dSW    =  diag_prod(S, W) 
    dSHS   =  diag_prod(S, H @ S)
    dTW    =  diag_prod(T, W)
    dWHW   =  diag_prod(W, HW)
    dSRmHW =  diag_prod(S, R - HW)
    dTmHRS =  diag_prod(T - H.T @ W, S)

    ests = (torch.trace(H) #* torch.ones(m, device=Om.device, dtype=Om.dtype)  # this is done using  broadcasting
            - dSHS
            + (-dTW + dWHW
               + torch.conj(dSW) * dSRmHW
               + torch.abs(dSW) ** 2 * dSHS
               + dTmHRS * dSW) * scale)

    tr = torch.mean(ests).item()
    err = torch.std(ests).item() / (m ** 0.5)

    return tr, err

def xnystrace_helper(Y: torch.Tensor, Om: torch.Tensor, improved: bool):
    """
    Nyström-based trace estimation in PyTorch.

    Args:
        Y (torch.Tensor): n x m matrix.
        Om (torch.Tensor): n x m test matrix (e.g., Gaussian).
        improved (bool): Whether to use the improved normalization scheme.

    Returns:
        t (float): Estimated trace.
        err (float): Standard error of the estimate.
    """
    n, m = Y.shape
    I = torch.eye(m, device=Y.device, dtype=Y.dtype) # (m,m) identity matrix 

    nu = torch.finfo(Y.dtype).eps * torch.norm(Y, 'fro') / (n**(0.5))
    Y += nu * Om  # Numerical stability shift

    Q, R = torch.linalg.qr(Y)  #  QR decomposition: torch.linalg.qr defaults to mode='reduced'
    
    #print("qr:", torch.dist(Q @ R, Y))
    H = Om.T @ Y # H = Omᵗ A Om

    # Symmetrize H before Cholesky
    #H_sym = 0.5 * (H + H.T) 
    #C = torch.linalg.cholesky(H_sym, upper=True)  # upper triangular
    C, info = torch.linalg.cholesky_ex(H, upper=True) 
    #print("chol:", torch.dist(C.T @ C, H_sym))

    # Solve B = R @ inv(C)
    #C_inv = torch.linalg.pinv(C)  # torch.linalg.lstsq(C,I, driver='gels').solution 
    #B = torch.linalg.solve_triangular(C.T, R.T, upper=False).T # R @ C_inv
    B = torch.linalg.solve(C, R, left=False)

    # Normalization
    if improved:
        QQ, RR = torch.linalg.qr(Om)  # QR of Om
        WW = QQ.T @ Om  
        RR_inv  =  torch.linalg.inv(RR)  # torch.linalg.lstsq(RR, I, driver='gels').solution # 
        SS = cnormc(RR_inv.T) 
        scale = (n - m + 1) / (n - torch.norm(WW, dim=0)**2 + (diag_prod(SS, WW))**2)
    else:
        scale = torch.ones(m, dtype=Y.dtype, device=Y.device)

    W = Q.T @ Om
    invH = torch.linalg.inv(H)  # torch.linalg.lstsq(H,I, driver='gels').solution 
    diagH = torch.diag(invH)**(-0.5) 

    S = (torch.linalg.solve(C.T, B, left=False) ) * diagH   #S = (torch.linalg.solve_triangular(C, B.T, upper=True).T)* diagH or S = (B @  C_inv.T ) * diagH
    dSW = diag_prod(S, W)
    ests = torch.norm(B, 'fro')**2 - torch.norm(S, dim=0)**2 + torch.abs(dSW)**2 * scale - nu * n

    trace = ests.mean().item()
    err = ests.std(unbiased=False).item() / torch.sqrt(torch.tensor(m, dtype=Y.dtype)).item()

    return trace, err



def xtrace_torch(A, m, sketchMethod= 'improved'):
    """
    Translates the MATLAB xtrace function into PyTorch.

    Args:
        A (torch.Tensor or callable): The matrix or a function representing matrix-vector multiplication.
        m (int): Number of test vectors 
        *varargin: Variable arguments passed to process_matrix and generate_test_matrix.

    Returns:
        tuple: (t, err) where
            t (torch.Tensor): Trace estimate.
            err (torch.Tensor): Error estimate.
    """
    n = A.shape[1]
    m_floored = m // 2
    dtype = A.dtype
    device = A.device
    Omega, improved = generate_test_matrix_torch(n, m_floored, sketchMethod= sketchMethod, dtype= dtype,device =device )
    Y = A @ Omega # m/2  matvecs 
    Q, R = torch.linalg.qr(Y, mode='reduced')
    Z = A @ Q    # m/2 matvecs
    t, err = xtrace_helper(Omega, Z, Q, R, improved)
    return t, err

def xnystrace_torch(A, m, sketchMethod='improved'):
    """
    Translates the MATLAB xtrace function into PyTorch.

    Args:
        A (torch.Tensor or callable): The matrix or a function representing matrix-vector multiplication.
        m (int): Number of test vectors 
        *varargin: Variable arguments passed to process_matrix and generate_test_matrix.

    Returns:
        tuple: (t, err) where
            t (torch.Tensor): Trace estimate.
            err (torch.Tensor): Error estimate.
    """
    n = A.shape[1]
    dtype = A.dtype
    device = A.device
    Omega, improved = generate_test_matrix_torch(n, m, sketchMethod= sketchMethod, dtype= dtype,device =device )
    Y = A @ Omega # m  matvecs 
    t, err = xnystrace_helper(Y, Omega, improved)
    return t, err



def NaiveXNysTrace(A, m,  sketchMethod='improved', dtype=torch.float32, device='cpu'): # for testing only 
    """
    Estimate trace of SPD matrix A using a naive Nyström-style method.

    Args:
        A (torch.Tensor): n x n symmetric positive definite matrix.
        m (int): Number of test vectors.
        dtype (torch.dtype): Floating point precision.
        device (str): 'cpu' or 'cuda'.

    Returns:
        trace_estimate (float): Estimated trace.
        trace_std (float): Standard error.
    """
    dtype = A.dtype
    device = A.device
    n = A.shape[0]
    Omega, _ = generate_test_matrix_torch(n, m, sketchMethod=sketchMethod, dtype=dtype, device=device)
    Y = A @ Omega

    trace_estimates = []

    for i in range(m):
        # Remove i-th column
        idx = [j for j in range(m) if j != i]
        Omega_i = Omega[:, idx]  # n x (m-1)
        Y_i = Y[:, idx]          # n x (m-1)
        omega_i = Omega[:, i]    # n

        # Compute pseudo-inverse safely
        try:
            M = Omega_i.T @ Y_i  # (m-1) x (m-1)
            #M_inv = torch.linalg.pinv(M)  # Regularized pseudoinverse
            #A_i = Y_i @ M_inv @ Y_i.T     # n x n
            A_i = Y_i @ (torch.linalg.lstsq(M, Y_i.T, driver='gels').solution)
        except RuntimeError:
            continue  # skip this iteration if inversion fails

        tr_Ai = torch.trace(A_i)
        r_i = omega_i @ ((A - A_i) @ omega_i)
        tr_i = tr_Ai + r_i
        trace_estimates.append(tr_i.item())

    trace_estimates = torch.tensor(trace_estimates, dtype=dtype)
    trace_mean = trace_estimates.mean().item()
    trace_std = trace_estimates.std(unbiased=True).item() / torch.sqrt(torch.tensor(len(trace_estimates), dtype=dtype)).item()
    
    return trace_mean, trace_std



def generate_test_matrix_torch(n, m,b=1, sketchMethod= 'improved', dtype= torch.float32, device='cpu'):
    """
    Torch version from generate_test_matrix function.

    Args:
        n (int): Dimension.
        m (int): Number of test vectors.s.
        b (int) batch size 

    Returns:
        tuple: (Om, improved) where
            Om (torch.Tensor): The generated test matrix (n x m).
            improved (bool): The boolean value of the 'improved' flag.
    """

    improved = False

    if sketchMethod== 'improved':
        improved = True

        Om = n**(0.5) * cnormc(torch.randn(b, n, m), dim = 1 ).to(dtype)

    elif sketchMethod in ['rademacher', 'signs']:

        Om = -1.0 + 2 * torch.randint(0, 2, (b, n, m)).to(dtype)

    elif sketchMethod == 'gaussian':
        Om = torch.randn(n, m).to(dtype) 

    elif sketchMethod in ['unif', 'sphere']:
        Om = n**(0.5) * cnormc(torch.randn(b, n, m), dim = 1).to(dtype) 
    
    elif sketchMethod == 'orth':
        Om, _, _ = torch.linalg.svd(torch.randn(b, n, m).to(dtype), full_matrices=False) # matlab uses orth(). TODO try driver 'gesvda'
        #Om, _ =torch.linalg.qr(torch.randn(n, m), mode='reduced')
        Om *= n**(0.5)        

    else:
        raise ValueError(f'"{sketchMethod}" not recognized as matrix type')         

    if b == 1:
        Om = Om.squeeze(0) # (1, n, m) to (n, m)

    return Om, improved

def trace_estimation_comparison_torch(n=1000, num_trials = 100, sketchMethod= 'improved', steps=15):
    """
    Compares trace estimation methods (Hutch++, Hutchinson, and XTrace) on synthetic matrices.

    Args:
        n (int, optional): Size of the matrices. Defaults to 1000.
        num_trials (int, optional): Number of trials for trace estimators. Defaults to 100.

    """
    As_torch = {}
    names = ['flat', 'poly', 'slowexp', 'fastexp', 'smallstep', 'bigstep']

    # 1. Flat eigenvalues
    As_torch[names[0]] = rand_with_evals(torch.linspace(1, 3, n))

    # 2. Polynomial decay eigenvalues
    evals_poly = 1/(torch.arange(1, n + 1) ** (2))
    As_torch[names[1]] = rand_with_evals(evals_poly)

    # 3. Slow exponential decay eigenvalues
    evals_slowexp = (0.9 ** torch.arange(0, n))
    As_torch[names[2]] = rand_with_evals(evals_slowexp)

    # 4. Fast exponential decay eigenvalues
    evals_fastexp = (0.7 ** torch.arange(0, n))
    As_torch[names[3]] = rand_with_evals(evals_fastexp)

    # 5. Small step eigenvalues
    num_ones = int(np.ceil(n / 20))
    evals_smallstep = torch.cat([torch.ones(num_ones), 1e-3 * torch.ones(n - num_ones)])
    As_torch[names[4]] = rand_with_evals(evals_smallstep)

    # 6. Big step eigenvalues
    evals_bigstep = torch.cat([torch.ones(num_ones), 1e-8 * torch.ones(n - num_ones)])
    As_torch[names[5]] = rand_with_evals(evals_bigstep)

    # 
    min_sketch_size = 18
    max_sketch_size = 5 * (n // 10)
    sketch_sizes = torch.linspace(min_sketch_size, max_sketch_size, steps=steps).ceil().int().unique()

    true_traces = {name: torch.trace(matrix).item() for name, matrix in As_torch.items()}

    relative_errors = {
        name: {'Hutch++': [], 'XTrace': [], 'Hutchinson': [], 'XNysTrace':[]}
        for name in As_torch.keys()
    }

    num_trials = num_trials

    # Perform estimations and calculate errors
    for name, matrix in As_torch.items():
        print(f"Processing matrix: {name} (True Trace: {true_traces[name]:.4f})")
        true_trace = true_traces[name]

        for sketch_size in sketch_sizes:
            #print(f"  Sketch size: {sketch_size}")
            
            errors_hutchpp   = []
            errors_xtrace    = []
            errors_hutch     = []
            errors_xnystrace = []

            for _ in range(num_trials):
                # --- Hutch ---
                estimated_trace_hutch = hutch(matrix, sketch_size)
                error_hutch = abs(estimated_trace_hutch - true_trace)
                rel_error_hutch = error_hutch / abs(true_trace) #if abs(true_trace) > 1e-13 else error_hutchpp
                errors_hutch.append(rel_error_hutch)

                # --- Hutch++ ---
                estimated_trace_hutchpp = simple_hutchplusplus(matrix, sketch_size)
                error_hutchpp = abs(estimated_trace_hutchpp - true_trace)
                rel_error_hutchpp = error_hutchpp / abs(true_trace) #if abs(true_trace) > 1e-13 else error_hutchpp
                errors_hutchpp.append(rel_error_hutchpp)

                # --- XTrace ---
                estimated_trace_xtrace, _ = xtrace_torch(matrix, sketch_size, sketchMethod=sketchMethod)
                error_xtrace = abs(estimated_trace_xtrace - true_trace)
                rel_error_xtrace = error_xtrace / abs(true_trace) #if abs(true_trace) > 1e-13 else error_xtrace
                errors_xtrace.append(rel_error_xtrace)

                # --- Xnystrace ---
                estimated_trace_xnystrace, _ = xnystrace_torch(matrix, sketch_size, sketchMethod=sketchMethod)
                error_xnystrace= abs(estimated_trace_xnystrace - true_trace)
                rel_error_xnystrace = error_xnystrace / abs(true_trace) #if abs(true_trace) > 1e-13 else error_xtrace
                errors_xnystrace.append(rel_error_xnystrace)

            # Store the mean relative errors
            relative_errors[name]['Hutch++'].append(torch.tensor(errors_hutchpp).mean().item())
            relative_errors[name]['XTrace'].append(torch.tensor(errors_xtrace).mean().item())
            relative_errors[name]['XNysTrace'].append(torch.tensor(errors_xnystrace).mean().item())
            relative_errors[name]['Hutchinson'].append(torch.tensor(errors_hutch).mean().item())


    #Plotting the results
    print("\nPlotting results...")
    num_matrices = len(As_torch)
    num_cols = 2 # Number of columns in the plot grid
    num_rows = (num_matrices + num_cols - 1) // num_cols # Calculate rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 5))
    axes = axes.flatten() # Flatten the 2D array of axes for easy indexing

    for i, name in enumerate(As_torch.keys()):
        ax = axes[i]
        ax.plot(sketch_sizes, relative_errors[name]['Hutchinson'], label='Hutchinson', marker='o')
        ax.plot(sketch_sizes, relative_errors[name]['Hutch++'], label='Hutch++', marker='*')
        ax.plot(sketch_sizes, relative_errors[name]['XTrace'], label='XTrace', marker='x')
        ax.plot(sketch_sizes, relative_errors[name]['XNysTrace'], label='XNysTrace', marker='^')

        ax.set_xlabel('Sketch Size (m)')
        ax.set_ylabel('Relative Error')
        ax.set_title(f'Relative Error vs. Sketch Size for "{name}" Matrix')
        ax.legend(loc='lower left', framealpha=0.5) 
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        # Use log scale for y-axis as errors often span orders of magnitude
        ax.set_yscale('log')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('TraceEstimationComparison.png', dpi=300)
    plt.show()

    print("Comparison complete.")        

    return 

if __name__ == "__main__":
    import torch # type: ignore
    import time
    torch.manual_seed(0)
    #trace_estimation_comparison_torch(n=1000, num_trials=100)
    test_batched_hutchplusplus()
