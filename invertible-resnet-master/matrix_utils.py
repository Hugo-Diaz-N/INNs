"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import numpy as np
import torch # type: ignore
torch.set_default_dtype(torch.float64)
import time
import math
from torch.autograd.functional import vjp, jvp # type: ignore
from scipy.linalg import logm
#from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn # type: ignore
import matplotlib.pyplot as plt
from torch.linalg import norm

from Jacobian2Matrix import JacobianLinearOperator
from Jacobian2Matrix import ConvJacobianOperator, ConvJacobianLinearOperator2,  flatten_tensor
from random_utils_nn import hutch, batched_hutchplusplus, batched_hutchplusplus2, sample_sketch_tensor

from chebyshev_utils import  conv_chebyshev_series_action_from_function, ChebMatvecOperator

from linear_operator.operators import LinearOperator

class MatvecOperator(LinearOperator):
    """
    Create a LinearOperator representing M @ v = (I + A^T)(I + A) @ v = v + A@v +A^T @ v + A^T(A@v)
    """
    def __init__(self, A):
        super().__init__(dtype=A.dtype, device=A.device)  # or use .device of A_mv
        self.A = A
        self._shape = A.shape[1:] # A.shape = (B, d,d)

    def __matmul__(self, x):
        # Compute (I + Aᵗ)(I + A)x = x + A @ x + Aᵗ @ x + Aᵗ @ (A @ x)
        xplusAx = x + self.A @ x
        z = xplusAx + self.A.T@(xplusAx) # .transpose(1, 2) 
        return z

    def _size(self):
        return self._shape
    

# Instead of using zero_gradients, use this to manually zero gradients
def zero_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

# def zero_gradients(x): # https://discuss.pytorch.org/t/from-torch-autograd-gradcheck-import-zero-gradients/127462/2
#     if isinstance(x, torch.Tensor):
#         if x.grad is not None:
#             x.grad.detach_()
#             x.grad.zero_()
#     elif isinstance(x, collections.abc.Iterable):
#         for elem in x:
#             zero_gradients(elem)

            
# def exact_matrix_logarithm_trace(Fx, x):
#     """
#     Computes slow-ass Tr(Ln(d(Fx)/dx))
#     :param Fx: output of f(x)
#     :param x: input
#     :return: Tr(Ln(I + df/dx))
#     """
#     bs = Fx.size(0)
#     outVector = torch.sum(Fx, 0).view(-1)
#     outdim = outVector.size()[0]
#     indim = x.view(bs, -1).size()
#     jac = torch.empty([bs, outdim, indim[1]], dtype=torch.float)
#     # for each output Fx[i] compute d(Fx[i])/d(x)
#     for i in range(outdim):
#         zero_gradients(x)
#         jac[:, i, :] = torch.autograd.grad(outVector[i], x,
#                                            retain_graph=True)[0].view(bs, outdim)
#     jac = jac.cpu().numpy()
#     iden = np.eye(jac.shape[1])
#     log_jac = np.stack([logm(jac[i] + iden) for i in range(bs)])
#     trace_jac = np.diagonal(log_jac, axis1=1, axis2=2).sum(1)
#     return trace_jac




def exact_matrix_logarithm_trace(f, x):
    """
    Computes Tr(log(I + df/dx)) where Fx = f(x).
    Slow and memory-intensive.
    """
    f.eval() 
    Fx = f(x)
    bs = Fx.size(0)
    outVector = torch.sum(Fx, 0).view(-1)
    outdim = outVector.size(0)
    indim = x.view(bs, -1).size(1)
    assert outdim == indim, "Function must be square to compute log-determinant"

    #print("Exact alg: shape x", x.shape)
    device = x.device
    jac = torch.empty([bs, outdim, indim], dtype=torch.float, device=device)

    for i in range(outdim): # there is a function for this: torch.autograd.functional.jacobian
        grad_outputs = torch.zeros_like(outVector)
        grad_outputs[i] = 1.0
        grad_i = torch.autograd.grad(outVector, x, grad_outputs=grad_outputs, # i-th component of output, "derivated" wrt x \in (bs, d_in)
                                     retain_graph=True, create_graph=False)[0]
        jac[:, i, :] = grad_i.view(bs, indim)

    jac = jac.detach()#.cpu()#.numpy()
    #print("jac",  jac.shape)
    iden = np.eye(jac.shape[1])
    #print("Batch size", bs) # torch.logdet
    # log_jac = np.stack([logm(jac[i] + iden,disp=False)[0] for i in range(bs)])
    # trace_jac = np.trace(log_jac, axis1=1, axis2=2)
    logDet_jac = np.stack([torch.logdet(jac[i] + iden) for i in range(bs)])
    #print("logDet_jac shape", logDet_jac.shape)
    return torch.from_numpy(logDet_jac).to(x.device)


def power_series_full_jac_exact_trace(Fx, x, k):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation with full
    jacobian and exact trace
    
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :return: Tr(Ln(I + df/dx))
    """
    _, jac = compute_log_det(x, Fx)
    jacPower = jac
    summand = torch.zeros_like(jacPower)
    for i in range(1, k+1):
        if i > 1:
            jacPower = torch.matmul(jacPower, jac)
        if (i + 1) % 2 == 0:
            summand += jacPower / (float(i))
        else: 
            summand -= jacPower / (float(i)) 
    trace = torch.diagonal(summand, dim1=1, dim2=2).sum(1)
    return trace

def power_series_matrix_logarithm_trace(Fx, x, k, n, seed=None):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation
    biased but fast
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :param n: number of Hitchinson's estimator samples
    :seed (int, optional): Random seed for reproducibility
    :return: Tr(Ln(I + df/dx))
    """
    if seed is not None:
        torch.manual_seed(seed)
    # trace estimation including power series
    #f.eval()
    #Fx = f(x)  # Shape: (B, C_out, H, W)
    outSum = Fx.sum(dim=0)
    dim = list(outSum.shape)
    dim.insert(0, n)
    dim.insert(0, x.size(0))
    u = torch.randn(dim).to(x.device)
    trLn = 0
    for j in range(1, k + 1):
        if j == 1:
            vectors = u
        # compute vector-jacobian product
        vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                       retain_graph=True, create_graph=True)[0] for i in range(n)]
        # compute summand
        vectors = torch.stack(vectors, dim=1)
        vjp4D = vectors.view(x.size(0), n, 1, -1)
        u4D = u.view(x.size(0), n, -1, 1)
        summand = torch.matmul(vjp4D, u4D)
        # add summand to power series
        if (j + 1) % 2 == 0:
            trLn += summand / float(j)
        else:
            trLn -= summand / float(j)
    trace = trLn.mean(dim=1).squeeze()
    return trace


def power_series_matrix_logarithm_trace_JT(func, x, k, n, sketch='gaussian', sigma=0.8, seed=None):
    """
    Approximates Tr(log(I + J_f(x))) using a stochastic trace estimator
    based on the power series expansion of log(I + A).

    log(I + A) = A - A^2/2 + A^3/3 - ... for ||A|| < 1

    This is useful for estimating log-determinants of Jacobians of convolutional networks.

    Args:
        func: Callable neural network module, maps x to f(x)
        x (Tensor): Input tensor of shape (B, C_in, H, W)
        k (int): Number of terms in the truncated power series
        n (int): Number of Hutchinson samples for trace estimation
        sketch (str): 'gaussian' (default) or 'rademacher' for random probe distribution
        seed (int, optional): Random seed for reproducibility

    Returns:
        Tensor of shape (B,) with trace estimates per batch sample.
    """
    if seed is not None:
        torch.manual_seed(seed)

    func.eval()
    B = x.size(0)
    Fx = func(x)                  # Output: (B, C_out, H, W)
    output_shape = Fx.shape[1:]   # (C_out, H, W)
    # Sample sketch tensor u of shape (B, n, *output_shape)
    u = sample_sketch_tensor((B, n, *output_shape), device=x.device, dtype=x.dtype, method=sketch)

    # Create Jacobian linear operator object
    J_x = ConvJacobianOperator(func, x)

    # Power series estimation
    v = u.clone()
    u = u.view( x.size(0), n, -1)
    trace_est = 0.0
 
    for j in range(1, k + 1):
        v = J_x.T @ v  # Shape: (B, n, C_in* H* W)
        inner = torch.einsum('bnd,bnd->bn', u, v)  # Inner product over all but first two dims → (B, n)
        #v = v.reshape(B ,n,*output_shape)
        coeff = 1.0 / j
        if (j + 1) % 2 != 0:  # if j is odd: +, else: -
            coeff *= -1

        trace_est += coeff * inner

    # Average over samples
    trace_est = trace_est.mean(dim=1)  # (B,)
    return trace_est


def Chebyshev_matrix_logarithm_trace_JT(func, x, k, n, sketch='gaussian', sigma=0.8, seed=None):
    """
    Approximates Tr(log(I + J_f(x))) using a stochastic trace estimator
    based on the Chebyshev polynomial expansion of log(I + A), where A = J_f(x)
    is the Jacobian of the function `func` evaluated at `x`.

    This method assumes that ||A|| < 1 so that the Chebyshev expansion converges.

    Args:
        func (Callable): Neural network module mapping x to f(x)
        x (Tensor): Input tensor of shape (B, C_in, H, W)
        k (int): Number of Chebyshev polynomial terms
        n (int): Number of Hutchinson samples for stochastic trace estimation
        sketch (str): Distribution for random probe vectors; either 'gaussian' or 'rademacher'
        sigma (float): Spectral radius estimate of A, must satisfy 0 < sigma < 1
        seed (int, optional): Random seed for reproducibility

    Returns:
        Tensor: Trace estimate of shape (B,)
    """
    assert 0 < sigma < 1, "sigma must be strictly between 0 and 1 to ensure convergence"

    if seed is not None:
        torch.manual_seed(seed)

    a = (1 - sigma) ** 2
    b = (1 + sigma) ** 2

    func.eval()

    # Defines the log-scaling function for the Chebyshev domain transformation
    f = lambda x: np.log(((b-a)/2)*x + (b+a)/2)

    #f =  np.log1p # log(1+x)

    # Output shape is required to define the sketch tensor
    Fx = func(x)  # Output: (B, C_out, H, W)
    output_shape = Fx.shape[1:]
    B = x.size(0)
    # Sample sketch tensor u of shape (B, n, *output_shape)
    Omega = sample_sketch_tensor((B, n, *output_shape), device=x.device, dtype=x.dtype, method=sketch) 
    Omega =  flatten_tensor(Omega) # same as Omega.view(B, n, -1)  # Flatten the last dimensions for matrix multiplication   
    #print("shape u", Omega.shape)
    J_x = ConvJacobianOperator(func, x)  # Must support batched transpose application
    M = ChebMatvecOperator(J_x, a=a, b=b) # (I+J^T)(I+J)
    #print("Jx shape:",J_x.shape)
    pJOmega = conv_chebyshev_series_action_from_function(M, f, k, Omega)
    #print("p(J)Omega shape ", pJOmega.shape)

    trace_est = 0.5* torch.einsum('bnd,bnd->bn', Omega.view(x.size(0), n, -1), pJOmega)


    return trace_est.mean(dim=1)  # (B,)



# from functorch import vjp, vmap # try 

# def f(x):
#     return Fx  # again, needs the actual function, not precomputed Fx

# # Get vjp function
# vjp_fn = vjp(f, x)[1]

# # vmap the vjp_fn over the batch of vectors
# vectors = vmap(vjp_fn)(vectors.mT)  # careful with dimensions


def power_series_matrix_logarithm_trace_hutchpp(Fx, x, k, num_queries =15):
    """
    Estimates trace(log(I + J)) using Hutch++ and VJPs of J = df/dx.

    Parameters:
    Fx : torch.Tensor
        Output of f(x), shape (B, C, H, W), where x requires grad.
    x : torch.Tensor
        Input tensor with requires_grad=True.
    k : int
        Number of terms in the power series expansion.
    num_queries : int
        Number of Hutch++ samples (matrix-vector products).

    Returns:
    trLn : torch.Tensor
        Estimated trace per batch element (shape B,).
    """
    B = x.shape[0]
    d = x.view(B, -1).shape[1]  # input dimension per sample
    print("Random Approach")
    print("shape Fx:",Fx.shape)
    print("shape x:",x.shape)
    trLn = torch.zeros(B, device=x.device)

    # s_cols = int(torch.ceil(torch.tensor(n / 3.0)).item())
    # g_cols = int(torch.floor(torch.tensor(n / 3.0)).item())
    s_cols = math.ceil(num_queries / 3.0)
    g_cols = math.floor(num_queries / 3.0)

    S = 2 * torch.randint(0, 2, (B, s_cols, *Fx.shape[1:]), dtype=Fx.dtype, device=Fx.device) - 1
    G = 2 * torch.randint(0, 2, (B, g_cols, *Fx.shape[1:]), dtype=Fx.dtype, device=Fx.device) - 1

    def vjp(vectors, Bsize=B):
        grads = []
        for i in range(vectors.shape[1]): # number of samples 
            grad_i = torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                         retain_graph=True, create_graph=True)[0]
            grads.append(grad_i.view(Bsize, -1))
        return torch.stack(grads, dim=1)  # (B, n, d)

    vectors_s = vjp(S)
    vectors_g = vjp(G)
    #print("shape vjp(S): ",vectors_s.shape)
    for j in range(1, k + 1): 
        if j > 1:
            # Reshape vectors to match Fx shape
            vectors_s_reshaped = vectors_s.view(B, s_cols, *Fx.shape[1:])
            vectors_g_reshaped = vectors_g.view(B, g_cols, *Fx.shape[1:])
            vectors_s = vjp(vectors_s_reshaped)
            vectors_g = vjp(vectors_g_reshaped)

        # QR on vectors_s[b].T ∈ ℝ^{d × s_cols}
        Q = torch.stack([torch.linalg.qr(vectors_s[b].T, mode='reduced')[0] for b in range(B)], dim=0)  # (B, d, s)

        # Project G onto orthogonal complement of Q
        G_flat = G.view(B, g_cols, -1).transpose(1, 2)   # (B, d, g)
        QtG = torch.matmul(Q.transpose(1, 2), G_flat)    # (B, s, g)
        QQtG = torch.matmul(Q, QtG)                      # (B, d, g)
        G_proj = G_flat - QQtG                           # (B, d, g)

        # Trace from Q block: tr(Qᵀ A Q)
        vectors = Q.transpose(1, 2).view(B, s_cols, *Fx.shape[1:])
        VJ = vjp(vectors)    # A*Q                           # (B, s, d)
        trQ = torch.sum(Q.transpose(1, 2) * VJ, dim=(1, 2))  # (B,)
        # Trace from projected G block: (1/g) tr(G_proj^T A G_proj)
        if g_cols > 0:
            G_proj_t = G_proj.transpose(1, 2)            # (B, g, d)
            VJg = vjp(G_proj_t.view(B, g_cols, *Fx.shape[1:]))  # (B, g, d)
            trG = torch.sum(G_proj_t * VJg, dim=(1, 2)) / g_cols  # (B,)
        else:
            print("g_cols=0, setting Tr(G)=0")
            trG = torch.zeros_like(trQ)
        trace_j = trQ + trG

        # Power series for log(1 + J): alternating series
        trLn = trLn + ((-1)**(j+1)) * trace_j / j

    return trLn



def compute_log_det(inputs, outputs):
    log_det_fn = log_det_2x2 if inputs.size()[1] == 2 else log_det_other
    batch_size = outputs.size(0)
    outVector = torch.sum(outputs,0).view(-1)
    outdim = outVector.size()[0]
    jac = torch.stack([torch.autograd.grad(outVector[i], inputs,
                                     retain_graph=True, create_graph=True)[0].view(batch_size, outdim) for i in range(outdim)], dim=1)
    log_det = torch.stack([log_det_fn(jac[i,:,:]) for i in range(batch_size)], dim=0)
    return log_det, jac


def log_det_2x2(x):
    return torch.log(x[0,0]*x[1,1]-x[0,1]*x[1,0])


def log_det_other(x):
    return torch.logdet(x)


def weak_bound(sigma, d, n_terms):
    """
    Returns a bound on |Tr(Ln(A)) - PowerSeries(A, n_terms)|
    :param sigma: lipschitz constant of block
    :param d: dimension of data
    :param n_terms: number of terms in our estimate
    :return:
    """
    inf_sum = -np.log(1. - sigma)
    fin_sum = 0.
    for k in range(1, n_terms + 1):
        fin_sum += (sigma**k) / k

    return d * (inf_sum - fin_sum)





# def chebyshev_coeffs(f, k=3):
#     """
#     Compute the first (k+1) Chebyshev coefficients for a function f(x)
#     on the interval [-1, 1] using Clenshaw-Curtis quadrature.

#     Args:
#         f (callable): Function to approximate on [-1, 1].
#         k (int): Highest degree Chebyshev coefficient to compute.

#     Returns:
#         coeffs (np.ndarray): Array of shape (k+1,) with Chebyshev coefficients.
#     """
#     n_quad = k+1 # number of quadrature points
#     theta = np.pi * (np.arange(n_quad) + 0.5) / n_quad
#     x = np.cos(theta)
#     fx = f(x)

#     coeffs = np.zeros(k + 1)
#     for j in range(k + 1):
#         Tj = np.cos(j * theta)
#         coeffs[j] = (2 / n_quad) * np.sum(fx * Tj)

#     coeffs[0] /= 2
#     return coeffs

# def chebyshev_coeffs(f, degree=3, dt=0.0001): # original code: https://github.com/erhallma/erhallma.github.io
#     """
#         Compute the first (k+1) Chebyshev coefficients for a function f(x)
#         on the interval [-1, 1] 
#         Args:
#             f (callable): Function to approximate on [-1, 1].
#             k (int): Highest degree Chebyshev coefficient to compute.

#         Returns:
#             coeffs (np.ndarray): Array of shape (k+1,) with Chebyshev coefficients.
#     """
#     ts = torch.arange(0, torch.pi, dt)
#     x = torch.cos(ts)
#     fx = f(x)
    
#     coeffs = torch.zeros(degree + 1)
#     for k in range(degree + 1):
#         integrand = fx * torch.cos(k * ts)
#         coeffs[k] = (2 / torch.pi) * torch.trapezoid(integrand, dx=dt)
#     coeffs[0] /= 2
#     return coeffs.numpy()






# def chebyshev_series_action_from_function(A: torch.Tensor, f, k: int, Omega: torch.Tensor) -> torch.Tensor:
#     """
#     Computes p(A) @ Omega where p(x) is the Chebyshev approximation of f(x)
#     of degree k on [-1, 1].

#     Args:
#         A (torch.Tensor): Square matrix of shape (n, n).
#         f (Callable): Function to approximate.
#         k (int): Degree of the Chebyshev polynomial.
#         Omega (torch.Tensor): Matrix of shape (n, m).

#     Returns:
#         torch.Tensor: Result of applying the Chebyshev approximation p(A) to Omega.
#     """
#     coeffs = chebyshev_coeffs(f, k)  # This should return a 1D torch.Tensor or list of floats
#     K = len(coeffs)

#     assert A.shape[0] == A.shape[1],     "Matrix A must be square"
#     assert A.shape[1] == Omega.shape[0], "Matrix A and Omega must align for multiplication"

#     T0 = Omega.clone()
#     result = coeffs[0] * T0

#     if K == 1:
#         return result

#     T1 = A @ Omega
#     result = result + coeffs[1] * T1

#     for j in range(2, K):
#         Tj = 2 * A @ T1 - T0
#         result = result + coeffs[j] * Tj
#         T0, T1 = T1, Tj

#     return result



def trace_taylor_series_action(A: torch.Tensor, Omega: torch.Tensor, coeffs: list[float]) -> float:
    """
    Approximate trace(f(A)) using trace estimation with a test matrix Omega. f(A)= sum_k c_k A^k is a Taylor polynomial:
        trace(f(A)) ≈ (1/m) * trace(Omega^T @ (sum_k c_k A^k) @ Omega)

    Args:
        A (torch.Tensor): Square matrix of shape (n, n).
        Omega (torch.Tensor): Test matrix of shape (n, m).
        coeffs (list[float]): List of Taylor coefficients [c0, c1, ..., cK].

    Returns:
        float: Approximation of trace(f(A)).
    """
    assert A.shape[0] == A.shape[1], "A must be square"
    assert A.shape[1] == Omega.shape[0], "Omega must align with A for multiplication"

    K = len(coeffs)
    Ak_Omega = Omega.clone()
    result = coeffs[0] * Ak_Omega

    for k in range(1, K):
        Ak_Omega = A @ Ak_Omega
        result = result + coeffs[k] * Ak_Omega

    trace_est = torch.sum(Omega * result).item() / Omega.shape[1]
    return trace_est



if __name__ == "__main__":
    import torch # type: ignore
    import time




    # __all__ = ['expm', 'cosm', 'sinm', 'tanm', 'coshm', 'sinhm', 'tanhm', 'logm',
    #        'funm', 'signm', 'sqrtm', 'fractional_matrix_power', 'expm_frechet',
    #        'expm_cond', 'khatri_rao']



