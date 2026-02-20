import torch
import torch.nn as nn
import torch.func as func

def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flatten (B, n, C, H, W) into (B, n, d) where d = C * H * W.
    """
    B, n = tensor.shape[:2]
    return tensor.view(B, n, -1)

def unflatten_vector(vector: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Unflatten (B, n, d) into (B, n, C, H, W) given shape = (C, H, W).
    """
    B, n, d = vector.shape
    C, H, W = shape
    assert d == C * H * W, "Mismatch in unflatten shape"
    return vector.view(B, n, C, H, W)


class JacobianLinearOperator:
    def __init__(self, f, x):
        """
        Create a batched Jacobian operator for a function f at points x.

        Args:
            f: Callable, maps (d_in,) → (d_out,)
            x: torch.Tensor of shape (B, d_in)
        """
        self.f = f
        self.x = x.detach().requires_grad_(True)
        self.shape = (x.shape[0], f(x[0:1]).shape[1], x.shape[1])  # (B, d_out, d_in)
        self.dtype = x.dtype
        self.device = x.device

    def __matmul__(self, other):
        """
        Implements A @ Omega = J_f(x) @ Omega (A itself is a tensor. Each slice is: J(x) for x in Batch)

        other  : Tensor of shape (B, K, d_in) # Each slice is multiplied by K vectors (Omega). 
        returns: Tensor of shape (B, K, d_out)
        """
        return self.jvp(other)

    @property
    def T(self):
        return self.Transposed(self)

    def jvp(self, tangents):
        """
        J_f(x) @ [v1 ... v_K]
        tangents: Tensor of shape (B, K, d_in)
        returns : Tensor of shape (B, K, d_out)
        """
        B, K, d_in = tangents.shape
        x_exp = self.x.unsqueeze(1).expand(-1, K, -1).reshape(B * K, d_in)
        t_flat = tangents.reshape(B * K, d_in)
        _, jvp_out = func.jvp(self.f, (x_exp,), (t_flat,))
        return jvp_out.view(B, K, -1)

    def vjp(self, cotangents):
        """
        [v1 ... v_K].T @ J_f(x)
        cotangents: Tensor of shape (B, K, d_out)
        returns: Tensor of shape (B, K, d_in)
        """
        B, K, d_out = cotangents.shape
        d_in = self.x.shape[1]
        x_exp = self.x.unsqueeze(1).expand(-1, K, -1).reshape(B * K, d_in)
        v_flat = cotangents.reshape(B * K, d_out)
        _, vjp_fn = func.vjp(self.f, x_exp)
        vjp_out, = vjp_fn(v_flat)
        return vjp_out.view(B, K, d_in)

    class Transposed:
        def __init__(self, parent):
            self.parent = parent
            self.shape = (parent.shape[0], parent.shape[2], parent.shape[1])  # (B, d_in, d_out)
            self.dtype = parent.dtype
            self.device = parent.device

        def __matmul__(self, other):
            # Computes vᵢ @ J_f(xᵢ)
            return self.parent.vjp(other)

class ConvJacobianOperator:
    def __init__(self, f: nn.Module, x: torch.Tensor):
        """
        Efficient batched JVP and VJP for convolutional networks.
        
        Args:
            f (nn.Module): A convolutional network mapping (B, C, H, W) → (B, C', H', W').
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        """
        self.f = f
        self.x = x.detach().requires_grad_(True)
        self.in_shape = x.shape                      # (B, C, H, W)
        self.out_shape = f(x).shape                  # (B, C', H', W')

        self.B, self.C, self.H, self.W = self.in_shape
        self.Cout, self.Hout, self.Wout = self.out_shape[1:]

        self.shape = (self.B, self.Cout * self.Hout * self.Wout, self.C * self.H * self.W)
        self.device = x.device
        self.dtype = x.dtype

    def jvp(self, tangents: torch.Tensor) -> torch.Tensor:
        """
        Jacobian-vector product: J_f(x) @ tangents

        Args:
            tangents: shape (B, K, C, H, W) or (B, K, C*H*W)

        Returns:
            Tensor of shape (B, K, C'*H'*W')
        """
        B, K = tangents.shape[:2]
        C, H, W = self.C, self.H, self.W

        if tangents.ndim == 3:
            tangents = tangents.view(B, K, C, H, W)
        elif tangents.shape[2:] != (C, H, W):
            raise ValueError(f"Expected tangents of shape (B, K, {C}, {H}, {W}) or (B, K, {C*H*W})")

        x_exp = self.x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W)
        t_exp = tangents.reshape(B * K, C, H, W)

        _, jvp_out = func.jvp(self.f, (x_exp,), (t_exp,))
        return jvp_out.view(B, K, -1)

    def vjp(self, cotangents: torch.Tensor) -> torch.Tensor:
        """
        Vector-Jacobian product: cotangents @ J_f(x)

        Args:
            cotangents: shape (B, K, C', H', W') or (B, K, C'*H'*W')

        Returns:
            Tensor of shape (B, K, C*H*W)
        """
        B, K = cotangents.shape[:2]
        Cout, Hout, Wout = self.Cout, self.Hout, self.Wout
        C, H, W = self.C, self.H, self.W

        if cotangents.ndim == 3:
            cotangents = cotangents.view(B, K, Cout, Hout, Wout)
        elif cotangents.shape[2:] != (Cout, Hout, Wout):
            raise ValueError(f"Expected cotangents of shape (B, K, {Cout}, {Hout}, {Wout}) or (B, K, {Cout*Hout*Wout})")

        x_exp = self.x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W)
        c_exp = cotangents.reshape(B * K, Cout, Hout, Wout)

        _, vjp_fn = func.vjp(self.f, x_exp)
        vjp_out, = vjp_fn(c_exp)

        return vjp_out.view(B, K, -1)

    @property
    def T(self):
        return self.Transposed(self)

    class Transposed:
        def __init__(self, parent):
            self.parent = parent
            self.shape = tuple(reversed(parent.shape))
            self.device = parent.device
            self.dtype = parent.dtype

        def __matmul__(self, cotangents):
            return self.parent.vjp(cotangents)

    def __matmul__(self, tangents):
        return self.jvp(tangents)

        

class ConvJacobianLinearOperatorOld:
    def __init__(self, f, x):
        """
        Batched Jacobian operator for convolutional network f at inputs x.
        Args:
            f: Callable, maps (B, C, H, W) → (B, C', H', W')
            x: torch.Tensor of shape (B, C, H, W)
        """
        self.f = f
        self.x = x.detach().requires_grad_(True)
        B, C, H, W = x.shape
        y = f(x)
        Cout, Hout, Wout = y.shape[1:]
        self.shape = (B, Cout , Hout , Wout)  # (B, d_out, d_in) # (B, Cout * Hout * Wout, C * H * W)
        self.dtype = x.dtype
        self.device = x.device

    def dim(self): #  (B, d_out, d_in). TODO do something more elegant 
        return 3

    @property
    def ndim(self):
        return 3
    def __matmul__(self, tangents):
        """
        A @ Omega = J_f(x) @ Omega
        Args:
            tangents: Tensor of shape (B, K, C, H, W)
        Returns:
            Tensor of shape (B, K, C', H', W') TODOchange to (B,K, d)
        """
        return self.jvp(tangents)

    @property
    def T(self):
        return self.Transposed(self)

    def jvp(self, tangents):
        """
        Batched Jacobian-vector product.
        Args:
            tangents: Tensor of shape (B, K, C, H, W)
        Returns:
            Tensor of shape (B, K, C', H', W')
        """
        B, K, C, H, W = tangents.shape
        x_exp = self.x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W)
        t_exp = tangents.reshape(B * K, C, H, W)
        _, jvp_out = func.jvp(self.f, (x_exp,), (t_exp,))
        #print(jvp_out.shape)
        return jvp_out.view(B, K,-1) #jvp_out.view(B, K, *jvp_out.shape[1:])  # (B, K, C', H', W') # TODO add doc 

    def vjp(self, cotangents):
        """
        Vector-Jacobian product.
        Args:
            cotangents: Tensor of shape (B, K, C', H', W')
        Returns:
            Tensor of shape (B, K, C, H, W)
        """
        B, K, Cout, Hout, Wout = cotangents.shape
        C, H, W = self.x.shape[1:]
        x_exp = self.x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W)
        v_exp = cotangents.reshape(B * K, Cout, Hout, Wout)
        _, vjp_fn = func.vjp(self.f, x_exp)
        vjp_out, = vjp_fn(v_exp)
        #print(vjp_out.shape)
        return vjp_out.view(B,K,-1) #vjp_out.view(B, K, C, H, W) # TODO add doc 

    class Transposed:
        def __init__(self, parent):
            self.parent = parent
            self.shape = parent.shape #(parent.shape[0], parent.shape[2], parent.shape[1])
            self.dtype = parent.dtype
            self.device = parent.device
            self.dim = parent.dim
            self.ndim = parent.ndim

        def __matmul__(self, cotangents):
            return self.parent.vjp(cotangents)
                
# Class for batched Jacobians for convolutional networks, where:
# x has shape (B, C, H, W), e.g., torch.Size([13, 3, 32, 32])
# f(x) has shape (B, C_out, H_out, W_out), e.g., (13, 12, 16, 16) for a strided conv block
# we need to modify how we flatten and reshape the inputs and outputs, because PyTorch's autograd.functional.jvp and vjp work on 2D tensors with shape (N, D) (not 4D).
class ConvJacobianLinearOperator2:
    def __init__(self, f, x):
        """
        Batched Jacobian operator for convolutional network f at inputs x.
        Args:
            f: Callable, maps (B, C, H, W) → (B, C', H', W')
            x: torch.Tensor of shape (B, C, H, W)
        """
        self.f = f
        self.x = x.detach().requires_grad_(True)
        B, C, H, W = x.shape
        y = f(x)
        Cout, Hout, Wout = y.shape[1:]
        self.shape = (B, Cout * Hout * Wout, C * H * W)  # (B, d_out, d_in)
        self.dtype = x.dtype
        self.device = x.device

    def dim(self): #  (B, d_out, d_in). TODO do something more elegant 
        return 3

    @property
    def ndim(self):
        return 3
    def __matmul__(self, tangents):
        """
        A @ Omega = J_f(x) @ Omega
        Args:
            tangents: Tensor of shape (B, K, C, H, W)
        Returns:
            Tensor of shape (B, K, C', H', W')
        """
        return self.jvp(tangents)

    @property
    def T(self):
        return self.Transposed(self)

    def jvp(self, tangents):
        """
        Batched Jacobian-vector product.
        Args:
            tangents: Tensor of shape (B, K, C, H, W)
        Returns:
            Tensor of shape (B, K, C', H', W')
        """
        B, K, C, H, W = tangents.shape
        x_exp = self.x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W)
        t_exp = tangents.reshape(B * K, C, H, W)
        _, jvp_out = func.jvp(self.f, (x_exp,), (t_exp,))
        return jvp_out.view(B, K, *jvp_out.shape[1:])  # (B, K, C', H', W')

    def vjp(self, cotangents):
        """
        Vector-Jacobian product.
        Args:
            cotangents: Tensor of shape (B, K, C', H', W')
        Returns:
            Tensor of shape (B, K, C, H, W)
        """
        B, K, Cout, Hout, Wout = cotangents.shape
        C, H, W = self.x.shape[1:]
        x_exp = self.x.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, C, H, W)
        v_exp = cotangents.reshape(B * K, Cout, Hout, Wout)
        _, vjp_fn = func.vjp(self.f, x_exp)
        vjp_out, = vjp_fn(v_exp)
        return vjp_out.view(B, K, C, H, W)

    class Transposed:
        def __init__(self, parent):
            self.parent = parent
            self.shape = (parent.shape[0], parent.shape[2], parent.shape[1])
            self.dtype = parent.dtype
            self.device = parent.device

        def __matmul__(self, cotangents):
            return self.parent.vjp(cotangents)

        
# ----- Define simple convolutional network -----# For testing 
    # ========== Convolutional Network ==========
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),  # 3→4 channels
            nn.ReLU(),
            nn.Conv2d(4, 2, kernel_size=3, padding=1),  # 4→2 channels
        )

    def forward(self, x):
        return self.net(x)        
        

# Example usage:
if __name__ == "__main__":
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.func as func  # For functional jacobian/vector-Jacobian operations
    from torch.testing import assert_close

    def relative_error(a, b):
        return (a - b).norm() / (b.norm() + 1e-12)

    # ========== Autoencoder Network ==========
    print("🔧 Initializing autoencoder model for JacobianLinearOperator tests...")
    d_in = 10
    hidden_dim = 20
    bottleneck_dim = 5
    f = nn.Sequential(
        nn.Linear(d_in, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, bottleneck_dim),
        nn.ReLU(),
        nn.Linear(bottleneck_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, d_in)
    )

    B, K = 4, 2
    x = torch.randn(B, d_in, requires_grad=True)
    A = JacobianLinearOperator(f, x)

    # Test shape
    d_out = f(x[:1]).shape[1]
    expected_shape = (B, d_out, d_in)
    print(f"\n🧪 Testing shape of JacobianLinearOperator...")
    print(f"A.shape = {A.shape}, expected = {expected_shape}")
    assert A.shape == expected_shape
    print("✅ Shape test passed.")

    # Test JVP
    print("\n🧪 Testing JVP for JacobianLinearOperator...")
    tangents = torch.randn(B, K, d_in)
    jvp_auto = A @ tangents
    jvp_manual = torch.stack([
        func.jvp(f, (x[i],), (tangents[i, k],))[1].unsqueeze(0)
        for i in range(B) for k in range(K)
    ]).view(B, K, -1)
    jvp_err = relative_error(jvp_auto, jvp_manual).item()
    print(f"JVP relative error: {jvp_err:.2e}")
    assert jvp_err < 1e-6
    print("✅ JVP test passed.")

    # Test VJP
    print("\n🧪 Testing VJP for JacobianLinearOperator...")
    cotangents = torch.randn(B, K, d_out)
    vjp_auto = A.T @ cotangents
    vjp_manual = torch.stack([
        func.vjp(f, x[i])[1](cotangents[i, k])[0].unsqueeze(0)
        for i in range(B) for k in range(K)
    ]).view(B, K, -1)
    vjp_err = relative_error(vjp_auto, vjp_manual).item()
    print(f"VJP relative error: {vjp_err:.2e}")
    assert vjp_err < 1e-6
    print("✅ VJP test passed.")

    print("\n🎉 All tests passed for JacobianLinearOperator.")

    # ========== ConvJacobianLinearOperator2 Tests ==========

    torch.manual_seed(0)
    B, C, H, W = 5, 3, 8, 8
    K = 3
    f = SimpleConvNet()
    x = torch.randn(B, C, H, W)
    A = ConvJacobianLinearOperator2(f, x)

    print("\n🧪 Testing ConvJacobianLinearOperator2 shape...")
    Dout = f(x[0:1]).numel()
    Din = x[0].numel()
    assert A.shape == (B, Dout, Din)
    print(f"✅ Shape correct: {A.shape}")

    print("\n🧪 Testing JVP (forward pass)...")
    Omega = torch.randn(B, K, C, H, W)
    Y = A @ Omega
    Y_expected = torch.stack([
        func.jvp(f, (x[i:i+1].requires_grad_(),), (Omega[i, k].unsqueeze(0),))[1]
        for i in range(B) for k in range(K)
    ]).view(B, K, *Y.shape[2:])
    assert (Y - Y_expected).abs().max() < 1e-5
    print("✅ JVP test passed.")

    print("\n🧪 Testing VJP (adjoint)...")
    K_adj = 4
    cotangents = torch.randn(B, K_adj, *Y.shape[2:])
    vjp_out = A.T @ cotangents
    assert vjp_out.shape == (B, K_adj, C, H, W)
    print("✅ VJP test passed.")

    # ========== Flatten & Unflatten ==========
    print("\n🧪 Testing flatten/unflatten functions...")
    B, n, C, H, W = 2, 3, 4, 5, 6
    x = torch.randn(B, n, C, H, W)
    flat = flatten_tensor(x)
    assert flat.shape == (B, n, C * H * W)
    unflat = unflatten_vector(flat, (C, H, W))
    assert unflat.shape == x.shape and torch.allclose(x, unflat)
    print("✅ Flatten/unflatten functions correct.")

    # ========== Compare Old and New Conv Operator ==========
    print("\n🧪 Comparing ConvJacobianOperator (new) with ConvJacobianLinearOperatorOld (reference)...")
    B, K = 4, 3
    C, H, W = 3, 8, 8
    x = torch.randn(B, C, H, W)
    v = torch.randn(B, K, C, H, W)

    J_new = ConvJacobianOperator(f, x)
    J_old = ConvJacobianLinearOperatorOld(f, x)

    print("→ Testing JVP agreement...")
    out_new = J_new @ flatten_tensor(v)
    out_old = J_old @ v
    assert_close(out_new, out_old, atol=1e-5, rtol=1e-4)
    print("✅ JVPs match.")

    print("→ Testing VJP agreement...")
    Cout, Hout, Wout = f(x).shape[1:]
    w = torch.randn(B, K, Cout, Hout, Wout)
    out_new = J_new.T @ flatten_tensor(w)
    out_old = J_old.T @ w
    assert_close(out_new, out_old, atol=1e-5, rtol=1e-4)
    print("✅ VJPs match.")

    print("\n🎉 All ConvJacobian tests passed.")


 # TODO add histogram
    # import matplotlib.pyplot as plt
# plt.rcParams['axes.axisbelow'] = True
# x = torch.randn(1, d_in, requires_grad=True)

# A = JacobianLinearOperator(f, x)
# _, m, n = A.shape
# 1/0
# Omega = torch.randn(n)  # Single output vector

# # Instantiate Jacobian operator
# J_x = JacobianLinearOperator(sine_xAT, x)
# # Test inner product identity
# n_trials = 500
# errors = []

# for _ in range(n_trials):
#     v = torch.randn(n)
#     z = torch.randn(m)

#     Jv = J_x @ v               # J(x) @ v
#     JTz = J_x.T @ z         # J(x)^T @ z

#     lhs = torch.dot(Jv, z).item()
#     rhs = torch.dot(v, JTz).item()
    
#     denom = max(abs(lhs), abs(rhs))
#     error = abs(lhs - rhs) / denom
#     errors.append(error)

# # Plot histogram of relative errors
# plt.figure(figsize=(12, 6))
# plt.hist(errors, bins=200, log=True, edgecolor='black')
# plt.xlabel("Relative Adjoint Error")
# plt.ylabel("Frequency (log scale)")
# plt.title("Histogram of Relative Errors for Adjoint Test")
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()
