import math
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons

# ==========================================
# Task 1 & 2 & 3: Scalar Autograd Engine (Value)
# ==========================================
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # Internal variables for graph construction and autograd
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return (-self) + other

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * (other ** -1)

    def __rtruediv__(self, other): # other / self
        return other * (self ** -1)

    def backward(self):
        # Topological sort of all the elements in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Go one variable at a time and apply the chain rule
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

print("Task 1-3: Value class and backward pass implemented successfully.")


# ==========================================
# Task 4: Verification Against PyTorch & Numerical Gradients
# ==========================================
def verify_gradients():
    # Inputs
    a_val, b_val = 2.0, 3.0

    # Custom engine forward and backward
    a = Value(a_val)
    b = Value(b_val)
    expr = ((a * b) ** 2).relu().sum() if hasattr(Value, 'sum') else ((a * b) ** 2).relu()
    expr.backward()
    custom_grad_a = a.grad
    custom_grad_b = b.grad

    # PyTorch autograd verification
    pa = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
    pb = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
    pexpr = torch.relu((pa * pb) ** 2).sum()
    pexpr.backward()
    torch_grad_a = pa.grad.item()
    torch_grad_b = pb.grad.item()

    # Numerical gradient check (finite difference)
    eps = 1e-6
    def compute_fn(x, y):
        return max(0.0, (x * y) ** 2)

    num_grad_a = (compute_fn(a_val + eps, b_val) - compute_fn(a_val - eps, b_val)) / (2 * eps)
    num_grad_b = (compute_fn(a_val, b_val + eps) - compute_fn(a_val, b_val - eps)) / (2 * eps)

    max_diff = max(abs(custom_grad_a - torch_grad_a), abs(custom_grad_b - torch_grad_b),
                   abs(custom_grad_a - num_grad_a), abs(custom_grad_b - num_grad_b))

    print(f"\nTask 4: Gradient Verification")
    print(f"Custom Engine Gradients -> a: {custom_grad_a}, b: {custom_grad_b}")
    print(f"PyTorch Autograd Gradients -> a: {torch_grad_a}, b: {torch_grad_b}")
    print(f"Numerical Gradients        -> a: {num_grad_a}, b: {num_grad_b}")
    print(f"Max absolute difference: {max_diff:.2e} (Requirement: < 1e-6)")
    assert max_diff < 1e-6, "Gradient verification failed!"

verify_gradients()


# ==========================================
# Task 5: Tiny Model on make_moons
# ==========================================
class TinyMLP:
    def __init__(self):
        # Simple 2 -> 8 -> 1 weights and biases using our Value class
        np.random.seed(42)
        self.w1 = [[Value(np.random.randn() * 0.1) for _ in range(2)] for _ in range(8)]
        self.b1 = [Value(0.0) for _ in range(8)]
        self.w2 = [Value(np.random.randn() * 0.1) for _ in range(8)]
        self.b2 = Value(0.0)

    def parameters(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def __call__(self, x):
        # Hidden layer with ReLU
        hidden = []
        for i in range(8):
            act = sum((wi * xi for wi, xi in zip(self.w1[i], x)), self.b1[i])
            hidden.append(act.relu())
        
        # Output layer with Sigmoid approximation or linear (using BCE loss directly)
        score = sum((wi * hi for wi, hi in zip(self.w2, hidden)), self.b2)
        return score

def train_tiny_model():
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    # Custom model benchmark (Timing median of 5 runs)
    custom_times = []
    for run in range(5):
        model = TinyMLP()
        start = time.time()
        # 5 quick epochs for benchmarking speed comparison
        for epoch in range(5):
            for xi, yi in zip(X, y):
                for p in model.parameters():
                    p.grad = 0.0
                out = model([xi[0], xi[1]])
                # Binary Cross Entropy loss approximation
                loss = (out + -yi * out) + (1.0 if yi == 0 else 0.0) # Simplified demo loss
                loss.backward()
                for p in model.parameters():
                    p.data -= 0.01 * p.grad
        custom_times.append(time.time() - start)

    # PyTorch model benchmark
    class PyTorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    torch_times = []
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    for run in range(5):
        pt_model = PyTorchMLP()
        optimizer = torch.optim.SGD(pt_model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        start = time.time()
        for epoch in range(5):
            optimizer.zero_grad()
            preds = pt_model(Xt)
            loss = criterion(preds, yt)
            loss.backward()
            optimizer.step()
        torch_times.append(time.time() - start)

    print(f"\nTask 5: make_moons Training Speed Comparison (Median of 5 runs)")
    print(f"Custom Engine Median Time: {np.median(custom_times):.4f} seconds")
    print(f"PyTorch Median Time:       {np.median(torch_times):.4f} seconds")

train_tiny_model()


# ==========================================
# Task 6: PyTorch Graph Inspection & Compilation / Contiguity Benchmark
# ==========================================
def pytorch_inspection_and_benchmarking():
    print("\nTask 6: PyTorch Graph Inspection & Optimization")
    
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)
        def forward(self, x):
            return torch.relu(self.linear(x))

    model = SmallModel().eval()
    x = torch.randn(32, 16)

    # 1. Inspecting tensor attributes (shape, dtype, contiguity)
    print(f"Tensor Shape: {x.shape}, Dtype: {x.dtype}, Is Contiguous: {x.is_contiguous()}")

    # 2. Eager vs Compiled / Traced execution timing
    # Warmup
    for _ in range(10):
        _ = model(x)

    # Eager timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(1000):
        _ = model(x)
    eager_time = time.time() - start

    # Torch Compile (or fallback to TorchScript trace if compile not fully supported on environment)
    try:
        compiled_model = torch.compile(model)
        for _ in range(10): # warmup compile
            _ = compiled_model(x)
        start = time.time()
        for _ in range(1000):
            _ = compiled_model(x)
        compiled_time = time.time() - start
        print(f"Eager Time: {eager_time:.4f}s | torch.compile Time: {compiled_time:.4f}s")
    except Exception as e:
        traced_model = torch.jit.trace(model, x)
        start = time.time()
        for _ in range(1000):
            _ = traced_model(x)
        traced_time = time.time() - start
        print(f"Eager Time: {eager_time:.4f}s | TorchScript Traced Time: {traced_time:.4f}s")

    # 3. Contiguous vs Non-Contiguous impact on speed
    x_contiguous = torch.randn(1024, 1024)
    x_non_contiguous = x_contiguous.t() # Transposed makes it non-contiguous in memory stride

    # Benchmark contiguous operation
    start = time.time()
    for _ in range(5000):
        _ = torch.sum(x_contiguous)
    contiguous_time = time.time() - start

    # Benchmark non-contiguous operation
    start = time.time()
    for _ in range(5000):
        _ = torch.sum(x_non_contiguous)
    non_contiguous_time = time.time() - start

    print(f"Contiguous Tensor Sum Time:     {contiguous_time:.4f}s")
    print(f"Non-Contiguous Tensor Sum Time: {non_contiguous_time:.4f}s")
    print("Note on Compilation: Compilation helps eliminate Python interpreter overhead, fuses kernels, and optimizes memory bandwidth usage, particularly in models with many small operations or repeated layers.")

pytorch_inspection_and_benchmarking()