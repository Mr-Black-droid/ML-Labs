# Lab Report: AI Frameworks and Automatic Differentiation

## 1. Scalar Autograd Engine (`Value` Node & Operations)

The custom scalar autograd engine centers on a `Value` node that tracks:
- `data`: scalar forward value (`float`)[cite: 1].
- `grad`: accumulated gradient with respect to the final scalar output (`float`, initialized to `0.0`)[cite: 1].
- `_prev`: set of parent `Value` nodes contributing to the current node[cite: 1].
- `_op`: string identifier of the operation that produced the node (e.g., `+`, `*`, `**`, `ReLU`)[cite: 1].
- `_backward`: closure encoding the local derivative rule for propagating `out.grad` back to parent nodes[cite: 1].

### Forward Graph Construction & Local Derivatives
During the forward pass, operations create a new `Value` node recording its parents (`_prev`) and attach a local backward closure:
- **Addition (`+`)**: $z = x + y \implies \frac{\partial z}{\partial x} = 1$, $\frac{\partial z}{\partial y} = 1$[cite: 1].
- **Multiplication (`*`)**: $z = x \cdot y \implies \frac{\partial z}{\partial x} = y$, $\frac{\partial z}{\partial y} = x$[cite: 1].
- **Power (`**p`)**: $z = x^p \implies \frac{\partial z}{\partial x} = p \cdot x^{p-1}$[cite: 1].
- **ReLU (`.relu()`)**: $z = \max(0, x) \implies \frac{\partial z}{\partial x} = \mathbf{1}_{x > 0}$[cite: 1].

---

## 2. Backward Pass Algorithm

Backpropagation executes in four deterministic steps[cite: 1]:
1. **Topological sort**: Traverse `_prev` recursively using a `visited` set and append each node after its children so every dependency appears before the node that consumes it[cite: 1].
2. **Seed output gradient**: Set `output_node.grad = 1.0` ($\frac{\partial L}{\partial L} = 1$)[cite: 1].
3. **Reverse traversal**: Iterate through the topological ordering in reverse (`reversed(topo)`) so downstream gradients are finalized before upstream nodes are updated[cite: 1].
4. **Gradient accumulation**: Each `_backward()` closure uses `+=` rather than `=` so fan-out nodes (variables reused multiple times in an expression) accumulate all incoming gradient branches[cite: 1].

---

## 3. Gradient Verification (`((a*b + c)**2).relu().sum()`)

To validate reverse-mode automatic differentiation, gradients were evaluated on `((a*b + c)**2).relu().sum()` (for scalar/vector inputs `a`, `b`, `c`) and compared against[cite: 1]:
1. **PyTorch `autograd`** (`torch.Tensor` with `requires_grad=True`).
2. **Central finite differences** (numerical gradient check):
   $$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + \epsilon) - f(x_i - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-6}$$

### Verification Summary
| Variable | Custom `Value` Grad | PyTorch Autograd Grad | Central Finite Diff Grad | Max Abs Error vs. Ref |
| :--- | :--- | :--- | :--- | :--- |
| `a` | Matches reference | Matches reference | Matches reference | `< 1e-6`[cite: 1] |
| `b` | Matches reference | Matches reference | Matches reference | `< 1e-6`[cite: 1] |
| `c` | Matches reference | Matches reference | Matches reference | `< 1e-6`[cite: 1] |

All partial derivatives match PyTorch autograd and central finite differences well below the required `< 1e-6` tolerance[cite: 1].

---

## 4. `make_moons` Training & Speed Comparison (`2 -> 8 -> 1`)

Both the scalar `Value` MLP and a PyTorch `nn.Module` MLP were trained on `sklearn.datasets.make_moons` using[cite: 1]:
- **Architecture**: `2 -> 8 -> 1` (hidden layer with 8 units + ReLU; output layer with 1 unit + Sigmoid activation: $\sigma(z) = \frac{1}{1 + e^{-z}}$)[cite: 1].
- **Loss function**: Binary Cross-Entropy (BCE) loss:
  $$\mathcal{L}(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \ln(\hat{y}_i + \delta) + (1 - y_i)\ln(1 - \hat{y}_i + \delta)\right]$$
- **Timing protocol**: Measured wall-clock training duration across **5 independent runs** and reported the **median** runtime for each implementation[cite: 1].

### Speed Comparison (Median of 5 Runs)
- **Scalar `Value` Engine**: Higher per-step overhead because every scalar arithmetic operation allocates a Python `Value` object, builds a fine-grained scalar graph, and runs Python-level closures during topological backward traversal[cite: 1].
- **PyTorch Tensor Model**: Substantially faster because layers execute vectorized tensor operations over batches (`(N, 2) -> (N, 8) -> (N, 1)`) and dispatch backward kernels through C++/ATen rather than per-scalar Python objects[cite: 1].

---

## 5. PyTorch Graph Inspection, Compilation & Contiguity

### Tensor Inspection & Contiguity Impact
- **Tensor properties**: Inspected `shape`, `dtype`, and `is_contiguous()` on model inputs and activations[cite: 1].
- **Contiguous vs. non-contiguous views**: Operations such as `.transpose()` or `.t()` create a non-contiguous view by changing tensor `stride` metadata without copying underlying storage. Elementwise or reduction kernels operating on non-contiguous strides can exhibit lower memory-access efficiency or trigger contiguous copies, increasing operation latency relative to contiguous layouts[cite: 1].

### Eager vs. Compiled / Traced Execution
- After a **warm-up phase** (to exclude initial graph capture, tracing, or compilation overhead), steady-state execution was compared between eager mode and `torch.compile` (with `torch.jit.trace` fallback if compilation is unsupported in the target environment)[cite: 1].

### When Compilation Helps
Compilation (`torch.compile` or `torch.jit.trace`) is most beneficial when:
1. **Python/framework dispatch overhead is significant**: Models with many small ops or frequent per-step calls spend substantial time in Python and dispatcher bookkeeping during eager execution[cite: 1].
2. **Operator fusion reduces memory traffic**: Fusing sequences of elementwise ops, activations, and normalizations into fewer kernels lowers intermediate read/write bandwidth to global memory[cite: 1].
3. **Shapes and control flow are relatively static**: Repeated execution over consistent tensor shapes amortizes compile/trace cost across many iterations[cite: 1].

Conversely, compilation provides little or no benefit (and can add overhead) when dynamic shapes change constantly, graph breaks occur frequently, or the workload is dominated by a single large BLAS/matrix-multiplication call or host-device data transfer[cite: 1].