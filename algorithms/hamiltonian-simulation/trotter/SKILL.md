---
name: trotter
description: Trotter-Suzuki product-formula Hamiltonian simulation, approximating e^{-iHt} via structured short-time exponential products with controllable order and step count.
---

## One-Step Run Example Command
```bash
python ./scripts/algorithm.py
```

# Trotter-Suzuki Hamiltonian Simulation Skill Guide

## Overview

Trotter-Suzuki approximates matrix exponential evolution

$$
U(t) = e^{-iHt}
$$

by decomposing the Hamiltonian into Pauli terms and replacing the full exponential with structured products of easier exponentials.

### Key Insight

You do not implement $e^{-i(H_1 + H_2 + \cdots + H_L)t}$ directly. Instead, you build a repeated short-time product formula over terms $H_\ell$, and increase accuracy by:
1. Using more time slices.
2. Using higher even Suzuki order.

### Why Trotter-Suzuki Matters:

1. It is the most direct gate-level baseline for Hamiltonian simulation.
2. It provides explicit control over accuracy versus circuit depth.
3. It works naturally with Pauli-string decompositions.
4. It is a strong reference implementation to benchmark advanced methods.

### Real Applications:

1. Time evolution of spin models and lattice Hamiltonians.
2. Digital quantum simulation in chemistry and materials.
3. Baseline for comparing QDrift, Taylor-LCU, and QSP methods.
4. Teaching and validating decomposition strategies in small systems.

## Learning Objectives

After using this skill, you should be able to:
1. Explain first-order versus even higher-order Suzuki formulas.
2. Understand how order and steps change error and depth.
3. Use the repository `TrotterAlgorithm` class correctly for experiments.
4. Extract the circuit from the result dictionary.
5. Build reproducible comparisons across parameter sweeps.

## Prerequisites

### Essential knowledge:

1. Hermitian Hamiltonians and unitary time evolution.
2. Pauli-string decomposition basics.
3. Quantum circuit composition and gate-sequence interpretation.

### Mathematical comfort:

1. Matrix norms, especially spectral norm.
2. Exponential of operators and series intuition.
3. Asymptotic error behavior with step refinement.

## Using the Provided Implementation

### Quick Start Example

```python
import numpy as np
from unitarylab_algorithms.hamiltonian_simulation.trotter.algorithm import TrotterAlgorithm

# 2-qubit Heisenberg Hamiltonian as a numpy matrix
H = np.array([[2.0, 1.0],
              [1.0, 3.0]])
t = 1.0
error = 1e-8

algo = TrotterAlgorithm()
result = algo.run(H=H, t=t, error=error, order=2, steps=1000, backend='torch')

print("status:", result['status'])          # 'ok' on success
print("circuit_path:", result['circuit_path'])
print("output files:", result['plot'])      # list of {format, filename} dicts
print("circuit:", result['circuit'])
```

### Core Parameters Explained

#### Constructor

```python
class TrotterAlgorithm(BaseAlgorithm):
    def __init__(self, text_mode: str = 'plain', algo_dir: str = None) -> None:
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Output text rendering mode (e.g. `'plain'`, `'legacy'`). |
| `algo_dir` | `str\|None` | `None` | Directory for saving result files. Auto-derived from file path if `None`. |

#### `run()` Parameters

```python
def run(self, H: np.ndarray, t: float, error: float,
        order: int = 1, steps: int = 1000,
        backend='torch', device='cpu', dtype=np.complex128):
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `H` | `np.ndarray` | required | Hermitian Hamiltonian matrix (square). |
| `t` | `float` | required | Total evolution time in $e^{-iHt}$. |
| `error` | `float` | required | Target approximation error used to adaptively bound step count. |
| `order` | `int` | `1` | Suzuki-Trotter order. Must be `1` or an even integer (`2,4,6,...`). |
| `steps` | `int` | `1000` | Upper bound on number of Trotter steps; adaptive formula may reduce this further. |
| `backend` | `str` | `'torch'` | Simulation backend for matrix computation. |
| `device` | `str` | `'cpu'` | Device for the backend (e.g. `'cpu'`, `'cuda'`). |
| `dtype` | | `np.complex128` | Numerical dtype for the evolution matrix. |

### Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` on error. |
| `circuit_path` | `list` | Paths to saved circuit diagram files (`[trotter_full, trotter_slice]`). |
| `plot` | `list` | List of `{"format": str, "filename": str}` dicts for all saved output files. |
| `circuit` | `Circuit` | The fully assembled Trotter circuit (`qc`). |
| `Approximate evolution matrix` | `np.ndarray` | $U_{\text{approx}} = $ Trotter-circuit unitary. |
| `Exact evolution matrix` | `np.ndarray` | $U_{\text{exact}} = e^{-iHt}$ from `scipy.linalg.expm`. |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}} - U_{\text{exact}}\|_F$. |

## Understanding the Core Components

### 1) Input validation and adaptive step rule

From `run()` in `algorithm.py`:

```python
# _format_system validates: finite t, positive error, square Hermitian H,
# pads dimension to next power-of-2 if needed.
dim, H, n = self._format_system(H, t, error)
alpha = np.linalg.norm(H, 2)
steps = int(min(
    5**order * np.power(t * n * alpha, 1 + 1.0 / order)
              * np.power(error, -1.0 / order) * 1.5,
    steps
))
```

Interpretation:
1. `_format_system` enforces Hermiticity, finite `t`, positive `error`, and power-of-2 padding.
2. Spectral norm `alpha = ||H||_2` drives the adaptive step formula.
3. The adaptive formula upper-bounds the actual step count by the caller-supplied `steps`.

### 2) Suzuki recursion engine

From `_recurse` in `algorithm.py`:

```python
if order == 1:
    return decomposition
elif order == 2:
    halves = [(p, c / 2) for p, c in decomposition[:-1]]
    full = [decomposition[-1]]
    return halves + full + list(reversed(halves))
else:
    reduction = 1 / (4 - 4 ** (1 / (order - 1)))
    outer = 2 * self._recurse(order - 2, [(p, c * reduction) for p, c in decomposition])
    inner = self._recurse(order - 2, [(p, c * (1 - 4 * reduction)) for p, c in decomposition])
    return outer + inner + outer
```

Interpretation:
1. Order-2 uses a symmetric palindrome composition.
2. Higher even order uses recursive composition with Suzuki reduction coefficient.
3. This construction expands gate count rapidly as order increases.

### 3) Slice expansion and full-circuit assembly

From `_expand` and `run()` in `algorithm.py`:

```python
# _expand: scale coefficients by 1/steps, then recurse for one slice
scaled_decomposition = [(p, c / steps) for p, c in decomposition]
one_slice = self._recurse(order, scaled_decomposition)
return one_slice
```

```python
# run(): decompose H*t, build slice circuit, repeat steps times
decomposition = pauli_string_decomposition(H * t)
sequence = self._expand(decomposition, order, steps)

for pauli_str, angle in sequence:
    gate = pauli_string_evolution(pauli_str, angle)
    trotter.append(gate, range(n))

qc = Circuit(reg, name='Trotter Decomposition')
qc.append(trotter.repeat(steps), range(n))
U_approx = qc.get_matrix(backend=backend, device=device, dtype=dtype)
```

Interpretation:
1. `pauli_string_decomposition(H * t)` absorbs the time factor into the Pauli coefficients.
2. One slice uses coefficients divided by `steps`; the slice circuit is `trotter`.
3. The full circuit `qc` wraps `trotter.repeat(steps)` — one block repeated.
4. Both circuits are saved: `trotter_full` (the repeated `qc`) and `trotter_slice` (`trotter`).

### 4) Notes on external package functions (brief)

1. `pauli_string_decomposition` (from `unitarylab.library.pauli_operator`) decomposes the matrix into Pauli strings with coefficients.
2. `pauli_string_evolution` builds the circuit for a single Pauli-string exponential $e^{-i\theta P}$.
3. Error is computed in `run()` as the Frobenius norm `norm(U_approx - expm(-1j * H * t), ord='fro')`.

## Hands-On Example: Hamiltonian Simulation

Use a parameter sweep to compare orders.

```python
import numpy as np
from unitarylab_algorithms.hamiltonian_simulation.trotter.algorithm import TrotterAlgorithm

# 2-qubit Hamiltonian matrix
H = np.array([[0.5 + 0.2,  0.3],
              [0.3,       -0.5 + 0.1]])

for order in [1, 2, 4]:
    algo = TrotterAlgorithm()
    result = algo.run(H=H, t=1.0, error=1e-8, order=order, steps=1000, backend='torch')
    frob_err = result['Frobenius norm of error']
    print(f"order={order}, status={result['status']}, Frobenius error={frob_err:.2e}")
    print("saved files:", result['plot'])
```

What to look for:
1. Higher order reduces Frobenius norm error for the same step budget.
2. The `plot` field lists the saved output files (txt report).
3. `circuit_path` holds paths to the SVG/PNG circuit diagrams.

## Mathematical Deep Dive

Assume local decomposition:

$$
H = \sum_{j=1}^{N} h_j H_j
$$

and non-commuting terms satisfy

$$
e^{-i(H_1+H_2)t} \neq e^{-iH_1t}e^{-iH_2t}.
$$

The Lie-Trotter product limit gives:

$$
e^{-i(H_1+H_2)t} = \lim_{r\to\infty}\left(e^{-iH_1 t/r}e^{-iH_2 t/r}\right)^r.
$$

For $\Delta t=t/r$, first-order and second-order product formulas are:

$$
S_1(\Delta t) = \prod_{\ell=1}^{L} e^{-iH_\ell \Delta t}
$$

$$
S_2(\Delta t)=\left(\prod_{\ell=1}^{L-1}e^{-iH_\ell\Delta t/2}\right)e^{-iH_L\Delta t}\left(\prod_{\ell=L-1}^{1}e^{-iH_\ell\Delta t/2}\right).
$$

Higher even-order Suzuki recursion can be written as:

$$
S_{2k+2}(t)=\left(S_{2k}(p_k t)\right)^2 S_{2k}((1-4p_k)t)\left(S_{2k}(p_k t)\right)^2,
\quad
p_k=\left(4-4^{1/(2k+1)}\right)^{-1}.
$$

Equivalent implementation form in this code is:

$$
S_{2k}(\Delta t)=S_{2k-2}(p_k\Delta t)^2\,S_{2k-2}((1-4p_k)\Delta t)\,S_{2k-2}(p_k\Delta t)^2.
$$

Error expressions (local step error and global error) follow the standard scaling:

$$
\varepsilon_{\Delta t}^{(1)} \sim O(\|H\|^2\Delta t^2),
\quad
\varepsilon_{t}^{(1)} \sim O(\|H\|^2 t\Delta t)
$$

$$
\varepsilon_{\Delta t}^{(2)} \sim O(\|H\|^3\Delta t^3),
\quad
\varepsilon_{t}^{(2)} \sim O(\|H\|^3 t\Delta t^2)
$$

$$
\varepsilon_{\Delta t}^{(2k)} \sim O(\|H\|^{2k+1}\Delta t^{2k+1}),
\quad
\varepsilon_{t}^{(2k)} \sim O(\|H\|^{2k+1} t\Delta t^{2k}).
$$

Implementation-consistent notes:
1. Use the notation $\Delta t=t/r$ consistently when reasoning about formulas. In code, this is implemented by first scaling coefficients by `1/self.steps` in `_expand`, then repeating the one-slice circuit with `qc.repeat(self.steps)`.
2. The recursive Suzuki coefficient in the code,
   `reduction = 1 / (4 - 4 ** (1 / (order - 1)))`,
   is the same structure as the standard recursive coefficient $p_k$ after index mapping.
3. Practical behavior may differ from asymptotic formulas when step clamping is active, because the current implementation enforces `self.steps <= 100`.


## Reference Implementation (PennyLane)

PennyLane provides `qml.ApproxTimeEvolution` for first-order Trotterized
Hamiltonian time evolution.

### Minimal PennyLane Example
```python
import pennylane as qml

n_wires = 2
dev = qml.device("default.qubit", wires=n_wires)

coeffs = [0.1, 0.2, 0.3]
ops = [
    qml.Z(0) @ qml.Z(1),
    qml.X(0),
    qml.X(1),
]

hamiltonian = qml.Hamiltonian(coeffs, ops)

@qml.qnode(dev)
def circuit(t):
    qml.ApproxTimeEvolution(hamiltonian, t, n=2)
    return [qml.expval(qml.Z(i)) for i in range(n_wires)]

result = circuit(0.5)
print(result)
```


## Real-World Applications

1. Spin-chain evolution benchmarking.
2. Digital simulation baseline for chemistry Hamiltonians.
3. Hardware-aware gate budgeting and algorithm comparison.
4. Educational demos of deterministic product formulas.
