---
name: trotter
description: "Trotter-Suzuki product-formula Hamiltonian simulation, approximating e^{-iHt} via structured short-time exponential products with controllable order and step count. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Trotter-Suzuki Hamiltonian Simulation Skill Guide

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Trotter-Suzuki Hamiltonian Simulation Skill Guide.

After using this skill, you should be able to:
1. Explain first-order versus even higher-order Suzuki formulas.
2. Understand how order and steps change error and depth.
3. Use the repository `TrotterAlgorithm` class correctly for experiments.
4. Extract the circuit from the result dictionary.
5. Build reproducible comparisons across parameter sweeps.

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

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

## Prerequisites

### Essential knowledge:

1. Hermitian Hamiltonians and unitary time evolution.
2. Pauli-string decomposition basics.
3. Quantum circuit composition and gate-sequence interpretation.

### Mathematical comfort:

1. Matrix norms, especially spectral norm.
2. Exponential of operators and series intuition.
3. Asymptotic error behavior with step refinement.

## Reference Implementation Example

## Core Parameters Explained

### Constructor

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

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` on error. |
| `circuit_path` | `list` | Paths to saved circuit diagram files (`[trotter_full, trotter_slice]`). |
| `plot` | `list` | List of `{"format": str, "filename": str}` dicts for all saved output files. |
| `circuit` | `Circuit` | The fully assembled Trotter circuit (`qc`). |
| `Approximate evolution matrix` | `np.ndarray` | $U_{\text{approx}} = $ Trotter-circuit unitary. |
| `Exact evolution matrix` | `np.ndarray` | $U_{\text{exact}} = e^{-iHt}$ from `scipy.linalg.expm`. |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}} - U_{\text{exact}}\|_F$. |

## Understanding the Key Quantum Components

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
3. Practical behavior depends on the caller-provided step count. The implementation does not enforce a `steps <= 100` clamp; `run()` defaults to `steps=1000`.

## Hands-On Example

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

1. Spin-chain evolution benchmarking.
2. Digital simulation baseline for chemistry Hamiltonians.
3. Hardware-aware gate budgeting and algorithm comparison.
4. Educational demos of deterministic product formulas.

## Reference Implementation

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

## Implementation Architecture

`TrotterAlgorithm` in `algorithm.py` implements the Suzuki-Trotter product formula in four stages.

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Validation & Scaling | `_format_system(H, t, error)` pads dimension to power-of-2; computes `alpha = np.linalg.norm(H, 2)`; adaptively clamps `steps = min(5**order * (t * n * alpha)**(1+1/order) * error**(-1/order) * 1.5, steps)` | Prepares system and determines step count from error budget |
| 2 — Pauli Decomposition | `pauli_string_decomposition(H * t)` absorbs time factor into coefficients | Expands $Ht$ into weighted Pauli strings |
| 3 — Suzuki Recursion | `_recurse(order, scaled_decomposition)` builds one slice: order-1 returns raw decomposition; order-2 applies symmetric palindrome; higher even orders use recursive Suzuki reduction with $p_k = (4 - 4^{1/(2k+1)})^{-1}$ | Constructs the order-$k$ product formula for one slice |
| 4 — Circuit Assembly & Verification | One slice circuit is built via `_expand()`, then repeated `steps` times via `trotter.repeat(steps)`; `qc.get_matrix()` extracts $U_{\text{approx}}$; Frobenius norm error vs `expm` | Assembles full circuit and benchmarks accuracy |

**Key design decision:** Both the full circuit (`trotter_full`) and the single-slice circuit (`trotter_slice`) are saved as SVG diagrams, enabling inspection of the product-formula structure at both scales.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Hamiltonian $H$ | `H` — square Hermitian, padded to power-of-2 |
| Total evolution time $t$ | `t` parameter |
| Adaptive step count | `min(5**order * (t * n * alpha)**(1+1/order) * error**(-1/order) * 1.5, steps)` |
| Spectral norm $\alpha = \|H\|_2$ | `alpha = np.linalg.norm(H, 2)` |
| Pauli decomposition | `pauli_string_decomposition(H * t)` absorbs $t$ into coefficients |
| Per-slice coefficients | `[(p, c / steps) for p, c in decomposition]` |
| Order-2 Suzuki (symmetric) | Halves coefficients for all but last term; appends reversed halves |
| Higher even order recursion | `reduction = 1 / (4 - 4**(1/(order-1)))`; `2*S_{2k-2}(p_k) + S_{2k-2}(1-4p_k) + 2*S_{2k-2}(p_k)` |
| Circuit assembly | `qc.append(trotter.repeat(steps), range(n))` |
| Exact reference | `scipy.linalg.expm(-1j * H * t)` |
| Error metric | `norm(U_approx - U_exact, ord='fro')` |

## Minimal Manual Implementation

```python
import numpy as np
from scipy.linalg import expm

def trotter_simulation_skeleton(H, t, steps=100, order=2):
    """Simplified Trotter-Suzuki Hamiltonian simulation.

    In practice, the full implementation uses Pauli decomposition and
    gate-level circuit construction. This skeleton uses dense matrices
    to illustrate the product-formula structure.
    """
    n = H.shape[0]
    dt = t / steps

    # Decompose H into terms (simplified: use the matrix itself)
    # In practice: pauli_string_decomposition(H)
    U_slice = np.eye(n, dtype=complex)

    if order == 1:
        U_slice = expm(-1j * H * dt)
    elif order == 2:
        # Symmetric second-order: e^{-iH_1*dt/2} ... e^{-iH_n*dt} ... e^{-iH_1*dt/2}
        U_half = expm(-1j * H * dt / 2)
        U_slice = U_half @ U_half  # simplified for single-term H
    # Higher even orders use recursive Suzuki composition

    # Repeat slice 'steps' times
    U_approx = np.linalg.matrix_power(U_slice, steps)
    U_exact = expm(-1j * H * t)
    return U_approx, np.linalg.norm(U_approx - U_exact, 'fro')
```

Note: This skeleton uses dense matrix exponentiation. The actual implementation decomposes $H$ into Pauli strings and constructs per-gate exponentials via `pauli_string_evolution()`, then repeats the slice circuit `steps` times.

## Debugging Tips

1. **Order must be 1 or even**: `order=3` raises an error. Valid orders: $1, 2, 4, 6, \dots$. Higher even orders reduce error exponentially but increase gate count rapidly via the Suzuki recursion.
2. **Step count**: The implementation does not clamp `steps` to 100. The `run()` default is `steps=1000`; pass an explicit value to control the Trotter resolution.
3. **Order 4+ gate explosion**: The recursive Suzuki formula doubles the gate count at each recursion level. For order 4 on a Hamiltonian with 10 Pauli terms, one slice already has $O(10 \times 5) = 50$ gates. Before choosing `order=6`, check whether the error reduction justifies the depth increase.
4. **Frobenius vs spectral norm**: The implementation reports Frobenius norm error, which is always $\geq$ spectral norm error. A "large" Frobenius error for a high-dimensional system may correspond to acceptable spectral-norm accuracy.
5. **Pauli decomposition overhead**: `pauli_string_decomposition(H * t)` absorbs the time factor into coefficient magnitudes. Large $t$ can produce large coefficients, which may affect numerical stability in the gate-level construction — verify the coefficient range before circuit assembly.
