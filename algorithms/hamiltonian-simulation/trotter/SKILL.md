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
3. Use the repository `SuzukiTrotterAlgorithm` class correctly for experiments.
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
from unitarylab.algorithms import SuzukiTrotterAlgorithm

# 2-qubit Heisenberg-like Hamiltonian as grouped Pauli terms
# Each group is a list of (pauli_string, coefficient) tuples
grouped_paulis = [
    [("ZI", 0.5), ("IZ", 0.5)],
    [("XX", 0.3), ("YY", 0.3)],
]
total_time = 1.0

algo = SuzukiTrotterAlgorithm(order=2, reps=1)
result = algo.run(
    grouped_paulis=grouped_paulis,
    total_time=total_time,
    backend='torch',
)

print("status:", result['status'])
print(result.get('plot', ''))
```

### Core Parameters Explained

#### Constructor

```python
class SuzukiTrotterAlgorithm:
    def __init__(self, order: int = 1, reps: int = 1) -> None:
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `order` | `int` | `1` | Suzuki-Trotter order. Must be `1` or an even integer (`2,4,6,...`). |
| `reps` | `int` | `1` | Number of repetitions of the Trotter step. |

#### `run()` Parameters

```python
def run(self, grouped_paulis, total_time, backend='torch', algo_dir=None):
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `grouped_paulis` | `List[List[Tuple[str, float]]]` | required | Hamiltonian as grouped Pauli terms. Each group is a list of `(pauli_string, coefficient)` tuples. |
| `total_time` | `float` | required | Evolution time $t$ in $e^{-iHt}$. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str\|None` | `None` | Output directory for circuit diagrams. |

### Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `circuit` | `GateSequence` | The assembled Trotter circuit. |
| `plot` | `str` | ASCII art result panel. |

## Understanding the Core Components

### 1) Constructor checks and adaptive step rule

From `algorithms/trotter/trotter.py`:

```python
if order > 1 and order % 2 == 1:
    raise ValueError("Order must be 1 or an even integer.")
self.order = order
self.alpha = np.linalg.norm(H, 2)
self.steps = int(min(max(steps, 5**order * np.power(t * self.target_qubits * self.alpha, 1 + 1.0 / order) * np.power(target_error, -1.0 / order) * 1.5), 1e2))
```

Interpretation:
1. Odd order above 1 is explicitly forbidden.
2. The code uses spectral norm `alpha = ||H||_2`.
3. Step count uses a heuristic tied to order, norm, time, qubits, and target error.

### 2) Suzuki recursion engine

From `_recurse` in `algorithms/trotter/trotter.py`:

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

From `_expand` and `_run`:

```python
scaled_decomposition = [(p, c / self.steps) for p, c in decomposition]
one_slice = self._recurse(self.order, scaled_decomposition)
return one_slice
```

```python
decomposition = self._pauli_decompose(self.H, self.t)
sequence = self._expand(decomposition)

for pauli_str, angle in sequence:
    gate = pauli_string_evolution(pauli_str, angle)
    qc.append(gate, range(self.target_qubits))

self._circuit = qc.repeat(self.steps)
self._evolution_result = self._circuit.get_matrix()
```

Interpretation:
1. One slice uses coefficients divided by `steps`.
2. Slice is then repeated by `qc.repeat(self.steps)`.
3. Evolution matrix comes directly from the assembled gate sequence.

### 4) Notes on external package functions (brief)

1. `_pauli_decompose` is inherited from `HamiltonianSimulationResult`; it handles banded-real optimization and generic decomposition.
2. `pauli_string_evolution` builds the circuit for a single Pauli-string exponential.
3. `total_error` is computed in the base class by comparing to `scipy.linalg.expm(-1j * H * t)`.

## Hands-On Example: Hamiltonian Simulation

Use a parameter sweep to visualize tradeoffs.

```python
from unitarylab.algorithms import SuzukiTrotterAlgorithm

# 2-qubit Hamiltonian as grouped Pauli terms
grouped_paulis = [
    [("ZI", 0.5), ("IZ", -0.3)],
    [("XX", 0.2), ("YY", 0.2), ("ZZ", 0.1)],
]

for order in [1, 2, 4]:
    algo = SuzukiTrotterAlgorithm(order=order, reps=1)
    result = algo.run(
        grouped_paulis=grouped_paulis,
        total_time=1.0,
        backend='torch',
    )
    print(f"order={order}, status={result['status']}")
    print(result.get('plot', ''))
```

What to look for:
1. Higher order produces more complex circuits but better approximation.
2. The `plot` field provides a summary of the decomposition.

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
