---
name: trotter
description: "Use when: you need product-formula Hamiltonian simulation with Trotter-Suzuki order and step control, code-level guidance, and practical error-depth tradeoff tuning."
---

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
3. Use the repository `Trotter` class correctly for experiments.
4. Extract circuit, matrix result, and spectral-norm error.
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
from engine.algorithms.trotter import Trotter

# Example 2x2 Hermitian Hamiltonian
H = np.array([
    [1.0, 0.3],
    [0.3, -1.0]
], dtype=complex)

sim = Trotter(
    H=H,
    t=1.0,
    target_error=1e-3,
    order=2,
    steps=80,
)

print("method:", sim.method)
print("target_qubits:", sim.target_qubits)
print("effective_steps:", sim.steps)

U_approx = sim.evolution_result
err = sim.total_error

print("U_approx shape:", U_approx.shape)
print("spectral-norm error:", err)
```

### Core Parameters Explained

```python
class Trotter(HamiltonianSimulationResult):
    def __init__(
        self,
        H: np.ndarray,
        t: float,
        target_error: float,
        order: int = 1,
        steps: int = 1000
    ) -> None:
        ...
```

Parameter meaning:
1. `H`: input Hermitian Hamiltonian.
2. `t`: evolution time in $e^{-iHt}$.
3. `target_error`: target tolerance used by step heuristic.
4. `order`: must be `1` or an even integer (`2,4,6,...`).
5. `steps`: user-requested baseline step count; implementation may increase and clamp.

Important implementation detail:
1. `self.steps` is computed by heuristic and clamped with `min(..., 1e2)`.
2. Therefore effective steps can differ from the value passed in.

```python
self.steps = int(min(max(
    steps,
    5**order
    * np.power(t * self.target_qubits * self.alpha, 1 + 1.0 / order)
    * np.power(target_error, -1.0 / order)
    * 1.5
), 1e2))
```

Return dictionary contains:

This class constructor returns an object, not a dictionary. For experiment logging, you can standardize outputs as:

```python
result = {
    "method": sim.method,
    "target_qubits": sim.target_qubits,
    "order": sim.order,
    "steps": sim.steps,
    "circuit": sim.circuit,
    "evolution_result": sim.evolution_result,
    "total_error": sim.total_error,
}
```

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
import numpy as np
from algorithms.trotter import Trotter

H = np.array([
    [0.7, 0.2, 0.0, 0.0],
    [0.2, -0.7, 0.3, 0.0],
    [0.0, 0.3, 0.4, 0.1],
    [0.0, 0.0, 0.1, -0.4],
], dtype=complex)

configs = [
    (1, 20),
    (1, 60),
    (2, 20),
    (2, 60),
]

rows = []
for order, steps in configs:
    sim = Trotter(H=H, t=1.0, target_error=1e-3, order=order, steps=steps)
    rows.append((order, steps, sim.steps, sim.total_error))

for order, req_steps, eff_steps, err in rows:
    print(f"order={order}, requested_steps={req_steps}, effective_steps={eff_steps}, error={err:.3e}")
```

What to look for:
1. `effective_steps` may differ from requested `steps` due to heuristic.
2. Increasing steps usually lowers error.
3. Higher order can lower error per step but increases per-slice complexity.

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

## Summary Checklist

1. [ ] Confirm `H` is Hermitian and dimensions are as expected.
2. [ ] Choose `order=1` or even `order>=2` only.
3. [ ] Verify `sim.steps` after construction.
4. [ ] Record both `total_error` and circuit size for each run.
5. [ ] Run a small sweep before production-size experiments.

## Real-World Applications

1. Spin-chain evolution benchmarking.
2. Digital simulation baseline for chemistry Hamiltonians.
3. Hardware-aware gate budgeting and algorithm comparison.
4. Educational demos of deterministic product formulas.
