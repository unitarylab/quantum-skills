---
name: reverse
description: "Reverse-mode statevector gradient and QGT computation for parameterized circuits using Qiskit Algorithms classes ReverseEstimatorGradient and ReverseQGT. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Reverse-Mode Gradient

## Overview

| Property | Detail |
|---|---|
| Category | Gradient / Quantum Geometric Tensor |
| Standard APIs | `ReverseEstimatorGradient`, `ReverseQGT` |
| Framework | `qiskit_algorithms.gradients` |
| Core idea | Reverse sweep on statevectors to compute derivatives without parameter-shift circuits |

This algorithm computes expectation gradients and QGT entries by traversing parameterized gates in reverse order. It is optimized for small circuits and exact statevector evaluation.

## Reference Implementation Example

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit_algorithms.gradients.utils import DerivativeType

theta = ParameterVector("theta", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

values = [0.5, 1.0]

# Gradient
observable = SparsePauliOp("ZZ")
grad = ReverseEstimatorGradient(derivative_type=DerivativeType.REAL)
grad_result = grad.run([qc], [observable], [values]).result()
print(grad_result.gradients)

# QGT
qgt = ReverseQGT(phase_fix=True, derivative_type=DerivativeType.COMPLEX)
qgt_result = qgt.run([qc], [values]).result()
print(qgt_result.qgts)
```

## Core Parameters Explained

### `ReverseEstimatorGradient`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `derivative_type` | `DerivativeType` | No | `DerivativeType.REAL` | Output projection: `REAL`, `IMAG`, or `COMPLEX` |

### `ReverseQGT`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `phase_fix` | `bool` | No | `True` | Apply phase-fix subtraction term |
| `derivative_type` | `DerivativeType` | No | `DerivativeType.COMPLEX` | Return real, imaginary, or complex QGT |

## Return Fields

### `ReverseEstimatorGradient.run()`

| Input | Type |
|---|---|
| `circuits` | `Sequence[QuantumCircuit]` |
| `observables` | `Sequence[BaseOperator]` |
| `parameter_values` | `Sequence[Sequence[float]]` |
| `parameters` | `Sequence[Sequence[Parameter]] \| None` |

| Output object | Key fields |
|---|---|
| `EstimatorGradientResult` | `gradients`, `metadata`, `precision` |

### `ReverseQGT.run()`

| Input | Type |
|---|---|
| `circuits` | `Sequence[QuantumCircuit]` |
| `parameter_values` | `Sequence[Sequence[float]]` |
| `parameters` | `Sequence[Sequence[Parameter]] \| None` |

| Output object | Key fields |
|---|---|
| `QGTResult` | `qgts`, `metadata`, `derivative_type`, `precision` |

## Implementation Architecture

| Component | Role |
|---|---|
| `split()` | Splits a circuit into per-parameter unitary blocks |
| `derive_circuit()` | Builds analytic derivative terms `(coeff, QuantumCircuit)` |
| `bind()` | Binds numeric parameter values into circuits |
| `ReverseEstimatorGradient` | Reverse sweep with two statevectors for $O(P)$ parameter scaling |
| `ReverseQGT` | Reverse sweep with three statevectors for $O(P^2)$ parameter scaling |

Engineering constraints:

- API-declared parameterized gates: `rx`, `ry`, `rz`, `cp`, `crx`, `cry`, `crz`.
- Practical compatibility note (`qiskit_algorithms==0.4.0`): circuits containing parameterized two-qubit controlled rotations/phases (`cp`, `crx`, `cry`, `crz`) may raise `KeyError` during internal gradient-parameter preprocessing.
- Robust workaround for current environments: use only parameterized `rx`/`ry`/`rz`, and keep entanglers non-parameterized (e.g., `cx`) in demo circuits.
- Circuits should use unique parameters per gate path; unsupported gates must be decomposed first.
- Runtime scales exponentially with qubit count due to statevector simulation.
- No external estimator backend is required by users; classes internally satisfy base interfaces.

## Mathematical Deep Dive

For expectation value $f(\theta)=\langle\psi(\theta)|\hat O|\psi(\theta)\rangle$, for parameter $\theta_j$:

$$
\frac{\partial f}{\partial\theta_j}=2\,\Re\!\left(\sum_k c_k\langle\lambda|\phi_k\rangle\right)
$$

with derivative decomposition:

$$
\frac{\partial U_j}{\partial\theta_j}=\sum_k c_k G_k
$$

For QGT:

$$
\mathrm{QGT}_{ij}=\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle
$$

If `phase_fix=True`, the phase-fix term is explicitly subtracted in the computed metric.

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Reverse-Mode Gradient.

Reverse-mode gradient computes expectation value gradients and QGT entries by traversing parameterized gates in reverse order using statevector-level differentiation. Unlike parameter-shift or LCU, it does not require additional circuit evaluations — it operates directly on statevectors, making it efficient for small circuits.

Use this skill when:
- Circuits are small enough for statevector simulation (exponential scaling with qubit count)
- No external estimator/sampler backend is desired (the algorithm internally satisfies the base interfaces)
- Both gradient and QGT are needed from the same reverse sweep

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Prerequisites

- Qiskit parameterized circuits (no external estimator/sampler primitive needed)
- Understanding of reverse-mode automatic differentiation applied to quantum circuits
- Statevector simulation background — runtime scales as $O(2^n)$ with qubit count $n$
- Familiarity with `split()`, `derive_circuit()`, and `bind()` for gate-level differentiation

## Understanding the Key Quantum Components

1. **Gate splitting via `split()`**: Decomposes a circuit into per-parameter unitary blocks, isolating the action of each parameterized gate for individual differentiation.
2. **Analytic derivative via `derive_circuit()`**: For each parameterized gate $U_j(\theta_j)$, computes the derivative decomposition $\partial U_j/\partial\theta_j = \sum_k c_k G_k$ as a list of `(coefficient, QuantumCircuit)` pairs.
3. **Parameter binding via `bind()`**: Substitutes numeric parameter values into the derivative circuits, preparing them for statevector evaluation.
4. **Reverse sweep (Estimator)**: Two statevectors are maintained — one forward-propagated and one backward-propagated — to compute all $n$ parameter gradients in $O(P)$ time (where $P$ is the number of parameterized gates).
5. **Reverse sweep (QGT)**: Three statevectors are used — forward, backward, and an intermediate — to compute the full $n \times n$ QGT matrix in $O(P^2)$ time.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Circuit $U(\theta)$ | Input `QuantumCircuit` with parameterized gates |
| Gate decomposition $U = \prod_j U_j(\theta_j)$ | `split()` — extracts per-parameter blocks |
| Derivative $\partial U_j/\partial\theta_j = \sum c_k G_k$ | `derive_circuit()` — returns `(coeff, QuantumCircuit)` pairs |
| Numeric parameter substitution | `bind()` — binds `parameter_values` into derivative circuits |
| Gradient $\partial\langle O\rangle/\partial\theta_j$ | `ReverseEstimatorGradient` — reverse sweep on two statevectors |
| QGT $Q_{ij}$ | `ReverseQGT` — reverse sweep on three statevectors |
| `DerivativeType.REAL/IMAG/COMPLEX` | Constructor `derivative_type`; controls output projection |
| Phase-fix term | `phase_fix=True` in `ReverseQGT` |

## Hands-On Example

Compare reverse-mode gradient with finite difference for validation:

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, FiniteDiffEstimatorGradient
from qiskit_algorithms.gradients.utils import DerivativeType

theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

values = [0.5, 1.0]
observable = SparsePauliOp("ZZ")

# Reverse-mode (analytic)
rev_grad = ReverseEstimatorGradient(derivative_type=DerivativeType.REAL)
rev_result = rev_grad.run([qc], [observable], [values]).result()

# Finite difference (numerical, for comparison)
fd_grad = FiniteDiffEstimatorGradient(None, epsilon=1e-4, method="central")
# (Note: FiniteDiffEstimatorGradient requires an estimator in the real API;
#  this simplified example illustrates the comparison pattern.)

print("Reverse-mode gradient:", rev_result.gradients[0])
```

Expected behavior: reverse-mode produces exact gradients (no truncation error) for statevector-compatible circuits.

## Minimal Manual Implementation

```python
import numpy as np

def reverse_mode_gradient_simplified(circuit_states, observable_matrix):
    """Simplified reverse-mode gradient structure.

    In practice, the full implementation:
    1. Splits the circuit into per-parameter blocks via split()
    2. Computes derivative circuits via derive_circuit()
    3. Binds numeric values via bind()
    4. Performs a reverse sweep over statevectors

    This skeleton illustrates the mathematical structure.
    """
    n_params = len(circuit_states)  # one state per parameter block
    grad = np.zeros(n_params)

    # Forward statevector (accumulated through all gates)
    psi_forward = circuit_states[-1]  # |ψ(θ)⟩ after all gates

    # Backward statevector: O|ψ⟩ propagated backward
    psi_backward = observable_matrix @ psi_forward  # O|ψ⟩

    for j in reversed(range(n_params)):
        # Derivative of gate j: ∂U_j/∂θ_j = Σ c_k G_k
        # Contract forward and backward states through derivative operators
        deriv_contrib = sum(
            c_k * np.vdot(psi_backward, G_k @ psi_forward)
            for c_k, G_k in derive_circuit_generators(j)
        )
        grad[j] = 2.0 * np.real(deriv_contrib)

    return grad
```

Note: This skeleton omits the `split()` / `derive_circuit()` / `bind()` pipeline. The actual implementation traverses gates in reverse, maintaining statevector pairs with $O(P)$ complexity.

## Debugging Tips

1. **Parameterized gate restrictions**: Circuits using parameterized two-qubit gates (`cp`, `crx`, `cry`, `crz`) may raise `KeyError` during gradient-parameter preprocessing in `qiskit_algorithms==0.4.0`. Stick to parameterized `rx`/`ry`/`rz` with non-parameterized entanglers (`cx`).
2. **Exponential scaling**: Runtime and memory scale as $O(2^n)$ with qubit count. For $n > 10$, reverse-mode may become impractical — consider parameter-shift or LCU with a sampler primitive instead.
3. **No external primitive needed**: Unlike other gradient methods, reverse-mode does not accept an estimator/sampler in the constructor. The classes internally satisfy the `BaseEstimatorGradient`/`BaseQGT` interfaces — passing a primitive will raise `TypeError`.
4. **Unsupported gate decomposition**: `derive_circuit()` only handles gates in the API-declared set (`rx`, `ry`, `rz`, `cp`, `crx`, `cry`, `crz`). Decompose any other parameterized gates before calling `run()`.
5. **Phase-fix default**: `ReverseQGT` defaults to `phase_fix=True`. If you need the raw QGT without phase-fix subtraction, set `phase_fix=False` — but note that most natural gradient optimizers expect the phase-fixed version.
