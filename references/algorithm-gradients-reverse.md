# Reverse-Mode Gradient

## Scope and Reference Use

Reverse-mode gradient computes expectation value gradients and QGT entries by traversing parameterized gates in reverse order using statevector-level differentiation. Unlike parameter-shift or LCU, it does not require additional circuit evaluations — it operates directly on statevectors, making it efficient for small circuits.

Scope:
- Circuits are small enough for statevector simulation (exponential scaling with qubit count)
- No external estimator/sampler backend is desired (the algorithm internally satisfies the base interfaces)
- Gradient and QGT are needed through their separate reverse-sweep classes

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

API source: [Qiskit Algorithms 0.4.0](https://github.com/qiskit-community/qiskit-algorithms/blob/0.4.0/qiskit_algorithms/gradients/reverse/reverse_gradient.py); these classes belong to `qiskit_algorithms`, not `unitarylab_algorithms` or `unitarylab.library`. The accompanying UnitaryLab teaching implementation has a separate interface. Examples target the V2 primitives available in Qiskit 2.6.1.

Reference scripts: [reverse_algorithm.py](../scripts/reverse_algorithm.py) (API usage example); [reverse_implementation.py](../scripts/reverse_implementation.py) (independent teaching implementation).

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
- In 0.4.0, `gradient_lookup()` implements `rx`, `ry`, `rz`, `crx`, `cry`, and `crz`, but has no `CPGate` branch even though `cp` is declared supported. A retained `cp` operation can therefore reach `NotImplementedError`; inspect preprocessing and decomposition rather than assuming all declared gates work identically.
- The examples use parameterized `rx`/`ry`/`rz` with fixed `cx` entanglers, avoiding that declared/implemented mismatch.
- Preprocessing assigns unique parameters per differentiated gate and postprocessing applies the chain rule; callers may reuse parameters. Unsupported parameterized gates must be decomposable to the supported set.
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
from qiskit.primitives import StatevectorEstimator
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
fd_grad = FiniteDiffEstimatorGradient(StatevectorEstimator(), epsilon=1e-4, method="central")
fd_result = fd_grad.run([qc], [observable], [values]).result()

print("Reverse-mode gradient:", rev_result.gradients[0])
print("Finite difference:", fd_result.gradients[0])
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

This is incomplete pseudocode: `derive_circuit_generators` is a placeholder and forward/backward state updates are omitted. It is not a runnable reverse-mode implementation. The actual reverse sweep is linear in differentiated blocks for fixed qubit count, while every statevector operation retains exponential qubit cost.

## Debugging Tips

1. **Declared/implemented gate mismatch**: `cp` appears in the supported list but lacks a derivative lookup branch in 0.4.0. Controlled rotations `crx`/`cry`/`crz` do have analytic lookup branches; do not treat an unverified blanket failure claim for all controlled gates as an API restriction.
2. **Exponential scaling**: Statevector memory grows as $2^n$ and runtime also depends on gate and parameter counts. There is no universal ten-qubit cutoff. Parameter-shift/LCU alternatives still require their Estimator or Sampler interfaces as appropriate.
3. **No external primitive needed**: The constructors take derivative settings, not an estimator/sampler argument. Use the documented named arguments; a positional primitive could be misinterpreted as another parameter rather than reliably raising `TypeError`.
4. **Unsupported gate decomposition**: The implemented derivative lookup covers `rx`, `ry`, `rz`, `crx`, `cry`, and `crz`. Public preprocessing may decompose other parameterized gates; the low-level `derive_circuit()` helper also expects unique parameters, which public preprocessing normally supplies.
5. **Phase-fix default**: `ReverseQGT` defaults to `phase_fix=True`. If you need the raw QGT without phase-fix subtraction, set `phase_fix=False` — but note that most natural gradient optimizers expect the phase-fixed version.
