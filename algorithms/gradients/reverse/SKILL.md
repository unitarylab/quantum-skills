---
name: reverse
description: Reverse-mode statevector gradient and QGT computation for parameterized circuits using Qiskit Algorithms classes ReverseEstimatorGradient and ReverseQGT.
---

# Reverse-Mode Gradient

## Algorithm Overview

| Property | Detail |
|---|---|
| Category | Gradient / Quantum Geometric Tensor |
| Standard APIs | `ReverseEstimatorGradient`, `ReverseQGT` |
| Framework | `qiskit_algorithms.gradients` |
| Core idea | Reverse sweep on statevectors to compute derivatives without parameter-shift circuits |

This algorithm computes expectation gradients and QGT entries by traversing parameterized gates in reverse order. It is optimized for small circuits and exact statevector evaluation.

## Mathematical Principles

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

## Core Parameters

### `ReverseEstimatorGradient`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `derivative_type` | `DerivativeType` | No | `DerivativeType.REAL` | Output projection: `REAL`, `IMAG`, or `COMPLEX` |

### `ReverseQGT`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `phase_fix` | `bool` | No | `True` | Apply phase-fix subtraction term |
| `derivative_type` | `DerivativeType` | No | `DerivativeType.COMPLEX` | Return real, imaginary, or complex QGT |

## Inputs and Outputs

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

## Implementation Description

| Component | Role |
|---|---|
| `split()` | Splits a circuit into per-parameter unitary blocks |
| `derive_circuit()` | Builds analytic derivative terms `(coeff, QuantumCircuit)` |
| `bind()` | Binds numeric parameter values into circuits |
| `ReverseEstimatorGradient` | Reverse sweep with two statevectors for $O(P)$ parameter scaling |
| `ReverseQGT` | Reverse sweep with three statevectors for $O(P^2)$ parameter scaling |

Engineering constraints:

- Supported parameterized gates: `rx`, `ry`, `rz`, `cp`, `crx`, `cry`, `crz`.
- Circuits should use unique parameters per gate path; unsupported gates must be decomposed first.
- Runtime scales exponentially with qubit count due to statevector simulation.
- No external estimator backend is required by users; classes internally satisfy base interfaces.

## Sample Code

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
