---
name: qfi
description: Compute the Quantum Fisher Information (QFI) matrix for a pure parameterized quantum state using the Quantum Geometric Tensor (QGT). Extracts the real part of the QGT and scales it by 4.
---

# Quantum Fisher Information (QFI)

## Algorithm Overview

| Property | Detail |
|---|---|
| **Category** | Gradient / Geometric Tensor |
| **Primary Class** | `QFI` |
| **Backing Class** | `LinCombQGT` (concrete `BaseQGT` implementation) |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Compute the QFI matrix of a pure parameterized quantum state; used in natural gradient descent, variational optimization, and quantum geometric learning |

`QFI` is an abstract class that wraps a `BaseQGT` instance and extracts the real part of the Quantum Geometric Tensor, scaled by 4. The standard concrete backend is `LinCombQGT`, which uses a linear-combination-of-unitaries technique via an `Estimator` primitive.

---

## Mathematical Principles

For a pure parameterized state $|\psi(\theta)\rangle$, the QFI matrix is defined as:

$$\mathrm{QFI}_{ij} = 4\,\mathrm{Re}\!\left[\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle\right]$$

where $\partial_i \equiv \frac{\partial}{\partial \theta_i}$.

This is obtained from the Quantum Geometric Tensor (QGT):

$$\mathrm{QGT}_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle$$

Relationship: $\mathrm{QFI} = 4\,\mathrm{Re}(\mathrm{QGT})$

The QFI matrix is real, symmetric, and positive semi-definite. It serves as the metric tensor of the quantum state manifold.

---

## Core Parameters

### `QFI` Constructor

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `qgt` | `BaseQGT` | Yes | — | Quantum Geometric Tensor backend (typically `LinCombQGT`) |
| `precision` | `float \| None` | No | `None` | Overrides the `BaseQGT`'s precision; uses `BaseQGT` default if `None` |

### `LinCombQGT` Constructor (standard `BaseQGT` backend)

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `phase_fix` | `bool` | No | `True` | Whether to subtract the phase-fix term $\langle\partial_i\psi\|\psi\rangle\langle\psi\|\partial_j\psi\rangle$ |
| `derivative_type` | `DerivativeType` | No | `DerivativeType.COMPLEX` | Derivative type; `QFI` internally forces `DerivativeType.REAL` during execution |
| `precision` | `float \| None` | No | `None` | Estimator precision; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

---

## Inputs and Outputs

### `QFI.run()`

**Input**:

| Argument | Type | Description |
|---|---|---|
| `circuits` | `Sequence[QuantumCircuit]` | Parameterized circuits to compute QFI for |
| `parameter_values` | `Sequence[Sequence[float]]` | Parameter values for each circuit |
| `parameters` | `Sequence[Sequence[Parameter] \| None] \| None` | Subset of parameters to differentiate; `None` differentiates all parameters |
| `precision` | `float \| Sequence[float] \| None` | Per-call precision override; falls back to `QFI.precision` then primitive default |

**Returns**: `AlgorithmJob` — call `.result()` to obtain a `QFIResult`.

**Output** (`QFIResult`):

| Field | Type | Description |
|---|---|---|
| `qfis` | `list[np.ndarray]` | QFI matrices, one per circuit; shape `(num_params, num_params)` |
| `metadata` | `dict[str, Any]` | Additional job metadata from the underlying QGT |
| `precision` | `float \| Sequence[float]` | Precision resolved from the primitive |

---

## Implementation Notes

- `QFI` is an **abstract class**; instantiate via a concrete `BaseQGT` such as `LinCombQGT`.
- During `_run`, `QFI` temporarily sets `qgt.derivative_type = DerivativeType.REAL`, then restores the original value after the job completes.
- The QFI matrix is computed as `4 * qgt_result.real` for each circuit result.
- `LinCombQGT` requires gates from its `SUPPORTED_GATES` list (`rx`, `ry`, `rz`, `cx`, `h`, etc.). Circuits with unsupported gates must be decomposed before being passed in.
- A single `AlgorithmJob` wraps the underlying estimator job; errors from the estimator surface as `AlgorithmError`.
- `QFI.precision` is a settable property that propagates to the underlying `BaseQGT` at call time.

---

## Sample Code

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import QFI, LinCombQGT

# Build a parameterized circuit
theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

parameter_values = [0.5, 1.0]

# Construct QFI using LinCombQGT as the QGT backend
estimator = StatevectorEstimator()
qgt = LinCombQGT(estimator)
qfi = QFI(qgt)

# Run and retrieve results
job = qfi.run([qc], [parameter_values])
result = job.result()

print(result.qfis)      # [array([[...], [...]])]  shape (2, 2) per circuit
print(result.precision) # precision used by the estimator
```
