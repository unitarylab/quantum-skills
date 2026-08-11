---
name: qfi
description: "Compute the Quantum Fisher Information (QFI) matrix for a pure parameterized quantum state using the Quantum Geometric Tensor (QGT). Extracts the real part of the QGT and scales it by 4. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Quantum Fisher Information (QFI)

## Overview

| Property | Detail |
|---|---|
| **Category** | Gradient / Geometric Tensor |
| **Primary Class** | `QFI` |
| **Backing Class** | `LinCombQGT` (concrete `BaseQGT` implementation) |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Compute the QFI matrix of a pure parameterized quantum state; used in natural gradient descent, variational optimization, and quantum geometric learning |

`QFI` is an abstract class that wraps a `BaseQGT` instance and extracts the real part of the Quantum Geometric Tensor, scaled by 4. The standard concrete backend is `LinCombQGT`, which uses a linear-combination-of-unitaries technique via an `Estimator` primitive.

---

## Reference Implementation Example

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

## Core Parameters Explained

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

## Return Fields

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

## Implementation Architecture

- `QFI` is an **abstract class**; instantiate via a concrete `BaseQGT` such as `LinCombQGT`.
- During `_run`, `QFI` temporarily sets `qgt.derivative_type = DerivativeType.REAL`, then restores the original value after the job completes.
- The QFI matrix is computed as `4 * qgt_result.real` for each circuit result.
- `LinCombQGT` requires gates from its `SUPPORTED_GATES` list (`rx`, `ry`, `rz`, `cx`, `h`, etc.). Circuits with unsupported gates must be decomposed before being passed in.
- A single `AlgorithmJob` wraps the underlying estimator job; errors from the estimator surface as `AlgorithmError`.
- `QFI.precision` is a settable property that propagates to the underlying `BaseQGT` at call time.

---

## Mathematical Deep Dive

For a pure parameterized state $|\psi(\theta)\rangle$, the QFI matrix is defined as:

$$\mathrm{QFI}_{ij} = 4\,\mathrm{Re}\!\left[\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle\right]$$

where $\partial_i \equiv \frac{\partial}{\partial \theta_i}$.

This is obtained from the Quantum Geometric Tensor (QGT):

$$\mathrm{QGT}_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle$$

Relationship: $\mathrm{QFI} = 4\,\mathrm{Re}(\mathrm{QGT})$

The QFI matrix is real, symmetric, and positive semi-definite. It serves as the metric tensor of the quantum state manifold.

---

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Quantum Fisher Information (QFI).

QFI computes the quantum Fisher information matrix — the real part of the Quantum Geometric Tensor (QGT) scaled by 4. It wraps a `BaseQGT` backend (typically `LinCombQGT`) and is used in natural gradient descent, variational optimization, and quantum geometric learning.

Use this skill when:
- Computing the metric tensor of a parameterized quantum state manifold
- Implementing quantum natural gradient descent
- Analyzing parameter sensitivity and state distinguishability

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Prerequisites

- NumPy
- Qiskit parameterized circuits with gates from the chosen QGT backend's supported set
- Estimator primitive (`BaseEstimatorV2`)
- Understanding of the Quantum Geometric Tensor and its relationship to the QFI: $\text{QFI} = 4\,\text{Re}(\text{QGT})$

## Understanding the Key Quantum Components

1. **QFI as a QGT wrapper**: `QFI` is an abstract class that delegates to a concrete `BaseQGT` instance (typically `LinCombQGT`). It does not perform any quantum computation directly — it post-processes QGT results.
2. **DerivativeType override**: During `_run()`, `QFI` temporarily sets `qgt.derivative_type = DerivativeType.REAL` (since only the real part of the QGT contributes to QFI), then restores the original value after the job completes.
3. **Scaling factor**: QFI = $4 \times \mathrm{Re}(\mathrm{QGT})$. The factor 4 comes from the standard definition of QFI for pure states relative to the QGT.
4. **Backend responsibility**: The `BaseQGT` backend (`LinCombQGT`) handles gate compatibility checks, ancilla-augmented circuit construction, and primitive job submission. QFI only scales and extracts the real part.
5. **Precision propagation**: `QFI.precision` is a settable property that propagates to the underlying `BaseQGT` at call time, allowing per-experiment precision control without reconstructing the backend.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| QGT $Q_{ij}$ | `qgt.run(...).result().qgts` — complex-valued, computed by backend |
| QFI $F_{ij} = 4\,\mathrm{Re}(Q_{ij})$ | `result.qfis` — `4 * qgt_result.real` per circuit |
| Derivative type forced to REAL | `_run()` temporarily sets `qgt.derivative_type = DerivativeType.REAL` |
| QGT backend | Constructor argument `qgt`; typically `LinCombQGT(estimator)` |
| Precision override | `QFI.precision` property → propagates to `qgt.precision` |
| Result format | `QFIResult` with `.qfis` (list of `np.ndarray`) |

## Hands-On Example

Compare QFI and QGT to verify the $4 \times \mathrm{Re}$ relationship:

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import QFI, LinCombQGT

theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

values = [0.5, 1.0]
estimator = StatevectorEstimator()

# QGT
qgt = LinCombQGT(estimator)
qgt_result = qgt.run([qc], [values]).result()
qgt_matrix = qgt_result.qgts[0]

# QFI
qfi = QFI(qgt)
qfi_result = qfi.run([qc], [values]).result()
qfi_matrix = qfi_result.qfis[0]

# Verify relationship
expected_qfi = 4.0 * np.real(qgt_matrix)
print("QFI == 4*Re(QGT):", np.allclose(qfi_matrix, expected_qfi))
print("QFI shape:", qfi_matrix.shape)
```

## Minimal Manual Implementation

```python
import numpy as np

def compute_qfi(qgt_backend, circuits, parameter_values):
    """Compute QFI from QGT backend — mirrors QFI._run() logic.

    In practice, the Qiskit QFI class temporarily overrides the QGT's
    derivative_type to REAL, then scales the real part by 4.
    """
    # Store original derivative type
    original_dtype = getattr(qgt_backend, 'derivative_type', None)

    # Force REAL (only real part contributes to QFI)
    qgt_backend.derivative_type = "REAL"  # conceptually DerivativeType.REAL

    # Run QGT
    qgt_job = qgt_backend.run(circuits, parameter_values)
    qgt_result = qgt_job.result()

    # Restore original derivative type
    if original_dtype is not None:
        qgt_backend.derivative_type = original_dtype

    # QFI = 4 * Re(QGT)
    qfis = [4.0 * np.real(qgt_mat) for qgt_mat in qgt_result.qgts]
    return qfis
```

## Debugging Tips

1. **Backend derivative_type reset**: If you set `qgt.derivative_type = DerivativeType.COMPLEX` and then run `QFI`, the derivative type is reset to `REAL` internally. After `QFI.run()`, check that the backend's derivative type has been restored to your original value.
2. **Gate compatibility**: Gate restrictions come from the QGT backend, not from QFI itself. If `qgt.run()` fails with unsupported gate errors, the issue is in the `LinCombQGT` (or other backend) configuration.
3. **QFI is always real**: Unlike QGT which can be complex, QFI matrices are always real and symmetric. If you see complex values in `result.qfis`, the backend's `REAL` override may not have been applied — verify the `QFI` class version.
4. **Precision chaining**: `QFI.precision` → `qgt.precision` → `estimator.precision`. If you set precision on the estimator directly, it may be overridden by `QFI.precision`. Set precision on `QFI` to ensure consistent propagation.
