# Linear Combination of Unitaries (LCU) Gradient

## Scope and Reference Use

The LCU method constructs augmented circuits with an ancilla qubit to evaluate parameter gradients analytically. Each parameterized gate is replaced by a controlled derivative operator, enabling exact gradient computation via expectation value measurements.

Scope:
- Analytic derivatives without finite-difference truncation error
- Circuits use gates from the LCU supported gate set
- Quantum Geometric Tensor (QGT) computation is also needed alongside gradients

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

API source: [Qiskit Algorithms 0.4.0](https://github.com/qiskit-community/qiskit-algorithms/blob/0.4.0/qiskit_algorithms/gradients/lin_comb/lin_comb_estimator_gradient.py); these classes belong to `qiskit_algorithms`, not `unitarylab_algorithms` or `unitarylab.library`. The accompanying UnitaryLab teaching implementation has a separate interface. Examples target the V2 primitives available in Qiskit 2.6.1.

Reference scripts: [linear_combination_algorithm.py](../scripts/linear_combination_algorithm.py) (API usage example); [linear_combination_implementation.py](../scripts/linear_combination_implementation.py) (independent teaching implementation).

## Overview

| Property | Detail |
|---|---|
| **Category** | Analytic Gradient Estimation / Quantum Geometric Tensor |
| **Primitives** | `LinCombEstimatorGradient`, `LinCombSamplerGradient`, `LinCombQGT` |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Analytically compute parameter gradients and QGT without finite-difference truncation error |

The LCU method augments circuits with an ancilla to evaluate analytic derivative overlaps. It avoids finite-difference truncation error for supported gate generators; floating-point, finite-shot, and device errors still affect estimates.

---

## Reference Implementation Example

### Estimator Gradient

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import LinCombEstimatorGradient
from qiskit_algorithms.gradients.utils import DerivativeType

# Build a parameterized circuit using only SUPPORTED_GATES
theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

observable = SparsePauliOp("ZZ")
parameter_values = [0.5, 1.0]

estimator = StatevectorEstimator()
gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.REAL)

result = gradient.run([qc], [observable], [parameter_values]).result()
print(result.gradients)   # [array([dE/dθ₀, dE/dθ₁])]
```

### Sampler Gradient

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.gradients import LinCombSamplerGradient

theta = ParameterVector("θ", 1)
qc = QuantumCircuit(1)
qc.ry(theta[0], 0)
qc.measure_all()

parameter_values = [0.5]

sampler = StatevectorSampler()
gradient = LinCombSamplerGradient(sampler)

result = gradient.run([qc], [parameter_values]).result()
print(result.gradients)   # [[{0: dP(0)/dθ₀, 1: dP(1)/dθ₀}]]
```

### Quantum Geometric Tensor

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import LinCombQGT
from qiskit_algorithms.gradients.utils import DerivativeType

theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

parameter_values = [0.5, 1.0]

estimator = StatevectorEstimator()
qgt = LinCombQGT(estimator, phase_fix=True, derivative_type=DerivativeType.COMPLEX)

result = qgt.run([qc], [parameter_values]).result()
print(result.qgts)   # [array([[QGT₀₀, QGT₀₁], [QGT₁₀, QGT₁₁]], dtype=complex)]
```

## Core Parameters Explained

### `LinCombEstimatorGradient(...)` constructor

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision; uses primitive default if `None` |
| `derivative_type` | `DerivativeType` | No | `DerivativeType.REAL` | Selects real, imaginary, or complex gradient output |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `LinCombSamplerGradient(...)` constructor

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampler` | `BaseSamplerV2` | Yes | — | Sampler primitive for probability distribution computation |
| `shots` | `int \| None` | No | `None` | Overrides default primitive shot count; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `LinCombQGT(...)` constructor

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive used for both QGT and phase-fix gradient |
| `phase_fix` | `bool` | No | `True` | Whether to subtract the phase-fix term from the QGT |
| `derivative_type` | `DerivativeType` | No | `DerivativeType.COMPLEX` | Selects real, imaginary, or complex QGT output |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### Supported Gates

All three classes share the same `SUPPORTED_GATES` constraint. Input circuits must only use gates from this set (or gates decomposable into it):

`rx`, `ry`, `rz`, `rzx`, `rzz`, `ryy`, `rxx`, `cx`, `cy`, `cz`, `ccx`, `swap`, `iswap`, `h`, `t`, `s`, `sdg`, `x`, `y`, `z`

---

## Return Fields

### `LinCombEstimatorGradient`

**Input** (via `.run()`):

| Argument | Type | Description |
|---|---|---|
| `circuits` | `Sequence[QuantumCircuit]` | Parameterized circuits to differentiate |
| `observables` | `Sequence[BaseOperator]` | Observables whose expectation value gradients are computed |
| `parameter_values` | `Sequence[Sequence[float]]` | Current parameter values for each circuit |
| `parameters` | `Sequence[Sequence[Parameter]] \| None` | Subset of parameters to differentiate; `None` differentiates all |

**Output**: `EstimatorGradientResult`

| Field | Type | Description |
|---|---|---|
| `gradients` | `list[np.ndarray]` | Gradient arrays, one per circuit; dtype is `float` (`REAL`/`IMAG`) or `complex` (`COMPLEX`); shape `(num_params,)` |
| `metadata` | `list[dict]` | Per-circuit metadata; includes `"parameters"` and `"derivative_type"` keys |
| `precision` | `float \| list[float]` | Precision used (resolved from primitive if not explicitly set) |

### `LinCombSamplerGradient`

**Input** (via `.run()`):

| Argument | Type | Description |
|---|---|---|
| `circuits` | `Sequence[QuantumCircuit]` | Parameterized circuits with measurements |
| `parameter_values` | `Sequence[Sequence[float]]` | Current parameter values for each circuit |
| `parameters` | `Sequence[Sequence[Parameter] \| None] \| None` | Subset of parameters to differentiate; `None` differentiates all |

**Output**: `SamplerGradientResult`

| Field | Type | Description |
|---|---|---|
| `gradients` | `list[list[dict[int, float]]]` | Per-circuit gradient distributions; each element is a list of dicts mapping bitstring int to gradient value |
| `metadata` | `list[dict]` | Per-circuit metadata; includes `"parameters"` key |
| `shots` | `int \| list[int]` | Shot count used (resolved from primitive if not explicitly set) |

### `LinCombQGT`

**Input** (via `.run()`):

| Argument | Type | Description |
|---|---|---|
| `circuits` | `Sequence[QuantumCircuit]` | Parameterized circuits |
| `parameter_values` | `Sequence[Sequence[float]]` | Current parameter values for each circuit |
| `parameters` | `Sequence[Sequence[Parameter]] \| None` | Subset of parameters; `None` uses all |

**Output**: `QGTResult`

| Field | Type | Description |
|---|---|---|
| `qgts` | `list[np.ndarray]` | Matrices of shape `(num_params, num_params)`; real-valued for REAL/IMAG projection and complex for COMPLEX mode. |
| `metadata` | `list[dict]` | Per-circuit metadata; includes `"parameters"` key |
| `derivative_type` | `DerivativeType` | Projection used for the returned QGT. |
| `precision` | float or sequence | Resolved estimator precision. |

---

## Implementation Architecture

- All three classes extend their respective base classes from `qiskit_algorithms.gradients.base`.
- The LCU method augments the input circuit with **one ancilla qubit** via `_make_lin_comb_gradient_circuit`. A separate augmented circuit is generated per target parameter.
- All augmented circuit evaluations for a single gradient call are batched into **one primitive job** to minimize overhead.
- LCU gradient circuits are cached in `_lin_comb_cache` (keyed by `_circuit_key(circuit)`) and reused across calls on the same circuit.
- `LinCombQGT` caches QGT circuits separately in `_lin_comb_qgt_circuit_cache` using `(param_i, param_j)` pairs; only upper-triangular entries are computed, then reflected.
- `LinCombQGT` internally instantiates a `LinCombEstimatorGradient` (with `DerivativeType.COMPLEX`) to compute the phase-fix term when `phase_fix=True`.
- For `DerivativeType.COMPLEX`, gradient circuits are evaluated twice (once for real, once for imaginary observables), doubling the PUB count.
- For `LinCombSamplerGradient`, measurements are injected into the augmented circuit (`add_measurement=True`); for `LinCombEstimatorGradient`, they are not (`add_measurement=False`).
- If a `transpiler` is provided, circuits are transpiled before execution and observables are re-laid-out via `observable.apply_layout(circuit.layout)`.
- Input circuits must only contain gates from `SUPPORTED_GATES`. Unsupported gates will cause a preprocessing error during `_preprocess()`.

---

## Mathematical Deep Dive

Let $|\psi(\theta)\rangle$ be the state produced by a parameterized circuit at parameter vector $\theta \in \mathbb{R}^n$, and let $O$ be an observable.

**Estimator gradient** (expectation value derivative):

$$\frac{\partial}{\partial \theta_i}\langle\psi(\theta)|O|\psi(\theta)\rangle = 2\,\mathrm{Re}\!\left[\langle\psi(\theta)|O|\partial_{\theta_i}\psi(\theta)\rangle\right]$$

The three `DerivativeType` modes compute:

| `DerivativeType` | Formula |
|---|---|
| `REAL` | $2\operatorname{Re}\langle\psi\rvert O\lvert\partial_i\psi\rangle$ |
| `IMAG` | $2\operatorname{Im}\langle\psi\rvert O\lvert\partial_i\psi\rangle$ |
| `COMPLEX` | $2\langle\psi\rvert O\lvert\partial_i\psi\rangle$ |

**Quantum Geometric Tensor (QGT)**:

$$\text{QGT}_{ij} = \langle\partial_i\psi|\partial_j\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle$$

The phase-fix term $\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle$ is subtracted when `phase_fix=True` (default).

**Circuit evaluations per parameter** (for $n$ parameters):

| Class | `DerivativeType` | Circuit Evaluations |
|---|---|---|
| `LinCombEstimatorGradient` | `REAL` / `IMAG` | $n$ |
| `LinCombEstimatorGradient` | `COMPLEX` | $2n$ |
| `LinCombSamplerGradient` | — | $n$ |
| `LinCombQGT` | `REAL` / `IMAG` | $n(n+1)/2$ |
| `LinCombQGT` | `COMPLEX` | $n(n+1)$ |

These counts concern independent differentiated gate parameters after preprocessing and the base derivative-overlap terms. Reused parameters can expand the internal circuit, and `phase_fix=True` adds a separate complex-gradient evaluation. They are not shot counts or universal costs in the number of requested source parameters.

---

## Prerequisites

- NumPy
- Qiskit parameterized circuits composed of gates from the LCU supported gate set
- Estimator primitives (`BaseEstimatorV2`) for expectation-value gradients; Sampler primitives for probability gradients
- Understanding of the linear combination of unitaries (LCU) framework: controlled derivative operators, ancilla qubits, and analytic gradient reconstruction

## Understanding the Key Quantum Components

1. **Supported gate set**: `rx`, `ry`, `rz`, `rzx`, `rzz`, `ryy`, `rxx`, `cx`, `cy`, `cz`, `ccx`, `swap`, `iswap`, `h`, `t`, `s`, `sdg`, `x`, `y`, `z` — broader than parameter-shift's set (includes `ccx`, `swap`, `iswap`).
2. **Ancilla-augmented circuits**: `_make_lin_comb_gradient_circuit()` adds one ancilla qubit and replaces each target parameterized gate with a controlled derivative operator. A separate augmented circuit is generated per target parameter.
3. **Circuit caching**: Augmented circuits are cached in `_lin_comb_cache` (keyed by `_circuit_key(circuit)`) and reused across calls — eliminates redundant circuit construction for repeated gradient evaluations on the same circuit structure.
4. **DerivativeType modes**: `REAL` computes $2\,\mathrm{Re}[\langle\psi|O|\partial_i\psi\rangle]$; `IMAG` computes the imaginary part; `COMPLEX` evaluates the augmented circuit twice (real and imaginary observables), doubling the PUB count.
5. **QGT integration**: `LinCombQGT` caches parameter-pair circuits. The complex QGT is completed with the conjugate-transposed upper triangle, producing a Hermitian matrix; its real projection is symmetric and its imaginary projection antisymmetric.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Parameter vector $\theta$ | `parameter_values` — `Sequence[Sequence[float]]` |
| Gradient $\partial\langle O\rangle/\partial\theta_i$ | `result.gradients[i]` — dtype depends on `DerivativeType` |
| Augmented circuit per parameter | `_make_lin_comb_gradient_circuit()` — one ancilla qubit added |
| Circuit cache key | `_circuit_key(circuit)` — hashes circuit structure for reuse |
| `DerivativeType.REAL/IMAG/COMPLEX` | Constructor `derivative_type`; controls output dtype and PUB count |
| QGT $Q_{ij}$ | `result.qgts[i]` — shape `(n, n)`; Hermitian in `COMPLEX` mode, real symmetric in `REAL` mode. |
| Phase-fix term | `phase_fix=True` subtracts $\langle\partial_i\psi\vert\psi\rangle\langle\psi\vert\partial_j\psi\rangle$. |
| Supported gates check | `_preprocess()` — validates circuits against `SUPPORTED_GATES` |

## Hands-On Example

Compute both gradient and QGT for the same circuit:

```python
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import LinCombEstimatorGradient, LinCombQGT
from qiskit_algorithms.gradients.utils import DerivativeType

theta = ParameterVector("θ", 3)
qc = QuantumCircuit(3)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.rz(theta[2], 2)
qc.cx(0, 1)
qc.cx(1, 2)

values = [0.5, -0.3, 1.2]
observable = SparsePauliOp("ZZZ")
estimator = StatevectorEstimator()

# Gradient
grad = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.REAL)
grad_result = grad.run([qc], [observable], [values]).result()
print("Gradient:", grad_result.gradients[0])

# QGT
qgt = LinCombQGT(estimator, phase_fix=True, derivative_type=DerivativeType.COMPLEX)
qgt_result = qgt.run([qc], [values]).result()
print("QGT shape:", qgt_result.qgts[0].shape)  # (3, 3)
print("QGT is Hermitian:", np.allclose(qgt_result.qgts[0], qgt_result.qgts[0].conj().T))
```

## Minimal Manual Implementation

```python
import numpy as np

def lin_comb_gradient_simplified(overlap_fn, theta, derivative_type="real"):
    """Postprocess supplied overlaps <psi|O|partial_i psi>.

    The callback must supply the analytic complex overlap; this fragment
    does not construct or simulate the ancilla LCU circuits.
    """
    n = len(theta)
    grad = np.zeros(n, dtype=complex if derivative_type == "complex" else float)

    for i in range(n):
        overlap = overlap_fn(theta, param_idx=i)
        if derivative_type == "real":
            grad[i] = 2.0 * np.real(overlap)
        elif derivative_type == "imag":
            grad[i] = 2.0 * np.imag(overlap)
        else:  # complex
            grad[i] = 2.0 * overlap

    return grad
```

Note: This skeleton illustrates the conceptual structure. The actual implementation in `qiskit_algorithms` constructs ancilla-augmented circuits via `_make_lin_comb_gradient_circuit()` and batches all evaluations into one primitive job.

## Debugging Tips

1. **Unsupported gates**: LCU's `SUPPORTED_GATES` set differs from parameter-shift's. Gates like `ccx` and `swap` are supported here but not in parameter-shift. Verify your circuits are compatible.
2. **COMPLEX doubles PUB count**: When using `DerivativeType.COMPLEX`, each augmented circuit is evaluated twice (real + imaginary observables), so the primitive job has $2n$ entries instead of $n$. Plan primitive resources accordingly.
3. **Circuit cache diagnostics**: Caches use circuit-derived keys. For inconsistent repeated results, compare circuit structure, parameter expressions/order, primitive settings, and cache key before attributing differences to stale data.
4. **Phase-fix for QGT**: When `phase_fix=False`, the QGT differs from the standard metric definition. Most variational algorithms (natural gradient, QNG) expect `phase_fix=True` (default).
5. **Ancilla qubit indexing**: The augmented circuit has `num_qubits + 1` qubits. If your circuit already uses all available hardware qubits, the ancilla addition will exceed capacity — verify qubit count before execution.
