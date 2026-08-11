---
name: linear-combination
description: "Analytic quantum gradient and Quantum Geometric Tensor (QGT) estimation via linear combination of unitaries (LCU). Supports Estimator (expectation value gradients), Sampler (probability distribution gradients), and QGT primitives with configurable derivative types. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Linear Combination of Unitaries (LCU) Gradient

## Overview

| Property | Detail |
|---|---|
| **Category** | Analytic Gradient Estimation / Quantum Geometric Tensor |
| **Primitives** | `LinCombEstimatorGradient`, `LinCombSamplerGradient`, `LinCombQGT` |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Analytically compute parameter gradients and QGT of parameterized quantum circuits without numerical error |

The LCU method constructs augmented circuits with an ancilla qubit to evaluate parameter gradients analytically. Each parameterized gate is replaced by a controlled derivative operator, allowing exact gradient computation via expectation value measurements. Unlike finite difference methods, LCU introduces no numerical approximation error but requires gates from a supported set.

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

### `LinCombEstimatorGradient.run()`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision; uses primitive default if `None` |
| `derivative_type` | `DerivativeType` | No | `DerivativeType.REAL` | Selects real, imaginary, or complex gradient output |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `LinCombSamplerGradient.run()`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampler` | `BaseSamplerV2` | Yes | — | Sampler primitive for probability distribution computation |
| `shots` | `int \| None` | No | `None` | Overrides default primitive shot count; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `LinCombQGT.run()`

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
| `qgts` | `list[np.ndarray]` | QGT matrices, one per circuit; shape `(num_params, num_params)`; dtype `complex` |
| `metadata` | `list[dict]` | Per-circuit metadata; includes `"parameters"` key |

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
| `REAL` | $2\,\mathrm{Re}[\langle\psi(\omega)|O(\theta)|\partial_\omega\psi(\omega)\rangle]$ |
| `IMAG` | $2\,\mathrm{Im}[\langle\psi(\omega)|O(\theta)|\partial_\omega\psi(\omega)\rangle]$ |
| `COMPLEX` | $2\langle\psi(\omega)|O(\theta)|\partial_\omega\psi(\omega)\rangle$ |

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

---

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Linear Combination of Unitaries (LCU) Gradient.

The LCU method constructs augmented circuits with an ancilla qubit to evaluate parameter gradients analytically. Each parameterized gate is replaced by a controlled derivative operator, enabling exact gradient computation via expectation value measurements.

Use this skill when:
- Exact analytic gradients are required without numerical approximation error
- Circuits use gates from the LCU supported gate set
- Quantum Geometric Tensor (QGT) computation is also needed alongside gradients

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

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
5. **QGT integration**: `LinCombQGT` caches QGT circuits separately using `(param_i, param_j)` pairs. Only upper-triangular entries are computed, then reflected to fill the symmetric matrix.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Parameter vector $\theta$ | `parameter_values` — `Sequence[Sequence[float]]` |
| Gradient $\partial\langle O\rangle/\partial\theta_i$ | `result.gradients[i]` — dtype depends on `DerivativeType` |
| Augmented circuit per parameter | `_make_lin_comb_gradient_circuit()` — one ancilla qubit added |
| Circuit cache key | `_circuit_key(circuit)` — hashes circuit structure for reuse |
| `DerivativeType.REAL/IMAG/COMPLEX` | Constructor `derivative_type`; controls output dtype and PUB count |
| QGT $Q_{ij}$ | `result.qgts[i]` — shape `(n, n)`, symmetric, dtype `complex` when `COMPLEX` |
| Phase-fix term | `phase_fix=True` subtracts $\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle$ |
| Supported gates check | `_preprocess()` — validates circuits against `SUPPORTED_GATES` |

## Hands-On Example

Compute both gradient and QGT for the same circuit:

```python
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

def lin_comb_gradient_simplified(eval_fn, theta, derivative_type="real"):
    """Simplified LCU-style gradient: analytically exact via derivative operators.

    In practice, the full implementation constructs ancilla-augmented circuits
    per parameter. This skeleton shows the mathematical structure.
    """
    n = len(theta)
    grad = np.zeros(n, dtype=complex if derivative_type == "complex" else float)

    for i in range(n):
        # In real LCU: build circuit with ancilla, controlled derivative at position i
        # Here we represent the analytic derivative pair evaluation conceptually
        re_val = eval_fn(theta, param_idx=i, observable="X_on_ancilla")
        if derivative_type == "real":
            grad[i] = 2.0 * np.real(re_val)
        elif derivative_type == "imag":
            grad[i] = 2.0 * np.imag(re_val)
        else:  # complex
            im_val = eval_fn(theta, param_idx=i, observable="Y_on_ancilla")
            grad[i] = re_val + 1j * im_val  # combines both ancilla measurements

    return grad
```

Note: This skeleton illustrates the conceptual structure. The actual implementation in `qiskit_algorithms` constructs ancilla-augmented circuits via `_make_lin_comb_gradient_circuit()` and batches all evaluations into one primitive job.

## Debugging Tips

1. **Unsupported gates**: LCU's `SUPPORTED_GATES` set differs from parameter-shift's. Gates like `ccx` and `swap` are supported here but not in parameter-shift. Verify your circuits are compatible.
2. **COMPLEX doubles PUB count**: When using `DerivativeType.COMPLEX`, each augmented circuit is evaluated twice (real + imaginary observables), so the primitive job has $2n$ entries instead of $n$. Plan primitive resources accordingly.
3. **Circuit cache invalidation**: Cached augmented circuits are keyed by circuit structure. If you modify a circuit after the first `run()` call, the cache may return stale results. Create a new gradient object if the circuit changes.
4. **Phase-fix for QGT**: When `phase_fix=False`, the QGT differs from the standard metric definition. Most variational algorithms (natural gradient, QNG) expect `phase_fix=True` (default).
5. **Ancilla qubit indexing**: The augmented circuit has `num_qubits + 1` qubits. If your circuit already uses all available hardware qubits, the ancilla addition will exceed capacity — verify qubit count before execution.
