---
name: parameter-shift
description: "Analytic quantum gradient estimation via the parameter shift rule. Supports both Estimator (expectation value gradients) and Sampler (probability distribution gradients) primitives. Requires circuits composed exclusively of supported gate types. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Parameter Shift Gradient

## Overview

| Property | Detail |
|---|---|
| **Category** | Gradient Estimation |
| **Primitives** | `ParamShiftEstimatorGradient`, `ParamShiftSamplerGradient` |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Analytically compute parameter gradients of parameterized quantum circuits |

The parameter shift rule computes exact analytic gradients of parameterized quantum circuits by evaluating the circuit at two shifted parameter values ($\theta \pm \pi/2$). Unlike numerical methods, it introduces no approximation error and is directly executable on quantum hardware. It is restricted to circuits whose gates belong to a supported set of single-parameter rotation and controlled gates.

---

## Reference Implementation Example

### Estimator Gradient

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

# Build a simple parameterized circuit
theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

observable = SparsePauliOp("ZZ")
parameter_values = [0.5, 1.0]

estimator = StatevectorEstimator()
gradient = ParamShiftEstimatorGradient(estimator)

result = gradient.run([qc], [observable], [parameter_values]).result()
print(result.gradients)   # [array([dE/dθ₀, dE/dθ₁])]
```

### Sampler Gradient

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.gradients import ParamShiftSamplerGradient

theta = ParameterVector("θ", 1)
qc = QuantumCircuit(1)
qc.ry(theta[0], 0)
qc.measure_all()

parameter_values = [0.5]

sampler = StatevectorSampler()
gradient = ParamShiftSamplerGradient(sampler)

result = gradient.run([qc], [parameter_values]).result()
print(result.gradients)   # [[{0: dP(0)/dθ₀, 1: dP(1)/dθ₀}]]
```

## Core Parameters Explained

### `ParamShiftEstimatorGradient.run()`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `ParamShiftSamplerGradient.run()`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampler` | `BaseSamplerV2` | Yes | — | Sampler primitive for probability distribution computation |
| `shots` | `int \| None` | No | `None` | Overrides default primitive shot count; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

---

## Return Fields

### `ParamShiftEstimatorGradient`

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
| `gradients` | `list[np.ndarray]` | Gradient arrays, one per circuit; shape `(num_params,)` |
| `metadata` | `list[dict]` | Per-circuit metadata, includes `"parameters"` key |
| `precision` | `float \| list[float]` | Precision used (resolved from primitive if not explicitly set) |

### `ParamShiftSamplerGradient`

**Input** (via `.run()`):

| Argument | Type | Description |
|---|---|---|
| `circuits` | `Sequence[QuantumCircuit]` | Parameterized circuits to differentiate |
| `parameter_values` | `Sequence[Sequence[float]]` | Current parameter values for each circuit |
| `parameters` | `Sequence[Sequence[Parameter] \| None] \| None` | Subset of parameters to differentiate; `None` differentiates all |

**Output**: `SamplerGradientResult`

| Field | Type | Description |
|---|---|---|
| `gradients` | `list[list[dict[int, float]]]` | Per-circuit gradient distributions; each element is a list of dicts mapping bitstring int to gradient value |
| `metadata` | `list[dict]` | Per-circuit metadata, includes `"parameters"` key |
| `shots` | `int \| list[int]` | Shot count used (resolved from primitive if not explicitly set) |

---

## Implementation Architecture

- Both classes extend `BaseEstimatorGradient` / `BaseSamplerGradient` from `qiskit_algorithms.gradients.base`.
- Circuits are preprocessed via `_preprocess()`, which decomposes unsupported gates into the `SUPPORTED_GATES` set before differentiation.
- Shifted parameter values are generated by `_make_param_shift_parameter_values()` from `qiskit_algorithms.gradients.utils`.
- All $2n$ shifted circuit evaluations for a single gradient call are batched into **one primitive job**.
- Gradient is computed as: `gradient[i] = (evs[:n//2] - evs[n//2:]) / 2` over the batched result array.
- If a `transpiler` is provided, circuits are transpiled before execution. For `ParamShiftEstimatorGradient`, observables are re-laid-out via `observable.apply_layout(circuit.layout)`.
- Gates outside `SUPPORTED_GATES` are decomposed during preprocessing; circuits using unsupported gates that cannot be decomposed will raise an error.

**Supported gates:**

`x`, `y`, `z`, `h`, `rx`, `ry`, `rz`, `p`, `cx`, `cy`, `cz`, `ryy`, `rxx`, `rzz`, `rzx`

---

## Mathematical Deep Dive

Let $f(\theta)$ be the expectation value or sampling probability of a parameterized circuit with parameter vector $\theta \in \mathbb{R}^n$. For any gate generator with eigenvalues $\pm\frac{1}{2}$, the exact gradient with respect to parameter $\theta_i$ is:

$$\frac{\partial f}{\partial \theta_i} = \frac{f\!\left(\theta + \frac{\pi}{2}\hat{e}_i\right) - f\!\left(\theta - \frac{\pi}{2}\hat{e}_i\right)}{2}$$

where $\hat{e}_i$ is the $i$-th standard basis vector.

**Circuit evaluations per gradient call** (for $n$ parameters to differentiate):

| Quantity | Value |
|---|---|
| Shifted evaluations per parameter | 2 |
| Total circuit evaluations | $2n$ |
| Shift magnitude | $\pi/2$ (fixed) |

The rule is exact — it does not depend on step size and carries no truncation error.

---

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Parameter Shift Gradient.

Parameter shift computes exact analytic gradients of parameterized quantum circuits by evaluating at two shifted parameter values ($\theta \pm \pi/2$). It supports both Estimator (expectation value gradients) and Sampler (probability distribution gradients) primitives.

Use this skill when:
- Exact analytic gradients are required (no numerical approximation error)
- Circuits use gates from the supported gate set
- The target hardware supports the parameter shift rule natively

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Prerequisites

- NumPy
- Qiskit parameterized circuits composed of gates from the supported gate set
- Estimator primitives (`BaseEstimatorV2`) or Sampler primitives (`BaseSamplerV2`)
- Understanding of the analytic parameter shift rule: $\frac{\partial f}{\partial\theta_i} = \frac{f(\theta + \pi/2\,\hat{e}_i) - f(\theta - \pi/2\,\hat{e}_i)}{2}$

## Understanding the Key Quantum Components

1. **Supported gate set**: `rx`, `ry`, `rz`, `rzx`, `rzz`, `ryy`, `rxx`, `cx`, `cy`, `cz`, `h`, `x`, `y`, `z`, `p`. Circuits with gates outside this set are decomposed during `_preprocess()` via `SUPPORTED_GATES` resolution.
2. **Shifted parameter values**: Generated by `_make_param_shift_parameter_values()` — for each target parameter $\theta_i$, two values are produced: $\theta_i + \pi/2$ and $\theta_i - \pi/2$, with all other parameters held at their nominal values.
3. **Batched primitive job**: All $2n$ shifted circuit evaluations are submitted as a single primitive job to minimize overhead. The result array interleaves plus and minus shifts.
4. **Gradient reconstruction**: `gradient[i] = (evals[:n//2] - evals[n//2:]) / 2` over the batched result — the factor $1/2$ (not $1$) accounts for the $\pi/2$ shift convention combined with the standard parameter-shift formula.
5. **Estimator vs Sampler paths**: Estimator mode adds observables to the PUB list and returns `EstimatorGradientResult`. Sampler mode requires measurement instructions on circuits and returns `SamplerGradientResult` with per-bitstring gradient dicts.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Parameter vector $\theta$ | `parameter_values` — `Sequence[Sequence[float]]` |
| Gradient $\partial f/\partial \theta_i$ | `result.gradients[i]` in output `EstimatorGradientResult` |
| Shift magnitude $\pi/2$ | Hardcoded in `_make_param_shift_parameter_values()` from `qiskit_algorithms.gradients.utils` |
| Shifted evaluation $f(\theta \pm \pi/2\,\hat{e}_i)$ | Plus/minus sub-arrays in the batched primitive job result |
| Target parameter subset | `parameters` argument — `Sequence[Sequence[Parameter]]`; `None` differentiates all |
| Gate compatibility check | `_preprocess()` — decomposes unsupported gates into `SUPPORTED_GATES` set |
| Result metadata (parameter order) | `result.metadata[i]["parameters"]` |

## Hands-On Example

Selectively differentiate only a subset of parameters:

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

theta = ParameterVector("θ", 3)
qc = QuantumCircuit(3)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.ry(theta[2], 2)
qc.cx(0, 1)
qc.cx(1, 2)

observable = SparsePauliOp("ZZZ")
estimator = StatevectorEstimator()
gradient = ParamShiftEstimatorGradient(estimator)

# Differentiate only θ₀ and θ₂ — skip θ₁
result = gradient.run(
    [qc], [observable], [[0.3, -0.2, 0.7]],
    parameters=[[theta[0], theta[2]]]
).result()

print(result.gradients[0])          # shape (2,): [∂E/∂θ₀, ∂E/∂θ₂]
print(result.metadata[0]["parameters"])  # confirms order
```

Expected behavior: output length matches requested parameter count; output order matches `parameters=[[...]]`.

## Minimal Manual Implementation

```python
import numpy as np

def param_shift_gradient(eval_fn, theta, shift=np.pi/2):
    """Compute exact analytic gradient via parameter shift rule.

    Args:
        eval_fn: callable(theta) -> float (expectation value or probability)
        theta: parameter vector of length n
        shift: shift magnitude (default π/2 for standard gates)

    Returns:
        gradient vector of length n
    """
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        theta_plus = np.array(theta, dtype=float)
        theta_plus[i] += shift
        theta_minus = np.array(theta, dtype=float)
        theta_minus[i] -= shift
        grad[i] = (eval_fn(theta_plus) - eval_fn(theta_minus)) / 2.0
    return grad
```

This matches the core logic: evaluate at $\pm \pi/2$, take half the difference. For production use, the Qiskit implementation batches all evaluations into a single primitive job.

## Debugging Tips

1. **Unsupported gates error**: Circuits must use only gates in `SUPPORTED_GATES`. Unsupported gates are auto-decomposed during `_preprocess()`, but decomposition may fail for exotic gates. Verify with `set(qc.count_ops().keys()) <= SUPPORTED_GATES`.
2. **Gradient ordering**: The output `gradients[i]` order matches `parameters[i]`, not `circuit.parameters`. If `parameters=None`, all circuit parameters are differentiated in `circuit.parameters` order.
3. **Estimator/Sampler mismatch**: `ParamShiftEstimatorGradient` requires `observables` in `.run()`; `ParamShiftSamplerGradient` does not. Passing observables to the sampler version, or omitting them from the estimator version, raises `TypeError`.
4. **Shift magnitude**: The $\pi/2$ shift is fixed — it cannot be adjusted. For gates with different eigenvalue spectra, use a different gradient method (e.g., LCU or finite difference).
5. **Batch job size**: With $n$ parameters, each gradient call submits $2n$ circuit evaluations in one job. For large $n$, memory usage scales linearly — monitor primitive job size.
