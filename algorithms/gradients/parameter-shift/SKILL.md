---
name: parameter-shift
description: Analytic quantum gradient estimation via the parameter shift rule. Supports both Estimator (expectation value gradients) and Sampler (probability distribution gradients) primitives. Requires circuits composed exclusively of supported gate types.
---

# Parameter Shift Gradient

## Algorithm Overview

| Property | Detail |
|---|---|
| **Category** | Gradient Estimation |
| **Primitives** | `ParamShiftEstimatorGradient`, `ParamShiftSamplerGradient` |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Analytically compute parameter gradients of parameterized quantum circuits |

The parameter shift rule computes exact analytic gradients of parameterized quantum circuits by evaluating the circuit at two shifted parameter values ($\theta \pm \pi/2$). Unlike numerical methods, it introduces no approximation error and is directly executable on quantum hardware. It is restricted to circuits whose gates belong to a supported set of single-parameter rotation and controlled gates.

---

## Mathematical Principles

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

## Core Parameters

### `ParamShiftEstimatorGradient`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `ParamShiftSamplerGradient`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampler` | `BaseSamplerV2` | Yes | — | Sampler primitive for probability distribution computation |
| `shots` | `int \| None` | No | `None` | Overrides default primitive shot count; uses primitive default if `None` |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

---

## Inputs and Outputs

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

## Implementation Notes

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

## Sample Code

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