---
name: finite-difference
description: Quantum gradient estimation via finite difference method. Supports both Estimator (expectation value gradients) and Sampler (probability distribution gradients) primitives with central, forward, and backward difference schemes.
---

# Finite Difference Gradient

## Algorithm Overview

| Property | Detail |
|---|---|
| **Category** | Gradient Estimation |
| **Primitives** | `FiniteDiffEstimatorGradient`, `FiniteDiffSamplerGradient` |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Numerically approximate parameter gradients of parameterized quantum circuits |

Finite difference gradient methods numerically approximate the derivative of a function by evaluating it at perturbed parameter values. They are applicable to any differentiable quantum circuit and do not require an analytically differentiable gate set, making them a universal but statistically noisier alternative to analytic gradient methods.

---

## Mathematical Principles

Let $f(\theta)$ be the expectation value or sampling probability of a parameterized circuit at parameter vector $\theta \in \mathbb{R}^n$, and let $\epsilon > 0$ be the perturbation step size.

**Central difference** (second-order accurate):

$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + \epsilon \hat{e}_i) - f(\theta - \epsilon \hat{e}_i)}{2\epsilon}$$

**Forward difference** (first-order accurate):

$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + \epsilon \hat{e}_i) - f(\theta)}{\epsilon}$$

**Backward difference** (first-order accurate):

$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta) - f(\theta - \epsilon \hat{e}_i)}{\epsilon}$$

where $\hat{e}_i$ is the $i$-th standard basis vector.

**Circuit evaluations per parameter** (for $n$ parameters to differentiate):

| Method | Circuit Evaluations |
|---|---|
| `central` | $2n$ |
| `forward` | $n + 1$ |
| `backward` | $n + 1$ |

---

## Core Parameters

### `FiniteDiffEstimatorGradient`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `epsilon` | `float` | Yes | — | Perturbation step size; must be positive |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision; uses primitive default if `None` |
| `method` | `Literal["central", "forward", "backward"]` | No | `"central"` | Finite difference scheme |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `FiniteDiffSamplerGradient`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampler` | `BaseSamplerV2` | Yes | — | Sampler primitive for probability distribution computation |
| `epsilon` | `float` | Yes | — | Perturbation step size; must be positive |
| `shots` | `int \| None` | No | `None` | Overrides default primitive shot count; uses primitive default if `None` |
| `method` | `Literal["central", "forward", "backward"]` | No | `"central"` | Finite difference scheme |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

---

## Inputs and Outputs

### `FiniteDiffEstimatorGradient`

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

### `FiniteDiffSamplerGradient`

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
- All perturbed circuit evaluations for a single gradient call are batched into **one primitive job** to minimize overhead.
- The perturbation offset matrix is constructed as `np.identity(circuit.num_parameters)[indices, :]` where `indices` are the positions of the target parameters in `circuit.parameters`.
- When `method="central"`, the PUB list is `[θ + εI, θ − εI]`; when `method="forward"` or `"backward"`, it is `[θ₀, θ ± εI]`.
- If a `transpiler` is provided, circuits are transpiled before execution. For `FiniteDiffEstimatorGradient`, observables are also re-laid-out via `observable.apply_layout(circuit.layout)`.
- `epsilon` must satisfy `epsilon > 0`; a `ValueError` is raised otherwise.
- `method` must be one of `"central"`, `"forward"`, `"backward"`; a `TypeError` is raised otherwise.
- Gradient computation is purely numerical — no gate-level decomposition or analytic shift rules are used.

---

## Sample Code

### Estimator Gradient

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient

# Build a simple parameterized circuit
theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

observable = SparsePauliOp("ZZ")
parameter_values = [0.5, 1.0]

estimator = StatevectorEstimator()
gradient = FiniteDiffEstimatorGradient(estimator, epsilon=1e-2, method="central")

result = gradient.run([qc], [observable], [parameter_values]).result()
print(result.gradients)   # [array([dE/dθ₀, dE/dθ₁])]
```

### Sampler Gradient

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.gradients import FiniteDiffSamplerGradient

theta = ParameterVector("θ", 1)
qc = QuantumCircuit(1)
qc.ry(theta[0], 0)
qc.measure_all()

parameter_values = [0.5]

sampler = StatevectorSampler()
gradient = FiniteDiffSamplerGradient(sampler, epsilon=1e-2, method="central")

result = gradient.run([qc], [parameter_values]).result()
print(result.gradients)   # [[{0: dP(0)/dθ₀, 1: dP(1)/dθ₀}]]
```