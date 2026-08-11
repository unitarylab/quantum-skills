---
name: finite-difference
description: "Quantum gradient estimation via finite difference method. Supports both Estimator (expectation value gradients) and Sampler (probability distribution gradients) primitives with central, forward, and backward difference schemes. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Finite Difference Gradient

## Overview

| Property | Detail |
|---|---|
| **Category** | Gradient Estimation |
| **Primitives** | `FiniteDiffEstimatorGradient`, `FiniteDiffSamplerGradient` |
| **Framework** | Qiskit Algorithms (`qiskit_algorithms.gradients`) |
| **Use Case** | Numerically approximate parameter gradients of parameterized quantum circuits |

Finite difference gradient methods numerically approximate the derivative of a function by evaluating it at perturbed parameter values. They are applicable to any differentiable quantum circuit and do not require an analytically differentiable gate set, making them a universal but statistically noisier alternative to analytic gradient methods.

---

## Reference Implementation Example

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

## Core Parameters Explained

### `FiniteDiffEstimatorGradient.run()`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `estimator` | `BaseEstimatorV2` | Yes | — | Estimator primitive for expectation value computation |
| `epsilon` | `float` | Yes | — | Perturbation step size; must be positive |
| `precision` | `float \| None` | No | `None` | Overrides default primitive precision; uses primitive default if `None` |
| `method` | `Literal["central", "forward", "backward"]` | No | `"central"` | Finite difference scheme |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

### `FiniteDiffSamplerGradient.run()`

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `sampler` | `BaseSamplerV2` | Yes | — | Sampler primitive for probability distribution computation |
| `epsilon` | `float` | Yes | — | Perturbation step size; must be positive |
| `shots` | `int \| None` | No | `None` | Overrides default primitive shot count; uses primitive default if `None` |
| `method` | `Literal["central", "forward", "backward"]` | No | `"central"` | Finite difference scheme |
| `transpiler` | `Transpiler \| None` | No | `None` | Optional transpiler with a `.run()` method |
| `transpiler_options` | `dict[str, Any] \| None` | No | `None` | Keyword arguments forwarded to `transpiler.run()` |

---

## Return Fields

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

## Implementation Architecture

- Both classes extend `BaseEstimatorGradient` / `BaseSamplerGradient` from `qiskit_algorithms.gradients.base`.
- All perturbed circuit evaluations for a single gradient call are batched into **one primitive job** to minimize overhead.
- The perturbation offset matrix is constructed as `np.identity(circuit.num_parameters)[indices, :]` where `indices` are the positions of the target parameters in `circuit.parameters`.
- When `method="central"`, the PUB list is `[θ + εI, θ − εI]`; when `method="forward"` or `"backward"`, it is `[θ₀, θ ± εI]`.
- If a `transpiler` is provided, circuits are transpiled before execution. For `FiniteDiffEstimatorGradient`, observables are also re-laid-out via `observable.apply_layout(circuit.layout)`.
- `epsilon` must satisfy `epsilon > 0`; a `ValueError` is raised otherwise.
- `method` must be one of `"central"`, `"forward"`, `"backward"`; a `TypeError` is raised otherwise.
- Gradient computation is purely numerical — no gate-level decomposition or analytic shift rules are used.

---

## Mathematical Deep Dive

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

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Finite Difference Gradient.

Finite difference numerically approximates parameter gradients of quantum circuits by evaluating at perturbed parameter values. It supports three schemes — central, forward, backward — and works with any differentiable gate, making it a universal fallback when analytic gradient methods are not applicable.

Use this skill when:
- Circuits contain gates outside the analytic gradient supported sets
- A universal, gate-agnostic gradient method is needed
- Step-size control ($\epsilon$) is important for balancing accuracy and noise

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Prerequisites

- NumPy
- Qiskit parameterized circuits (any gate type — no gate set restrictions)
- Estimator primitives (`BaseEstimatorV2`) or Sampler primitives (`BaseSamplerV2`)
- Understanding of finite difference methods: truncation error scales as $O(\epsilon^2)$ for central, $O(\epsilon)$ for forward/backward

## Understanding the Key Quantum Components

1. **Perturbation step size $\epsilon$**: Controls the balance between truncation error (large $\epsilon$) and noise amplification (small $\epsilon$). Must be positive — `ValueError` if $\epsilon \leq 0$.
2. **Difference schemes**: Central difference uses $2n$ evaluations and is second-order accurate. Forward and backward use $n+1$ evaluations and are first-order accurate.
3. **Perturbation offset matrix**: Constructed as `np.identity(circuit.num_parameters)[indices, :]` where `indices` are the positions of target parameters — maps parameter indices to perturbation directions.
4. **Batched primitive job**: For central difference, the PUB list is `[θ + εI, θ − εI]`; for forward, `[θ₀, θ + εI]`; for backward, `[θ₀, θ − εI]`. All evaluations are submitted as one job.
5. **Universal applicability**: No gate set restrictions — works on any parameterized circuit. This is the key advantage over parameter shift and LCU methods.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Parameter vector $\theta$ | `parameter_values` — `Sequence[Sequence[float]]` |
| Perturbation step $\epsilon$ | Constructor argument `epsilon`; must be `> 0` |
| Method (central/forward/backward) | Constructor argument `method`; `Literal["central", "forward", "backward"]` |
| Perturbation directions $\hat{e}_i$ | `np.identity(circuit.num_parameters)[indices, :]` |
| $f(\theta + \epsilon\hat{e}_i)$, $f(\theta - \epsilon\hat{e}_i)$ | Plus/minus sub-arrays in batched primitive result |
| Target parameter subset | `parameters` argument; `None` differentiates all |
| Gradient output $\partial f/\partial\theta_i$ | `result.gradients[i]` in `EstimatorGradientResult` |

## Hands-On Example

Compare different difference schemes on the same circuit:

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import FiniteDiffEstimatorGradient

theta = ParameterVector("θ", 2)
qc = QuantumCircuit(2)
qc.rx(theta[0], 0)
qc.rz(theta[1], 1)
qc.cx(0, 1)

observable = SparsePauliOp("ZZ")
estimator = StatevectorEstimator()

for method in ["central", "forward", "backward"]:
    gradient = FiniteDiffEstimatorGradient(estimator, epsilon=1e-3, method=method)
    result = gradient.run([qc], [observable], [[0.5, 1.0]]).result()
    print(f"{method:>10}: {result.gradients[0]}")
```

Expected behavior: central difference yields the most accurate result; forward and backward produce similar values with slightly larger error.

## Minimal Manual Implementation

```python
import numpy as np

def finite_diff_gradient(eval_fn, theta, epsilon=1e-3, method="central"):
    """Numerically approximate gradient via finite difference.

    Args:
        eval_fn: callable(theta) -> float
        theta: parameter vector of length n
        epsilon: perturbation step size
        method: "central", "forward", or "backward"

    Returns:
        approximate gradient vector of length n
    """
    n = len(theta)
    grad = np.zeros(n)
    f0 = eval_fn(theta) if method in ("forward", "backward") else None

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        if method == "central":
            fp = eval_fn(theta + epsilon * ei)
            fm = eval_fn(theta - epsilon * ei)
            grad[i] = (fp - fm) / (2 * epsilon)
        elif method == "forward":
            fp = eval_fn(theta + epsilon * ei)
            grad[i] = (fp - f0) / epsilon
        elif method == "backward":
            fm = eval_fn(theta - epsilon * ei)
            grad[i] = (f0 - fm) / epsilon
    return grad
```

## Debugging Tips

1. **$\epsilon$ too large**: Truncation error dominates. Start with $\epsilon = 10^{-2}$ and reduce while monitoring gradient stability.
2. **$\epsilon$ too small**: Noise (shot noise on real hardware, floating-point on simulators) amplifies the $1/\epsilon$ division. On noisy hardware, larger $\epsilon$ with more shots is often better than small $\epsilon$ with few shots.
3. **Forward vs central accuracy**: Forward difference is $O(\epsilon)$, central is $O(\epsilon^2)$. For statevector simulations, prefer central; for noisy hardware, forward may require fewer evaluations with comparable precision.
4. **Method validation**: `method` must be exactly `"central"`, `"forward"`, or `"backward"` — a `TypeError` is raised otherwise.
5. **$\epsilon \leq 0$**: Raises `ValueError`. Always validate $\epsilon > 0$ before constructing the gradient object.
