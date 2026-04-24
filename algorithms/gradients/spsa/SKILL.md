---
name: spsa
description: Concise guide to the local SPSA estimator and sampler gradient implementations for parameterized quantum circuits.
---

# SPSA Gradient Estimation

## Purpose
Use this skill for the SPSA gradient implementations in this folder:
- `scripts/spsa_estimator_gradient.py`
- `scripts/spsa_sampler_gradient.py`

They estimate gradients of:
- expectation values via `SPSAEstimatorGradient`
- sampled output distributions via `SPSASamplerGradient`

SPSA is useful when circuits have many parameters, because each batch needs only `2 * batch_size` circuit evaluations, independent of parameter count.

## One-Step Run Command
There is no standalone script entry point in this folder.

Use the classes as library components inside the full package.

## Overview
For each random perturbation vector `delta in {+1, -1}^d`, the implementation evaluates:
- `f(theta + epsilon * delta)`
- `f(theta - epsilon * delta)`

It then forms a central-difference estimate and averages across `batch_size` perturbations.

## Prerequisites
- NumPy
- Qiskit parameterized circuits
- Estimator or Sampler primitives
- Basic finite-difference gradient intuition

## Using the Provided Implementation
The real entry classes are:
- `SPSAEstimatorGradient`
- `SPSASamplerGradient`

Public usage goes through inherited `run(...)`; the algorithm itself is implemented in `_run(...)`.

Minimal estimator example:

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import SPSAEstimatorGradient

theta = ParameterVector("theta", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)

estimator = StatevectorEstimator()
obs = SparsePauliOp("ZZ")

grad = SPSAEstimatorGradient(estimator=estimator, epsilon=0.01, batch_size=4, seed=123)
result = grad.run(
    circuits=[qc],
    observables=[obs],
    parameter_values=[[0.3, -0.2]],
    parameters=[[theta[0], theta[1]]],
    precision=None,
).result()

print(result.gradients)
```

Minimal sampler example:

```python
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.gradients import SPSASamplerGradient

theta = ParameterVector("theta", 2)
qc = QuantumCircuit(2)
qc.ry(theta[0], 0)
qc.ry(theta[1], 1)
qc.cx(0, 1)
qc.measure_all()

sampler = StatevectorSampler()

grad = SPSASamplerGradient(sampler=sampler, epsilon=0.01, batch_size=4, seed=123)
result = grad.run(
    circuits=[qc],
    parameter_values=[[0.3, -0.2]],
    parameters=[[theta[0], theta[1]]],
    shots=None,
).result()

print(result.gradients)
```

## Core Parameters Explained

### `SPSAEstimatorGradient`
- `estimator`: expectation-value primitive
- `epsilon`: perturbation size, must be positive
- `batch_size=1`: number of SPSA samples to average
- `seed=None`: RNG seed
- `precision=None`: estimator precision override
- `transpiler=None`, `transpiler_options=None`: optional transpilation

Run inputs:
- `circuits`
- `observables`
- `parameter_values`
- `parameters`
- `precision`

Return fields:
- `gradients`
- `metadata`
- `precision`

### `SPSASamplerGradient`
- `sampler`: sampling primitive
- `epsilon`: perturbation size, must be positive
- `batch_size=1`: number of SPSA samples to average
- `seed=None`: RNG seed
- `shots=None`: sampler shot override
- `transpiler=None`, `transpiler_options=None`: optional transpilation

Run inputs:
- `circuits`
- `parameter_values`
- `parameters`
- `shots`

Return fields:
- `gradients`
- `metadata`
- `shots`

## Implementation Architecture

### Estimator path
`SPSAEstimatorGradient._run(...)` does the following:
1. Normalize `precision`.
2. Optionally transpile circuits and remap observables with circuit layout.
3. For each circuit, sample `batch_size` random sign vectors.
4. Build `theta + epsilon * delta` and `theta - epsilon * delta` parameter sets.
5. Submit one batched estimator job.
6. Compute
   $$
   \frac{f_+ - f_-}{2\epsilon}
   $$
   and divide by the perturbation signs to recover per-parameter gradients.
7. Average across the batch and keep only requested parameters.

Simplified reconstruction:

```python
for each circuit:
    deltas = random_pm_one_vectors(batch_size)
    plus = [theta + epsilon * d for d in deltas]
    minus = [theta - epsilon * d for d in deltas]
    pubs.append((circuit, observable, plus + minus, precision))

results = estimator.run(pubs).result()

gradient = mean(((f_plus - f_minus) / (2 * epsilon)) / delta over batch)
```

### Sampler path
`SPSASamplerGradient._run(...)` is similar, but works on probability distributions:
1. Normalize `shots`.
2. Optionally transpile circuits.
3. Build plus/minus parameter sets from random sign vectors.
4. Submit one batched sampler job.
5. Convert counts to probabilities.
6. Compute per-bitstring distribution differences.
7. Reconstruct each requested parameter gradient and average over the batch.

Simplified reconstruction:

```python
for each circuit:
    deltas = random_pm_one_vectors(batch_size)
    plus = [theta + epsilon * d for d in deltas]
    minus = [theta - epsilon * d for d in deltas]
    pubs.append((circuit, plus + minus, shots))

results = sampler.run(pubs).result()

for each target parameter j:
    gradient_j = average_k(diff_distribution_k * deltas[k][j])
```

## Understanding the Key Quantum Components
- Parameterized circuit `U(theta)`
- Estimator objective: expectation value `f(theta)`
- Sampler objective: output distribution `p_theta(z)`
- Random perturbation vector `delta in {+1, -1}^d`
- Quantum evaluations at `theta +/- epsilon * delta`
- Classical post-processing for gradient recovery

## Theory-to-Code Mapping
- `theta`: `parameter_values`
- `delta`: random `+1/-1` vectors built with NumPy RNG
- `epsilon`: constructor argument `epsilon`
- `f(theta + epsilon * delta)`, `f(theta - epsilon * delta)`: `plus` and `minus` evaluations
- requested parameters: `parameters`
- batch average: NumPy mean or dict-value averaging

## Mathematical Deep Dive
For scalar objective `f(theta)`:

$$
\hat g_i(\theta) = \frac{f(\theta + \epsilon \delta) - f(\theta - \epsilon \delta)}{2\epsilon\,\delta_i}
$$

For `batch_size = B`:

$$
\hat g(\theta) = \frac{1}{B}\sum_{k=1}^B \frac{f(\theta + \epsilon \delta^{(k)}) - f(\theta - \epsilon \delta^{(k)})}{2\epsilon} \odot \delta^{(k)}
$$

Since `delta_i in {+1, -1}`, dividing by `delta_i` is equivalent to multiplying by `delta_i`.

For sampler gradients, the same idea is applied per output bitstring probability.

## Hands-On Example
If you request only a parameter subset, the output keeps that order:

```python
result = grad.run(
    circuits=[qc],
    observables=[obs],
    parameter_values=[[0.1, -0.4, 0.7]],
    parameters=[[theta[0], theta[2]]],
    precision=None,
).result()

print(result.gradients[0])
print(result.metadata[0]["parameters"])
```

Expected behavior:
- output length matches requested parameter count
- output order matches `parameters=[[...]]`

## Implementing Your Own Version

```python
import numpy as np

def spsa_gradient(eval_fn, theta, epsilon, batch_size, rng):
    grads = []
    for _ in range(batch_size):
        delta = (-1) ** rng.integers(0, 2, len(theta))
        f_plus = eval_fn(theta + epsilon * delta)
        f_minus = eval_fn(theta - epsilon * delta)
        grads.append(((f_plus - f_minus) / (2 * epsilon)) / delta)
    return np.mean(np.asarray(grads), axis=0)
```

This matches the core logic of the estimator implementation; the sampler version replaces scalar outputs with sparse probability dictionaries.

## Debugging Tips
- `epsilon <= 0` raises `ValueError`.
- In estimator mode, the input lists must have matching lengths.
- Requested parameters must exist in the circuit.
- Too-small `epsilon` can make noise dominate.
- Larger `batch_size` reduces variance but increases runtime.
- In estimator mode, transpilation changes layout, so observables are remapped.
- In sampler mode, gradients are built from normalized counts, so shot count matters.
