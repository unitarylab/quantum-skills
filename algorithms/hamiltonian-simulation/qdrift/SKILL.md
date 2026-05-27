---
name: qdrift
description: QDrift randomized Hamiltonian simulation, approximating e^{-iHt} by stochastically sampling Pauli-term evolutions with probability proportional to coefficient magnitude.
---

## One-Step Run Example Command
```bash
python ./scripts/algorithm.py
```

# QDrift Hamiltonian Simulation Skill Guide

## Overview

QDrift is a randomized Hamiltonian simulation method for approximating

$$
U(t)=e^{-iHt}
$$

using randomly sampled short Pauli evolutions.

### Key Insight

Instead of applying every Pauli term each slice, QDrift samples a single term at each step with probability proportional to coefficient magnitude, then applies a uniformly scaled angle. This replaces deterministic Trotter ordering with stochastic averaging.

### Why QDrift Matters:

1. Sampling-based structure can reduce dependence on number of Hamiltonian terms in some regimes.
2. It provides a practical randomized baseline against deterministic product formulas.
3. It is straightforward to implement and easy to parallelize over repeated trials.
4. It naturally supports statistical error analysis through multiple seeds.

### Real Applications:

1. Sparse or large-term-count Hamiltonian simulation studies.
2. Monte Carlo style benchmarking across random trajectories.
3. Resource tradeoff experiments under depth constraints.
4. Comparative studies with Trotter/Taylor/QSP pipelines.

## Learning Objectives

After using this skill, you should be able to:
1. Derive and implement QDrift sampling probabilities.
2. Understand how `steps` controls variance and depth.
3. Use reproducible random seeds for fair comparisons.
4. Interpret matrix-level outputs and spectral-norm error.
5. Build aggregate statistics over repeated runs.

## Prerequisites

### Essential knowledge:

1. Pauli decomposition of Hermitian matrices.
2. Basic probability distributions and random sampling.
3. Quantum gate-sequence execution concepts.

### Mathematical comfort:

1. Expectation and variance basics.
2. Norm-based approximation error metrics.
3. Scaling intuition with sample count.

## Using the Provided Implementation

### Quick Start Example

```python
import numpy as np
from unitarylab_algorithms import QDriftAlgorithm

# 2x2 Hermitian Hamiltonian matrix
H = np.array([[2, 1],
              [1, 3]], dtype=float)

algo = QDriftAlgorithm()
result = algo.run(
    H=H,
    t=1.0,
    error=1e-8,
    steps=5000,
    backend='torch',
)

print("status:", result['status'])
print("Frobenius norm of error:", result['Frobenius norm of error'])
for f in result['plot']:
    print(f"Saved {f['format']} file: {f['filename']}")
```

### Core Parameters Explained

#### Constructor

```python
class QDriftAlgorithm:
    def __init__(self, text_mode: str = "plain", algo_dir: str = None) -> None:
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output text format mode. |
| `algo_dir` | `str\|None` | `None` | Directory for saving result files. Auto-generated if `None`. |

#### `run()` Parameters

```python
def run(self, H: np.ndarray, t: float, error: float, steps: int = 5000, backend='torch', device='cpu', dtype=np.complex128):
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `H` | `np.ndarray` | required | Hermitian Hamiltonian matrix (2D square array). Non-power-of-2 dimensions are zero-padded. |
| `t` | `float` | required | Total evolution time. |
| `error` | `float` | required | Desired approximation error (currently reserved; does not auto-set `steps`). |
| `steps` | `int` | `5000` | Number of random Pauli samples. Larger values reduce variance and increase circuit depth. |
| `backend` | `str` | `'torch'` | Simulation backend for `qc.get_matrix()`. |
| `device` | `str` | `'cpu'` | Device for backend computation. |
| `dtype` | `type` | `np.complex128` | Dtype for matrix computation. |

### Return Fields

The `run()` method returns a dictionary built by `_build_return_dict(success, circuit_path, filepath, circuit)`. The `self.output` fields are merged into the result via `result.update(self.output)`, so all keys below are accessible directly on the returned dict:

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` otherwise. |
| `circuit_path` | `str` | Local path to the saved circuit diagram (SVG). |
| `plot` | `list` | List of saved result files, each as `{"format": str, "filename": str}` (format is the 3-char file extension). |
| `circuit` | `Circuit` | The assembled QDrift circuit object. |
| `Approximate evolution matrix` | `np.ndarray` | Approximate unitary $U_{\text{approx}}$ from the random QDrift circuit (`qc.get_matrix()`). |
| `Exact evolution matrix` | `np.ndarray` | Exact reference unitary $e^{-iHt}$ computed via `scipy.linalg.expm`. |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}} - U_{\text{exact}}\|_F$ — Frobenius norm of the difference. |

## Understanding the Core Components

### 1) Probability and angle construction

From `_expand(decomposition, t, steps)` in `algorithm.py`:

```python
pauli_strings = [p for p, _ in decomposition]
coeffs = np.array([c.real for _, c in decomposition], dtype=float)

lam = np.sum(np.abs(coeffs))
probs = np.abs(coeffs) / lam

indices = np.random.choice(len(decomposition), size=steps, p=probs)

sequence = []
for idx in indices:
    pauli_str = pauli_strings[idx]
    sign = np.sign(coeffs[idx])
    angle = sign * lam * t / steps
    sequence.append((pauli_str, angle))
```

Interpretation:
1. Probability mass follows coefficient magnitude.
2. Every sampled gate uses same magnitude scale `lam * t / steps`, modulated by sign.
3. `np.random.choice(..., p=probs)` is the stochastic core of QDrift.

### 2) Circuit assembly from random sequence

From `run()` in `algorithm.py`:

```python
decomposition = pauli_string_decomposition(H)
sequence = self._expand(decomposition, t, steps)

reg = Register('K', n)
qc = Circuit(reg, name='QDrift Decomposition')
for pauli_str, angle in sequence:
    gate = pauli_string_evolution(pauli_str, angle)
    qc.append(gate, range(n))

U_approx = qc.get_matrix(backend=backend, device=device, dtype=dtype)
U_exact = expm(-1j * H * t)
U_error = norm(U_approx - U_exact, ord='fro')
```

Interpretation:
1. The evolution result corresponds to one random trajectory.
2. `np.random.choice` is unseeded by default; results vary between runs.
3. For stable evaluation, fix a random seed externally via `np.random.seed(...)` before calling `run()`.

### 3) External package function notes (brief)

1. `pauli_string_decomposition` from `unitarylab.library.pauli_operator` decomposes the matrix into Pauli terms.
2. `pauli_string_evolution` creates the gate implementation for each sampled term.
3. Error is assessed via Frobenius norm against `scipy.linalg.expm(-1j * H * t)`.

## Hands-On Example: Hamiltonian Simulation

Measure variance across random seeds and step counts.

```python
import numpy as np
from unitarylab_algorithms import QDriftAlgorithm

# 2-qubit Heisenberg-like Hamiltonian (4×4 matrix)
XX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=float)
ZZ = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=float)
H = XX + ZZ

for steps in [1000, 5000]:
    for seed in [0, 1, 2]:
        np.random.seed(seed)
        algo = QDriftAlgorithm()
        result = algo.run(
            H=H,
            t=1.0,
            error=1e-8,
            steps=steps,
            backend='torch',
        )
        err = result['Frobenius norm of error']
        print(f"steps={steps}, seed={seed}, Frobenius error={err:.3e}")
```

What to look for:
1. Larger `steps` reduces the Frobenius-norm error on average.
2. Different seeds produce different random trajectories with some variance.

## Mathematical Deep Dive

For positive-coefficient decomposition,

$$
H = \sum_{\ell=1}^{L} h_\ell H_\ell,
\quad
h_\ell > 0,
\quad
\lambda = \sum_{\ell=1}^{L} h_\ell.
$$

Sampling probabilities are

$$
p_\ell = \frac{h_\ell}{\lambda}.
$$

The equivalent generalized form used in code is:

$$
H = \sum_{j=1}^{L} c_j P_j
$$

with

$$
\lambda = \sum_{j=1}^{L} |c_j|,
\quad
p_j = \frac{|c_j|}{\lambda}
$$

Set

$$
t_{\mathrm{step}} = \lambda t / N
$$

and build the random product

$$
U_{\text{qdrift}}(t)=\prod_{k=1}^{N}\exp(-iH_{j_k}t_{\mathrm{step}}).
$$

For signed coefficients in this implementation, each sampled gate is

$$
e^{-i\,\mathrm{sign}(c_j)\,\lambda t / N\; P_j}
$$

for total `N = steps` samples.

A commonly cited complexity expression in this randomized setting is

$$
N = O\left(\frac{(\lambda t)^2}{\epsilon}\right),
$$

while first-order deterministic product formulas are often compared using

$$
O\left(\frac{L^2(\lambda t)^2}{\epsilon}\right)
$$

style dependence.

Practical interpretation:
1. One trajectory is random and biased toward large-magnitude terms.
2. Increasing `N` improves concentration around target evolution.
3. Empirical error assessment should include repeated seeds and statistics.

Implementation-consistent notes:
1. The positive-coefficient presentation $H=\sum h_j H_j$ is a special case. Current code supports signed real Pauli coefficients by sampling with $|c_j|$ and encoding sign in the rotation angle.
2. The common statement "no explicit dependence on $L$" refers to the randomized quantum-step complexity form. In practical implementation, classical decomposition and sampling still iterate across terms.
3. In this code path, `target_error` is not used to automatically choose `steps`; users should tune `steps` experimentally.

## Real-World Applications

1. Noisy simulation pipelines where randomized depth patterns are useful.
2. Fast baseline approximations for large Pauli expansions.
3. Statistical benchmarking for algorithm selection.
4. Randomized circuit studies in NISQ-style analyses.
