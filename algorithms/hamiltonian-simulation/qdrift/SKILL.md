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
from engine.algorithms import QDriftAlgorithm

# 2-qubit Hamiltonian as (pauli_string, coefficient) list
H_list = [
    ("ZI", 0.45),
    ("IZ", 0.45),
    ("XX", 0.1),
]

algo = QDriftAlgorithm(seed=1234)
result = algo.run(
    H_list=H_list,
    t=1.0,
    epsilon=1e-3,
    n_qubits=2,
    backend='torch',
)

print("status:", result['status'])
print("error:", result['error'])
print(result.get('plot', ''))
```

### Core Parameters Explained

#### Constructor

```python
class QDriftAlgorithm:
    def __init__(self, seed: int = 666) -> None:
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `seed` | `int` | `666` | Random seed for reproducibility. |

#### `run()` Parameters

```python
def run(self, H_list, t, epsilon, n_qubits, backend='torch', algo_dir=None):
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `H_list` | `List[Tuple[str, float]]` | required | Hamiltonian as a list of `(pauli_string, coefficient)` tuples. |
| `t` | `float` | required | Total evolution time. |
| `epsilon` | `float` | required | Target error; used to compute sampling step count $N = \lceil 2\lambda^2 t^2 / \epsilon \rceil$. |
| `n_qubits` | `int` | required | Number of qubits. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str\|None` | `None` | Output directory for circuit diagram. |

### Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `error` | `float` | Estimated approximation error. |
| `circuit` | `GateSequence` | The assembled QDrift circuit (one random trajectory). |
| `circuit_path` | `str` | Path to circuit SVG. |
| `message` | `str` | Summary message. |
| `plot` | `str` | ASCII art result panel. |

## Understanding the Core Components

### 1) Probability and angle construction

From `_expand` in `algorithms/qdrift/qdrift.py`:

```python
pauli_strings = [p for p, _ in decomposition]
coeffs = np.array([c.real for _, c in decomposition], dtype=float)

lam = np.sum(np.abs(coeffs))
probs = np.abs(coeffs) / lam

indices = np.random.choice(len(decomposition), size=self.steps, p=probs)

sequence = []
for idx in indices:
    pauli_str = pauli_strings[idx]
    sign = np.sign(coeffs[idx])
    angle = sign * lam * t / self.steps
    sequence.append((pauli_str, angle))
```

Interpretation:
1. Probability mass follows coefficient magnitude.
2. Every sampled gate uses same magnitude scale `lam * t / steps`, modulated by sign.
3. `np.random.choice(..., p=probs)` is the stochastic core of QDrift.

### 2) Circuit assembly from random sequence

From `_run`:

```python
decomposition = self._pauli_decompose(self.H, self.t)
sequence = self._expand(decomposition, self.t)

for pauli_str, angle in sequence:
    gate = pauli_string_evolution(pauli_str, angle)
    qc.append(gate, range(self.target_qubits))

self._circuit = qc
self._evolution_result = self._circuit.get_matrix()
```

Interpretation:
1. The evolution result corresponds to one random trajectory.
2. Different seeds produce different approximation instances.
3. For stable evaluation, average over multiple independent runs.

### 3) External package function notes (brief)

1. `_pauli_decompose` comes from base class and handles decomposition details.
2. `pauli_string_evolution` creates the gate implementation for each sampled term.
3. `total_error` comes from base class and compares to exact `expm(-1jHt)`.

## Hands-On Example: Hamiltonian Simulation

Measure variance across seeds and step counts.

```python
from engine.algorithms import QDriftAlgorithm

H_list = [
    ("ZI", 0.35),
    ("IZ", 0.35),
    ("XX", 0.15),
    ("YY", 0.15),
]

for epsilon in [0.1, 0.01]:
    for seed in [0, 1, 2]:
        algo = QDriftAlgorithm(seed=seed)
        result = algo.run(
            H_list=H_list,
            t=1.0,
            epsilon=epsilon,
            n_qubits=2,
            backend='torch',
        )
        print(f"epsilon={epsilon}, seed={seed}, error={result['error']:.3e}")
```

What to look for:
1. Smaller `epsilon` leads to more sampling steps and lower error.
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
