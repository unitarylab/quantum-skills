---
name: qdrift
description: "QDrift randomized Hamiltonian simulation, approximating e^{-iHt} by stochastically sampling Pauli-term evolutions with probability proportional to coefficient magnitude. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# QDrift Hamiltonian Simulation Skill Guide

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement QDrift Hamiltonian Simulation Skill Guide.

After using this skill, you should be able to:
1. Derive and implement QDrift sampling probabilities.
2. Understand how `steps` controls variance and depth.
3. Use reproducible random seeds for fair comparisons.
4. Interpret matrix-level outputs and spectral-norm error.
5. Build aggregate statistics over repeated runs.

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

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

## Prerequisites

### Essential knowledge:

1. Pauli decomposition of Hermitian matrices.
2. Basic probability distributions and random sampling.
3. Quantum gate-sequence execution concepts.

### Mathematical comfort:

1. Expectation and variance basics.
2. Norm-based approximation error metrics.
3. Scaling intuition with sample count.

## Reference Implementation Example

## Core Parameters Explained

### Constructor

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

## Return Fields

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

## Understanding the Key Quantum Components

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

## Hands-On Example

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

1. Noisy simulation pipelines where randomized depth patterns are useful.
2. Fast baseline approximations for large Pauli expansions.
3. Statistical benchmarking for algorithm selection.
4. Randomized circuit studies in NISQ-style analyses.

## Implementation Architecture

`QDriftAlgorithm` in `algorithm.py` implements the randomized simulation in three stages plus export.

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Validation & Decomposition | `_format_system(H, t, error)` pads dimension to power-of-2; `pauli_string_decomposition(H)` extracts `(pauli_str, coeff)` pairs | Prepares Hermitian $H$ and its Pauli decomposition |
| 2 — Random Sampling | `_expand(decomposition, t, steps)` computes $\lambda = \sum \|c_j\|$, probabilities $p_j = \|c_j\|/\lambda$, samples `steps` indices via `np.random.choice`, builds `sequence` of `(pauli_str, angle)` with `angle = sign(c_j) * lam * t / steps` | Core QDrift stochastic mechanism |
| 3 — Circuit Assembly | Iterates `sequence`, applies `pauli_string_evolution(pauli_str, angle)` to each term; wraps in `qc = Circuit(reg, name='QDrift Decomposition')` | Builds the full random trajectory circuit |
| 4 — Verification & Export | `qc.get_matrix(backend, device, dtype)` extracts $U_{\text{approx}}$; `expm(-1j * H * t)` computes $U_{\text{exact}}$; Frobenius norm error; saves circuit SVG and text report | Benchmarks against exact evolution |

**Key design decision:** `np.random.choice` is unseeded by default — each `run()` produces a different random trajectory. For reproducible results, call `np.random.seed(seed)` before `algo.run()`.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Pauli decomposition $H = \sum c_j P_j$ | `pauli_string_decomposition(H)` from `unitarylab.library.pauli_operator` |
| Coefficient norm sum $\lambda = \sum \|c_j\|$ | `lam = np.sum(np.abs(coeffs))` |
| Sampling probabilities $p_j = \|c_j\| / \lambda$ | `probs = np.abs(coeffs) / lam` |
| Random index sampling | `np.random.choice(len(decomposition), size=steps, p=probs)` |
| Per-step angle $\text{sign}(c_j) \cdot \lambda t / N$ | `angle = sign * lam * t / steps` |
| Pauli evolution gate | `pauli_string_evolution(pauli_str, angle)` |
| Approximate unitary $U_{\text{qdrift}}$ | `qc.get_matrix(backend, device, dtype)` |
| Exact reference $e^{-iHt}$ | `scipy.linalg.expm(-1j * H * t)` |
| Error metric | `norm(U_approx - U_exact, ord='fro')` |
| Complexity expression | $N = O((\lambda t)^2 / \epsilon)$ for randomized setting |

## Minimal Manual Implementation

```python
import numpy as np

def qdrift_simulation(pauli_terms, coeffs, t, steps, rng=None):
    """Simplified QDrift: sample Pauli terms by coefficient magnitude.

    Args:
        pauli_terms: list of Pauli string identifiers
        coeffs: list of real coefficients
        t: total evolution time
        steps: number of random samples N
        rng: numpy random generator (optional)

    Returns:
        sequence of (pauli_str, angle) pairs
    """
    if rng is None:
        rng = np.random.default_rng()
    coeffs = np.abs(np.array(coeffs, dtype=float))
    lam = np.sum(coeffs)
    probs = coeffs / lam
    indices = rng.choice(len(pauli_terms), size=steps, p=probs)
    sequence = []
    for idx in indices:
        sign = np.sign(coeffs[idx]) if coeffs[idx] != 0 else 1.0
        angle = sign * lam * t / steps
        sequence.append((pauli_terms[idx], angle))
    return sequence
```

This matches the core `_expand()` logic: probability mass follows coefficient magnitude; every sampled gate uses the same magnitude scale `lam * t / steps` modulated by sign.

## Debugging Tips

1. **Non-reproducible results**: QDrift uses unseeded `np.random.choice` by default. For consistent trajectories across runs, call `np.random.seed(seed)` before each `algo.run()`.
2. **Insufficient steps**: Large Frobenius error typically means `steps` is too small. The error scales as $O(1/\sqrt{N})$ for a single trajectory — quadrupling `steps` roughly halves the expected error.
3. **Variance across seeds**: Different random seeds produce different trajectories with varying error. For benchmarking, average across 5-10 seeds and report mean ± std of the Frobenius error.
4. **Complex coefficients**: The current implementation uses `coeffs.real` for probability computation. Hamiltonians with significant imaginary Pauli coefficients may produce biased sampling — verify the decomposition produces real-dominant coefficients.
5. **Backend selection**: `backend='torch'` is the default and most tested path. Other backends may have different performance characteristics for `qc.get_matrix()`.
