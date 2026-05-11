---
name: qsp
description: QSP-based Hamiltonian simulation approximates e^{-iHt} by block-encoding the Hamiltonian and applying polynomial spectral transformations via interleaved signal-processing rotations, achieving high precision with efficiently bounded circuit depth.
---


# QSP Hamiltonian Simulation Skill Guide

## Overview

QSP-based Hamiltonian simulation approximates the time-evolution operator

$$
U(t) = e^{-iHt}
$$

by converting it into a polynomial approximation problem on the eigenvalues of the block-encoded Hamiltonian.

### Key Insight

Rather than decomposing $H$ term-by-term as in Trotterization, QSP encodes the full spectral action of $e^{-iHt}$ as a polynomial transformation applied to a block-encoding oracle. The Hamiltonian is block-encoded once, and separate QSP circuits approximate $\beta\cos(tH)$ and $\beta\sin(tH)$ via Chebyshev series. An LCU construction merges them into a single-slice evolution circuit. When the polynomial degree is insufficient, the algorithm automatically increases the number of time slices.

### Why QSP Matters:

1. Achieves near-optimal query complexity for Hamiltonian simulation.
2. Approximates arbitrary analytic functions of a block-encoded operator.
3. Polynomial degree and time slices give fine-grained control over accuracy vs. depth.
4. Strong theoretical foundation for comparing against Trotter and QDrift methods.

### Real Applications:

1. Precision time evolution of molecular and lattice Hamiltonians.
2. Subroutine in quantum linear algebra and quantum walks.
3. Benchmarking advanced simulation methods against product formulas.
4. Digital simulation pipelines where gate budget is the primary constraint.

## Learning Objectives

After using this skill, you should be able to:
1. Explain how block encoding and polynomial approximation replace term-by-term Trotterization.
2. Understand how `degree` and `time_slices` jointly control accuracy and circuit depth.
3. Use the `QSPHSAlgorithm` class correctly for Hamiltonian simulation experiments.
4. Interpret the approximate and exact evolution matrices and the Frobenius-norm error.
5. Build reproducible comparisons of accuracy across parameter sweeps.

## Prerequisites

### Essential knowledge:

1. Hermitian Hamiltonians and unitary time evolution.
2. Block encoding of quantum operators.
3. Quantum circuit composition and controlled-unitary gate sequences.

### Mathematical comfort:

1. Chebyshev polynomial series and Bessel function expansions.
2. Matrix norms, especially the Frobenius norm.
3. Linear combination of unitaries (LCU) framework.

## Using the Provided Implementation

### Quick Start Example

```python
import numpy as np
from unitarylab_algorithms import QSPHSAlgorithm

# 2x2 Hermitian Hamiltonian
H = np.array([[2, 1],
              [1, 3]], dtype=complex)

algo = QSPHSAlgorithm(text_mode="plain")
result = algo.run(
    H=H,
    t=1.0,
    error=1e-8,
    degree=15,
    beta=0.7,
)

print("status      :", result["status"])
print("circuit_path:", result["circuit_path"])
print("file_path   :", result["file_path"])
```

### Core Parameters Explained

#### Constructor

```python
class QSPHSAlgorithm:
    def __init__(self, text_mode: str = "plain", algo_dir: str = None) -> None:
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output formatting mode for saved text reports. |
| `algo_dir` | `str\|None` | `None` | Directory for saving results. Auto-derived from CWD if `None`. |

#### `run()` Parameters

```python
def run(self, H, t, error, degree=15, beta=0.7):
    ...
```

| Parameter | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `H` | `np.ndarray` | required | Square, Hermitian; padded to next power-of-2 dim if needed | Hamiltonian matrix. |
| `t` | `float` | required | Finite real number | Total evolution time. |
| `error` | `float` | required | `> 0` | Target approximation error; drives degree and slice estimation. |
| `degree` | `int` | `15` | `≥ 1` | Upper bound on QSP polynomial degree per time slice. |
| `beta` | `float` | `0.7` | `(0, 1)` strictly | Preconditioning factor for numerical stability. |

### Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `file_path` | `str` | Path to saved text result file. |

Stored in `algo.output` after `run()`:

| Key | Type | Description |
|---|---|---|
| `Approximate evolution matrix` | `np.ndarray` | $U_{\text{approx}}$ composed from time slices. |
| `Exact evolution matrix` | `np.ndarray` | $e^{-iHt}$ computed via `scipy.linalg.expm`. |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}} - U_{\text{exact}}\|_F$. |

## Understanding the Core Components

### 1) Block encoding and degree estimation

From `run()` in `algorithm.py`:

```python
encoded_H = block_encode(H, method="nagy")
UH    = encoded_H.circuit     # circuit that block-encodes H/α
alpha = encoded_H.alpha        # scaling factor α
m     = encoded_H.total_qubits - n   # ancilla qubit count
```

```python
@staticmethod
def _estimate_required_degree(alpha: float, t: float, target_error: float) -> int:
    t_scaled = abs(alpha * t)
    return max(1, int(np.ceil(1.4 * t_scaled + np.log(1.0 / target_error))))
```

Interpretation:
1. `block_encode(H, method="nagy")` returns a normalized oracle with scaling factor `alpha`.
2. The required degree grows linearly with `|alpha * t_slice|` and logarithmically with `1/error`.
3. When the requested `degree` is too low, `time_slices` is doubled until the per-slice degree fits.

### 2) Chebyshev coefficient construction via Bessel functions

From `run()` in `algorithm.py`:

```python
t = alpha * slice_time   # dimensionless parameter s = α · t_slice

coef_cos = np.zeros(d + 1)
coef_cos[0] = jn(0, t) * beta
for i in range(1, d + 1):
    if i % 2 == 0:
        coef_cos[i] = jn(i, t) * 2 * (-1) ** (i / 2) * beta

coef_sin = np.zeros(d + 1)
for i in range(d + 1):
    if i % 2 != 0:
        coef_sin[i] = jn(i, t) * 2 * (-1) ** ((i - 1) / 2) * beta

cos_Ht = QSP(UH, n, m, coef_cos.copy(), 0)
sin_Ht = QSP(UH, n, m, coef_sin.copy(), 1)
```

Interpretation:
1. Even-index Bessel coefficients fill the cosine expansion; odd-index fill the sine expansion.
2. `beta` scales both polynomials to keep values inside $[-1, 1]$ for numerical stability.
3. `QSP(UH, n, m, coef, parity)` constructs the QSP circuit for the given Chebyshev series.

### 3) LCU combination and matrix composition

From `run()` in `algorithm.py`:

```python
qc = Circuit(n + m + 2)
qc.h(n + m + 1)                                             # |+⟩ on selection qubit
qc.s(n + m + 1)                                             # S gate → |+i⟩
qc.z(n + m + 1)                                             # phase adjustment
qc.append(cos_Ht, list(range(n + m + 1)), [n + m + 1], [0])  # cos part, control=0
qc.append(sin_Ht, list(range(n + m + 1)), [n + m + 1], [1])  # sin part, control=1
qc.h(n + m + 1)                                             # final Hadamard

u_slice = qc.get_matrix(n) * factor   # factor = 2 / beta

if time_slices > 1:
    U_approx = np.linalg.matrix_power(u_slice, time_slices)
else:
    U_approx = u_slice
```

Interpretation:
1. The LCU uses one extra qubit as a selection register to coherently combine cos and sin blocks.
2. `factor = 2/beta` rescales the output to recover the true $e^{-iHt}$ normalization.
3. Multiple slices are composed by matrix power rather than circuit repetition, enabling exact linear-algebra comparison.

### 4) Notes on external package functions (brief)

1. `block_encode(H, method="nagy")` is from `unitarylab.library`; it returns `encoded_H` with `.circuit`, `.alpha`, and `.total_qubits`.
2. `QSP(UH, n, m, coef, parity)` is from `unitarylab.library._qsp`; it builds the interleaved signal-processing circuit for the given coefficient array.
3. `Circuit.get_matrix(n)` extracts the $2^n \times 2^n$ unitary submatrix acting on the $n$ system qubits.
4. Error is computed in `run()` by comparing to `scipy.linalg.expm(-1j * H * t)` via Frobenius norm.

## Hands-On Example: Hamiltonian Simulation

Sweep `degree` and `t` to observe accuracy vs. circuit depth trade-offs.

```python
import numpy as np
from unitarylab_algorithms import QSPHSAlgorithm

H = np.array([[2, 1],
              [1, 3]], dtype=complex)

for t in [1.0, 3.0, 5.0]:
    for degree in [10, 20, 40]:
        algo = QSPHSAlgorithm(text_mode="plain")
        result = algo.run(H=H, t=t, error=1e-8, degree=degree, beta=0.7)
        frob_err = algo.output["Frobenius norm of error"]
        print(f"t={t:.1f}, degree={degree:>2d}, error={frob_err:.2e}, status={result['status']}")
```

What to look for:
1. Larger `t` requires a higher `degree` or more time slices to maintain accuracy.
2. Increasing `degree` reduces error until it saturates at the Bessel-approximation limit.
3. When `degree` is insufficient, the algorithm compensates by splitting into more time slices.

## Mathematical Deep Dive

The block encoding scales the Hamiltonian as

$$
\langle 0^m | U_H | 0^m \rangle = \frac{H}{\alpha}
$$

where $\alpha$ is the 1-norm scaling factor from `block_encode`. For each time slice of duration $t_{\text{slice}} = t / \text{time\_slices}$, the dimensionless parameter is

$$
s = \alpha \cdot t_{\text{slice}}
$$

The target functions are expanded as Chebyshev series using Bessel coefficients:

$$
\beta \cos(sx) \approx \sum_{k \text{ even}} c_k T_k(x), \qquad c_0 = \beta J_0(s),\quad c_k = 2(-1)^{k/2}\beta J_k(s)
$$

$$
\beta \sin(sx) \approx \sum_{k \text{ odd}} c_k T_k(x), \qquad c_k = 2(-1)^{(k-1)/2}\beta J_k(s)
$$

The LCU combines both blocks to recover the full complex exponential:

$$
U_{\text{approx}} = \frac{2}{\beta}\bigl(\beta\cos(sH) - i\,\beta\sin(sH)\bigr) \approx e^{-iHt_{\text{slice}}}
$$

Accuracy is measured by the Frobenius norm of the residual:

$$
\varepsilon = \|U_{\text{approx}}^{\text{time\_slices}} - e^{-iHt}\|_F
$$

The required polynomial degree per slice is estimated as:

$$
d_{\text{req}} = \left\lceil 1.4 \cdot |\alpha \cdot t_{\text{slice}}| + \ln(1/\varepsilon) \right\rceil
$$

Overall circuit complexity is dominated by:

$$
O\!\left(d_{\text{req}} \times \text{time\_slices} \times \text{cost}(U_H)\right)
$$

Implementation-consistent notes:
1. `beta` keeps polynomial values inside $[-1, 1]$; the factor `2/beta` undoes this rescaling at the matrix level.
2. The `QSP_MAX_TIME_SLICES` environment variable (default `4096`) caps the automatic slice expansion loop.
3. `H` must be Hermitian to within `atol=1e-12`; non-power-of-2 dimensions are zero-padded before encoding.

## Real-World Applications

1. High-precision digital simulation of molecular electronic structure.
2. Ground-state preparation via imaginary-time evolution via polynomial approximation.
3. Quantum linear systems algorithms that use $e^{-iHt}$ as a subroutine.
4. Circuit resource estimation studies comparing QSP, Trotter, and QDrift on the same Hamiltonian.