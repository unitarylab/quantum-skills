---
name: taylor

description: "Simulate the time evolution of a quantum system using Taylor series expansion."
---

# Taylor Hamiltonian Simulation Skill Guide

## Algorithm Overview

**Category:** Hamiltonian Simulation — Algebraic Expansion Methods

**Purpose:** Approximate the time-evolution operator $e^{-iHt}$ by truncating its Taylor series and expressing the result as a Linear Combination of Unitaries (LCU). The Hamiltonian is first decomposed into a weighted sum of Pauli strings; the Taylor series is then built order-by-order via dynamic programming; finally the combined operator is implemented as a single LCU circuit.

**Core Idea:**

1. Split the total evolution time into $r$ equal slices to keep each slice's spectral weight small.
2. Expand $e^{-iH(t/r)}$ as a degree-$K$ Taylor series over Pauli string products.
3. Elevate the single-slice approximation to the $r$-th power using `pauli_string_power`.
4. Construct the LCU circuit from the resulting weighted Pauli unitaries.

---

## Mathematical Principles

**Taylor expansion of the evolution operator:**

$$
e^{-iHt} = \sum_{k=0}^{K} \frac{(-iHt)^k}{k!} + \mathcal{O}\!\left(\frac{(\alpha t)^{K+1}}{(K+1)!}\right)
$$

where $\alpha = \|H\|_2$ is the spectral norm.

**Time-slicing** reduces the per-slice parameter $\lambda = \alpha t$ to $\lambda / r$, allowing a lower truncation order $K$:

$$
r = \left\lfloor \frac{\alpha t}{0.5} \right\rfloor + 1, \qquad \lambda_{\text{slice}} = \frac{\alpha t}{r}
$$

**Adaptive degree selection:**

$$
K = \min\!\Bigl(\max\!\bigl(K_{\text{init}},\; \lceil 1.5\,\lambda + 1.5\ln(1/\varepsilon) \rceil\bigr),\; 15\Bigr)
$$

**Pauli decomposition** of the Hamiltonian:

$$
H = \sum_{j} c_j P_j, \qquad P_j \in \{I, X, Y, Z\}^{\otimes n}
$$

**LCU representation** of the truncated slice operator:

$$
U_{\text{approx}}^{(1)} \approx \sum_{\ell} w_\ell V_\ell, \qquad w_\ell \in \mathbb{R}_{\geq 0},\; V_\ell \text{ unitary}
$$

$$
e^{-iHt} \approx \bigl(U_{\text{approx}}^{(1)}\bigr)^r
$$

**Error metric** (Frobenius norm):

$$
\varepsilon = \bigl\|U_{\text{approx}}^r - e^{-iHt}\bigr\|_F
$$

---

## Core Parameters

### Constructor — `TaylorAlgorithm`

```python
class TaylorAlgorithm:
    def __init__(self, text_mode: str = "plain", algo_dir: str = None) -> None:
        ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output formatting mode for saved text reports (`"plain"` or `"legacy"`). |
| `algo_dir` | `str \| None` | `None` | Directory for saving results. Auto-derived from CWD if `None`. |

### `run()` Method

```python
def run(self, H: np.ndarray, t: float, error: float, degree: int = 15) -> dict:
    ...
```

| Parameter | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `H` | `np.ndarray` | required | Square, Hermitian; padded to next power-of-2 if needed | Hamiltonian matrix. |
| `t` | `float` | required | Finite real number | Total evolution time. |
| `error` | `float` | required | `> 0` | Target approximation error; used to compute the adaptive expansion degree. |
| `degree` | `int` | `15` | `≥ 1`, capped at `15` | Initial guess for the Taylor truncation order. Adjusted internally. |

---

## Inputs and Outputs

### Inputs

| Name | Type | Requirements |
|---|---|---|
| `H` | `np.ndarray` (complex128) | Square, Hermitian (checked to `atol=1e-12`); non-power-of-2 dims are zero-padded. |
| `t` | `float` | Any finite real number. |
| `error` | `float` | Strictly positive. |
| `degree` | `int` | Positive integer; the runtime value is clipped to `[computed_min, 15]`. |

### Return Value of `run()`

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `file_path` | `str` | Path to the saved text result file. |

### `algo.output` Fields (after `run()`)

| Key | Type | Description |
|---|---|---|
| `Approximate evolution matrix` | `np.ndarray` | $U_{\text{approx}}^r$ — the LCU-based time-evolution approximation. |
| `Exact evolution matrix` | `np.ndarray` | $e^{-iHt}$ computed via `scipy.linalg.expm`. |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}}^r - e^{-iHt}\|_F$. |

---

## Implementation Description

### Execution Flow

**Stage 1 — Input validation and formatting**
- Asserts `H` is square, Hermitian, and finite-`t`.
- Zero-pads `H` to the next power-of-2 dimension if necessary.
- Computes spectral norm `alpha = np.linalg.norm(H, 2)` and `lam = alpha * t`.
- Determines the number of time slices: `r = int(lam / 0.5) + 1`.
- Adjusts `degree` adaptively.

**Stage 2 — Pauli decomposition**
- Calls `pauli_string_decomposition(H * t / r)` to expand the per-slice Hamiltonian into a list of `(pauli_string, coefficient)` tuples.

**Stage 3 — Taylor series construction**
- Builds `degree + 1` order maps (`ans_term_map[k]`) via dynamic programming:
  - Order 0: identity.
  - Order $k$: multiply each order-$(k{-}1)$ Pauli string by each decomposed term, accumulating coefficients with the factor $-i/k$.
- Collapses all orders into a single `ans_term_list` of `(pauli_string, complex_coeff)` pairs.

**Stage 4 — Power elevation**
- Calls `pauli_string_power(ans_term_list, r)` to raise the single-slice operator to the $r$-th power symbolically.

**Stage 5 — LCU circuit construction**
- For each `(pauli_string, coeff)` term:
  - Extracts `magnitude = |coeff|` as the LCU weight.
  - Extracts `phase = arg(coeff)` and wraps it into a global phase gate via `_make_U_rotation`.
- Constructs the full LCU circuit using `LCU(lcu_terms)`.

**Stage 6 — Matrix extraction and error estimation**
- Extracts the $2^n \times 2^n$ block from the full LCU matrix: `lcu_matrix[i*m, j*m]` where `m = len(LCU_terms)`.
- Rescales by the sum of LCU weights `s` to recover the true normalization.
- Computes the exact evolution with `scipy.linalg.expm(-1j * H * t)`.
- Reports the Frobenius norm error.

### Engineering Constraints

| Constraint | Detail |
|---|---|
| Maximum Taylor degree | Hard-capped at `15` by implementation. |
| Hamiltonian dimension | Must be power-of-2; zero-padding is applied automatically. |
| Hermiticity tolerance | `atol = 1e-12`. |
| LCU term count | Grows exponentially with `degree` and the number of Pauli terms; large Hamiltonians increase memory use. |

### Key Internal Functions

| Function | Source | Role |
|---|---|---|
| `pauli_string_decomposition(H)` | `unitarylab.library.pauli_operator` | Decomposes `H` into weighted Pauli strings. |
| `pauli_string_multiply(s1, s2)` | `unitarylab.library.pauli_operator` | Multiplies two Pauli strings; returns result string and phase. |
| `pauli_string_power(terms, r)` | `unitarylab.library.pauli_operator` | Raises a Pauli-string operator to integer power `r`. |
| `pauli_string_circuit(s)` | `unitarylab.library.pauli_operator` | Builds the quantum circuit for a Pauli string. |
| `LCU(terms)` | `unitarylab.library` | Constructs the LCU quantum circuit from `(circuit, weight)` pairs. |
| `Circuit.get_matrix()` | `unitarylab` | Extracts the full unitary matrix from a circuit. |

---

## Sample Code

### Minimal Example

```python
import numpy as np
from unitarylab_algorithms import TaylorAlgorithm

# 2×2 Hermitian Hamiltonian
H = np.array([[2, 1],
              [1, 3]], dtype=complex)

algo = TaylorAlgorithm(text_mode="plain")
result = algo.run(
    H=H,
    t=1.0,
    error=1e-8,
    degree=15,
)

print("status      :", result["status"])
print("circuit_path:", result["circuit_path"])
print("file_path   :", result["file_path"])
print("Frobenius error:", algo.output["Frobenius norm of error"])
```

### Accuracy Sweep — `degree` vs. `t`

```python
import numpy as np
from unitarylab_algorithms import TaylorAlgorithm

H = np.array([[2, 1],
              [1, 3]], dtype=complex)

for t in [1.0, 3.0, 5.0]:
    for degree in [5, 10, 15]:
        algo = TaylorAlgorithm(text_mode="plain")
        result = algo.run(H=H, t=t, error=1e-8, degree=degree)
        frob_err = algo.output["Frobenius norm of error"]
        print(f"t={t:.1f}, degree={degree:>2d}, error={frob_err:.2e}, status={result['status']}")
```

**Expected observations:**
1. Larger `t` increases `lam`, which raises `r` (more slices) and the required `degree`.
2. Increasing `degree` reduces error until it saturates near machine precision.
3. The effective degree is clamped at 15; very large `t` is compensated by more time slices.