---
name: taylor

description: "Simulate the time evolution of a quantum system using Taylor series expansion. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Taylor Hamiltonian Simulation Skill Guide

## Overview

**Category:** Hamiltonian Simulation — Algebraic Expansion Methods

**Purpose:** Approximate the time-evolution operator $e^{-iHt}$ by truncating its Taylor series and expressing the result as a Linear Combination of Unitaries (LCU). The Hamiltonian is first decomposed into a weighted sum of Pauli strings; the Taylor series is then built order-by-order via dynamic programming; finally the combined operator is implemented as a single LCU circuit.

**Core Idea:**

1. Split the total evolution time into $r$ equal slices to keep each slice's spectral weight small.
2. Expand $e^{-iH(t/r)}$ as a degree-$K$ Taylor series over Pauli string products.
3. Elevate the single-slice approximation to the $r$-th power using `pauli_string_power`.
4. Construct the LCU circuit from the resulting weighted Pauli unitaries.

---

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Taylor Hamiltonian Simulation.

Taylor simulation approximates $e^{-iHt}$ by truncating its Taylor series to degree $K$, decomposing the Hamiltonian into Pauli strings, constructing the truncated operator via dynamic programming, and implementing it as a Linear Combination of Unitaries (LCU) circuit.

Use this skill when:
- LCU-based Hamiltonian simulation with controllable truncation order is needed
- The Hamiltonian can be decomposed into a manageable number of Pauli terms
- Comparing Taylor-series methods against Trotter, QDrift, or QSP approaches

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Reference Implementation Example

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
print("plot        :", result["plot"])
print("Frobenius error:", result["Frobenius norm of error"])
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

## Core Parameters Explained

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

`run()` returns a dictionary built by `_build_return_dict()`. The output fields from `algo.output` are merged directly into the same dict via `result.update(self.output)`.

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` on error. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `plot` | `list[dict]` | List of saved result files, each `{"format": str, "filename": str}`. |
| `circuit` | `Circuit` | The constructed LCU circuit object. |
| `Approximate evolution matrix` | `np.ndarray` | $U_{\text{approx}}^r$ — the LCU-based time-evolution approximation. |
| `Exact evolution matrix` | `np.ndarray` | $e^{-iHt}$ computed via `scipy.linalg.expm`. |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}}^r - e^{-iHt}\|_F$. |

The same output fields are also accessible via `algo.output` after `run()` completes.

---

## Return Fields

`run()` returns a dictionary built by `_build_return_dict()`. The output fields from `algo.output` are merged directly into the same dict.

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` on error |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram |
| `plot` | `list[dict]` | List of saved result files, each `{"format": str, "filename": str}` |
| `circuit` | `Circuit` | The constructed LCU circuit object |
| `Approximate evolution matrix` | `np.ndarray` | $U_{\text{approx}}^r$ — the LCU-based time-evolution approximation |
| `Exact evolution matrix` | `np.ndarray` | $e^{-iHt}$ computed via `scipy.linalg.expm` |
| `Frobenius norm of error` | `float` | $\|U_{\text{approx}}^r - e^{-iHt}\|_F$ |

## Implementation Architecture

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

## Mathematical Deep Dive

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

## Prerequisites

- Hamiltonian simulation fundamentals: $e^{-iHt} = \sum_{k=0}^\infty (-iHt)^k/k!$
- Pauli decomposition of Hermitian matrices into weighted Pauli strings
- Linear Combination of Unitaries (LCU) circuit construction
- Dynamic programming for efficient polynomial series construction
- Python: `numpy`, `scipy.linalg.expm`

## Understanding the Key Quantum Components

1. **Time-slicing**: The total evolution is split into $r = \lfloor \alpha t / 0.5 \rfloor + 1$ slices where $\alpha = \|H\|_2$. Each slice evolves for $t/r$, keeping $\lambda_{\text{slice}} = \alpha t / r$ small so a low-degree Taylor truncation suffices.
2. **Taylor series via dynamic programming**: Order $k$ is built from order $k-1$ by multiplying each existing Pauli string by each decomposed Hamiltonian term, accumulating with factor $-i/k$. This avoids recomputing all products from scratch.
3. **Power elevation**: `pauli_string_power(ans_term_list, r)` symbolically raises the single-slice operator to the $r$-th power, composing $r$ slices into the full evolution.
4. **LCU circuit assembly**: Each `(pauli_string, coeff)` term contributes one unitary to the LCU. The magnitude $|coeff|$ becomes the LCU weight; the phase $\arg(coeff)$ is wrapped into a global phase gate.
5. **Hard degree cap**: The Taylor degree is capped at 15. For very large $t$, the algorithm compensates by increasing $r$ (more slices) rather than exceeding this cap.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Hamiltonian $H$ | `H` — 2D `np.ndarray`, square Hermitian |
| Spectral norm $\alpha = \|H\|_2$ | `alpha = np.linalg.norm(H, 2)` |
| Scaled parameter $\lambda = \alpha t$ | `lam = alpha * t` |
| Number of slices $r$ | `r = int(lam / 0.5) + 1` |
| Per-slice Hamiltonian | `pauli_string_decomposition(H * t / r)` |
| Taylor degree $K$ | Adaptive: $\min(\max(K_{\text{init}}, \lceil 1.5\lambda + 1.5\ln(1/\epsilon)\rceil), 15)$ |
| Order-$k$ terms | `ans_term_map[k]` — built via dynamic programming |
| Power elevation | `pauli_string_power(ans_term_list, r)` |
| LCU construction | `LCU(lcu_terms)` from `unitarylab.library` |
| Matrix extraction | `lcu_matrix[i*m, j*m]` where $m$ = number of LCU terms |
| Frobenius error | $\|U_{\text{approx}}^r - e^{-iHt}\|_F$ |

## Hands-On Example

Compare Taylor against exact diagonalization for varying degree:

```python
import numpy as np
from unitarylab_algorithms import TaylorAlgorithm

# 4x4 Hamiltonian (2 qubits)
H = np.array([[1.0, 0.5, 0.0, 0.0],
              [0.5, 2.0, 0.3, 0.0],
              [0.0, 0.3, 1.5, 0.2],
              [0.0, 0.0, 0.2, 2.5]], dtype=complex)

for degree in [3, 5, 10, 15]:
    algo = TaylorAlgorithm(text_mode="plain")
    result = algo.run(H=H, t=2.0, error=1e-8, degree=degree)
    frob_err = algo.output["Frobenius norm of error"]
    print(f"degree={degree:>2d}, Frobenius error={frob_err:.2e}, status={result['status']}")
```

Expected behavior: higher degree reduces error until it saturates near the LCU normalization floor.

## Minimal Manual Implementation

```python
import numpy as np
from scipy.linalg import expm

def taylor_simulation_skeleton(H, t, degree=5):
    """Simplified Taylor-series Hamiltonian simulation.

    In practice, the full implementation uses Pauli decomposition,
    dynamic programming for order construction, and LCU assembly.
    This skeleton illustrates the core mathematical structure.
    """
    n = H.shape[0]
    # Time-slicing
    alpha = np.linalg.norm(H, 2)
    r = max(1, int(alpha * t / 0.5) + 1)
    t_slice = t / r

    # Taylor series of e^{-iH*t_slice}
    U_slice = np.zeros((n, n), dtype=complex)
    H_power = np.eye(n, dtype=complex)
    factorial = 1.0
    for k in range(degree + 1):
        U_slice += ((-1j * t_slice) ** k / factorial) * H_power
        H_power = H_power @ H
        factorial *= (k + 1)

    # Power elevation: U ≈ (U_slice)^r
    U_approx = np.linalg.matrix_power(U_slice, r)
    U_exact = expm(-1j * H * t)
    return U_approx, np.linalg.norm(U_approx - U_exact, 'fro')
```

Note: This skeleton uses dense matrix exponentiation for clarity. The actual implementation decomposes $H$ into Pauli strings and constructs an LCU circuit — avoiding explicit $2^n \times 2^n$ matrix formation.

## Debugging Tips

1. **LCU term explosion**: The number of LCU terms grows exponentially with `degree` and the number of Pauli terms in the decomposition. For Hamiltonians with many Pauli terms, LCU memory usage can become prohibitive — monitor `len(lcu_terms)`.
2. **Degree auto-adjustment**: The effective degree may be lower than the requested value if the adaptive formula determines the per-slice $\lambda$ is already small enough. Check `algo.output` fields for the actual degree used.
3. **Slicing dominates at large t**: For $t > 5$, the number of slices $r$ grows linearly. Very large $r$ increases both LCU construction time and the power-elevation cost. Consider whether Trotter or QSP may be more efficient for large-$t$ simulations.
4. **Hermiticity check**: The implementation verifies $H$ is Hermitian to `atol=1e-12`. Nearly-Hermitian matrices with floating-point asymmetries may fail — symmetrize with `H = (H + H.conj().T) / 2` before passing.
5. **Degree cap at 15**: If error remains large even at `degree=15`, the bottleneck is likely the time-slicing factor or the LCU normalization, not the Taylor truncation. Increase slices by checking the adaptive `r` computation.
