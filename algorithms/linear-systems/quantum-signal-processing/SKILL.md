---
name: quantum-signal-processing
description: A quantum algorithm for signal processing tasks, leveraging quantum phase estimation and amplitude amplification techniques to achieve efficient signal analysis and transformation. This skill includes implementations and educational resources for understanding and utilizing quantum signal processing algorithms in various applications.
---
# Quantum Signal Processing (QSP)

## Purpose

Quantum Signal Processing implements a polynomial transformation $P(x)$ of a signal $x$ using a sequence of signal-processing rotations and signal operators. This implementation demonstrates QSP applied to Hamiltonian simulation: approximating $e^{-i\tau x}$ for a signal variable $x$.

Use this skill when you need to:
- Apply a polynomial function to the eigenvalues of a quantum operator.
- Implement Hamiltonian simulation via a phase sequence (QSVT predecessor).

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

QSP constructs a single-qubit circuit interleaving:
- **Phase rotations** $R_z(2\phi_k)$: parameterized by a phase sequence $\Phi = (\phi_0, \phi_1, \ldots, \phi_d)$.
- **Signal operator** $W(x) = R_x(2\arccos(x))$: encoding the signal $x$.

The circuit implements the $(0,0)$ matrix element of:
$$U_\Phi(x) = e^{i\phi_0 Z}\prod_{k=1}^d [W(x) \cdot e^{i\phi_k Z}] \approx \begin{pmatrix} P(x) & \cdot \\ \cdot & \cdot \end{pmatrix}$$

The phase sequence $\Phi$ is optimized (via L-BFGS-B) to approximate the target polynomial $P(x) \approx e^{-i\tau x}$.

## Prerequisites

- Single-qubit rotations: $R_z$, $R_x$.
- Polynomial approximation concepts (Chebyshev expansion).
- Python: `numpy`, `scipy.optimize`, `GateSequence`, `Register`.

## Using the Provided Implementation

```python
from unitarylab.algorithms import QSPAlgorithm

algo = QSPAlgorithm(seed=42)
result = algo.run(
    target_tau=1.0,    # Evolution parameter τ: targets e^{-iτx}
    degree=10,         # Polynomial degree d
    x_value=0.5,       # Test signal point x ∈ [-1, 1]
    backend='torch'
)

print(result['error'])         # Absolute error |QSP(x) - exp(-iτx)|
print(result['circuit_path'])  # SVG circuit diagram
print(result['plot'])          # ASCII result panel
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `target_tau` | `float` | required | Hamiltonian evolution time $\tau$. The target function is $e^{-i\tau x}$. |
| `degree` | `int` | required | Polynomial degree $d$. Higher degree achieves smaller error for complex targets. |
| `x_value` | `float` | `0.5` | Test point $x \in [-1, 1]$ at which to evaluate the approximation error. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `x_value` must be in $[-1, 1]$ (the domain of the signal operator $W(x)$). Values outside this range are clipped internally.
- `degree` determines the number of signal $W(x)$ applications: $d+1$ total rotations in the circuit.
- The phase sequence `_find_phases()` is computed via numerical optimization (L-BFGS-B), not analytically. The result may vary slightly with different `seed`.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'`. |
| `error` | `float` | Absolute approximation error $|$QSP$(x) - e^{-i\tau x}|$ at `x_value`. |
| `circuit` | `GateSequence` | The built QSP circuit. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `message` | `str` | Result summary including error. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`QSPAlgorithm` in `algorithm.py` organizes the algorithm into five stages with one classical optimizer helper and one format helper.

**`run(target_tau, degree, x_value, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Phase Optimization | `_find_phases(target_tau, degree)` — runs L-BFGS-B minimization | Computes the QSP phase sequence $\Phi$ numerically |
| 2 — Circuit Construction | Creates `GateSequence(Register('q', 1))`; applies `gs.rz(2*phases[0], 0)` as initial phase; loops `degree` times applying `gs.rx(2*theta, 0)` (signal) then `gs.rz(2*phases[k], 0)` (phase rotation) | Builds the alternating signal/phase circuit |
| 3 — Simulation | `gs.execute()` → `final_state[0]` | Runs statevector evolution of the single qubit |
| 4 — Post-Processing | Compares `qsp_val = final_state[0]` to `ideal_val = exp(-i*tau*x_value)` | Computes absolute approximation error |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_find_phases(tau, d)`** — Fully classical numerical optimizer. Builds an internal matrix-product simulation of the QSP circuit for a grid of `2d+1` samples. Defines loss as average squared magnitude error vs. `exp(-i*tau*x)`. Runs `scipy.optimize.minimize` with L-BFGS-B starting from random initial phases (`np.random.randn(d+1) * 0.1`). Returns `(d+1)` phase values.
- **`format_result_ascii()`** — Renders the stored `self.last_result` as an ASCII panel. Note that `self.last_result` here is set as a plain dict (not via `_update_last_result`), a difference from other algorithms.

**Single-qubit circuit structure (Stage 2):**
```
Rz(2φ₀) → Rx(2θ) → Rz(2φ₁) → Rx(2θ) → Rz(2φ₂) → ... → Rx(2θ) → Rz(2φ_d)
```
Total gates: `2d + 1`. The `theta = arccos(clip(x_value, -1, 1))` is a fixed parameter computed from `x_value`.

**Important implementation note:** The simulation in `_find_phases` uses explicit 2×2 matrix products (NumPy) — not the `GateSequence` engine — so the optimization is fully classical and independent of the backend.

**Data flow:** `(target_tau, degree)` → `_find_phases()` → `phases` array → `GateSequence` with Rz/Rx gates → `execute()` → `final_state[0]` → `abs_error` vs. `ideal_val` → result dict.

## Understanding the Key Quantum Components
$$W(x) = \begin{pmatrix} x & i\sqrt{1-x^2} \\ i\sqrt{1-x^2} & x \end{pmatrix}$$
Implemented as $R_x(2\arccos(x))$ on the single qubit. This maps the scalar signal $x$ into a 2×2 rotation.

### 2. Phase Rotations $e^{i\phi_k Z}$
Implemented as $R_z(2\phi_k)$ gates. Interleaved with the signal operator, they provide the degrees of freedom to approximate arbitrary polynomials.

### 3. QSP Circuit Structure
```
Rz(2φ₀) → Rx(2θ) → Rz(2φ₁) → Rx(2θ) → ... → Rz(2φ_d)
```
where $\theta = \arccos(x)$. The $(0,0)$ matrix element of the product equals $P(x)$.

### 4. Phase Optimization
The phases $\Phi$ are found by minimizing:
$$L(\Phi) = \frac{1}{\tilde{d}}\sum_{j=1}^{\tilde{d}} |\langle 0|U_\Phi(x_j)|0\rangle - e^{-i\tau x_j}|^2$$
over $2d+1$ Chebyshev sample points $x_j = \cos\left(\frac{(2j-1)\pi}{4\tilde{d}}\right)$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Signal operator $W(x)$ = $R_x(2\theta)$ | `gs.rx(2 * theta, 0)` where `theta = arccos(x_value)` |
| Phase rotations $e^{i\phi_k Z}$ = $R_z(2\phi_k)$ | `gs.rz(2 * phases[k], 0)` in the main loop |
| Initial phase $\phi_0$ | `gs.rz(2 * phases[0], 0)` before loop |
| Phase sequence $\Phi = (\phi_0, \ldots, \phi_d)$ | `phases` returned by `_find_phases()` — `(d+1)` floats |
| Target function $e^{-i\tau x}$ | `ideal_val = np.exp(-1j * target_tau * x_value)` in Stage 4 |
| Loss function minimization | `minimize(loss, ...)` with `method='L-BFGS-B'` in `_find_phases()` |
| $(0,0)$ matrix element $\langle 0|U_\Phi(x)|0\rangle$ | `final_state[0]` after `gs.execute()` |
| Approximation error at test point | `abs_error = abs(qsp_val - ideal_val)` |
| Sample points $x_j = \cos((2j-1)\pi/(4\tilde d))$ | `x_samples = np.linspace(-1, 1, 2*d+1)` (uniform, not Chebyshev nodes) |

**Notes on implementation vs. theory:** The README describes exact Chebyshev sample points, but the code uses `np.linspace(-1, 1, 2*d+1)` (uniformly spaced) as the optimization grid. The loss function is identical in structure. The `_find_phases` uses 2×2 matrix products for efficiency, not the quantum simulator. The returned `error` is evaluated only at `x_value`, not over the full interval — it is a spot check, not a worst-case bound.

## Mathematical Deep Dive with $|P(x)| \leq 1$ for all $x \in [-1,1]$ and parity $d \bmod 2$, there exist phases $\Phi$ such that:
$$\langle 0|U_\Phi(x)|0\rangle = P(x)$$

**Hamiltonian simulation target:** $P(x) = e^{-i\tau x}$, approximated by a degree-$d$ truncated Chebyshev expansion. For $\tau$ fixed, error decreases exponentially in $d$.

**Complexity:** The circuit uses $d+1$ single-qubit gates plus $2d+1$ $R_z$ gates — total $O(d)$ gates.

## Hands-On Example

```python
from unitarylab.algorithms import QSPAlgorithm
import numpy as np

algo = QSPAlgorithm(seed=0)

# Approximate e^{-i*0.5*x} at several test points
for x_test in [0.0, 0.3, 0.7, 1.0]:
    result = algo.run(target_tau=0.5, degree=8, x_value=x_test)
    ideal = np.exp(-1j * 0.5 * x_test)
    print(f"x={x_test:.1f}: error={result['error']:.2e}, ideal_re={ideal.real:.4f}")
```

## Implementing Your Own Version

The following Python skeleton reconstructs the key components: phase optimization and the alternating QSP circuit.

### Step 1: Simulate QSP circuit classically for optimization

```python
# Simplified reconstruction — mirrors QSPAlgorithm._find_phases() internals
import numpy as np
from scipy.optimize import minimize

def qsp_matrix(x: float, phases: np.ndarray) -> np.ndarray:
    """Compute the QSP circuit product matrix for signal x and phase vector phases."""
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    # Signal operator W(x) = Rx(2*arccos(x))
    W = np.array([[x, 1j*np.sqrt(1-x**2)],
                  [1j*np.sqrt(1-x**2), x]])
    # Initial phase rotation Rz(2*phi_0)
    mat = np.array([[np.exp(1j*phases[0]), 0], [0, np.exp(-1j*phases[0])]])
    for k in range(1, len(phases)):
        # Pauli Rz gate: e^{i phi_k Z}
        Rz_k = np.array([[np.exp(1j*phases[k]), 0], [0, np.exp(-1j*phases[k])]])
        mat = Rz_k @ W @ mat
    return mat

def find_qsp_phases(tau: float, d: int, seed: int = 42) -> np.ndarray:
    """Find QSP phases that approximate exp(-i*tau*x) on [-1,1]."""
    np.random.seed(seed)
    x_samples = np.linspace(-1, 1, 2*d + 1)
    target_vals = np.exp(-1j * tau * x_samples)

    def loss(phi):
        total = 0.0
        for xi, ti in zip(x_samples, target_vals):
            mat = qsp_matrix(xi, phi)
            total += abs(mat[0, 0] - ti) ** 2
        return total / len(x_samples)

    res = minimize(loss, x0=np.random.randn(d+1)*0.1, method='L-BFGS-B',
                   options={'maxiter': 400})
    return res.x
```

### Step 2: Build and execute the QSP circuit

```python
# Simplified reconstruction — mirrors QSPAlgorithm.run() Stage 2 and Stage 3
from unitarylab.core import GateSequence, Register
import numpy as np

def build_qsp_circuit(phases: np.ndarray, x_value: float, backend: str = 'torch') -> GateSequence:
    """Construct the QSP circuit: Rz(2φ₀) → [Rx(2θ) → Rz(2φₖ)] × d."""
    d = len(phases) - 1
    theta = float(np.arccos(np.clip(x_value, -1.0, 1.0)))
    gs = GateSequence(Register('q', 1), backend=backend)
    # Initial phase rotation
    gs.rz(2 * phases[0], 0)
    for k in range(1, d + 1):
        gs.rx(2 * theta, 0)      # signal operator W(x) = Rx(2*arccos(x))
        gs.rz(2 * phases[k], 0)  # phase rotation
    return gs

# Full QSP pipeline
def qsp_approximate(tau: float, d: int, x_value: float, backend: str = 'torch'):
    phases = find_qsp_phases(tau, d)
    circ = build_qsp_circuit(phases, x_value, backend)
    state = circ.execute()
    qsp_val = np.asarray(state, dtype=complex).reshape(-1)[0]
    ideal = np.exp(-1j * tau * x_value)
    return qsp_val, abs(qsp_val - ideal)
```

## Debugging Tips

1. **Large `error` for small `degree`**: $e^{-i\tau x}$ requires degree $\sim O(\tau + \log(1/\epsilon))$ for $\epsilon$-approximation. Increase `degree`.
2. **Optimizer not converging**: The L-BFGS-B optimizer uses `maxiter=400`. Increase in `_find_phases()` if needed.
3. **`x_value` outside $[-1, 1]$**: Internally clipped by `np.clip`. Ensure your signal values are in the valid domain.
4. **Error non-zero at `x=0`**: $e^{-i\tau\cdot 0} = 1$; if error is large here with small $\tau$, the optimizer may have gotten stuck.
5. **Random phase initialization**: Change `seed` in `QSPAlgorithm(seed=...)` to try different starting points.
