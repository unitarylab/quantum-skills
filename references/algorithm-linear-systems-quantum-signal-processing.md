# Quantum Signal Processing (QSP)

## Scope and Reference Use

Quantum Signal Processing implements a polynomial transformation $P(x)$ of a scalar signal $x$ using a sequence of single-qubit signal-processing rotations and phase operators. This implementation demonstrates basic QSP applied to function approximation: approximating $\cos(t \cdot x)$ at a test point.

This reference covers:
- Demonstrate the core QSP signal/phase sequence on one qubit.
- Approximate a scalar function such as $\cos(t x)$ at a point $x \in [-1,1]$.
- Inspect phase-factor optimization and QSP circuit structure.

For a Hamiltonian matrix `H`, time evolution $e^{-iHt}$, block encoding, Chebyshev expansions of $\cos(tH)$ and $\sin(tH)$, or `QSPHSAlgorithm`, read [QSP-HS Hamiltonian Simulation](algorithm-hamiltonian-simulation-qsp.md). The scalar demonstration here is a different wrapper.

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

Wrapper source: `unitarylab_algorithms/linear_algebra/qsp/algorithm.py`.

Reference scripts: [quantum_signal_processing_algorithm.py](../scripts/quantum_signal_processing_algorithm.py) (API usage example); [quantum_signal_processing_qsp_implementation.py](../scripts/quantum_signal_processing_qsp_implementation.py) (independent teaching implementation).

## Overview

QSP constructs a single-qubit circuit interleaving:
- **SDK phase calls** `rz(2*phi_k, 0)`, parameterized by $\Phi=(\phi_0,\ldots,\phi_d)$.
- **SDK signal calls** `rx(2*arccos(x), 0)`, encoding the scalar signal.

The classical phase optimizer uses the positive-exponent matrix convention:
$$U_\Phi(x) = e^{i\phi_0 Z}\prod_{k=1}^d [W(x) \cdot e^{i\phi_k Z}] \approx \begin{pmatrix} P(x) & \cdot \\ \cdot & \cdot \end{pmatrix}$$

The SDK uses standard $R_x(\theta)=e^{-i\theta X/2}$ and $R_z(\theta)=e^{-i\theta Z/2}$. Its applied product is the adjoint of the optimizer product for the same phases, so its $(0,0)$ amplitude is conjugated; absolute error against real $\cos(tx)$ is unchanged by conjugation. This convention is visible in `unitarylab/backend/pygates/unitary.py` and should be preserved during a port.

## Prerequisites

- Single-qubit rotations: $R_z$, $R_x$.
- Polynomial approximation concepts (Chebyshev expansion).
- Python: `numpy`, `scipy.optimize`, `Circuit`, `Register`.

## Reference Implementation Example

```python
from unitarylab_algorithms import QSPAlgorithm

algo = QSPAlgorithm(text_mode="legacy")
result = algo.run(
    t=1.0,         # Evolution time t: targets cos(t * x)
    d=10,          # Polynomial degree d
    x=0.5,         # Test signal point x ∈ [-1, 1]
    backend='torch'
)

print(result['Absolute error'])    # Absolute error |QSP(x) - cos(t*x)|
print(result['Estimated value'])   # QSP estimated value at x
print(result['Ideal value'])       # cos(t * x) at test point
print(result['circuit_path'])      # SVG circuit diagram
print(result['plot'])              # List of saved output file dicts
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `t` | `float` | required | Evolution time. The target function is $\cos(t \cdot x)$. |
| `d` | `int` | required | Number of signal applications and polynomial degree bound. Larger values do not guarantee a better optimizer result. |
| `x` | `float` | `0.5` | Test point $x \in [-1, 1]$ at which to evaluate the approximation error. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device (e.g. `'cpu'`, `'cuda'`). |
| `dtype` | `type` | `np.complex128` | Numeric dtype for simulation. |

**Common misunderstandings:**
- `x` must be in $[-1, 1]$ (the domain of the signal operator $W(x)$). Values outside this range are clipped internally.
- `d` determines the number of signal $W(x)$ applications: `d` signal operators and `d+1` phase rotations in the circuit.
- The phase sequence `_find_phases()` is computed via numerical optimization (L-BFGS-B), not analytically. The result may vary across runs due to random initialization.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on normal wrapper return, independently of optimizer convergence or `Absolute error`. |
| `Estimated value` | `complex` | Circuit amplitude $\langle0\rvert U_{\mathrm{circuit}}(x)\lvert0\rangle$ at `x`. |
| `Ideal value` | `float` | Ideal value $\cos(t \cdot x)$ at test point `x`. |
| `Absolute error` | `float` | `abs(qsp_val - cos(t*x))` at the test point. |
| `Computation time (s)` | `float` | Wall-clock simulation execution time in seconds. |
| `circuit` | `Circuit` | The built QSP circuit object. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `plot` | `list` | List of saved output file dicts, each with `'format'` (last 3 chars of filename) and `'filename'` keys. |

## Implementation Architecture

`QSPAlgorithm` in `algorithm.py` organizes the algorithm into five stages with one classical optimizer helper.

**`run(t, d, x, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Phase Optimization | `_find_phases(t, d)` — runs L-BFGS-B minimization | Computes the QSP phase sequence $\Phi$ numerically |
| 2 — Circuit Construction | Creates `Circuit(Register('q', 1))`; applies `qc.rz(2*phases[0], 0)` as initial phase; loops `d` times applying `qc.rx(2*theta, 0)` (signal) then `qc.rz(2*phases[k], 0)` (phase rotation) | Builds the alternating signal/phase circuit |
| 3 — Simulation | `qc.execute(backend=backend, device=device, dtype=dtype)` → `final_state[0]` | Runs statevector evolution of the single qubit |
| 4 — Post-Processing | Compares `qsp_val = complex(final_state[0])` to `ideal_val = np.cos(t * x)` | Computes absolute approximation error |
| 5 — Export | `self.save_circuit(qc)` and `self.save_txt()` | Saves SVG circuit diagram and text result file |

**Helper Methods:**

- **`_find_phases(t, d)`** — Fully classical numerical optimizer. Builds an internal matrix-product simulation of the QSP circuit for a grid of `2d+1` samples. Defines loss as average squared magnitude error vs. `cos(t * x)`. Runs `scipy.optimize.minimize` with L-BFGS-B starting from random initial phases (`np.random.randn(d+1) * 0.1`). Returns `(d+1)` phase values.
- **`_build_return_dict(success, circuit_path, filepath, circuit)`** — Packages the result. Sets `status='ok'` if `success=True`, `'failed'` otherwise. Wraps `filepath` into a list of `{'format': filename[-3:], 'filename': filename}` dicts stored under `'plot'`. Merges `self.output` (containing `'Estimated value'`, `'Ideal value'`, `'Absolute error'`, `'Computation time (s)'`) into the returned dict.

**Single-qubit circuit structure (Stage 2):**
```
Rz(2φ₀) → Rx(2θ) → Rz(2φ₁) → Rx(2θ) → Rz(2φ₂) → ... → Rx(2θ) → Rz(2φ_d)
```
Total gates: `2d + 1`. The `theta = arccos(clip(x, -1, 1))` is a fixed parameter computed from `x`.

**Important implementation note:** The simulation in `_find_phases` uses explicit 2×2 matrix products (NumPy) — not the `Circuit` engine — so the optimization is fully classical and independent of the backend.

**Data flow:** `(t, d)` → `_find_phases()` → `phases` array → `Circuit` with Rz/Rx gates → `execute()` → `final_state[0]` → `abs_error` vs. `ideal_val = np.cos(t * x)` → `_build_return_dict()` → result dict.

## Understanding the Key Quantum Components

$$W(x) = \begin{pmatrix} x & i\sqrt{1-x^2} \\ i\sqrt{1-x^2} & x \end{pmatrix}$$
This optimizer matrix is $R_x(-2\arccos(x))$ under the SDK convention. The executed SDK signal call has the opposite sign: `rx(2*arccos(x), 0)`.

### 2. Phase Rotations $e^{i\phi_k Z}$
The optimizer uses $e^{i\phi_k Z}=\operatorname{diag}(e^{i\phi_k},e^{-i\phi_k})$; the SDK circuit calls `rz(2*phi_k, 0)`. QSP polynomials satisfy parity and boundedness constraints.

### 3. QSP Circuit Structure
```
Rz(2φ₀) → Rx(2θ) → Rz(2φ₁) → Rx(2θ) → ... → Rz(2φ_d)
```
where $\theta = \arccos(x)$. The $(0,0)$ matrix element of the product equals $P(x)$.

### 4. Phase Optimization
The phases $\Phi$ are found by minimizing:
$$L(\Phi)=\frac{1}{2d+1}\sum_{j=0}^{2d}|\langle0|U_\Phi(x_j)|0\rangle-\cos(t x_j)|^2,$$
over `np.linspace(-1, 1, 2*d+1)`. The wrapper uses the returned phases without checking `res.success`.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Scalar signal | Optimizer uses $W_+(x)$; circuit uses `rx(2*arccos(clip(x,-1,1)), 0)`, its adjoint. |
| Phase convention | Optimizer uses $e^{+i\phi_k Z}$; circuit `rz(2*phi_k, 0)` uses $e^{-i\phi_k Z}$. |
| Initial phase $\phi_0$ | `qc.rz(2 * phases[0], 0)` before loop |
| Phase sequence $\Phi = (\phi_0, \ldots, \phi_d)$ | `phases` returned by `_find_phases()` — `(d+1)` floats |
| Target function $\cos(t \cdot x)$ | `ideal_val = np.cos(t * x)` in Stage 4 |
| Loss function minimization | `minimize(loss, ...)` with `method='L-BFGS-B'` in `_find_phases()` |
| Circuit $(0,0)$ amplitude | `qsp_val = complex(final_state[0])` after `qc.execute()`; conjugated relative to the optimizer convention. |
| Approximation error at test point | `abs_error = float(np.abs(qsp_val - ideal_val))` |
| Phase-search sample grid | `x_samples = np.linspace(-1, 1, 2*d+1)`; uniform, not Chebyshev nodes. |

**Implementation limits:** `_find_phases()` evaluates a uniform grid with explicit 2×2 matrix products. The returned `Absolute error` measures the complex circuit amplitude only at `x`; it is not a worst-case interval bound. The separate teaching script fits and compares only the real part and uses a fixed seed, so its errors and status are not the wrapper contract.

## Mathematical Deep Dive

$$\langle 0|U_\Phi(x)|0\rangle = P(x)$$

**Function approximation target:** The wrapper numerically fits $\cos(tx)$ on a finite uniform grid; it does not compute a truncated Chebyshev expansion or certify a uniform approximation bound. Optimizer convergence, parity, and degree affect the achieved error.

**Circuit size:** $d$ signal `Rx` gates and $d+1$ phase `Rz` gates, totaling $2d+1$. This excludes classical phase-search cost.

## Hands-On Example

```python
from unitarylab_algorithms import QSPAlgorithm
import numpy as np

algo = QSPAlgorithm(text_mode="legacy")

# Approximate cos(0.5 * x) at several test points
for x_test in [0.0, 0.3, 0.7, 1.0]:
    result = algo.run(t=0.5, d=8, x=x_test)
    ideal = np.cos(0.5 * x_test)
    print(f"x={x_test:.1f}: error={result['Absolute error']:.2e}, ideal={ideal:.4f}")
```

## Minimal Manual Implementation

The following Python skeleton reconstructs the key components: phase optimization and the alternating QSP circuit.

### Step 1: Simulate QSP circuit classically for optimization

```python
# Simplified reconstruction — mirrors QSPAlgorithm._find_phases() internals
import numpy as np
from scipy.optimize import minimize

def qsp_matrix(x: float, phases: np.ndarray) -> np.ndarray:
    """Compute the QSP circuit product matrix for signal x and phase vector phases."""
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    # Optimizer W+(x) = Rx(-2*arccos(x)) under the SDK convention
    W = np.array([[x, 1j*np.sqrt(1-x**2)],
                  [1j*np.sqrt(1-x**2), x]])
    # Optimizer phase exp(+i*phi_0*Z), opposite sign to SDK rz(2*phi_0)
    mat = np.array([[np.exp(1j*phases[0]), 0], [0, np.exp(-1j*phases[0])]])
    for k in range(1, len(phases)):
        # Pauli Rz gate: e^{i phi_k Z}
        Rz_k = np.array([[np.exp(1j*phases[k]), 0], [0, np.exp(-1j*phases[k])]])
        mat = mat @ W @ Rz_k
    return mat

def find_qsp_phases(tau: float, d: int, seed: int = 42) -> np.ndarray:
    """Teaching phase search for cos(tau*x); seed/maxiter differ from the wrapper."""
    np.random.seed(seed)
    x_samples = np.linspace(-1, 1, 2*d + 1)
    target_vals = np.cos(tau * x_samples)

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
from unitarylab.core import Circuit, Register
import numpy as np

def build_qsp_circuit(phases: np.ndarray, x_value: float, backend: str = 'torch') -> Circuit:
    """Construct the QSP circuit: Rz(2φ₀) → [Rx(2θ) → Rz(2φₖ)] × d."""
    d = len(phases) - 1
    theta = float(np.arccos(np.clip(x_value, -1.0, 1.0)))
    qc = Circuit(Register('q', 1))
    # Initial phase rotation
    qc.rz(2 * phases[0], 0)
    for k in range(1, d + 1):
        qc.rx(2 * theta, 0)      # signal operator W(x) = Rx(2*arccos(x))
        qc.rz(2 * phases[k], 0)  # phase rotation
    return qc

# Full QSP pipeline
def qsp_approximate(t: float, d: int, x: float, backend: str = 'torch'):
    phases = find_qsp_phases(t, d)
    circ = build_qsp_circuit(phases, x, backend)
    state = circ.execute(backend=backend).state
    qsp_val = complex(state[0]) if hasattr(state, '__getitem__') else 0.0
    ideal = np.cos(t * x)
    return qsp_val, abs(qsp_val - ideal)
```

## Debugging Tips

1. **Large `Absolute error`**: Inspect degree, optimizer termination, target parity, and circuit/optimizer gate conventions. Increasing `d` can help but does not certify accuracy.
2. **Optimizer not converging**: The L-BFGS-B optimizer uses default iteration limits. Modify `_find_phases()` to pass `options={'maxiter': N}` to `minimize()` if needed.
3. **`x` outside $[-1, 1]$**: Internally clipped by `np.clip`. Ensure your signal values are in the valid domain.
4. **Error non-zero at `x=0`**: $\cos(t \cdot 0) = 1$; if error is large here with small `t`, the optimizer may have gotten stuck. Re-run to get a different random initialization.
5. **Random phase initialization**: `_find_phases()` uses `np.random.randn(d+1)*0.1` as starting phases. Results may vary across runs; simply re-run `algo.run()` to try a different initialization.

## Reference Implementation

The main implementation path in this project remains the **UnitaryLab QSP implementation** based on `QSPAlgorithm`, `Circuit`, and alternating `Rz/Rx` signal-processing rotations.

PennyLane is provided here only as a reference implementation for the more general **Quantum Singular Value Transformation (QSVT)** framework. Compared with the current UnitaryLab QSP example, PennyLane QSVT applies polynomial transformations to the singular values of a block-encoded matrix or operator.

### Example A: Minimal PennyLane QSVT Run with Matrix Block Encoding

```python
import pennylane as qml
import numpy as np

# Hermitian input matrix A
A = np.array([
    [0.1, 0.2],
    [0.2, -0.4],
])

# Polynomial P(x) = 0.2 + 0.3x^2
# Coefficients are ordered from lowest to highest power.
poly = np.array([0.2, 0.0, 0.3])

dev = qml.device("default.qubit", wires=[0, 1])

@qml.qnode(dev)
def circuit():
    qml.qsvt(
        A,
        poly,
        encoding_wires=[0, 1],
        block_encoding="embedding",
    )
    return qml.state()

result = circuit()
print(result)

```
