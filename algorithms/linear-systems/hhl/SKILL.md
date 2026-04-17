---
name: hhl
description: A quantum algorithm for solving linear systems of equations, providing exponential speedup over classical methods for certain types of problems. This skill includes efficient implementations and educational resources for understanding and utilizing quantum linear systems algorithms in various applications.
---

# HHL Algorithm (Harrow-Hassidim-Lloyd)

## Purpose

HHL solves a linear system $Ax = b$ where $A$ is Hermitian, producing a quantum state $|x\rangle \propto A^{-1}|b\rangle$. The quantum speedup is exponential in the problem size ($O(\log N)$ qubits vs. classical $O(N)$) under specific conditions.

Use this skill when you need to:
- Solve $Ax = b$ for a Hermitian matrix $A$ of size $N = 2^n$.
- Use HHL as a subroutine in quantum simulation or optimization.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

HHL proceeds in five steps:
1. **State preparation**: Encode $b$ as quantum state $|b\rangle$ in the system register.
2. **QPE**: Apply Quantum Phase Estimation using $U = e^{iAt}$ to extract eigenvalue phases of $A$ into a phase register.
3. **Controlled rotation**: Apply an ancilla rotation $R_y(2\arcsin(C/\lambda_j))$ controlled by the estimated eigenvalue, mapping amplitude proportional to $1/\lambda_j$.
4. **Inverse QPE**: Uncompute the phase register.
5. **Post-selection**: Measure the ancilla in $|1\rangle$; the system register collapses to $A^{-1}|b\rangle$.

## Prerequisites

- QPE (Quantum Phase Estimation).
- Matrix eigenvalues, Hermitian matrices, condition number.
- Python: `numpy`, `GateSequence`, `State`, and `QPEAlgorithm` (from `fundamental_algorithm.qpe`).

## Using the Provided Implementation

```python
from engine.algorithms import HHLAlgorithm
import numpy as np

# 2x2 Hermitian system
A = np.array([[1.5, 0.5], [0.5, 1.5]])
b = np.array([1.0, 0.0])

algo = HHLAlgorithm()
result = algo.run(
    A=A,
    b=b,
    d=4,           # Phase register bits
    backend='torch'
)

print(result['quantum_solution'])    # Quantum solution vector x_q
print(result['classical_solution'])  # Classical exact solution x_c
print(result['diff_norm'])           # ||x_q - x_c||
print(result['post_selection_prob'])    # Post-selection success probability
print(result['circuit_path'])        # SVG circuit diagram
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` | required | Hermitian matrix of size $N \times N$ where $N = 2^n$. |
| `b` | `np.ndarray` | required | Right-hand side vector of length $N$. |
| `d` | `int` | required | Phase register bits. More bits → better eigenvalue resolution. |
| `t` | `float` or `None` | `None` | Evolution time. Auto-computed from eigenvalues if `None`. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `A` must be **Hermitian** (`A == A†`). Non-Hermitian matrices will raise `ValueError`.
- `A` must be square with dimension a **power of 2**. Pad with zeros if needed.
- `t` is auto-tuned to avoid phase wraparound. Setting `t` too large will force auto-correction with a warning.
- The output `quantum_solution` contains the reconstructed classical vector, not the quantum state coefficients directly.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'success'` if post-selection probability is nonzero. |
| `solution_quantum` | `np.ndarray` | Reconstructed solution vector from quantum state. |
| `solution_classical` | `np.ndarray` | Exact classical solution $A^{-1}b$ for comparison. |
| `state_system`| `float`| ... |
| `post_selection_prob` | `float` | Probability of post-selecting ancilla $= |1\rangle$. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `message` | `str` | Result summary. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`HHLAlgorithm` in `algorithm.py` is the most architecturally complex single-algorithm file. It orchestrates five macro-stages with four specialized sub-circuit builders and reuses the `QPEAlgorithm.build_qpe_circuit()` module from the fundamental algorithms.

**`run(A, b, d, t, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Matrix Pre-processing | Validates $A$ (Hermitian, power-of-2 dims); normalizes $b$; extracts eigenvalues; computes `t` to avoid phase wraparound; computes `k_start` and `scale_factor`; sets `signed_phase_mode` for matrices with negative eigenvalues | Adaptive parameter tuning |
| 2 — Circuit Construction | Builds `GateSequence(1 + d + n_sys)` with ancilla at 0, phase at 1..d, system at d+1...; calls four sub-circuit builders (A–D) | Full HHL circuit assembly |
| 3 — Simulation | `gs.execute()` → `np.asarray(final_state)` | Runs statevector simulation |
| 4 — Post-Processing | `_postselect_solution_state(state_arr, scale_factor, d, n_sys)` extracts the ancilla=1 sub-state; computes `diff_norm` vs. `np.linalg.solve(A, b)` | Solution extraction and comparison |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Stage 2 — Four Sub-Circuit Steps:**

- **Step A: State Preparation** — `gs.initialize(b_state, system_qubits)` loads normalized $b/\|b\|$ into the system register.
- **Step B: QPE** — `_expi_hermitian(A, t)` computes $U = e^{iAt}$ numerically; `_unitary_circuit_from_matrix(U_mat)` wraps it in a `GateSequence` via `gs.unitary()`; `QPEAlgorithm().build_qpe_circuit(U_circ, d)` builds the full QPE sub-circuit; it is appended to `phase_qubits + system_qubits`.
- **Step C: Controlled Reciprocal Rotation** — `_controlled_reciprocal_rotation(d, t, k_start, signed_phase)` builds the ancilla rotation circuit. For each phase integer $k$, it applies X-flips to phase bits to select the $k$-th computational basis state, computes rotation angle $2\arcsin(C/k)$, and applies `mcry(theta, controls, ancilla=0)`. Then unflips X gates. This creates a distinct controlled-Ry for each eigenvalue bin.
- **Step D: Inverse QPE** — `gs.append(qpe_circ.dagger(), ...)` uncomputes the phase register, removing entanglement.

**Helper Methods:**

- **`_expi_hermitian(A, t)`** — Eigendecomposition: `np.linalg.eigh(A)` → `V @ diag(exp(i*2π*λ*t)) @ V†`.
- **`_unitary_circuit_from_matrix(U, backend)`** — Wraps an $N\times N$ matrix as a `GateSequence` via `gs.unitary(U, range(n))`.
- **`_controlled_reciprocal_rotation(d, t, k_start, signed_phase, backend)`** — Builds the ancilla rotation sub-circuit; handles both signed (negative eigenvalues) and unsigned phase modes.
- **`_decode_signed_phase_index(k, d)`** — Converts raw QPE bin index to signed integer (negative for bins above `grid//2`).
- **`_postselect_solution_state(state, scale, d, n)`** — Extracts the ancilla=1, phase=0 subspace: `state[1::2]` gives ancilla=1 slice; `[0::stride]` (with `stride = 2^d`) selects phase=0; scales by `scale_factor` to recover $A^{-1}b$.

**Data flow:** `(A, b)` → eigenvalue analysis → circuit assembly (A→B→C→D) → `execute()` → `_postselect_solution_state()` → `solution_quantum` → `_build_return()`.

## Understanding the Key Quantum Components
The normalized vector $b/\|b\|$ is loaded into the system register using `GateSequence.initialize()`.

### 2. Unitary Exponentiation ($U = e^{iAt}$)
The Hermitian matrix $A$ is exponentiated to produce a unitary $U = e^{iAt}$. QPE is applied to $U$; eigenphases of $U$ encode eigenvalues of $A$ as $\phi_j = \lambda_j t / (2\pi)$.

### 3. Quantum Phase Estimation
`QPEAlgorithm.build_qpe_circuit()` is called internally with $U = e^{iAt}$ and $d$ phase bits. The phase register encodes the binary approximation of $\phi_j$.

### 4. Controlled Reciprocal Rotation (Ancilla Rotation)
An ancilla qubit is rotated by angle $2\arcsin(C/\lambda_j)$ where $C$ is a normalization constant. After rotation, the ancilla amplitude $\sin(2\arcsin(C/\lambda_j)) \propto 1/\lambda_j$ encodes the inverse eigenvalue.

### 5. Inverse QPE
The phase register is uncomputed (disentangled) by running the QPE circuit backwards (`qpe_circ.dagger()`).

### 6. Post-selection
Measuring the ancilla in $|1\rangle$ collapses the system register to:
$$|x\rangle \propto \sum_j \frac{b_j}{\lambda_j}|u_j\rangle = A^{-1}|b\rangle$$

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| State preparation $|b\rangle = b/\|b\|$ | `gs.initialize(b_state, system_qubits)` — Step A |
| Evolution time $t$ (avoids wraparound) | Auto-computed: `t = target_phi_max / lam_max` in Stage 1 |
| Unitary $U = e^{iAt}$ | `_expi_hermitian(A, t)` → explicit matrix via NumPy eigendecomposition |
| QPE to encode eigenphases | `QPEAlgorithm().build_qpe_circuit(U_circ, d)` — Step B |
| Ancilla rotation angle $2\arcsin(C/\lambda_j)$ | `_controlled_reciprocal_rotation()` — one `mcry` per phase bin $k$ |
| Normalization constant $C = k_\text{start}$ | `C = k_start` inside `_controlled_reciprocal_rotation()` |
| Inverse QPE (uncompute phase) | `qpe_circ.dagger()` appended to same qubit indices — Step D |
| Post-selection on ancilla $= |1\rangle$ | `_postselect_solution_state()` — `state[1::2]` slice |
| Phase=0 post-selection | `vec_anc1[0::stride]` where `stride = 2^d` |
| Scale factor for classical solution | `scale_factor = norm_b * t * grid / k_start` |
| Signed phase mode (negative eigenvalues) | `signed_phase_mode` flag; `_decode_signed_phase_index()` maps bins > `grid//2` to negative |

**Notes on encapsulation:** The QPE sub-circuit is sourced from `QPEAlgorithm.build_qpe_circuit()` rather than rebuilt inline — this is the primary example of algorithm reuse in this codebase. The controlled reciprocal rotation is built without any classical oracle: it exhaustively creates one `mcry` gate per eigenvalue bin (total $2^d$ gates), which is the dominant circuit cost. The post-selection extraction is done via array slicing rather than symbolic measurement.

## Mathematical Deep Dive, $|b\rangle = \sum_j b_j |u_j\rangle$.

**QPE encoding:** $U|u_j\rangle = e^{i\lambda_j t}|u_j\rangle$. QPE reads off $\phi_j = \lambda_j t /(2\pi)$ into the phase register.

**Controlled rotation:** Ancilla becomes $\sqrt{1-(C/\lambda_j)^2}|0\rangle + (C/\lambda_j)|1\rangle$ for each eigencomponent.

**Post-selection output:** Tracing out the ancilla $=|1\rangle$:
$$|x\rangle = \frac{1}{\mathcal{N}}\sum_j \frac{Cb_j}{\lambda_j}|u_j\rangle, \quad \mathcal{N} = \sqrt{\sum_j C^2|b_j|^2/\lambda_j^2}$$

**Condition number dependence:** The post-selection probability scales as $\Theta(1/\kappa^2)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number.

## Hands-On Example

```python
from engine.algorithms import HHLAlgorithm
import numpy as np

# 4x4 example (requires 2 system qubits)
A = np.array([
    [2, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 4]
], dtype=float)
b = np.array([1, 1, 1, 1], dtype=float)

algo = HHLAlgorithm()
result = algo.run(A=A, b=b, d=5, backend='torch')

print(f"Quantum solution: {result['quantum_solution'].real.round(4)}")
print(f"Classical solution: {result['classical_solution'].real.round(4)}")
print(f"Post-selection prob: {result['post_selection_prob']:.4f}")
```

## Implementing Your Own Version

The following Python skeleton reconstructs the five structural stages of the HHL algorithm using the key building blocks.

### Stage A: State Preparation and Unitary Exponentiation

```python
# Simplified reconstruction — mirrors HHLAlgorithm._expi_hermitian() and _unitary_circuit_from_matrix()
import numpy as np
from engine.core import GateSequence

def expi_hermitian(A: np.ndarray, t: float) -> np.ndarray:
    """Compute U = exp(i*A*t) numerically via eigendecomposition."""
    eigenvalues, V = np.linalg.eigh(A)
    exp_diag = np.exp(1j * eigenvalues * t)
    return (V * exp_diag) @ V.conj().T    # V @ diag(exp) @ V†

def unitary_circuit(U_mat: np.ndarray, backend: str = 'torch') -> GateSequence:
    """Wrap a unitary matrix as a GateSequence."""
    n = int(np.log2(U_mat.shape[0]))
    gs = GateSequence(n, backend=backend)
    gs.unitary(U_mat, list(range(n)))
    return gs
```

### Stage B: QPE to Encode Eigenphases

```python
# Exact usage — calls QPEAlgorithm.build_qpe_circuit() as in the real implementation
from engine.algorithms import QPEAlgorithm

def encode_eigenphases(U_circ: GateSequence, d: int, backend: str = 'torch') -> GateSequence:
    """Build QPE sub-circuit for eigenvalue encoding."""
    return QPEAlgorithm().build_qpe_circuit(U_circ, d, prepare_target=None, backend=backend)
```

### Stage C: Controlled Reciprocal Rotation (ancilla rotation)

```python
# Simplified reconstruction — mirrors HHLAlgorithm._controlled_reciprocal_rotation()

def controlled_reciprocal_rotation(d: int, k_start: int, backend: str = 'torch') -> GateSequence:
    """For each eigenvalue bin k, rotate ancilla by 2*arcsin(k_start/k)."""
    gs = GateSequence(d + 1, backend=backend)   # d phase qubits + 1 ancilla
    ancilla = 0
    phase_qubits = list(range(1, d + 1))
    grid = 2 ** d
    for k in range(1, grid):
        angle = 2 * np.arcsin(k_start / k)
        # Select phase bin |k> by X-flipping bits where k_bit = 0
        k_bits = format(k, f'0{d}b')
        controls = []
        for bit_idx, bit in enumerate(reversed(k_bits)):
            if bit == '0':
                gs.x(phase_qubits[bit_idx])
            controls.append(phase_qubits[bit_idx])
        gs.mcry(angle, controls, ancilla)
        # Unflip
        for bit_idx, bit in enumerate(reversed(k_bits)):
            if bit == '0':
                gs.x(phase_qubits[bit_idx])
    return gs
```

### Stage D: Inverse QPE and Post-selection

```python
# Simplified reconstruction — mirrors how HHL assembles stages and extracts the solution

def extract_hhl_solution(state_arr: np.ndarray, d: int, n_sys: int, scale_factor: float):
    """Extract solution from statevector by post-selecting ancilla=1, phase=0."""
    # ancilla=1 block: indices [1::2] (ancilla qubit 0 = 1)
    stride = 2 ** d
    vec_anc1 = state_arr[1::2]
    # phase=0 block within ancilla=1: every stride-th element starting at 0
    solution = vec_anc1[0::stride][:2**n_sys]
    return solution * scale_factor
```

### Full Minimal Skeleton

```python
# Simplified reconstruction illustrating overall HHL control flow

import numpy as np
from engine.core import GateSequence
from engine.algorithms import QPEAlgorithm

def hhl_minimal(A, b, d, t=None, backend='torch'):
    n_sys = int(np.log2(len(b)))
    # Auto-tune t to avoid phase wraparound
    lam = np.linalg.eigvalsh(A)
    lam_max = np.max(np.abs(lam))
    if t is None:
        t = 0.45 / lam_max

    # Stage A: exp(iAt) as unitary circuit
    U_mat = expi_hermitian(A, t)
    U_circ = unitary_circuit(U_mat, backend)

    # Stage B: QPE sub-circuit
    qpe_circ = encode_eigenphases(U_circ, d, backend)

    # Full HHL circuit (simplified - excludes ancilla rotation sub-circuit assembly)
    # In the real implementation HHLAlgorithm._controlled_reciprocal_rotation()
    # and qpe_circ.dagger() are assembled sequentially
    pass  # full assembly in HHLAlgorithm.run()
```

1. **`A` not Hermitian**: Add `A = (A + A.conj().T) / 2` to symmetrize before calling.
2. **`N` not a power of 2**: Pad `A` and `b` to the next power of 2 with zeros.
3. **Large `diff_norm`**: Increase `d` for better phase resolution. Also check if $t$ is auto-corrected.
4. **`post_selection_prob` near zero**: The condition number $\kappa$ is very large. HHL's efficiency degrades with large $\kappa$.
5. **Negative eigenvalues**: The code detects negative eigenvalues and enables `signed_phase_mode` automatically, which constrains phase to $[0, 0.5)$.
