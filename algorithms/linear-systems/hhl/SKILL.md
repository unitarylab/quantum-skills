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

## Overview

HHL proceeds in five steps:
1. **State preparation**: Encode $b$ as quantum state $|b\rangle$ in the system register.
2. **QPE**: Apply Quantum Phase Estimation using $U = e^{i 2\pi A t}$ to extract eigenvalue phases of $A$ into a phase register. QPE reads eigenphases $\phi_j = \lambda_j t$ (in cycles), encoding them as phase register bins $k_j \approx \phi_j \cdot 2^d = \lambda_j \cdot t \cdot 2^d$.
3. **Controlled rotation**: Apply an ancilla rotation $R_y(2\arcsin(C/\lambda_j))$ controlled by the estimated eigenvalue, mapping amplitude proportional to $1/\lambda_j$.
4. **Inverse QPE**: Uncompute the phase register.
5. **Post-selection**: Measure the ancilla in $|1\rangle$; the system register collapses to $A^{-1}|b\rangle$.

## Prerequisites

- QPE (Quantum Phase Estimation).
- Matrix eigenvalues, Hermitian matrices, condition number.
- Python: `numpy`, `Circuit`, and `QPE` (from `unitarylab.library`).

## Using the Provided Implementation

```python
from unitarylab import Circuit
from unitarylab_algorithms import HHLAlgorithm
import numpy as np

# 2x2 Hermitian system
A = np.array([[1.5, 0.5], [0.5, 1.5]])
b = np.array([1.0, 0.0])

algo = HHLAlgorithm(text_mode="plain")
result = algo.run(
    A=A,
    b=b,
    d=4,           # Phase register bits
    backend='torch'
)

print(result['Estimated solution (quantum)'])    # Quantum solution vector x_q
print(result['Exact solution (classical)'])      # Classical exact solution x_c
print(result['L2 error'])                        # ||x_q - x_c||
print(result['Post-selection probability'])      # Post-selection success probability
print(result['circuit_path'])                    # SVG circuit diagram
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | `np.ndarray` | required | Hermitian matrix of size $N \times N$ where $N = 2^n$. |
| `b` | `np.ndarray` | required | Right-hand side vector of length $N$. |
| `d` | `int` | required | Phase register bits. More bits → better eigenvalue resolution. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device (e.g. `'cpu'`, `'cuda'`). |
| `dtype` | `type` | `np.complex128` | Numeric dtype for simulation. |

> **Note:** `algo_dir` is set in `HHLAlgorithm.__init__(algo_dir=...)`, not in `run()`. Evolution time `t` is always auto-computed from the matrix eigenvalues and is not a `run()` parameter.

**Common misunderstandings:**
- `A` must be **Hermitian** (`A == A†`). Non-Hermitian matrices will raise `ValueError`.
- `A` must be square with dimension a **power of 2**. Pad with zeros if needed.
- Evolution time `t` is auto-computed inside `run()` from eigenvalues (`t = target_phi_max / lam_max`), ensuring $\phi_{\max} = \lambda_{\max} t$ stays within $[0,1)$ to avoid phase wraparound. It is not a user-facing parameter.
- The output key `'Estimated solution (quantum)'` contains the reconstructed classical vector, not the raw quantum state coefficients directly.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` on failure. |
| `Estimated solution (quantum)` | `np.ndarray` | Reconstructed solution vector $x_q$ from quantum post-selection. |
| `Exact solution (classical)` | `np.ndarray` | Exact classical solution $A^{-1}b$ via `np.linalg.solve`. |
| `L2 error` | `float` | L2 norm \|\|x_q - x_classical\|\| between quantum and classical solutions. |
| `Post-selection probability` | `float` | Probability of post-selecting ancilla $= |1\rangle$ and phase $= |0\rangle$. |
| `Computation time (s)` | `float` | Low-level simulation execution time in seconds. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `plot` | `list` | List of output file dicts, each with `'format'` and `'filename'` keys. |
| `circuit` | `Circuit` | The assembled `Circuit` object for the full HHL circuit. |

## Implementation Architecture

`HHLAlgorithm` in `algorithm.py` is the most architecturally complex single-algorithm file. It orchestrates five macro-stages with four specialized sub-circuit builders and reuses `QPE` from `unitarylab.library`.

**`run(A, b, d, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Matrix Pre-processing | Validates $A$ (Hermitian, power-of-2 dims); normalizes $b$; extracts eigenvalues; computes `t` to avoid phase wraparound; computes `k_start` and `scale_factor`; sets `signed_phase_mode` for matrices with negative eigenvalues | Adaptive parameter tuning |
| 2 — Circuit Construction | Builds `Circuit(1 + d + n_sys)` with ancilla at 0, phase at 1..d, system at d+1...; calls four sub-circuit builders (A–D) | Full HHL circuit assembly |
| 3 — Simulation | `qc.execute()` → `np.asarray(final_state)` | Runs statevector simulation |
| 4 — Post-Processing | `_postselect_solution_state(state_arr, scale_factor, d, n_sys)` extracts the ancilla=1 sub-state; computes `diff_norm` vs. `np.linalg.solve(A, b)` | Solution extraction and comparison |
| 5 — Export | `qc.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Stage 2 — Four Sub-Circuit Steps:**

- **Step A: State Preparation** — `qc.initialize(b_state, system_qubits)` loads normalized $b/\|b\|$ into the system register.
- **Step B: QPE** — `_expi_hermitian(A, t)` computes $U = e^{i 2\pi A t}$ numerically; `_unitary_circuit_from_matrix(U_mat)` wraps it in a `Circuit` via `qc.unitary()`; `QPE(U_circ, d, return_circuit=True)` builds the full QPE sub-circuit; it is appended to `phase_qubits + system_qubits`.
- **Step C: Controlled Reciprocal Rotation** — `_controlled_reciprocal_rotation(d, t, k_start, signed_phase)` builds the ancilla rotation circuit. For each phase integer $k$, it applies X-flips to phase bits to select the $k$-th computational basis state, computes rotation angle $2\arcsin(C/k)$, and applies `mcry(theta, controls, ancilla=0)`. Then unflips X gates. This creates a distinct controlled-Ry for each eigenvalue bin.
- **Step D: Inverse QPE** — `qc.append(qpe_circ.dagger(), ...)` uncomputes the phase register, removing entanglement.

**Helper Methods:**

- **`_expi_hermitian(A, t)`** — Eigendecomposition: `np.linalg.eigh(A)` → `V @ diag(exp(i*2π*λ*t)) @ V†`.
- **`_unitary_circuit_from_matrix(U)`** — Wraps an $N\times N$ matrix as a `Circuit` via `qc.unitary(U, list(range(n)))`.
- **`_controlled_reciprocal_rotation(d, t, k_start, signed_phase)`** — Builds the ancilla rotation sub-circuit; handles both signed (negative eigenvalues) and unsigned phase modes.
- **`_decode_signed_phase_index(k, d)`** — Converts raw QPE bin index to signed integer (negative for bins above `grid//2`).
- **`_postselect_solution_state(state, scale, d, n)`** — Extracts the ancilla=1, phase=0 subspace: `state[1::2]` gives ancilla=1 slice; `[0::stride]` (with `stride = 2^d`) selects phase=0; scales by `scale_factor` to recover $A^{-1}b$.

**Data flow:** `(A, b)` → eigenvalue analysis → circuit assembly (A→B→C→D) → `execute()` → `_postselect_solution_state()` → `'Estimated solution (quantum)'` → `_build_return_dict()`.

## Understanding the Key Quantum Components
The normalized vector $b/\|b\|$ is loaded into the system register using `Circuit.initialize()`.

### 2. Unitary Exponentiation ($U = e^{i 2\pi A t}$)
The Hermitian matrix $A$ is exponentiated to produce a unitary $U = e^{i 2\pi A t}$. QPE is applied to $U$; eigenphases of $U$ encode eigenvalues of $A$ as $\phi_j = \lambda_j t$ (in cycles, i.e. fraction of $2\pi$). The phase bin index is $k_j \approx \phi_j \cdot 2^d = \lambda_j \cdot t \cdot 2^d$.

**Evolution time $t$:** The code auto-computes $t = \text{target\_phi\_max} / \lambda_{\max}$ so that the largest eigenphase $\phi_{\max} = \lambda_{\max} t = \text{target\_phi\_max}$ stays within QPE's resolvable range $[0, 1)$, preventing phase wraparound. For unsigned mode $\text{target\_phi\_max} = 1 - 1/2^d$; for signed mode $\text{target\_phi\_max} = 0.5 - 1/2^d$.

**Signed phase mode:** When $A$ has negative eigenvalues, $\lambda_j t$ can be negative. Since QPE measures phases in $[0, 1)$, negative phases wrap to the upper half of the interval. By constraining $|\lambda_{\max}| t < 0.5$, positive eigenvalues map to $\phi_j \in [0, 0.5)$ and negative eigenvalues map to $\phi_j \in (0.5, 1)$. The decoder `_decode_signed_phase_index(k, d)` converts raw bin indices back to signed integers (bins above $\text{grid}/2$ are interpreted as negative).

### 3. Quantum Phase Estimation
`QPE(U_circ, d, return_circuit=True)` is called internally with $U = e^{i 2\pi A t}$ and $d$ phase bits. The phase register encodes the binary approximation of $\phi_j = \lambda_j t$.

### 4. Controlled Reciprocal Rotation (Ancilla Rotation)
An ancilla qubit is rotated by angle $\theta = 2\arcsin(C/\lambda_j)$ where $C$ is a normalization constant. After $R_y(\theta)|0\rangle$, the $|1\rangle$ amplitude is $\sin(\theta/2) = \sin(\arcsin(C/\lambda_j)) = C/\lambda_j$, which is proportional to $1/\lambda_j$, encoding the inverse eigenvalue.

### 5. Inverse QPE
The phase register is uncomputed (disentangled) by running the QPE circuit backwards (`qpe_circ.dagger()`).

### 6. Post-selection
Measuring the ancilla in $|1\rangle$ collapses the system register to:
$$|x\rangle \propto \sum_j \frac{b_j}{\lambda_j}|u_j\rangle = A^{-1}|b\rangle$$

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| State preparation $|b\rangle = b/\|b\|$ | `qc.initialize(b_state, system_qubits)` — Step A |
| Evolution time $t = \text{target\_phi\_max} / \lambda_{\max}$ (ensures $\phi_{\max} < 1$, no wraparound) | Auto-computed in Stage 1; $\phi_j = \lambda_j t$ maps eigenvalues to QPE phase bins $k_j \approx \lambda_j t \cdot 2^d$ |
| Unitary $U = e^{i 2\pi A t}$ | `_expi_hermitian(A, t)` → explicit matrix via NumPy eigendecomposition |
| QPE to encode eigenphases $\phi_j = \lambda_j t$ | `QPE(U_circ, d, return_circuit=True)` — Step B |
| Ancilla rotation angle $\theta_k = 2\arcsin(C/k)$ per phase bin $k$ | `_controlled_reciprocal_rotation()` — one `mcry` per phase bin; $|1\rangle$ amplitude $= \sin(\theta_k/2) = C/k \propto 1/\lambda_j$ |
| Normalization constant $C = k_\text{start}$ | $C = k_\text{start}$ inside `_controlled_reciprocal_rotation()`; $k_\text{start} = \lfloor \phi_{\min} \cdot 2^d \rfloor - 10$ |
| Inverse QPE (uncompute phase) | `qpe_circ.dagger()` appended to same qubit indices — Step D |
| Post-selection on ancilla $= |1\rangle$ | `_postselect_solution_state()` — `state[1::2]` slice |
| Phase=0 post-selection | `vec_anc1[0::stride]` where `stride = 2^d` |
| Scale factor for classical solution | `scale_factor = \|b\| \cdot t \cdot 2^d / k_\text{start}` — compensates for the $C/k$ scaling and $b$ normalization |
| Signed phase mode (negative eigenvalues) | `signed_phase_mode` flag; constrains $\|\lambda_{\max}\|t < 0.5$ so positive $\phi_j \in [0, 0.5)$, negative $\phi_j \in (0.5, 1)$; `_decode_signed_phase_index()` maps bins $> 2^d/2$ to negative |

**Notes on encapsulation:** The QPE sub-circuit is sourced from `QPE(U_circ, d, return_circuit=True)` (imported from `unitarylab.library`) rather than rebuilt inline — this is the primary example of algorithm reuse in this codebase. The controlled reciprocal rotation is built without any classical oracle: it exhaustively creates one `mcry` gate per eigenvalue bin (total $2^d$ gates), which is the dominant circuit cost. The post-selection extraction is done via array slicing rather than symbolic measurement.

## Mathematical Deep Dive, $|b\rangle = \sum_j b_j |u_j\rangle$.

**QPE encoding:** $U|u_j\rangle = e^{i 2\pi \lambda_j t}|u_j\rangle$. QPE reads off $\phi_j = \lambda_j t$ (in cycles) into the phase register, producing phase bin $k_j \approx \phi_j \cdot 2^d = \lambda_j t \cdot 2^d$.

**Controlled rotation:** Ancilla becomes $\sqrt{1-(C/\lambda_j)^2}|0\rangle + (C/\lambda_j)|1\rangle$ for each eigencomponent.

**Post-selection output:** Tracing out the ancilla $=|1\rangle$:
$$|x\rangle = \frac{1}{\mathcal{N}}\sum_j \frac{Cb_j}{\lambda_j}|u_j\rangle, \quad \mathcal{N} = \sqrt{\sum_j C^2|b_j|^2/\lambda_j^2}$$

**Condition number dependence:** The post-selection probability scales as $\Theta(1/\kappa^2)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number.

## Hands-On Example

```python
from unitarylab_algorithms.linear_algebra.hhl.algorithm import HHLAlgorithm
import numpy as np

# 4x4 example (requires 2 system qubits)
A = np.array([
    [2, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 4]
], dtype=float)
b = np.array([1, 1, 1, 1], dtype=float)

algo = HHLAlgorithm(text_mode="plain")
result = algo.run(A=A, b=b, d=5, backend='torch')

print(f"Quantum solution: {result['Estimated solution (quantum)'].real.round(4)}")
print(f"Classical solution: {result['Exact solution (classical)'].real.round(4)}")
print(f"Post-selection prob: {result['Post-selection probability']:.4f}")
```

## Implementing Your Own Version

The following Python skeleton reconstructs the five structural stages of the HHL algorithm using the key building blocks.

### Stage A: State Preparation and Unitary Exponentiation

```python
# Simplified reconstruction — mirrors HHLAlgorithm._expi_hermitian() and _unitary_circuit_from_matrix()
import numpy as np
from unitarylab.core import Circuit

def expi_hermitian(A: np.ndarray, t: float) -> np.ndarray:
    """Compute U = exp(i * 2π * A * t) numerically via eigendecomposition."""
    eigenvalues, V = np.linalg.eigh(A)
    exp_diag = np.exp(1j * 2 * np.pi * eigenvalues * t)
    return (V * exp_diag) @ V.conj().T    # V @ diag(exp) @ V†

def unitary_circuit(U_mat: np.ndarray, backend: str = 'torch') -> Circuit:
    """Wrap a unitary matrix as a Circuit."""
    n = int(np.log2(U_mat.shape[0]))
    qc = Circuit(n)
    qc.unitary(U_mat, list(range(n)))
    return qc
```

### Stage B: QPE to Encode Eigenphases

```python
# Exact usage — mirrors QPE call in the real implementation
from unitarylab.library import QPE

def encode_eigenphases(U_circ: Circuit, d: int) -> Circuit:
    """Build QPE sub-circuit for eigenvalue encoding."""
    return QPE(U_circ, d, return_circuit=True)
```

### Stage C: Controlled Reciprocal Rotation (ancilla rotation)

```python
# Simplified reconstruction — mirrors HHLAlgorithm._controlled_reciprocal_rotation()

def controlled_reciprocal_rotation(d: int, k_start: int, backend: str = 'torch') -> Circuit:
    """For each eigenvalue bin k, rotate ancilla by 2*arcsin(k_start/k)."""
    qc = Circuit(d + 1)   # d phase qubits + 1 ancilla
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
                qc.x(phase_qubits[bit_idx])
            controls.append(phase_qubits[bit_idx])
        qc.mcry(angle, controls, ancilla)
        # Unflip
        for bit_idx, bit in enumerate(reversed(k_bits)):
            if bit == '0':
                qc.x(phase_qubits[bit_idx])
    return qc
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
from unitarylab.core import Circuit
from unitarylab.library import QPE

def hhl_minimal(A, b, d, backend='torch'):
    n_sys = int(np.log2(len(b)))
    # t is auto-tuned inside HHLAlgorithm.run() — not a user parameter
    lam = np.linalg.eigvalsh(A)
    lam_max = np.max(np.abs(lam))
    t = 0.45 / lam_max   # approximate; real code uses target_phi_max / lam_max

    # Stage A: exp(i * 2π * A * t) as unitary circuit
    U_mat = expi_hermitian(A, t)
    U_circ = unitary_circuit(U_mat)

    # Stage B: QPE sub-circuit
    qpe_circ = QPE(U_circ, d, return_circuit=True)

    # Full HHL circuit (simplified - excludes ancilla rotation sub-circuit assembly)
    # In the real implementation HHLAlgorithm._controlled_reciprocal_rotation()
    # and qpe_circ.dagger() are assembled sequentially
    pass  # full assembly in HHLAlgorithm.run()
```

## Troubleshooting

1. **`A` not Hermitian**: Add `A = (A + A.conj().T) / 2` to symmetrize before calling.
2. **`N` not a power of 2**: Pad `A` and `b` to the next power of 2 with zeros.
3. **Large `L2 error`**: Increase `d` for better phase resolution. Evolution time `t` is always auto-computed.
4. **`Post-selection probability` near zero**: The condition number $\kappa$ is very large. HHL's efficiency degrades with large $\kappa$.
5. **Negative eigenvalues**: The code detects negative eigenvalues and enables `signed_phase_mode` automatically, which constrains phase to $[0, 0.5)$.
