---
name: amplitude-estimation
description: A quantum algorithm for estimating the amplitude of a specific state in a quantum superposition, which can be used for various applications such as Monte Carlo simulations and optimization problems. Provides efficient implementations and educational resources for understanding and utilizing amplitude estimation in quantum algorithm development.
---

# Amplitude Estimation (QAE)

## Purpose

Quantum Amplitude Estimation (QAE) estimates the success probability $a$ of a quantum circuit $U$ (i.e., $U|0\rangle = \sqrt{a}|\text{good}\rangle + \sqrt{1-a}|\text{bad}\rangle$) with $O(1/\epsilon)$ circuit evaluations — a quadratic speedup over the classical $O(1/\epsilon^2)$ Monte Carlo sampling.

Use this skill when you need to:
- Precisely estimate a probability amplitude output by a quantum circuit.
- Apply QAE as a subroutine in quantum finance, quantum Monte Carlo, or other estimation tasks.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

QAE combines Grover/Amplitude Amplification with Quantum Phase Estimation (QPE):

1. The Grover operator $Q = S_\psi S_f$ acts as a rotation in the 2D space spanned by $|\text{good}\rangle$ and $|\text{bad}\rangle$, with eigenvalue phases $\pm 2\theta$ where $\sin\theta = \sqrt{a}$.
2. A QPE circuit with $d$ phase-register qubits measures $\phi \approx 2\theta/(2\pi)$.
3. The amplitude is recovered via $\hat{a} = \sin^2(\pi \phi)$.

## Prerequisites

- Understanding of Grover iteration (oracle + diffuser).
- Understanding of Quantum Phase Estimation (QPE) and the inverse QFT.
- Python: `numpy`, `GateSequence`, `Register`.

## Using the Provided Implementation

```python
from engine.algorithms import AmplitudeEstimationAlgorithm
from engine.core import GateSequence

# Build state preparation U (data register only, no ancilla)
U = GateSequence(2, name="PrepU", backend='torch')
U.ry(1.1, 0)
U.cx(0, 1)

algo = AmplitudeEstimationAlgorithm()
result = algo.run(
    U=U,
    good_zero_qubits=[0],   # Good state: qubit 0 in |0>
    d=6,                    # Phase register bits (higher d = better precision)
    backend='torch'
)

print(result['estimated_amplitude'])  # Estimated probability p
print(result['phi'])                  # Estimated phase phi
print(result['histogram'])            # Phase register measurement histogram
print(result['circuit_path'])         # SVG circuit diagram path
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `U` | `GateSequence` | required | State preparation circuit on data qubits (no ancilla). |
| `good_zero_qubits` | `List[int]` | required | Indices of qubits that must be $\|0\rangle$ to define the good state. |
| `d` | `int` | `6` | Number of qubits in the phase register. Precision scales as $\delta a \approx \pi/2^d$. |
| `backend` | `str` | `'torch'` | Simulation backend. Implementation forces `'torch'`. |
| `algo_dir` | `str` or `None` | `None` | Directory to save circuit diagram. |

**Common misunderstandings:**
- `d` controls precision, not the number of Grover iterations. Larger `d` means more controlled-$Q$ applications in QPE.
- The total qubit count is `d + n_data + 1` (phase + data + Grover ancilla).
- `estimated_amplitude` is $\sin^2(\pi\phi)$; do not confuse the phase $\phi$ with the amplitude.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | Always `'success'`. |
| `estimated_amplitude` | `float` | Estimated success probability $\hat{a} = \sin^2(\pi\phi)$. |
| `phi` | `float` | Estimated phase $\phi \in [0, 0.5]$ after folding. |
| `histogram` | `dict` | Bit-string → probability for phase register measurements. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `message` | `str` | Summary including best phase bit-string and estimated $\hat{a}$. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`AmplitudeEstimationAlgorithm` in `algorithm.py` organizes the QAE algorithm into five ordered stages within `run()`, plus a set of self-contained quantum-circuit building helpers.

**`run(U, good_zero_qubits, d, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Setup | Computes `n_data`, sets `total_qubits = d + n_data + 1` | Establishes register layout: phase (`d` qubits) + data (`n_data` qubits) + Grover ancilla (1 qubit) |
| 2 — Circuit Construction | Builds `prepare` (data init), calls `_grover_operator_from_zero_oracle(U, good_zero_qubits)` to get `G`, then calls `_qpe_circuit(G, d, prepare)` | Assembles the complete QPE circuit around the Grover operator |
| 3 — Simulation | `qpe_circ.execute()` → `np.asarray(state)` → `_phase_histogram(statevector, d)` | Runs statevector simulation; extracts phase register probability histogram |
| 4 — Classical Post-Processing | Reads top bit-string; computes `phi_raw = int(bits, 2) / 2^d`; folds to `[0, 0.5]`; computes `est_amp = sin²(π·φ)` | Converts QPE phase readout back to amplitude estimate |
| 5 — Export | `qpe_circ.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_grover_operator_from_zero_oracle(U, good_zero_qubits)`** — Builds a single Grover iteration `G` as a `GateSequence(n_data+1)`. Calls `_phase_oracle_all_zeros` (good-state phase flip), then `_diffusion_about_prepared_state` (reflection about $|\psi\rangle$). Appends a global phase correction (`X` + kickback ancilla) so that controlled-`G` functions correctly in QPE.
- **`_qpe_circuit(G, d, prepare_target)`** — Constructs the full QPE circuit. Appends state prep to target register, applies H to all phase qubits, applies controlled-`G^(2^k)` for each phase qubit `k`, then appends the iQFT via `_iqft_circuit(d)`.
- **`_iqft_circuit(n, do_swaps)`** — Builds the inverse QFT circuit directly from Hadamard gates and controlled-phase rotations `mcp`. Qubit-swap reversal is applied first.
- **`_phase_oracle_all_zeros(gs, zero_qubits, ancilla)`** — Phase-kickback oracle: prepares ancilla in $|-\rangle$, X-flips `zero_qubits`, applies MCX, unflips, unprepares ancilla. Same as amplitude amplification's oracle.
- **`_diffusion_about_prepared_state(gs, U, data_qubits, ancilla)`** — Applies `U†`, calls `_phase_oracle_all_zeros` on all data qubits, applies `U`. Same as amplitude amplification's diffuser.
- **`_phase_histogram(statevector, d)`** — Marginalizes the full statevector over the non-phase qubits by grouping indices `idx % 2^d` and summing probabilities. Returns a sorted dict of bit-string → probability.
- **`_update_last_result` / `_build_return`** — Store all runtime fields in `self.last_result` and package the result dict with the ASCII panel.

**Data flow:** `U` + `good_zero_qubits` → `_grover_operator_from_zero_oracle()` → `G` → `_qpe_circuit()` → `qpe_circ` → `execute()` → `_phase_histogram()` → `est_amp` → `_build_return()`.

## Understanding the Key Quantum Components
The data register is initialized with $U|0\rangle^n = \sqrt{a}|\text{good}\rangle + \sqrt{1-a}|\text{bad}\rangle$. The good state is defined by `good_zero_qubits` all being $|0\rangle$.

### 2. Grover Operator $Q$
The Grover operator is constructed internally as:
- **Phase oracle** $S_f$: Flips the phase of the good state using phase kickback from an ancilla qubit prepared in $|-\rangle$.
- **Diffuser** $S_\psi = 2|\psi\rangle\langle\psi| - I$: Reflects about the prepared state $|\psi\rangle = U|0\rangle$.

In the 2D invariant subspace, $Q$ acts as a rotation by angle $2\theta$:
$$Q|\psi_\pm\rangle = e^{\pm i2\theta}|\psi_\pm\rangle, \quad \sin\theta = \sqrt{a}$$

A global phase correction is applied to cancel the $(-1)$ introduced by the diffuser sign convention, so controlled-$Q$ works correctly in QPE.

### 3. Quantum Phase Estimation (QPE)
A $d$-qubit phase register is prepared in uniform superposition. Controlled powers of $Q$ ($Q^{2^0}, Q^{2^1}, \ldots, Q^{2^{d-1}}$) encode the phase $2\theta$ into the register. The inverse QFT (iQFT) converts phase to binary.

### 4. Phase Folding and Amplitude Recovery
QPE produces two symmetric peaks at $\phi \approx 2\theta/(2\pi)$ and $1 - 2\theta/(2\pi)$. The code folds to $[0, 0.5]$ via:
$$\phi = \min(\phi_{\text{raw}}, 1 - \phi_{\text{raw}})$$
Then:
$$\hat{a} = \sin^2(\pi\phi)$$

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| State $|\psi\rangle = U|0\rangle$ | `prepare = GateSequence(n_data+1)`; `prepare.append(U, data_indices)` |
| Good state: all `good_zero_qubits` in $|0\rangle$ | `good_zero_qubits` → `_phase_oracle_all_zeros(zero_qubits=good_zero_qubits)` |
| Grover operator $Q$ | `_grover_operator_from_zero_oracle(U, good_zero_qubits)` → returns `G` as a `GateSequence` |
| Global phase correction $(-1)$ | X-kick on ancilla appended at end of `_grover_operator_from_zero_oracle` |
| QPE phase register ($d$ bits) | `phase = list(range(d))` in `_qpe_circuit()`; Hadamard applied to all `d` bits |
| Controlled $Q^{2^k}$ applications | Inner loop `for _ in range(2**k): gs.append(cU, ...)` in `_qpe_circuit()` |
| Inverse QFT (iQFT) | `_iqft_circuit(d)` — appended to phase qubits in `_qpe_circuit()` |
| Phase readout $\phi$ | `phi_raw = int(best_bits, 2) / 2^d`; folded to `[0, 0.5]` |
| Amplitude formula $\hat{a} = \sin^2(\pi\phi)$ | `est_amp = float(np.sin(np.pi * phi) ** 2)` |
| Phase histogram | `_phase_histogram(statevector, d)` — marginalizes all non-phase qubits |
| Query complexity $O(1/\epsilon)$ | Implicit: `d` phase bits → $2^d$ controlled-`G` total applications → accuracy $\pi/2^d$ |

**Notes on encapsulation:** The iQFT is built directly from `mcp` and `h` gates in `_iqft_circuit()` rather than calling an external library. The Grover operator and QPE circuit are both fully self-contained `GateSequence` objects that can be inspected or drawn independently. The phase folding and $\sin^2$ recovery are purely classical post-processing steps after simulation.

## Mathematical Deep Dive

**Grover eigenvalues:**
$$Q|\psi_\pm\rangle = e^{\pm i2\theta}|\psi_\pm\rangle, \quad |\psi_\pm\rangle = \frac{1}{\sqrt{2}}(|\text{good}\rangle \pm i|\text{bad}\rangle)$$

**QPE readout:** With $d$ phase bits, the resolution is $\delta\phi = 1/2^d$, leading to amplitude accuracy:
$$\delta a \approx \pi \cdot \delta\phi = \frac{\pi}{2^d}$$

**Query complexity:** QPE with $d$ bits requires $O(2^d)$ queries to $Q$, which equals $O(1/\epsilon)$ for target accuracy $\epsilon$. Classical Monte Carlo requires $O(1/\epsilon^2)$.

**Peak success probability:** The main QPE peak appears with probability $\geq 4/\pi^2 \approx 0.405$.

## Hands-On Example

```python
from engine.algorithms import AmplitudeEstimationAlgorithm
from engine.core import GateSequence
import numpy as np

# True success probability: p = sin^2(pi/8) ≈ 0.146
# Single qubit state: |psi> = cos(theta)|0> + sin(theta)|1>
# Good state: qubit 0 in |0>
theta = np.pi / 8
U = GateSequence(1, name="prep", backend='torch')
U.ry(2 * theta, 0)  # cos(theta)|0> + sin(theta)|1>; good = |0>, p = cos^2(theta)

algo = AmplitudeEstimationAlgorithm()
result = algo.run(U=U, good_zero_qubits=[0], d=8, backend='torch')

print(f"True p = {np.cos(theta)**2:.6f}")
print(f"QAE estimate = {result['estimated_amplitude']:.6f}")
print(f"Phase phi = {result['phi']:.6f}")
print(result['plot'])
```

## Implementing Your Own Version

The following Python skeleton reconstructs the key components of the QAE algorithm and can be adapted into a compatible implementation.

### Step 1: Build the iQFT circuit

```python
# Simplified reconstruction — mirrors AmplitudeEstimationAlgorithm._iqft_circuit()
import numpy as np
from engine.core import GateSequence

def build_iqft(n: int, do_swaps: bool = True, backend: str = 'torch') -> GateSequence:
    gs = GateSequence(n, name=f"iQFT_{n}", backend=backend)
    if do_swaps:
        for i in range(n // 2):
            gs.swap(i, n - 1 - i)
    for j in range(n):
        for k in range(j):
            angle = -np.pi / (2 ** (j - k))
            gs.mcp(angle, k, j)   # controlled-phase rotation
        gs.h(j)
    return gs
```

### Step 2: Build the Grover operator with global phase correction

```python
# Simplified reconstruction — mirrors AmplitudeEstimationAlgorithm._grover_operator_from_zero_oracle()

def build_phase_oracle(n_data: int, good_zero_qubits: list, backend: str = 'torch') -> GateSequence:
    """Phase-kickback oracle: flips phase of states where good_zero_qubits are all |0>."""
    gs = GateSequence(n_data + 1, backend=backend)
    ancilla = n_data
    # 1. Prepare ancilla in |->
    gs.x(ancilla); gs.h(ancilla)
    # 2. Flip controlled qubits; MCX on ancilla
    for q in good_zero_qubits: gs.x(q)
    if len(good_zero_qubits) == 1:
        gs.cx(good_zero_qubits[0], ancilla)
    else:
        gs.mcx(good_zero_qubits, ancilla)
    for q in good_zero_qubits: gs.x(q)
    # 3. Unprepare ancilla
    gs.h(ancilla); gs.x(ancilla)
    return gs

def build_grover_operator(U: GateSequence, good_zero_qubits: list, backend: str = 'torch') -> GateSequence:
    """Single Grover iteration G = Diffuser ∘ Oracle, with global phase correction."""
    n_data = U.get_num_qubits()
    ancilla = n_data
    data = list(range(n_data))
    gs = GateSequence(n_data + 1, name="GroverIter", backend=backend)
    # Oracle
    oracle = build_phase_oracle(n_data, good_zero_qubits, backend)
    gs.append(oracle, list(range(n_data + 1)))
    # Diffuser: U† → all-zeros oracle → U
    gs.append(U.dagger(), data)
    all_zeros_oracle = build_phase_oracle(n_data, data, backend)
    gs.append(all_zeros_oracle, list(range(n_data + 1)))
    gs.append(U, data)
    # Global phase correction (-1) so controlled-G works in QPE
    gs.x(ancilla); gs.h(ancilla)
    gs.x(ancilla)
    gs.h(ancilla); gs.x(ancilla)
    return gs
```

### Step 3: Build the QPE circuit around G

```python
# Simplified reconstruction — mirrors AmplitudeEstimationAlgorithm._qpe_circuit()

def build_qpe_circuit(G: GateSequence, d: int,
                      prepare_target=None, backend: str = 'torch') -> GateSequence:
    """QPE circuit: controlled-G^(2^k) powers + iQFT on phase register."""
    n_target = G.get_num_qubits()
    gs = GateSequence(d + n_target, name=f"QPE_d{d}", backend=backend)
    phase = list(range(d))
    target = list(range(d, d + n_target))

    if prepare_target is not None:
        gs.append(prepare_target, target)

    for q in phase:
        gs.h(q)

    cG = G.control(1)
    for k in range(d):
        power = 2 ** k
        for _ in range(power):
            gs.append(cG, target + [phase[k]])

    iqft = build_iqft(d, do_swaps=True, backend=backend)
    gs.append(iqft, phase)
    return gs
```

### Step 4: Extract amplitude from phase histogram

```python
def phase_histogram(statevector: np.ndarray, d: int) -> dict:
    """Marginalizes all non-phase qubits; groups by phase register index."""
    probs = np.abs(statevector) ** 2
    counts = {}
    modulus = 2 ** d
    for idx, p in enumerate(probs):
        if p < 1e-12: continue
        k = idx % modulus
        bits = format(k, f'0{d}b')
        counts[bits] = counts.get(bits, 0.0) + float(p)
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))

def qae_full(U, good_zero_qubits, d, backend='torch'):
    """End-to-end QAE: returns estimated amplitude."""
    G = build_grover_operator(U, good_zero_qubits, backend)
    n_data = U.get_num_qubits()
    prepare = GateSequence(n_data + 1, backend=backend)
    prepare.append(U, list(range(n_data)))
    qpe_circ = build_qpe_circuit(G, d, prepare_target=prepare, backend=backend)
    state = qpe_circ.execute()
    sv = np.asarray(state, dtype=complex).reshape(-1)
    histogram = phase_histogram(sv, d)
    best_bits = next(iter(histogram))
    phi_raw = int(best_bits, 2) / (2 ** d)
    phi = min(phi_raw, 1.0 - phi_raw)
    return float(np.sin(np.pi * phi) ** 2), phi, histogram
```

## Debugging Tips

1. **Large estimation error**: Increase `d`. Each additional bit halves the phase resolution.
2. **Histogram has many small peaks instead of one dominant peak**: This is expected for small `d`; increase `d` to sharpen the peak.
3. **`estimated_amplitude` misses a factor**: Ensure the good state condition (all `good_zero_qubits` in $|0\rangle$) matches your circuit's actual target. A common mistake is using wrong qubit indices.
4. **Symmetric peaks**: The histogram may show two peaks at $\phi$ and $1-\phi$. The code folds both to the same estimate — this is correct behavior.
5. **Circuit size scaling**: Total circuit depth scales as $O(2^d)$ due to QPE. For `d=10`, expect a deep circuit.
