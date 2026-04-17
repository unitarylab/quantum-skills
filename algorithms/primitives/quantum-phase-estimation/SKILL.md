---
name: quantum-phase-estimation
description: A quantum phase estimation algorithm that can estimate the eigenvalues of a unitary operator with high precision, which is a fundamental component in many quantum algorithms such as Shor's algorithm and quantum simulation.
---

# Quantum Phase Estimation (QPE)

## Purpose

Given a unitary operator $U$ and one of its eigenstates $|\psi\rangle$ satisfying $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$, QPE uses $d$ auxiliary qubits and the inverse QFT to extract a binary approximation of the phase $\phi$ with precision $1/2^d$.

Use this skill when you need to:
- Estimate an eigenphase of a known unitary.
- Use QPE as a subroutine in HHL, QAE, or Shor's algorithm.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Initialize a $d$-qubit phase register in $|0\rangle^d$ and a target register in the eigenstate $|\psi\rangle$.
2. Apply Hadamard gates to all phase qubits, creating a uniform superposition.
3. Apply controlled powers of $U$: the $k$-th phase qubit controls $U^{2^k}$.
4. Apply the inverse QFT (iQFT) to the phase register.
5. Measure the phase register. The result is a $d$-bit binary representation of $\phi$.

The `QPEAlgorithm` class also exposes `build_qpe_circuit()` for embedding QPE as a subroutine in other algorithms (e.g., HHL).

## Prerequisites

- Understanding of controlled unitary gates.
- Understanding of the Quantum Fourier Transform (QFT) and iQFT.
- Python: `numpy`, `GateSequence`, `State`, project library `IQFT`.

## Using the Provided Implementation

```python
from engine.algorithms import QPEAlgorithm
from engine.core import GateSequence
import numpy as np

# Build a 1-qubit unitary with known phase phi = 1/4
# U = diag(1, e^{2pi*i*phi}) so with phi=0.25: U = diag(1, i) = S gate
U = GateSequence(1, name="S_gate", backend='torch')
U.s(0)   # S gate has phase e^{i*pi/2} = e^{2pi*i*0.25}

# Prepare eigenstate |1> (eigenstate of S is |1> with eigenvalue i=e^{i*pi/2})
prepare_psi = GateSequence(1, name="prep_1", backend='torch')
prepare_psi.x(0)   # |0> -> |1>

algo = QPEAlgorithm()
result = algo.run(
    U=U,
    d=4,                       # 4-bit phase precision (1/16 = 0.0625)
    prepare_target=prepare_psi,
    backend='torch'
)

print(result['estimated_phase'])        # Should be ~0.25
print(result['confidence_probability'])      # Probability of the best state
print(result['circuit_path'])   # SVG circuit diagram path
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `U` | `GateSequence` | required | Unitary operator whose phase is to be estimated. |
| `d` | `int` | required | Number of phase-register qubits. Precision is $1/2^d$. |
| `prepare_target` | `GateSequence` or `None` | `None` | Circuit preparing the eigenstate $\|\psi\rangle$. Defaults to $\|0\rangle$ if `None`. |
| `backend` | `str` | `'torch'` | Simulation backend. Forces `'torch'`. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `d` determines precision: to resolve $\phi$ to accuracy $\epsilon$, use $d \geq \lceil \log_2(1/\epsilon) \rceil$.
- If `prepare_target` is `None`, the target starts in $|0\rangle$. Only correct if $|0\rangle$ is an eigenstate of $U$.
- The phase register occupies the first $d$ qubits; the target register follows.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `estimated_phase` | `float` | Estimated phase as a decimal in $[0, 1)$. |
| `confidence_probability` | `float` | the `brest_prob`, Probability of the best-fit phase bit-string. |
| `circuit` | `GateSequence` | The full QPE circuit object. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `message` | `str` | Summary with estimated phase value. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`QPEAlgorithm` in `algorithm.py` splits the implementation into `run()` (the five-stage orchestrator) and `build_qpe_circuit()` (a reusable, exportable circuit builder).

**`run(U, d, prepare_target, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Setup | Extracts `n_target`, computes `total_qubits = d + n_target` | Establishes qubit layout: phase (`d` qubits) + target (`n_target` qubits) |
| 2 — Circuit Construction | Calls `self.build_qpe_circuit(U, d, prepare_target, backend=backend)` | Delegates all circuit building to the reusable helper |
| 3 — Simulation | `gs.execute()` → `np.asarray(final_state)` → `State(state_arr)` | Runs statevector simulation; wraps result in a `State` object |
| 4 — Phase Extraction | `state_obj.probabilities_dict(phase_qubits, endian='little', threshold=1e-8)` → picks best bit-string → `phi_est = int(bits,2) / 2^d` | Marginalizes target register; converts binary readout to decimal phase |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**`build_qpe_circuit(U, d, prepare_target, backend)` — Reusable Circuit Builder:**

This method is the core algorithmic component, designed to be called by other algorithms (e.g., HHL, QAE). It builds the circuit in four sub-steps:

1. If `prepare_target` is provided, appends it to the target register (`target_qubits = range(d, d+n_target)`).
2. Applies Hadamard to all `d` phase qubits (`phase_qubits = range(d)`).
3. Loops `k = 0..d-1`: builds `cU = U.control(1, '1')` and applies it `2^k` times controlled by `phase_qubits[k]`.
4. Appends `IQFT(d, backend=backend)` from `engine.library` to the phase register.

**Helper Methods:**

- **`_update_last_result` / `_build_return`** — Store runtime fields and package result dict with ASCII panel.
- `State.probabilities_dict(phase_qubits, endian='little', threshold=...)` — called in Stage 4 to extract phase register marginals from the full statevector.

**Data flow:** `U` → `build_qpe_circuit()` → `GateSequence` → `execute()` → `State.probabilities_dict()` → `phi_est` → `_build_return()`.

**Note:** `build_qpe_circuit()` can be called directly without `run()` to obtain the `GateSequence` for embedding in a parent algorithm.

## Understanding the Key Quantum Components
The $d$ phase qubits are initialized to $|0\rangle^d$ and Hadamard-transformed:
$$H^{\otimes d}|0\rangle^d = \frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1}|j\rangle$$

### 2. Controlled Unitary Powers
The $k$-th phase qubit (qubit $k$, $k = 0, \ldots, d-1$) controls $U^{2^k}$:
$$\text{if phase qubit}_k = |1\rangle, \text{ apply } U^{2^k} \text{ to target}$$
This decomposes the multi-controlled unitary $\sum_j|j\rangle\langle j|\otimes U^j$ into $d$ single-controlled gates, each efficiently realized by repeated application of $cU$.

### 3. Phase Kickback
Since $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$, applying controlled-$U^{2^k}$ to $|\psi\rangle$ controlled by $|1\rangle$ accumulates phase $e^{2\pi i\phi \cdot 2^k}$:
$$\frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1} e^{2\pi i\phi j}|j\rangle \otimes |\psi\rangle$$

### 4. Inverse QFT
The iQFT transforms the phase-encoded register to: if $\phi = k_0/2^d$ exactly, the output is $|k_0\rangle$ with probability 1. In general, the probability peaks near the closest $d$-bit approximation of $\phi$.

### 5. Subroutine Usage
`QPEAlgorithm.build_qpe_circuit(U, d, prepare_target, backend)` returns a standalone `GateSequence` for embedding QPE in other algorithms (e.g., HHL uses this directly).

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Phase register $|0\rangle^d$ → $H^{\otimes d}$ | `for q in phase_qubits: gs.h(q)` in `build_qpe_circuit()` |
| Eigenstate preparation $|\psi\rangle$ | `prepare_target` appended via `gs.append(prepare_target, target_qubits)` |
| Controlled $U^{2^k}$ | `cU = U.control(1, '1')` repeated `2^k` times per phase qubit `k` |
| Inverse QFT | `IQFT(d, backend)` from `engine.library`, appended to `phase_qubits` |
| Phase readout $\phi = k_0/2^d$ | `int(best_bits_str, 2) / (2 ** d)` in Stage 4 |
| Probability of best phase | `best_prob = sorted_phases[0][1]` from `state_obj.probabilities_dict()` |
| Phase precision $\delta\phi = 1/2^d$ | Implicit: determined by number of bits `d` in phase register |
| Subroutine for HHL / QAE | `build_qpe_circuit()` returns a standalone `GateSequence` embeddable externally |

**Notes on encapsulation:** The iQFT is sourced from `engine.library.IQFT` rather than constructed inline, unlike the amplitude estimation implementation which builds it locally. The controlled-unitary power is realized by looping `2^k` repetitions of `cU`, not by general unitary exponential; this is correct but exponentially expensive in `d`. The `endian='little'` argument in `probabilities_dict()` ensures the least-significant bit is on the left, consistent with the register ordering.

## Mathematical Deep Dive
$$|0\rangle^d|\psi\rangle \rightarrow \frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1} e^{2\pi i\phi j}|j\rangle|\psi\rangle$$

The iQFT then maps this to:
$$\sum_{k'=0}^{2^d-1}\left(\frac{1}{2^d}\sum_{j=0}^{2^d-1} e^{2\pi i j(\phi - k'/2^d)}\right)|k'\rangle|\psi\rangle$$

When $\phi = k_0/2^d$ exactly, the sum equals $\delta_{k', k_0}$ and the measurement outcome is $k_0$ with probability 1.

**Precision:** To estimate $\phi$ to $n$ bits of accuracy with probability $\geq 1 - \epsilon$, use $d = n + \lceil\log_2(2 + 1/(2\epsilon))\rceil$ phase bits.

**Library backend:** This implementation uses `engine.library.IQFT` for the inverse QFT.

## Hands-On Example

```python
from engine.algorithms import QPEAlgorithm
from engine.core import GateSequence
import numpy as np

# Estimate phase of T gate (T|1> = e^{i*pi/4}|1>, phi = 1/8 = 0.125)
U = GateSequence(1, name="T_gate", backend='torch')
U.t(0)

prepare_psi = GateSequence(1, name="prep1", backend='torch')
prepare_psi.x(0)  # |1> is eigenstate of T with phase pi/4

algo = QPEAlgorithm()
result = algo.run(U=U, d=6, prepare_target=prepare_psi, backend='torch')

print(f"Estimated phi = {result['estimated_phase']:.6f}")  # Expected ≈ 0.125
print(f"Best probability = {result['confidence_probability']:.4f}")
```

## Implementing Your Own Version

```python
from engine.core import GateSequence
from engine.library import IQFT

def qpe_circuit(U: GateSequence, d: int, prepare_target=None, backend='torch'):
    n_target = U.get_num_qubits()
    gs = GateSequence(d + n_target, backend=backend)
    phase = list(range(d))
    target = list(range(d, d + n_target))

    if prepare_target is not None:
        gs.append(prepare_target, target)

    for q in phase:
        gs.h(q)

    cU = U.control(1)
    for k in range(d):
        for _ in range(2 ** k):
            gs.append(cU, target + [phase[k]])

    gs.append(IQFT(d, backend=backend), phase)
    return gs
```

## Debugging Tips

1. **Wrong phase estimate**: Check that `prepare_target` actually prepares an eigenstate of `U`. If the initial state is a superposition of eigenstates, QPE will show multiple peaks.
2. **Phase not in $[0, 1)$**: The output `estimated_phase = int(best_bits, 2) / 2^d` is always in $[0, 1)$ by construction; no folding needed for QPE (unlike QAE).
3. **Precision insufficient**: Increase `d`. Each additional bit doubles the resolution.
4. **Gate count explodes**: $U^{2^{d-1}}$ is applied up to $2^{d-1}$ times. For deep $U$, use matrix exponentiation to build $U^{2^k}$ directly.
5. **IQFT sign convention**: The engine's `IQFT` matches the standard convention. Do not swap QFT and IQFT orders.
