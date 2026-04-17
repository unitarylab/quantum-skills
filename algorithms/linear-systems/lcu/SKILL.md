---
name: lcu
description: A quantum algorithm for solving linear systems of equations using the Linear Combination of Unitaries (LCU) method, providing exponential speedup over classical methods for certain types of problems. This skill includes efficient implementations and educational resources for understanding and utilizing the LCU algorithm in various applications.
---
# Linear Combination of Unitaries (LCU)

## Purpose

LCU probabilistically applies a non-unitary operator $M = \sum_j \alpha_j U_j$ to a quantum state by using an ancilla register, a SELECT oracle, and amplitude amplification post-selection. It is a key subroutine in QSVT, Hamiltonian simulation, and quantum linear solvers.

Use this skill when you need to:
- Apply a weighted sum of unitaries to a quantum state.
- Implement block-encodings for non-unitary operations.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

The LCU circuit has three components:
1. **$V$ (PREPARE):** Creates the superposition $V|0\rangle = \sum_j \sqrt{\alpha_j / s}|j\rangle$ in the ancilla register, where $s = \sum_j \alpha_j$.
2. **SELECT ($U_c$):** Controlled on ancilla $= |j\rangle$, applies $U_j$ to the system: $\sum_j |j\rangle|j\rangle_{\text{anc}} \otimes U_j|\psi\rangle$.
3. **$V^\dagger$ (UNPREPARE):** Rotates the ancilla back so that measuring all-zeros succeeds with probability $\|M|\psi\rangle\|^2/s^2$.

## Prerequisites

- Controlled unitary gates.
- Ancilla qubits and post-selection.
- Python: `numpy`, `GateSequence`, `State`.

## Using the Provided Implementation

```python
from engine.algorithms import LCUAlgorithm
from engine.core import GateSequence
import numpy as np

# Build two unitary operators U0, U1
n_sys = 1
U0 = GateSequence(n_sys, backend='torch')
U0.h(0)         # Hadamard

U1 = GateSequence(n_sys, backend='torch')
U1.x(0)         # Pauli-X

# Apply M = 0.6*H + 0.4*X to system
algo = LCUAlgorithm()
result = algo.run(
    alphas=[0.6, 0.4],
    unitaries=[U0, U1],
    n_sys=n_sys,
    initial_state=None,    # Starts in |0>
    backend='torch'
)

print(result['lcu_success_probability'])   # Post-selection probability
print(result['status'])         # 'ok' if success_prob > 1e-6
print(result['circuit_path'])   # SVG circuit diagram
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alphas` | `List[float]` | required | Non-negative weights $\alpha_j$ for each unitary $U_j$. |
| `unitaries` | `List[GateSequence]` | required | List of unitary circuits $U_j$. Must have the same number of qubits as `n_sys`. |
| `n_sys` | `int` | required | Number of qubits in the system (data) register. |
| `initial_state` | `GateSequence` or `None` | `None` | Circuit preparing the initial system state. Defaults to $\|0\rangle^{n_{\text{sys}}}$. |
| `backend` | `str` | `'torch'` | Simulation backend. `'torch'` recommended. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `alphas` must be non-negative (used as probability amplitudes after square root).
- The ancilla register size is $n_{\text{anc}} = \lceil\log_2 m\rceil$ where $m = $ `len(alphas)`.
- LCU is a probabilistic procedure; the success probability can be low for ill-conditioned operators.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if `success_prob > 1e-6`; `'failed'` otherwise. |
| `lcu_success_probability` | `float` | Probability of measuring ancilla $= |0\rangle^{n_{\text{anc}}}$. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `message` | `str` | Result summary. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`LCUAlgorithm` in `algorithm.py` builds the LCU circuit in five stages using two circuit constructor helpers: `_build_V` and `_build_select`.

**`run(alphas, unitaries, n_sys, initial_state, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Setup | Validates `len(alphas)==len(unitaries)`; computes `m`, `n_anc = ceil(log2(m))`, `s_norm = sum(alphas)` | Determines ancilla size |
| 2 — Circuit Construction | Creates `GateSequence(n_anc + n_sys)`; optionally appends `initial_state` to system; calls `_build_V`, `_build_select`, `V_circ.dagger()` in sequence | Assembles full PREPARE → SELECT → UNPREPARE circuit |
| 3 — Simulation | `qc.execute()` → `State(np.asarray(raw_result))` | Runs statevector simulation |
| 4 — Post-Processing | `state_obj.probabilities_dict(anc_qubits, endian='little')` → extracts probability of `'0'*n_anc` | Computes LCU success probability |
| 5 — Export | `qc.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_build_V(alphas, s_norm, m, n_anc, backend)`** — Constructs the PREPARE operator. Builds amplitude vector `state[j] = sqrt(alphas[j]/s_norm)` padded to `2^n_anc` elements. Calls `_state_preparation_tree(state)` to get a list of angle-decomposed Ry/McRy gate specs. Applies them as `qc.ry()` or `qc.mcry()` gates on the ancilla register.
- **`_state_preparation_tree(alpha)`** — Recursive tree decomposition for amplitude encoding. Computes `theta = 2 * arctan2(norm_b, norm_a)` at each binary tree node and records whether the gate is unconditional (root) or conditional (controlled by parent nodes). Returns a list of gate dicts with `{name, target, angle, control_qubit, control_value}`.
- **`_build_select(unitaries, m, n_anc, n_sys)`** — Constructs the SELECT operator. Loops over all `(j, U_j)` pairs; for each, calls `qc.append(U_j, target=sys_qubits, control=anc_qubits, control_sequence=format(j, f'0{n_anc}b'))`. This applies $U_j$ conditionally on ancilla = $|j\rangle$ in binary.
- **`_update_last_result` / `_build_return`** — Store fields and package result dict. Note: `_build_return` returns `status='success'` regardless of `success_prob` value; the low-probability case is reflected in the `success_prob` field.

**Data flow:** `(alphas, unitaries)` → `_build_V()` → `_build_select()` → `V†` → `execute()` → `probabilities_dict()` → `success_prob` → `_build_return()`.

## Understanding the Key Quantum Components
Maps $|0\rangle_{\text{anc}} \rightarrow \sum_j \sqrt{\alpha_j/s}|j\rangle_{\text{anc}}$ where $s = \sum_k \alpha_k$. Implemented as a state-preparation circuit on the ancilla.

### 2. SELECT Operator ($U_c$)
$$U_c = \sum_{j=0}^{m-1} |j\rangle\langle j|_{\text{anc}} \otimes U_j$$
Controlled on each basis state of the ancilla, applies the corresponding $U_j$ to the system. Implemented as a multiplexer (multi-controlled gate).

### 3. UNPREPARE Operator ($V^\dagger$)
The inverse of $V$. After combining PREPARE → SELECT → UNPREPARE, the state decomposes as:
$$|\Phi\rangle = |0\rangle_{\text{anc}} \otimes \frac{1}{s}M|\psi\rangle + |0\rangle_{\text{anc}}^{\perp} \otimes \ldots$$

### 4. Post-selection
Measuring the ancilla in $|0\rangle^{n_{\text{anc}}}$ succeeds with probability:
$$P_{\text{succ}} = \frac{\|M|\psi\rangle\|^2}{s^2}$$
On success, the system register holds $M|\psi\rangle/\|M|\psi\rangle\|$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| PREPARE: $V|0\rangle \to \sum_j\sqrt{\alpha_j/s}|j\rangle$ | `_build_V()` — amplitude tree → Ry/McRy gates on ancilla |
| Amplitude tree decomposition | `_state_preparation_tree(state)` — recursive `arctan2` angles |
| SELECT: $\sum_j|j\rangle\langle j|\otimes U_j$ | `_build_select()` — `qc.append(U_j, ..., control_sequence=format(j,'b'))` |
| UNPREPARE: $V^\dagger$ | `V_circ.dagger()` appended to ancilla qubits |
| Normalization $s = \sum_k\alpha_k$ | `s_norm = float(np.sum(alphas))` in Stage 1 |
| Post-selection on ancilla $= |0\rangle^{n_\text{anc}}$ | `probabilities_dict(anc_qubits)` → key `'0'*n_anc` |
| Success probability $P = \|M|\psi\rangle\|^2/s^2$ | `success_prob` from Stage 4 measurement |
| Ancilla size $n_\text{anc} = \lceil\log_2 m\rceil$ | `n_anc = int(np.ceil(np.log2(m)))` in Stage 1 |

**Notes on encapsulation:** The PREPARE operator uses a recursive classical tree computation (`_state_preparation_tree`) to decompose an arbitrary $2^{n_\text{anc}}$-element state into a Ry/McRy gate sequence. This is entirely classical computation; the quantum part is just applying those gates. The SELECT operator directly calls `qc.append()` with `control_sequence` to avoid building separate MCX structures. The `_build_return` always sets `status='success'`; the actual LCU outcome quality is in `lcu_success_probability`.

## Mathematical Deep Dive
$$V|0\rangle_{\text{anc}}|\psi\rangle = \sum_j \sqrt{\frac{\alpha_j}{s}}|j\rangle|\psi\rangle$$

After SELECT:
$$\sum_j \sqrt{\frac{\alpha_j}{s}}|j\rangle U_j|\psi\rangle$$

After UNPREPARE ($V^\dagger \otimes I$):
$$|0\rangle_{\text{anc}} \otimes \frac{1}{s}M|\psi\rangle + |0\rangle_{\text{anc}}^{\perp} \otimes \ldots$$

Post-selection success probability:
$$P = \frac{\|M|\psi\rangle\|^2}{s^2}$$

To boost $P$, apply amplitude amplification $O(s/\|M|\psi\rangle\|)$ times.

## Hands-On Example

```python
from engine.algorithms import LCUAlgorithm
from engine.core import GateSequence

n_sys = 2
I_circ = GateSequence(n_sys, backend='torch')   # Identity
X0_circ = GateSequence(n_sys, backend='torch')
X0_circ.x(0)

algo = LCUAlgorithm()
result = algo.run(
    alphas=[0.8, 0.2],
    unitaries=[I_circ, X0_circ],
    n_sys=n_sys,
    backend='torch',
    algo_dir='./results/lcu'
)
print(f"Success probability: {result['lcu_success_probability']:.4f}")
print(f"Status: {result['status']}")
```

## Implementing Your Own Version

The following Python skeleton reconstructs the three structural components of LCU: PREPARE, SELECT, and UNPREPARE.

### Step 1: Amplitude-tree PREPARE circuit

```python
# Simplified reconstruction — mirrors LCUAlgorithm._build_V() and _state_preparation_tree()
import numpy as np
from engine.core import GateSequence

def build_prepare(alphas, n_anc: int, backend: str = 'torch') -> GateSequence:
    """Build V|0> = sum_j sqrt(alpha_j/s)|j> on ancilla register."""
    s = sum(alphas)
    m = len(alphas)
    # Pad to 2^n_anc
    state = np.zeros(2**n_anc)
    for j in range(m):
        state[j] = np.sqrt(alphas[j] / s)
    # Use tree decomposition: at each node compute Ry angle theta = 2*arctan2(||right||, ||left||)
    # The real implementation calls _state_preparation_tree() which does this recursively
    # For a flat 2-element case:
    if m == 2:
        gs = GateSequence(n_anc, backend=backend)
        theta = 2 * np.arctan2(state[1], state[0])
        gs.ry(theta, 0)
        return gs
    # General case: delegate to the recursive tree builder in LCUAlgorithm
    raise NotImplementedError("Use LCUAlgorithm._build_V() for general m")
```

### Step 2: SELECT oracle (multiplexer)

```python
# Simplified reconstruction — mirrors LCUAlgorithm._build_select()

def build_select(unitaries, n_anc: int, n_sys: int, backend: str = 'torch') -> GateSequence:
    """SELECT = sum_j |j><j|_anc ⊗ U_j: apply U_j conditional on ancilla=|j>."""
    m = len(unitaries)
    gs = GateSequence(n_anc + n_sys, backend=backend)
    sys_qubits = list(range(n_anc, n_anc + n_sys))
    anc_qubits = list(range(n_anc))
    for j, U_j in enumerate(unitaries):
        control_seq = format(j, f'0{n_anc}b')   # e.g. '01' for j=1, n_anc=2
        gs.append(U_j, target=sys_qubits, control=anc_qubits, control_sequence=control_seq)
    return gs
```

### Step 3: Full LCU circuit assembly

```python
# Simplified reconstruction of the full PREPARE → SELECT → UNPREPARE structure

def lcu_circuit(alphas, unitaries, n_sys: int, initial_state=None, backend: str = 'torch'):
    """Full LCU circuit: PREPARE ⊗ I |psi> → SELECT → UNPREPARE ⊗ I."""
    m = len(alphas)
    n_anc = int(np.ceil(np.log2(max(m, 2))))
    gs = GateSequence(n_anc + n_sys, backend=backend)

    anc_qubits = list(range(n_anc))
    sys_qubits = list(range(n_anc, n_anc + n_sys))

    # 1. Initial system state
    if initial_state is not None:
        gs.append(initial_state, sys_qubits)

    # 2. PREPARE
    V_circ = build_prepare(alphas, n_anc, backend)
    gs.append(V_circ, anc_qubits)

    # 3. SELECT
    sel_circ = build_select(unitaries, n_anc, n_sys, backend)
    gs.append(sel_circ, list(range(n_anc + n_sys)))

    # 4. UNPREPARE (V†)
    gs.append(V_circ.dagger(), anc_qubits)

    return gs
```

## Debugging Tips

1. **`len(alphas) != len(unitaries)`**: Raises `ValueError`. Always keep these lists the same length.
2. **`initial_state.get_num_qubits() != n_sys`**: Raises `ValueError`. The initial state circuit must match `n_sys` exactly.
3. **Very low `success_prob`**: This is expected when $\|M|\psi\rangle\| \ll s$. Use amplitude amplification to boost.
4. **`status='failed'` (prob near 0)**: The operator $M$ may annihilate $|\psi\rangle$, or the unitaries cancel. Verify the operator structure.
5. **Memory limit**: Ancilla adds $\lceil\log_2 m\rceil$ qubits. For $m=8$ unitaries, this is only 3 ancilla qubits.
