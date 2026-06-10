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

## Overview

The LCU circuit has three components:
1. **$V$ (PREPARE):** Creates the superposition $V|0\rangle = \sum_j \sqrt{\alpha_j / s}|j\rangle$ in the ancilla register, where $s = \sum_j \alpha_j$.
2. **SELECT ($U_c$):** Controlled on ancilla $= |j\rangle$, applies $U_j$ to the system: $\sum_j |j\rangle|j\rangle_{\text{anc}} \otimes U_j|\psi\rangle$.
3. **$V^\dagger$ (UNPREPARE):** Rotates the ancilla back so that measuring all-zeros succeeds with probability $\|M|\psi\rangle\|^2/s^2$.

## Prerequisites

- Controlled unitary gates.
- Ancilla qubits and post-selection.
- Python: `numpy`, `Circuit`.

## Using the Provided Implementation

```python
from unitarylab_algorithms import LCUAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Build two unitary operators U0, U1
n_sys = 1
U0 = Circuit(n_sys, name='H')
U0.h(0)         # Hadamard

U1 = Circuit(n_sys, name='X')
U1.x(0)         # Pauli-X

# Apply M = 0.6*H + 0.4*X to system
algo = LCUAlgorithm(text_mode='plain')
result = algo.run(
    alphas=[0.6, 0.4],
    unitaries=[U0, U1],
    n_sys=n_sys,
    initial_state=None,    # Starts in |0>
    backend='torch'
)

print(result['Success probability'])   # Post-selection probability
print(result['status'])                # 'ok' on success
print(result['circuit_path'])          # SVG circuit diagram
print(result['Result state'])          # Post-selected system state
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alphas` | `List[float]` | required | Non-negative weights $\alpha_j$ for each unitary $U_j$. |
| `unitaries` | `List[Circuit]` | required | List of unitary circuits $U_j$. Must have the same number of qubits as `n_sys`. |
| `n_sys` | `int` | required | Number of qubits in the system (data) register. |
| `initial_state` | `Circuit` or `None` | `None` | Circuit preparing the initial system state. Defaults to $\|0\rangle^{n_{\text{sys}}}$. |
| `backend` | `str` | `'torch'` | Simulation backend. `'torch'` recommended. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `alphas` must be non-negative (used as probability amplitudes after square root).
- The ancilla register size is $n_{\text{anc}} = \lceil\log_2 m\rceil$ where $m = $ `len(alphas)`.
- LCU is a probabilistic procedure; the success probability can be low for ill-conditioned operators.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | Always `'ok'` (success/failure reflected in `'Success probability'`). |
| `Success probability` | `float` | Probability of measuring all ancilla qubits in $\|0\rangle$ — the LCU post-selection probability. |
| `Computation time (s)` | `float` | Wall-clock simulation time in seconds. |
| `Result state` | `np.ndarray` | Post-selected system state proportional to $M\|\psi\rangle$. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `plot` | `list` | List of output file dicts, each with `'format'` (last 3 chars of filename) and `'filename'` keys. |
| `circuit` | `Circuit` | The assembled `Circuit` object for the full LCU circuit. |

## Implementation Architecture

`LCUAlgorithm` in `algorithm.py` builds the LCU circuit in five stages using two circuit constructor helpers: `_build_V` and `_build_select`.

**`run(alphas, unitaries, n_sys, initial_state, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Setup | Validates `len(alphas)==len(unitaries)`; computes `m`, `n_anc = ceil(log2(m))`, `s_norm = sum(alphas)` | Determines ancilla size |
| 2 — Circuit Construction | Creates `Circuit(n_anc + n_sys, name='LCU_circuit')`; optionally appends `initial_state` to system; calls `_build_V`, `_build_select`, `V_circ.dagger()` in sequence | Assembles full PREPARE → SELECT → UNPREPARE circuit |
| 3 — Simulation | `qc.execute(backend=backend, device=device, dtype=dtype)` → `raw_result` | Runs statevector simulation |
| 4 — Post-Processing | `raw_result._phase_probabilities_from_state(anc_qubits, endian="little", threshold=0.0)` → extracts probability of `'0'*n_anc`; builds `output` dict with `'Success probability'`, `'Computation time (s)'`, `'Result state'` | Computes LCU success probability and post-selected state |
| 5 — Export | `self.save_circuit(qc)` and `self.save_txt()` | Saves SVG circuit diagram and text result file |

**Helper Methods:**

- **`_build_V(alphas, s_norm, m, n_anc)`** — Constructs the PREPARE operator. Builds amplitude vector `state[j] = sqrt(alphas[j]/s_norm)` padded to `2^n_anc` elements. Creates `Circuit(n_anc, name='V')` and calls `qc.initialize(state, range(n_anc))` to directly load the amplitude distribution.
- **`_build_select(unitaries, m, n_anc, n_sys)`** — Constructs the SELECT operator. Creates `Circuit(n_anc + n_sys, name='SELECT-U')`. Loops over all `(j, U_j)` pairs; for each, computes `ctrl_state = format(j, f"0{n_anc}b")` (reversed if `U.order == 'little'`) and calls `qc.append(U_j, target=sys_qubits, control=anc_qubits, control_state=ctrl_state)`. This applies $U_j$ conditionally on ancilla $= |j\rangle$ in binary.
- **`_build_return_dict(success, circuit_path, filepath, circuit)`** — Packages the result. Sets `status='ok'` if `success=True`, `'failed'` otherwise. Wraps `filepath` into a list of `{'format': filename[-3:], 'filename': filename}` dicts stored under `'plot'`. Merges `self.output` (containing `'Success probability'`, `'Computation time (s)'`, `'Result state'`) into the returned dict.

**Data flow:** `(alphas, unitaries)` → `_build_V()` → `_build_select()` → `V_circ.dagger()` → `qc.execute()` → `_phase_probabilities_from_state()` → `success_prob` → `update_output()` → `_build_return_dict()`.

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
| PREPARE operator | `_build_V()` constructs the state-preparation circuit. It maps `|0>` to `sum_j sqrt(alpha_j / s) |j>` through an amplitude-tree decomposition and controlled rotation gates on the ancilla/select register. |
| Amplitude tree decomposition | `_state_preparation_tree(state)` recursively computes rotation angles using `arctan2`, then converts the target amplitude distribution into a sequence of `Ry` / multi-controlled `Ry` operations. |
| SELECT operator | `_build_select()` implements `SELECT = sum_j |j><j| ⊗ U_j`. For each index `j`, it appends the corresponding controlled unitary `U_j` and uses `ctrl_state=format(j, f"0{n_anc}b")` to ensure the selected block is activated only when the ancilla/select register equals the binary encoding of `j`. |
| UNPREPARE operator | `V_circ.dagger()` is appended after the SELECT block to implement `V†`, mapping the prepared ancilla/select state back toward `|0>`. |
| Normalization factor | `s_norm = float(np.sum(alphas))` in Stage 1 computes `s = sum_k alpha_k`, which is used to normalize the LCU coefficients. |
| Ancilla/select register size | `n_anc = int(np.ceil(np.log2(m)))` in Stage 1 determines the number of ancilla/select qubits required to index all `m` unitary terms. |
| Post-selection condition | `raw_result._phase_probabilities_from_state(anc_qubits, endian="little", threshold=0.0)` is used to obtain the probability distribution on the ancilla/select register. The successful branch corresponds to the all-zero key: `'0' * n_anc`. |
| Success probability | `success_prob` is extracted from the Stage 4 measurement result. It represents the probability of successfully projecting the ancilla/select register onto `|0>^{n_anc}` after applying `PREPARE → SELECT → UNPREPARE`. |
| LCU output relation | After successful post-selection, the system state is proportional to `M|psi> / s`, where `M = sum_j alpha_j U_j`. Therefore, the success probability is related to `||M|psi>||^2 / s^2`. |

**Notes on encapsulation:** The PREPARE operator uses `qc.initialize(state, range(n_anc))` to directly load the amplitude distribution — no recursive gate decomposition in `_build_V` itself. The SELECT operator directly calls `qc.append()` with `control_state` and reverses `ctrl_state` when `U.order == 'little'`. `_build_return_dict` always receives `success=True` (hardcoded in `run()`), so `status` is always `'ok'`; the actual LCU outcome quality is in `result['Success probability']`.

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
from unitarylab_algorithms import LCUAlgorithm
from unitarylab.core import Circuit

n_sys = 2
I_circ = Circuit(n_sys, name='II')    # Identity
X0_circ = Circuit(n_sys, name='XI')
X0_circ.x(0)

algo = LCUAlgorithm(text_mode='plain')
result = algo.run(
    alphas=[0.8, 0.2],
    unitaries=[I_circ, X0_circ],
    n_sys=n_sys,
    backend='torch'
)
print(f"Success probability: {result['Success probability']:.4f}")
print(f"Status: {result['status']}")
print(f"Result state: {result['Result state']}")
```

## Implementing Your Own Version

The following Python skeleton reconstructs the three structural components of LCU: PREPARE, SELECT, and UNPREPARE.

### Step 1: Amplitude-tree PREPARE circuit

```python
# Simplified reconstruction — mirrors LCUAlgorithm._build_V() and _state_preparation_tree()
import numpy as np
from unitarylab.core import Circuit

def build_prepare(alphas, n_anc: int, backend: str = 'torch') -> Circuit:
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
        qc = Circuit(n_anc)
        theta = 2 * np.arctan2(state[1], state[0])
        qc.ry(theta, 0)
        return qc
    # General case: delegate to the recursive tree builder in LCUAlgorithm
    raise NotImplementedError("Use LCUAlgorithm._build_V() for general m")
```

### Step 2: SELECT oracle (multiplexer)

```python
# Simplified reconstruction — mirrors LCUAlgorithm._build_select()

def build_select(unitaries, n_anc: int, n_sys: int, backend: str = 'torch') -> Circuit:
    """SELECT = sum_j |j><j|_anc ⊗ U_j: apply U_j conditional on ancilla=|j>."""
    m = len(unitaries)
    qc = Circuit(n_anc + n_sys)
    sys_qubits = list(range(n_anc, n_anc + n_sys))
    anc_qubits = list(range(n_anc))
    for j, U_j in enumerate(unitaries):
        control_seq = format(j, f'0{n_anc}b')   # e.g. '01' for j=1, n_anc=2
        qc.append(U_j, target=sys_qubits, control=anc_qubits, control_state=control_seq)
    return qc
```

### Step 3: Full LCU circuit assembly

```python
# Simplified reconstruction of the full PREPARE → SELECT → UNPREPARE structure

def lcu_circuit(alphas, unitaries, n_sys: int, initial_state=None, backend: str = 'torch'):
    """Full LCU circuit: PREPARE ⊗ I |psi> → SELECT → UNPREPARE ⊗ I."""
    m = len(alphas)
    n_anc = int(np.ceil(np.log2(max(m, 2))))
    qc = Circuit(n_anc + n_sys)

    anc_qubits = list(range(n_anc))
    sys_qubits = list(range(n_anc, n_anc + n_sys))

    # 1. Initial system state
    if initial_state is not None:
        qc.append(initial_state, sys_qubits)

    # 2. PREPARE
    V_circ = build_prepare(alphas, n_anc, backend)
    qc.append(V_circ, anc_qubits)

    # 3. SELECT
    sel_circ = build_select(unitaries, n_anc, n_sys, backend)
    qc.append(sel_circ, list(range(n_anc + n_sys)))

    # 4. UNPREPARE (V†)
    qc.append(V_circ.dagger(), anc_qubits)

    return qc
```

## Debugging Tips

1. **`len(alphas) != len(unitaries)`**: Raises `ValueError`. Always keep these lists the same length.
2. **`initial_state.get_num_qubits() != n_sys`**: Raises `ValueError`. The initial state circuit must match `n_sys` exactly.
3. **Very low `success_prob`**: This is expected when $\|M|\psi\rangle\| \ll s$. Use amplitude amplification to boost.
4. **Very low `'Success probability'` (near 0)**: `status` is always `'ok'` since `_build_return_dict` is always called with `success=True`. Check `result['Success probability']` directly. If near 0, the operator $M$ may annihilate $|\psi\rangle$, or the unitaries cancel.
5. **Memory limit**: Ancilla adds $\lceil\log_2 m\rceil$ qubits. For $m=8$ unitaries, this is only 3 ancilla qubits.
