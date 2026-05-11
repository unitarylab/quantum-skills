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
- Python: `numpy`, `Circuit`, `State`, project library `IQFT`.

## Using the Provided Implementation

```python
from unitarylab_algorithms import QPEAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Build a 1-qubit unitary with known phase phi = 1/4
# U = diag(1, e^{2pi*i*phi}) so with phi=0.25: U = diag(1, i) = S gate
U = Circuit(1, name="S_gate", backend='torch')
U.s(0)   # S gate has phase e^{i*pi/2} = e^{2pi*i*0.25}

# Prepare eigenstate |1> (eigenstate of S is |1> with eigenvalue i=e^{i*pi/2})
prepare_psi = Circuit(1, name="prep_1", backend='torch')
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
| `U` | `Circuit` | required | Unitary operator whose phase is to be estimated. |
| `d` | `int` | required | Number of phase-register qubits. Precision is $1/2^d$. |
| `prepare_target` | `Circuit` or `None` | `None` | Circuit preparing the eigenstate $\|\psi\rangle$. Defaults to $\|0\rangle$ if `None`. |
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
| `circuit` | `Circuit` | The full QPE circuit object. |
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
4. Appends `IQFT(d, backend=backend)` from `unitarylab.library` to the phase register.

**Helper Methods:**

- **`_update_last_result` / `_build_return`** — Store runtime fields and package result dict with ASCII panel.
- `State.probabilities_dict(phase_qubits, endian='little', threshold=...)` — called in Stage 4 to extract phase register marginals from the full statevector.

**Data flow:** `U` → `build_qpe_circuit()` → `Circuit` → `execute()` → `State.probabilities_dict()` → `phi_est` → `_build_return()`.

**Note:** `build_qpe_circuit()` can be called directly without `run()` to obtain the `Circuit` for embedding in a parent algorithm.

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
`QPEAlgorithm.build_qpe_circuit(U, d, prepare_target, backend)` returns a standalone `Circuit` for embedding QPE in other algorithms (e.g., HHL uses this directly).

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Phase register $|0\rangle^d$ → $H^{\otimes d}$ | `for q in phase_qubits: gs.h(q)` in `build_qpe_circuit()` |
| Eigenstate preparation $|\psi\rangle$ | `prepare_target` appended via `gs.append(prepare_target, target_qubits)` |
| Controlled $U^{2^k}$ | `cU = U.control(1, '1')` repeated `2^k` times per phase qubit `k` |
| Inverse QFT | `IQFT(d, backend)` from `unitarylab.library`, appended to `phase_qubits` |
| Phase readout $\phi = k_0/2^d$ | `int(best_bits_str, 2) / (2 ** d)` in Stage 4 |
| Probability of best phase | `best_prob = sorted_phases[0][1]` from `state_obj.probabilities_dict()` |
| Phase precision $\delta\phi = 1/2^d$ | Implicit: determined by number of bits `d` in phase register |
| Subroutine for HHL / QAE | `build_qpe_circuit()` returns a standalone `Circuit` embeddable externally |

**Notes on encapsulation:** The iQFT is sourced from `unitarylab.library.IQFT` rather than constructed inline, unlike the amplitude estimation implementation which builds it locally. The controlled-unitary power is realized by looping `2^k` repetitions of `cU`, not by general unitary exponential; this is correct but exponentially expensive in `d`. The `endian='little'` argument in `probabilities_dict()` ensures the least-significant bit is on the left, consistent with the register ordering.

## Mathematical Deep Dive
$$|0\rangle^d|\psi\rangle \rightarrow \frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1} e^{2\pi i\phi j}|j\rangle|\psi\rangle$$

The iQFT then maps this to:
$$\sum_{k'=0}^{2^d-1}\left(\frac{1}{2^d}\sum_{j=0}^{2^d-1} e^{2\pi i j(\phi - k'/2^d)}\right)|k'\rangle|\psi\rangle$$

When $\phi = k_0/2^d$ exactly, the sum equals $\delta_{k', k_0}$ and the measurement outcome is $k_0$ with probability 1.

**Precision:** To estimate $\phi$ to $n$ bits of accuracy with probability $\geq 1 - \epsilon$, use $d = n + \lceil\log_2(2 + 1/(2\epsilon))\rceil$ phase bits.

**Library backend:** This implementation uses `unitarylab.library.IQFT` for the inverse QFT.

## Hands-On Example (UnitaryLab)

```python
from unitarylab_algorithms import QPEAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Estimate phase of T gate (T|1> = e^{i*pi/4}|1>, phi = 1/8 = 0.125)
U = Circuit(1, name="T_gate", backend='torch')
U.t(0)

prepare_psi = Circuit(1, name="prep1", backend='torch')
prepare_psi.x(0)  # |1> is eigenstate of T with phase pi/4

algo = QPEAlgorithm()
result = algo.run(U=U, d=6, prepare_target=prepare_psi, backend='torch')

print(f"Estimated phi = {result['estimated_phase']:.6f}")  # Expected ≈ 0.125
print(f"Best probability = {result['confidence_probability']:.4f}")
```

## Reference Implementation (Qiskit)

In addition to the UnitaryLab implementation above, the same quantum phase estimation idea can also be expressed using Qiskit’s `PhaseEstimation` workflow. This section is provided only as a reference example for users who want to compare different software ecosystems. The main implementation path of this skill remains the UnitaryLab version described above.

### Example A: Minimal Qiskit Phase Estimation Run
```python
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.phase_estimators import PhaseEstimation

# Unitary: T gate
# On eigenstate |1>, T|1> = e^{i*pi/4}|1> = e^{2pi*i*(1/8)}|1>
unitary = QuantumCircuit(1)
unitary.t(0)

# Prepare eigenstate |1>
state_prep = QuantumCircuit(1)
state_prep.x(0)

# Use 3 evaluation qubits
qpe = PhaseEstimation(
    num_evaluation_qubits=3,
    sampler=StatevectorSampler()
)

result = qpe.estimate(
    unitary=unitary,
    state_preparation=state_prep
)

print("Most likely phase:", result.phase)
print("All phases:", result.phases)
```
### Example B: Qiskit with Phase Filtering
```python
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.phase_estimators import PhaseEstimation

# Unitary: S gate
# On eigenstate |1>, S|1> = i|1> = e^{i*pi/2}|1> = e^{2pi*i*0.25}|1>
unitary = QuantumCircuit(1)
unitary.s(0)

# Prepare eigenstate |1>
state_prep = QuantumCircuit(1)
state_prep.x(0)

qpe = PhaseEstimation(
    num_evaluation_qubits=4,
    sampler=StatevectorSampler()
)

result = qpe.estimate(
    unitary=unitary,
    state_preparation=state_prep
)

print("Most likely phase:", result.phase)
print("Raw phase distribution:", result.phases)
print("Filtered phases:", result.filter_phases(cutoff=0.01, as_float=True))
```
## Reference Implementation (PennyLane)
In addition to the UnitaryLab implementation above, PennyLane also provides a
template-based standard QPE workflow through `QuantumPhaseEstimation`. This
section is included only as a reference example. The primary implementation path
of this skill remains the UnitaryLab version described above.

```python
import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np

# 使用 @ 组合复合算子
unitary = qml.RX(np.pi / 2, wires=[0]) @ qml.CNOT(wires=[0, 1])
eigenvector = np.array([-1 / 2, -1 / 2, 1 / 2, 1 / 2])

n_estimation_wires = 5
estimation_wires = range(2, n_estimation_wires + 2)

dev = qml.device("default.qubit", wires=n_estimation_wires + 2)

@qml.qnode(dev)
def circuit():
    qml.StatePrep(eigenvector, wires=[0, 1])
    QuantumPhaseEstimation(
        unitary,  # 已含 target_wires，无需再传
        estimation_wires=estimation_wires,
    )
    return qml.probs(estimation_wires)

phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires
print(phase_estimated)
```

### Other Qiskit PE Variants (Reference)

Qiskit also includes several phase-estimation-related variants beyond the standard
`PhaseEstimation` workflow:

- **`IterativePhaseEstimation`**  
  An iterative phase-estimation method that estimates the phase bit-by-bit using a single
  auxiliary qubit rather than a full multi-qubit evaluation register. It performs multiple rounds
  of controlled-unitary application and feedback-angle updates, and is useful when qubit resources
  are limited.  
  Official reference:  
  `https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.IterativePhaseEstimation.html#qiskit_algorithms.IterativePhaseEstimation`

- **`HamiltonianPhaseEstimation`**  
  A Hamiltonian-oriented phase-estimation variant for estimating eigenvalues of Hermitian operators.
  Instead of taking a unitary circuit directly, it accepts a Pauli or `SparsePauliOp` Hamiltonian,
  computes or uses an eigenvalue bound, scales the operator to avoid phase wrapping, exponentiates it
  into a unitary evolution, and then runs phase estimation internally.  
  Official reference:  
  `https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.HamiltonianPhaseEstimation.html#qiskit_algorithms.HamiltonianPhaseEstimation`


### Other PennyLane PE Variant (Reference)

PennyLane also provides an **iterative quantum phase estimation** interface through `qml.iterative_qpe(base, aux_wire, iters)`. Unlike standard QPE, this variant uses **one auxiliary qubit** and returns a list of **mid-circuit measurement values with qubit reset**. It is a resource-efficient alternative when qubit count is limited. 
Official reference: 
`https://docs.pennylane.ai/en/stable/code/api/pennylane.iterative_qpe.html`


## Minimal Manual Implementation (UnitaryLab) 

```python
from unitarylab.core import Circuit
from unitarylab.library import IQFT

def qpe_circuit(U: Circuit, d: int, prepare_target=None, backend='torch'):
    n_target = U.get_num_qubits()
    gs = Circuit(d + n_target, backend=backend)
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
5. **IQFT sign convention**: The unitarylab's `IQFT` matches the standard convention. Do not swap QFT and IQFT orders.
