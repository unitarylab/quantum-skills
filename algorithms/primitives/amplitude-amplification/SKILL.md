---
name: amplitude-amplification
description: A quantum algorithm that generalizes Grover's search algorithm, allowing for the amplification of the probability of desired outcomes in a quantum state. It is used to find marked items in an unsorted database with quadratic speedup compared to classical algorithms. This skill provides a comprehensive guide to understanding, implementing (using UnitaryLab's quantum simulator), and utilizing amplitude amplification in quantum computing applications.
---

# Amplitude Amplification

## Purpose

Amplitude Amplification generalizes Grover's search algorithm. Given a unitary operator $U$ that prepares a state with a small initial success probability $p$, this algorithm iteratively applies a Grover-style operator to amplify the probability of the "good" (target) states, achieving probability close to 1 after $O(1/\sqrt{p})$ iterations.

Use this skill when you need to:
- Boost the probability of sampling a desired outcome from a quantum circuit.
- Apply the quantum quadratic speedup over classical random sampling.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

The algorithm works by:
1. Preparing the initial state $|\psi\rangle = U|0\rangle$ using a user-supplied `GateSequence` $U$.
2. Defining a "good" state condition: qubits listed in `good_zero_qubits` must all be in state $|0\rangle$.
3. Repeating Grover iterations: each iteration applies an **oracle** (phase-flipper for good states) followed by a **diffuser** (reflection about $|\psi\rangle$).
4. Measuring the amplified probability in the data register.

The number of iterations is chosen automatically from the initial probability $p$ via:
$$k \approx \frac{\pi}{4\theta} - \frac{1}{2}, \quad \sin\theta = \sqrt{p}$$
or set manually via `reps`.

## Prerequisites

- Familiarity with quantum gates: H, X, multi-controlled-X (MCX), CX.
- Understanding of quantum state vectors and measurement probabilities.
- Python: `numpy`, project core classes `GateSequence`, `Register`, `State`.

## Using the Provided Implementation

```python
from unitarylab.algorithms import AmplitudeAmplificationAlgorithm
from unitarylab.core import GateSequence

# Prepare state U such that some target qubits land in |0>
# Example: 2-qubit state preparation
U = GateSequence(2, name="PrepU", backend='torch')
U.ry(0.6, 0)   # Partially rotates qubit 0
U.cx(0, 1)     # Entangles

algo = AmplitudeAmplificationAlgorithm()
result = algo.run(
    U=U,
    good_zero_qubits=[0, 1],   # Good state: both qubits = |0>
    p=0.1,                      # Initial success probability estimate
    backend='torch'
)

print(result['amplified_prob'])   # Amplified probability of good state
print(result['circuit_path'])     # SVG circuit diagram path
print(result['status'])           # 'ok' if amplification succeeded
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `U` | `GateSequence` | required | State preparation circuit (excludes ancilla qubit). |
| `good_zero_qubits` | `List[int]` | required | Indices of data qubits that define the good state by being $\|0\rangle$. |
| `p` | `float` | required | Estimated initial success probability. Must satisfy $0 < p < 1$. |
| `reps` | `int` or `None` | `None` | Manual override for number of Grover iterations. If `None`, computed from `p`. |
| `backend` | `str` | `'torch'` | Simulation backend. Only `'torch'` is supported. |
| `algo_dir` | `str` or `None` | `None` | Directory to save circuit diagram. Auto-set if `None`. |

**Common misunderstandings:**
- `good_zero_qubits` refers to indices within the **data register** (not the ancilla). The ancilla qubit is automatically appended at index `n_data`.
- `p` is used only to compute `reps` when `reps=None`. If your estimate of `p` is wrong, the amplification may over- or under-shoot.
- Setting `reps` manually bypasses the `p`-based calculation entirely.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if amplified probability exceeds initial `p`; `'failed'` otherwise. |
| `amplified_prob` | `float` | Measured probability of the good state after amplification. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `message` | `str` | Human-readable summary. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`AmplitudeAmplificationAlgorithm` in `algorithm.py` decomposes the full algorithm into five ordered stages inside `run()`, supported by four dedicated helper methods.

**`run(U, good_zero_qubits, p, reps, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Resolution | `n_data = U.get_num_qubits()`; `ancilla = n_data`; calls `_get_optimal_iterations(p)` if `reps is None` | Converts user-supplied probability into Grover iteration count $k$ |
| 2 — Circuit Construction | Creates `GateSequence(n_data+1)`, appends `U`, then loops `reps` times appending oracle and diffuser | Builds the full amplitude amplification circuit layer by layer |
| 3 — Simulation Execution | `gs.execute()` → `State(re_state)` → `state_obj.calculate_state(data_qubits)` | Runs the statevector simulation; marginalizes ancilla to get data-register probabilities |
| 4 — Post-Processing | Iterates `state_basis_dict`; sums probability for states where all `good_zero_qubits == '0'` | Classically extracts `target_prob`; sets `is_success = (target_prob > p)` |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_get_optimal_iterations(p)`** — Computes `k = max(0, round(π/(4·arcsin(√p)) − 0.5))`. Uses `round()` instead of `floor()` to avoid floating-point precision errors near integer boundaries.
- **`_build_oracle(gs, zero_qubits, ancilla)`** — Implements the phase-kickback oracle. Prepares ancilla in $|-\rangle$ via `_prepare_kickback_ancilla_minus`, applies X-flips on all `zero_qubits` to convert $|0\rangle$-control to $|1\rangle$-control, applies `gs.cx`/`gs.mcx` targeting the ancilla, then reverts X-flips and unprepares the ancilla. The net data-register effect is $|x\rangle \mapsto (-1)^{f(x)}|x\rangle$.
- **`_build_diffuser(gs, U, data_qubits, ancilla)`** — Implements the Grover diffuser as `U†` → `_build_oracle(all data_qubits)` → `U`. This reflects the state about $U|0\rangle^n$. It reuses `_build_oracle` on the full data register, which is the $2|0^n\rangle\langle 0^n|-I$ reflection.
- **`_update_last_result` / `_build_return`** — Store all runtime data into `self.last_result` and package the result dict (including the ASCII panel rendered by `format_result_ascii()`).

**Data flow:** `U` (user input) → `GateSequence` circuit → `execute()` → `State.calculate_state()` → `target_prob` → `_build_return()` → result dict.

## Understanding the Key Quantum Components

### 1. State Preparation ($U$)
The user-supplied `GateSequence` $U$ acts on the data register to prepare:
$$U|0\rangle^n = \sqrt{p}|\text{good}\rangle + \sqrt{1-p}|\text{bad}\rangle$$
One ancilla qubit is appended automatically at index `n_data`.

### 2. Oracle (Phase Kickback)
The oracle marks good states by flipping their phase via a kickback ancilla in the $|-\rangle$ state:
- Ancilla is prepared as $|-\rangle = HX|0\rangle$.
- A multi-controlled-X (MCX) gate targets the ancilla, controlled on all `good_zero_qubits` being $|0\rangle$.
- Pauli-X gates are applied before/after to convert the $|0\rangle$-control to $|1\rangle$-control.
- The net effect on the data register: $|x\rangle \rightarrow (-1)^{f(x)}|x\rangle$ where $f(x)=1$ iff all good qubits are $|0\rangle$.

### 3. Diffuser (Reflection about $|\psi\rangle$)
$$D = U(2|0^n\rangle\langle 0^n| - I)U^\dagger$$
Implemented as:
1. Apply $U^\dagger$ (inverse of state preparation).
2. Apply the all-zeros phase oracle.
3. Apply $U$ again.
This reflects the state about the prepared state $|\psi\rangle = U|0\rangle$.

### 4. Grover Iteration Loop
Each iteration applies oracle → diffuser. After $k$ iterations starting from angle $\theta = \arcsin(\sqrt{p})$:
$$G^k|\psi\rangle = \sin((2k+1)\theta)|\text{good}\rangle + \cos((2k+1)\theta)|\text{bad}\rangle$$
The good-state amplitude grows, reaching maximum near $(2k+1)\theta \approx \pi/2$.

### 5. Measurement
The data register probabilities are extracted via `State.calculate_state(data_qubits)`, marginalizing over the ancilla.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Initial state $|\psi\rangle = U|0\rangle$ | `gs.append(U, data_qubits)` in Stage 2 of `run()` |
| Good state: $f(x)=1$ iff all marked qubits are $|0\rangle$ | `good_zero_qubits` parameter; tested via `basis_str[q] == '0'` in Stage 4 |
| Oracle $O_f$: phase flip on good states | `_build_oracle(gs, zero_qubits, ancilla)` — MCX + kickback ancilla |
| Diffuser $D = 2|\psi\rangle\langle\psi| - I$ | `_build_diffuser(gs, U, data_qubits, ancilla)` — `U†` → all-zeros oracle → `U` |
| Angle $\theta = \arcsin(\sqrt{p})$ | `theta = math.asin(math.sqrt(p))` inside `_get_optimal_iterations()` |
| Optimal iteration count $k = \lfloor\pi/(4\theta) - 1/2\rceil$ | `int(round(math.pi / (4.0 * theta) - 0.5))` in `_get_optimal_iterations()` |
| Ancilla qubit for phase kickback | `ancilla = n_data` — automatically the last qubit in the `n_data+1` register |
| Final probability of good state | `target_prob` accumulated in Stage 4; returned as `amplified_prob` |
| Amplified probability $\sin^2((2k+1)\theta)$ | Not computed symbolically; arises from simulation measurement in Stage 3–4 |

**Notes on encapsulation:** The README describes the ancilla-based kickback oracle. In the code, the oracle is fully encapsulated in `_build_oracle`, and the diffuser reuses it via `_build_diffuser`. The $2|0^n\rangle\langle 0^n|-I$ reflection is not built with explicit Hadamard or X gates on all qubits — instead, it is also an oracle call with `zero_qubits = list(data_qubits)`. The global phase correction seen in some QAE variants is **not** applied here; this implementation targets probability amplification, not eigenphase use.

## Mathematical Deep Dive

Define the good and bad subspaces:
$$|w\rangle = \frac{1}{\sqrt{M}}\sum_{x \in G}|x\rangle, \quad |r\rangle = \frac{1}{\sqrt{N-M}}\sum_{x \notin G}|x\rangle$$

Initial state angle: $\sin\theta = \sqrt{p} = \sqrt{M/N}$

After $k$ Grover iterations:
$$G^k|\psi\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|r\rangle$$

Optimal iteration count for maximum amplitude:
$$k = \left\lfloor \frac{\pi}{4\theta} - \frac{1}{2} \right\rceil \approx \frac{\pi}{4}\frac{1}{\sqrt{p}} \quad (p \ll 1)$$

Query complexity: $O(1/\sqrt{p})$, quadratic speedup over classical $O(1/p)$.

## Hands-On Example (UnitaryLab)
```python
from unitarylab.algorithms import AmplitudeAmplificationAlgorithm
from unitarylab.core import GateSequence
import numpy as np

# Build a 3-qubit state with small success probability
# Target: qubits 0 and 1 in |0> simultaneously
U = GateSequence(3, name="MyPrep", backend='torch')
U.ry(0.3, 0)   # rotates qubit 0 slightly away from |0>
U.ry(0.3, 1)
U.ry(1.5, 2)   # qubit 2 not in target condition

# Initial success probability ≈ cos^2(0.15) * cos^2(0.15) ≈ 0.956 * 0.956 ≈ 0.91
# Use a smaller angle for a more interesting demo: p ~ 0.05
U2 = GateSequence(3, name="LowProb", backend='torch')
U2.ry(1.4, 0)
U2.ry(1.4, 1)

algo = AmplitudeAmplificationAlgorithm()
result = algo.run(
    U=U2,
    good_zero_qubits=[0, 1],  # Both qubit 0 and qubit 1 must be |0>
    p=0.05,                    # Estimated initial probability
    backend='torch',
    algo_dir='./results/amp_amp'
)

print(f"Amplified probability: {result['amplified_prob']:.4f}")
print(f"Status: {result['status']}")
print(result['plot'])
```
## Reference Implementation (Qiskit)

In addition to the UnitaryLab implementation above, the same amplitude amplification idea can also be expressed using Qiskit’s AmplificationProblem and Grover workflow. This section is provided only as a reference example for users who want to compare different software ecosystems. The main implementation path of this skill remains the UnitaryLab version described above.
### Example A: Minimal Qiskit Grover Run
```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.amplitude_amplifiers import AmplificationProblem, Grover

# Oracle marking |11> as the good state
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

problem = AmplificationProblem(
    oracle=oracle,
    is_good_state=["11"],
)

grover = Grover(
    iterations=1,
    sampler=StatevectorSampler()
)

result = grover.amplify(problem)

print("Top measurement:", result.top_measurement)
print("Oracle evaluation:", result.oracle_evaluation)
print("Max probability:", result.max_probability)
```
### Example B: Qiskit with Post-Processing
```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.amplitude_amplifiers import AmplificationProblem, Grover

# Oracle marking |111>
oracle = QuantumCircuit(3)
oracle.ccz(0, 1, 2)

problem = AmplificationProblem(
    oracle=oracle,
    objective_qubits=[0, 1, 2],
    is_good_state=lambda bitstr: bitstr == "111",
    post_processing=lambda bitstr: int(bitstr, 2),
)

grover = Grover(
    growth_rate=1.2,
    sampler=StatevectorSampler()
)

result = grover.amplify(problem)

print("Top measurement:", result.top_measurement)
print("Decoded assignment:", result.assignment)
print("Tried powers:", result.iterations)
```

## Reference Implementation (PennyLane)

PennyLane also provides an amplitude amplification template through
`qml.AmplitudeAmplification`. This section is included only as a reference
implementation for comparison with other quantum software frameworks. The main
implementation path of this skill remains the UnitaryLab version described above.

### Example A: Minimal PennyLane Amplitude Amplification Run (Grover Search)

```python
import pennylane as qml
import numpy as np

# Build the state-preparation operator U (uniform superposition over 3 qubits)
@qml.prod
def generator(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)

U = generator(wires=range(3))

# Oracle that marks |2⟩ (binary 010) with a phase flip
O = qml.FlipSign(2, wires=range(3))

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    generator(wires=range(3))                  # Prepare |Ψ⟩
    qml.AmplitudeAmplification(U, O, iters=3)  # Amplify |2⟩
    return qml.probs(wires=range(3))

probs = circuit()

print(np.round(probs, 3))
# Expected: dominant probability at index 2
```


## Minimal Manual Implementation (UnitaryLab) 

```python
from unitarylab.core import GateSequence
import math

def amplitude_amplification(U: GateSequence, good_zero_qubits, p: float, backend='torch'):
    n_data = U.get_num_qubits()
    ancilla = n_data
    reps = max(1, round(math.pi / (4 * math.asin(math.sqrt(p))) - 0.5))

    gs = GateSequence(n_data + 1, backend=backend)
    gs.append(U, list(range(n_data)))

    for _ in range(reps):
        # --- Oracle ---
        gs.x(ancilla); gs.h(ancilla)
        for q in good_zero_qubits: gs.x(q)
        if len(good_zero_qubits) == 1:
            gs.cx(good_zero_qubits[0], ancilla)
        else:
            gs.mcx(good_zero_qubits, ancilla)
        for q in good_zero_qubits: gs.x(q)
        gs.h(ancilla); gs.x(ancilla)

        # --- Diffuser ---
        gs.append(U.dagger(), list(range(n_data)))
        # (repeat oracle on all data qubits)
        gs.x(ancilla); gs.h(ancilla)
        all_data = list(range(n_data))
        for q in all_data: gs.x(q)
        gs.mcx(all_data, ancilla)
        for q in all_data: gs.x(q)
        gs.h(ancilla); gs.x(ancilla)
        gs.append(U, list(range(n_data)))

    return gs
```

## Debugging Tips

1. **`p` estimate is inaccurate**: If you misestimate $p$, `reps` may be wrong. Use `reps` manually if you know the true probability.
2. **No amplification (status='failed')**: Double-check that `good_zero_qubits` indices are correct and that the good state has nonzero initial probability.
3. **Good state never in $|0\rangle$**: If the target condition is "qubit in $|1\rangle$", apply an `X` gate to those qubits in $U$ before running, or add them to `good_zero_qubits` after an X pre-rotation.
4. **Circuit grows too large**: Each Grover iteration adds `O(n)` gates. For large `reps`, simulation may be slow.
5. **Ancilla index conflict**: The ancilla is always placed at index `n_data` (one beyond the data register). Do not include it in `good_zero_qubits`.
