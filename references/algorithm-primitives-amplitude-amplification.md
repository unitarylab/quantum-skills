# Amplitude Amplification

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

Amplitude Amplification generalizes Grover's search algorithm. Given a unitary operator $U$ that prepares a state with a small initial success probability $p$, this algorithm iteratively applies a Grover-style operator to amplify the probability of the "good" (target) states, achieving probability close to 1 after $O(1/\sqrt{p})$ iterations.

The documented algorithm-library interface accepts `U`, `good_zero_qubits`, `p`, and optional `reps`. Its built-in oracle marks exactly the data states whose listed qubits are all zero; it does not accept an arbitrary oracle or predicate.

For the standalone Grover search workflow, read [algorithm-primitives-grover.md](algorithm-primitives-grover.md).

Source pointers: [algorithm-library usage example](../scripts/amplitude_amplification_algorithm.py) and [independent teaching implementation](../scripts/amplitude_amplification_implementation.py). The latter exposes its own solver and does not establish the installed algorithm library's API contract.

Algorithm-library source: `AmplitudeAmplificationAlgorithm` (`unitarylab_algorithms/fundamental_algorithm/amplitude_amplification/algorithm.py`).

## Overview

The algorithm works by:
1. Preparing the initial state $|\psi\rangle = U|0\rangle$ using a user-supplied `Circuit` $U$.
2. Defining a "good" state condition: qubits listed in `good_zero_qubits` must all be in state $|0\rangle$.
3. Repeating Grover iterations: each iteration applies an **oracle** (phase-flipper for good states) followed by a **diffuser** (reflection about $|\psi\rangle$).
4. Measuring the amplified probability in the data register.

The number of iterations is chosen automatically from the initial probability $p$ via:
$$k \approx \frac{\pi}{4\theta} - \frac{1}{2}, \quad \sin\theta = \sqrt{p}$$
or set manually via `reps`.

## Prerequisites

- Familiarity with quantum gates: H, X, multi-controlled-X (MCX), CX.
- Understanding of quantum state vectors and measurement probabilities.
- Python: `numpy`, project core classes `Circuit`, `Register`.

## Reference Implementation Example

```python
from unitarylab_algorithms import AmplitudeAmplificationAlgorithm
from unitarylab.core import Circuit
import math

# Prepare |00> with probability p = 0.1.
p = 0.1
U = Circuit(2, name="PrepU")
U.ry(2 * math.acos(math.sqrt(p)), 0)

algo = AmplitudeAmplificationAlgorithm()
result = algo.run(
    U=U,
    good_zero_qubits=[0, 1],   # Good state: both qubits = |0>
    p=p,                        # Matches this U and good-state condition
    backend='torch'
)

print(result['Amplified Target Probability'])  # Amplified probability of good state
print(result['circuit_path'])                  # SVG circuit diagram path
print(result['status'])                        # 'ok' if amplification succeeded
print(result['plot'])                          # List of saved file dicts [{"format": ..., "filename": ...}]
```

## Core Parameters Explained

**`__init__(text_mode, algo_dir)` — Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Output text formatting mode. |
| `algo_dir` | `str` or `None` | `None` | Directory to save results; auto-derived from cwd if `None`. |

**`run(U, good_zero_qubits, p, reps, backend, device, dtype)` — Run parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `U` | `Circuit` | required | State preparation circuit (excludes ancilla qubit). |
| `good_zero_qubits` | `List[int]` | required | Indices of data qubits that define the good state by being $\|0\rangle$. |
| `p` | `float` | required | Estimated initial success probability; use $0<p<1$. The explicit interval check occurs only when `reps=None`; manual `reps` still evaluates `asin(sqrt(p))` and compares the result with `p`. |
| `reps` | `int` or `None` | `None` | Manual override for number of Grover iterations. If `None`, computed from `p`. |
| `backend` | `str` | `'torch'` | Forwarded to `Circuit.execute()`; support is defined by the installed SDK. |
| `device` | `str` | `'cpu'` | Device for simulation (e.g. `'cpu'` or `'cuda'`). |
| `dtype` | dtype | `np.complex128` | Numerical dtype for simulation. |

**Common misunderstandings:**
- `good_zero_qubits` refers to indices within the **data register** (not the ancilla). The ancilla qubit is automatically appended at index `n_data`.
- With `reps=None`, `p` determines the iteration count. A wrong estimate can cause amplification to over- or under-shoot.
- Setting `reps` manually bypasses only iteration-count selection. `p` is still used in angle logging and the `target_prob > p` status comparison; pass a valid probability matching the prepared state.
- Use a nonempty, unique list of valid data-qubit indices. The source has no explicit list validation; its empty-control branch applies Z to the kickback ancilla and does not implement the usual all-states-good phase oracle.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if amplified probability exceeds initial `p`; `'failed'` otherwise. |
| `Amplified Target Probability` | `float` | Sum over retained data-register marginals satisfying the source's string-index check; marginals below `1e-5` are omitted. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `plot` | `list` | List of saved file dicts, each with `'format'` and `'filename'` keys. |
| `circuit` | `Circuit` | The `Circuit` object constructed for the algorithm. |
| `Initial Success Probability` | `float` | The `p` value passed to `run()`. |
| `Repetitions` | `int` | Number of Grover iterations actually used. |
| `Computation Time (s)` | `float` | Quantum simulation wall-clock time in seconds. |
| `Data register size` | `int` | Number of data qubits (derived from `U.get_num_qubits()`). |

## Implementation Architecture

`AmplitudeAmplificationAlgorithm` in `algorithm.py` decomposes the full algorithm into five ordered stages inside `run()`, supported by four dedicated helper methods.

**`run(U, good_zero_qubits, p, reps, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Resolution | `n_data = U.get_num_qubits()`; `ancilla = n_data`; calls `_get_optimal_iterations(p)` if `reps is None` | Converts user-supplied probability into Grover iteration count $k$ |
| 2 — Circuit Construction | Creates `Circuit(n_data+1, name='Amplitude_Amplification')`, appends `U` to `data_qubits`, then loops `reps` times calling `_build_oracle(qc, good_zero_qubits, ancilla)` and `_build_diffuser(qc, U, data_qubits, ancilla)` | Builds the full amplitude amplification circuit layer by layer |
| 3 — Simulation Execution | `qc.execute(backend=backend, device=device, dtype=dtype)` → `result.marginal_probabilities(qubits=data_qubits, threshold=1e-5)` | Runs the statevector simulation; marginalizes ancilla to get data-register probabilities |
| 4 — Post-Processing | Iterates `state_basis_dict`; sums probability for states where all `good_zero_qubits == '0'`; sets `is_success = (target_prob > p)` | Classically extracts `target_prob` (`result['Amplified Target Probability']`) |
| 5 — Export | `self.save_circuit(qc)` / `self.save_txt()` → `_build_return_dict(is_success, circuit_path, filename, qc)` | Saves SVG circuit + text results; packages return dict |

**Helper Methods:**

- **`_get_optimal_iterations(p)`** — Computes `k = max(0, round(π/(4·arcsin(√p)) − 0.5))`, choosing a nearby integer iteration count. Rounding does not eliminate floating-point error.
- **`_build_oracle(qc, zero_qubits, ancilla)`** — Implements the phase-kickback oracle. Prepares ancilla in $|-\rangle$ via `_prepare_kickback_ancilla_minus`, applies X-flips on all `zero_qubits` to convert $|0\rangle$-control to $|1\rangle$-control, applies `qc.cx`/`qc.mcx` targeting the ancilla, then reverts X-flips and unprepares the ancilla. The net data-register effect is $|x\rangle \mapsto (-1)^{f(x)}|x\rangle$.
- **`_build_diffuser(qc, U, data_qubits, ancilla)`** — Applies `U†` → `_build_oracle(all data_qubits)` → `U`. The zero-state phase flip is $I-2|0^n\rangle\langle0^n|$, so the implemented diffuser is the negative of the conventional $2|\psi\rangle\langle\psi|-I$. This global sign does not affect amplification probabilities.
- **`_build_return_dict(success, circuit_path, filepath, circuit)`** — Packages the final return dict. Sets `status` to `'ok'` or `'failed'`, wraps file paths as `[{"format": ext, "filename": path}]`, and merges `self.output` fields (`Amplified Target Probability`, `Initial Success Probability`, `Repetitions`, `Computation Time (s)`, `Data register size`) into the result.

**Data flow:** `U` → `Circuit(n_data+1)` → `qc.execute(backend=backend, device=device, dtype=dtype)` → `result.marginal_probabilities(qubits=data_qubits, threshold=1e-5)` → `target_prob` (`result['Amplified Target Probability']`) → `_build_return_dict(is_success, circuit_path, filename, qc)` → result dict.

## Understanding the Key Quantum Components

### 1. State Preparation ($U$)
The user-supplied `Circuit` $U$ acts on the data register to prepare:
$$U|0\rangle^n = \sqrt{p}|\text{good}\rangle + \sqrt{1-p}|\text{bad}\rangle$$
One ancilla qubit is appended automatically at index `n_data`.

### 2. Oracle (Phase Kickback)
The oracle marks good states by flipping their phase via a kickback ancilla in the $|-\rangle$ state:
- Ancilla is prepared as $|-\rangle = HX|0\rangle$.
- A multi-controlled-X (MCX) gate targets the ancilla, controlled on all `good_zero_qubits` being $|0\rangle$.
- Pauli-X gates are applied before/after to convert the $|0\rangle$-control to $|1\rangle$-control.
- The net effect on the data register: $|x\rangle \rightarrow (-1)^{f(x)}|x\rangle$ where $f(x)=1$ iff all good qubits are $|0\rangle$.

### 3. Diffuser (Reflection about $|\psi\rangle$)
The conventional diffuser is $D = U(2|0^n\rangle\langle 0^n| - I)U^\dagger$; the circuit below implements $-D$.
Implemented as:
1. Apply $U^\dagger$ (inverse of state preparation).
2. Apply the all-zeros phase oracle.
3. Apply $U$ again.
Its probabilities agree with reflection about $|\psi\rangle = U|0\rangle$; controlled use would make the sign relevant.

### 4. Grover Iteration Loop
Each iteration applies oracle → diffuser. For the conventional operator $G=DO_f$, starting from $\theta=\arcsin(\sqrt p)$:
$$G^k|\psi\rangle = \sin((2k+1)\theta)|\text{good}\rangle + \cos((2k+1)\theta)|\text{bad}\rangle$$
The implemented diffuser contributes an additional global factor $(-1)^k$ to this state expression, leaving probabilities unchanged. The first probability maximum lies near $(2k+1)\theta=\pi/2$; integer iterations need not reach probability one.

### 5. Measurement
The data register probabilities are extracted via `result.marginal_probabilities(qubits=data_qubits, threshold=1e-5)`, marginalizing over the ancilla.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Initial state $|\psi\rangle = U|0\rangle$ | `qc.append(U, data_qubits)` in Stage 2 of `run()` |
| Good state: $f(x)=1$ iff all marked qubits are $|0\rangle$ | `good_zero_qubits` parameter; tested via `basis_str[q] == '0'` in Stage 4 |
| Oracle $O_f$: phase flip on good states | `_build_oracle(qc, zero_qubits, ancilla)` — MCX + kickback ancilla |
| Diffuser $D = 2|\psi\rangle\langle\psi| - I$ | `_build_diffuser(qc, U, data_qubits, ancilla)` — `U†` → all-zeros oracle → `U` |
| Angle $\theta = \arcsin(\sqrt{p})$ | `theta = math.asin(math.sqrt(p))` inside `_get_optimal_iterations()` |
| Optimal iteration count $k = \lfloor\pi/(4\theta) - 1/2\rceil$ | `int(round(math.pi / (4.0 * theta) - 0.5))` in `_get_optimal_iterations()` |
| Ancilla qubit for phase kickback | `ancilla = n_data` — automatically the last qubit in the `n_data+1` register |
| Final probability of good state | `target_prob` accumulated in Stage 4; returned as `result['Amplified Target Probability']` |
| Amplified probability $\sin^2((2k+1)\theta)$ | Not computed symbolically; arises from simulation measurement in Stage 3–4 |

**Known result-indexing mismatch:** The oracle addresses physical qubit `q`, but result accumulation tests `basis_str[q]`. SDK displayed marginal strings place qubit 0 on the right, so this test instead reads physical qubit `n_data-1-q`. For an asymmetric subset of good qubits, the reported sum and status can describe a different subspace from the amplified one. The all-data-qubits examples below avoid that mismatch. See the SDK's [bit-string convention](https://docs.unitarylab.com/en/docs/unitarylab-simulator-user-manual/circuit-workflow).

## Mathematical Deep Dive

For a general prepared state, let $\Pi_G$ project onto the good subspace and let $p=\|\Pi_G|\psi\rangle\|^2$. Its normalized good and bad components are $|w\rangle=\Pi_G|\psi\rangle/\sqrt p$ and $|r\rangle=(I-\Pi_G)|\psi\rangle/\sqrt{1-p}$. Only for uniform preparation is $p=M/N$.

Initial state angle: $\sin\theta = \sqrt{p}$.

After $k$ Grover iterations:
$$G^k|\psi\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|r\rangle$$

Optimal iteration count for maximum amplitude:
$$k = \left\lfloor \frac{\pi}{4\theta} - \frac{1}{2} \right\rceil \approx \frac{\pi}{4}\frac{1}{\sqrt{p}} \quad (p \ll 1)$$

Query complexity: $O(1/\sqrt{p})$, quadratic speedup over classical $O(1/p)$.

## Hands-On Example

```python
from unitarylab_algorithms import AmplitudeAmplificationAlgorithm
from unitarylab.core import Circuit
import numpy as np

# Each rotated qubit has probability cos²(theta) of being |0>.
theta = 1.2
U2 = Circuit(3, name="LowProb")
U2.ry(2 * theta, 0)
U2.ry(2 * theta, 1)
p = float(np.cos(theta) ** 4)

algo = AmplitudeAmplificationAlgorithm(algo_dir='./results/amp_amp')
result = algo.run(
    U=U2,
    good_zero_qubits=[0, 1, 2],  # All three must be zero; qubit 2 stays zero in U2
    p=p,                       # Exact initial probability for this preparation
    backend='torch'
)

print(f"Amplified probability: {result['Amplified Target Probability']:.4f}")
print(f"Status: {result['status']}")
print(result['plot'])   # [{'format': 'txt', 'filename': '...'}]
```

## Minimal Manual Implementation

```python
from unitarylab.core import Circuit
import math

def amplitude_amplification(U: Circuit, good_zero_qubits, p: float, backend='torch'):
    n_data = U.get_num_qubits()
    ancilla = n_data
    reps = max(0, round(math.pi / (4 * math.asin(math.sqrt(p))) - 0.5))

    qc = Circuit(n_data + 1)
    qc.append(U, list(range(n_data)))

    for _ in range(reps):
        # --- Oracle ---
        qc.x(ancilla); qc.h(ancilla)
        for q in good_zero_qubits: qc.x(q)
        if len(good_zero_qubits) == 1:
            qc.cx(good_zero_qubits[0], ancilla)
        else:
            qc.mcx(good_zero_qubits, ancilla)
        for q in good_zero_qubits: qc.x(q)
        qc.h(ancilla); qc.x(ancilla)

        # --- Diffuser ---
        qc.append(U.dagger(), list(range(n_data)))
        # (repeat oracle on all data qubits)
        qc.x(ancilla); qc.h(ancilla)
        all_data = list(range(n_data))
        for q in all_data: qc.x(q)
        qc.mcx(all_data, ancilla)
        for q in all_data: qc.x(q)
        qc.h(ancilla); qc.x(ancilla)
        qc.append(U, list(range(n_data)))

    return qc
```

## Debugging Tips

1. **`p` estimate is inaccurate**: If you misestimate $p$, `reps` may be wrong. Use `reps` manually if you know the true probability.
2. **No amplification (status='failed')**: Double-check that `good_zero_qubits` indices are correct and that the good state has nonzero initial probability.
3. **Target condition uses one bits**: The interface only marks zero conditions. Appending X gates to the preparation changes the data representation; any such task-specific encoding must also be reflected in outcome interpretation. An arbitrary predicate requires a different implementation.
4. **Circuit grows too large**: Each iteration includes `U`, `U†`, and two controlled phase-oracle constructions. Gate cost depends on preparation depth and MCX decomposition as well as data size and `reps`.
5. **Ancilla index conflict**: The ancilla is always placed at index `n_data` (one beyond the data register). Do not include it in `good_zero_qubits`.

## Reference Implementation

Qiskit's [AmplificationProblem](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.AmplificationProblem.html) and [Grover](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.Grover.html) provide a separate comparison interface accepting an explicit phase oracle and a V2 sampler. This capability does not extend `AmplitudeAmplificationAlgorithm`'s zero-qubit interface.
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

PennyLane's [AmplitudeAmplification](https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeAmplification.html) template takes preparation and phase-oracle operators as separate inputs. It is a comparison interface with its own wire and optional fixed-point settings.

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
