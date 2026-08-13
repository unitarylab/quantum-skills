---
name: grover
description: "Grover's search algorithm for finding a marked computational-basis state in an unstructured search space with quadratic query speedup. This is the uniform-superposition, fixed-target special case of amplitude amplification. This skill provides a comprehensive guide to understanding, implementing, and utilizing Grover's algorithm in quantum computing applications."
---

# Grover Search

> **Implementation scope:** This skill describes the current `GroverAlgorithm` implementation: a uniform-superposition Grover search over $2^n$ computational-basis states with exactly one marked target bit string. It is **not** a general amplitude-amplification interface and does not support arbitrary state preparation, multiple marked states, or custom good-state predicates.

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Grover's Search.

Grover's search finds a marked item in an unstructured search space. In this implementation, the search space is the $2^n$ computational-basis states of an $n$-qubit data register, and the marked item is a single target bit string.

Grover search is the uniform-superposition, fixed-target special case of amplitude amplification:

- state preparation is $U = H^{\otimes n}$,
- the initial success probability is $p = 1/2^n$ for one marked state,
- the oracle marks one computational-basis target,
- the diffuser reflects about the uniform superposition.

Use this skill when you need to:
- Search for a target bit string such as `'101'`.
- Demonstrate Grover's quadratic speedup for unstructured search.
- Use the standalone `GroverAlgorithm` class rather than the general amplitude-amplification interface.

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Overview

The algorithm proceeds as follows:

1. Prepare the uniform superposition $|s\rangle = H^{\otimes n}|0^n\rangle$.
2. Mark the target state with a phase oracle.
3. Apply the diffuser, a reflection about $|s\rangle$.
4. Repeat oracle plus diffuser for the automatically computed near-optimal iteration count. For one marked state with initial success probability $p = 1/2^n$, the ideal count is approximately $\frac{\pi}{4\arcsin(\sqrt{p})} - \frac{1}{2}$, which reduces to $\frac{\pi}{4}\sqrt{2^n}$ for large $N$.
5. Simulate the circuit and return the most likely basis state.

For one target state, the initial success probability is
$$
p = \frac{1}{2^n}.
$$
Writing $\sin\theta = \sqrt{p}$, after $k$ Grover iterations the target-state probability is ideally
$$
\sin^2((2k+1)\theta).
$$

## Prerequisites

- Familiarity with quantum gates: H, X, multi-controlled-X (MCX).
- Understanding of quantum state vectors and measurement probabilities.
- Basic knowledge of Grover's algorithm: oracle, diffuser, and the $\pi/4\sqrt{N}$ iteration count.
- Python: `numpy`, project core classes `Circuit`.

## Reference Implementation Example

```python
from unitarylab_algorithms import GroverAlgorithm

algo = GroverAlgorithm(text_mode="plain")
result = algo.run(
    n=3,
    target="101",
    backend="torch",
    device="cpu",
)

print(result["status"])
print(result["Result"])
print(result["Amplified target-state probability"])
print(result["circuit_path"])
```

## Core Parameters Explained

**`__init__(text_mode, algo_dir)` — Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Output text formatting mode. |
| `algo_dir` | `str` or `None` | `None` | Directory to save results; auto-derived from cwd if `None`. |

**`run(n, target, backend, device, dtype)` — Run parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | required | Number of data-register qubits. The search space has size $N=2^n$. |
| `target` | `str` | required | Target computational-basis state as a binary string containing only `'0'` and `'1'`. Its length must equal `n`. For example, `n=3`, `target="101"`. Do **not** pass an integer such as `target=5`. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device (e.g. `'cpu'`, `'cuda'`). |
| `dtype` | dtype | `np.complex128` | Numeric dtype for simulation. |

> **Input expectation:** The implementation expects `target` to already be a valid binary string of length `n`. When generating code, always pass `target` as a string, not as an integer. Passing an integer or a string of wrong length may produce subtle errors that `run()` does not explicitly guard against.

**Common misunderstandings:**
- `target` must be a binary string of length `n` containing only `'0'` and `'1'`. For example, `n=3`, `target="101"`. Do not pass an integer such as `target=5`.
- The implementation is for a **single** marked target state. Multiple marked states are better described through the general amplitude-amplification framework.
- Too many Grover iterations can rotate past the target state and reduce success probability.
- The returned `Result` is the most likely state from statevector probabilities, not a finite-shot measurement sample.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | Public return status from `BaseAlgorithm`: `'ok'` when `run()` completes normally; execution exceptions propagate instead of producing a result dictionary. Check `Result == target` to determine whether the intended target was found. |
| `Amplified target-state probability` | `float` | Probability of the most likely state (`argmax` over full statevector) after Grover iterations. In normal successful runs this corresponds to the target state's dominant probability, but it is the probability of whatever state was selected by `argmax`, not necessarily the user-intended target. Check `Result == target` to confirm. |
| `Result` | `str` | Most likely computational-basis state inferred from the simulated statevector probabilities via `argmax`. It is **not** a finite-shot measurement sample. |
| `circuit_path` | `str` | Path to the saved circuit diagram. |
| `plot` | `list` | Saved output file metadata. |
| `circuit` | `Circuit` | The assembled Grover circuit. |

**Status note:** Internally, `run()` records `self.status` as `'success'` or `'partial_success'` according to whether the most-likely state matches `target`. The public return dictionary is built through `BaseAlgorithm._build_return_dict(True, ...)`, so `result["status"]` is `'ok'` for any normal completion. For generated examples or tests, compare `result["Result"]` with the input `target`; do not use the public `status` field alone to infer target recovery.

## Implementation Architecture

`GroverAlgorithm.run(n, target, backend, device, dtype)` builds a standalone Grover circuit in five stages.

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 | Builds `U = H^n`, sets `p = 1 / 2^n`, computes `reps = _get_optimal_iterations(p)` | Uniform-state preparation and iteration count |
| 2 | Creates `Circuit(n + 1, name="Grover")`, appends `U`, then repeats oracle and diffuser | Grover circuit assembly |
| 3 | Executes the statevector simulation | Quantum simulation |
| 4 | Computes probabilities, selects `argmax`, formats the basis state | Classical result extraction |
| 5 | Saves circuit and text output | Export |

The oracle uses a kickback ancilla prepared in $|-\rangle$. The method `_build_oracle(qc, target_qubits_index, target_qubits_value, ancilla)` applies `qc.mcx(...)` controlled on the target bit pattern, so it can mark targets containing both `0` and `1` bits without manually surrounding controls by X gates.

The diffuser is implemented as:

```text
U^\dagger -> all-zero phase oracle -> U
```

where `U = H^n`, so this is the usual reflection about the uniform superposition.

**Data flow:** `(n, target)` → iteration count from `p = 1/2^n` → circuit assembly (H^n → oracle → diffuser × reps) → `execute()` → statevector probabilities → `argmax` → `'Result'` → `_build_return_dict()`.

## Understanding the Key Quantum Components

### 1. State Preparation ($U = H^{\otimes n}$)

The initial state is the uniform superposition over all $2^n$ computational basis states:

$$|s\rangle = H^{\otimes n}|0^n\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$$

The initial success probability for one marked state is $p = |\langle \text{target}|s\rangle|^2 = 1/2^n$.

### 2. Oracle (Phase Kickback)

The oracle marks the target state by flipping its phase via a kickback ancilla in the $|-\rangle$ state:

- Ancilla is prepared as $|-\rangle = HX|0\rangle$.
- A multi-controlled-X (MCX) gate targets the ancilla, controlled on the target bit pattern.
- `_build_oracle(qc, target_qubits_index, target_qubits_value, ancilla)` applies `qc.mcx(...)` so targets containing both `0` and `1` bits are handled without manually surrounding controls by X gates.
- The net effect on the data register: $|x\rangle \mapsto (-1)^{f(x)}|x\rangle$ where $f(x) = 1$ iff $x = \text{target}$, $f(x) = 0$ otherwise.

### 3. Diffuser (Reflection about $|s\rangle$)

The Grover diffuser reflects the state about the uniform superposition:

$$D = 2|s\rangle\langle s| - I = H^{\otimes n}(2|0^n\rangle\langle 0^n| - I)H^{\otimes n}$$

Implemented as:
1. Apply $H^{\otimes n}$ (which is $U^\dagger = U$ since Hadamard is self-inverse).
2. Apply the all-zeros phase oracle.
3. Apply $H^{\otimes n}$ again.

### 4. Grover Iteration

Each iteration applies oracle → diffuser. Starting from $\sin\theta = \sqrt{p} = 2^{-n/2}$, after $k$ iterations:

$$G^k|s\rangle = \sin((2k+1)\theta)|\text{target}\rangle + \cos((2k+1)\theta)|\text{other}\rangle$$

The target-state amplitude grows, reaching a maximum near $(2k+1)\theta \approx \pi/2$, giving optimal iteration count:

$$k_{\text{opt}} \approx \frac{\pi}{4\sqrt{p}} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{2^n}$$

### 5. Measurement

The data register is measured in the computational basis. In this implementation, statevector probabilities are computed directly from `qc.execute()` as `np.abs(result.state) ** 2`, and the most likely basis state is returned as `Result`.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Uniform superposition $|s\rangle = H^{\otimes n}|0^n\rangle$ | `U = Circuit(n); U.h(list(range(n)))` in Stage 1 |
| Initial probability $p = 1/2^n$ | `p = 1 / (2 ** n)` in `run()` Stage 1 |
| Optimal iteration count $k = \lfloor\pi/(4\arcsin\sqrt{p}) - 1/2\rceil$ | `_get_optimal_iterations(p)` → `int(round(math.pi / (4 * math.asin(math.sqrt(p))) - 0.5))` |
| Oracle $O$: phase flip on target state | `_build_oracle(qc, target_qubits_index, target_qubits_value, ancilla)` — MCX + kickback ancilla in $\|-\rangle$ |
| Diffuser \(D = 2|s\rangle\langle s| - I\) | `_build_diffuser(qc, U, data_qubits, ancilla)` — `U†` → all-zeros oracle → `U` |
| After \(k\) iterations: amplified probability | Stage 4 computes full-statevector probabilities and reports the probability of the `argmax` state as `Amplified target-state probability`; validate success with `Result == target`. |
| Ancilla qubit for phase kickback | Always index `n` (one beyond data register) |
| Post-measurement result extraction | `argmax` over statevector probabilities → `Result` as binary string |

**Notes on encapsulation:** The oracle is fully encapsulated in `_build_oracle`, and the diffuser reuses the same oracle pattern on all data qubits. The iteration count `_get_optimal_iterations(p)` uses `round()` rather than `floor()` to avoid floating-point precision errors near integer boundaries. Global phase corrections are not applied; this implementation targets probability amplification for measurement.

## Mathematical Deep Dive

Define the uniform superposition and target subspace:

$$|s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle, \quad N=2^n$$

For a single target state $|\omega\rangle$, define the orthogonal complement:

$$|s'\rangle = \frac{1}{\sqrt{N-1}}\sum_{x \neq \omega}|x\rangle$$

The initial state in the 2D Grover plane:

$$|s\rangle = \sin\theta\,|\omega\rangle + \cos\theta\,|s'\rangle, \quad \sin\theta = \frac{1}{\sqrt{N}}$$

The Grover operator $G = D \cdot O$ is a rotation by $2\theta$ in this plane:

$$G = \begin{pmatrix} \cos 2\theta & -\sin 2\theta \\ \sin 2\theta & \cos 2\theta \end{pmatrix}$$

After $k$ iterations:

$$G^k|s\rangle = \sin((2k+1)\theta)|\omega\rangle + \cos((2k+1)\theta)|s'\rangle$$

Optimal $k$ for maximum amplitude:

$$(2k+1)\theta \approx \frac{\pi}{2} \implies k = \left\lfloor\frac{\pi}{4\theta} - \frac{1}{2}\right\rceil \approx \frac{\pi}{4}\sqrt{N}$$

Query complexity: $O(\sqrt{N})$, quadratic speedup over classical $O(N)$.

## Hands-On Example

```python
from unitarylab_algorithms import GroverAlgorithm

# Single target search over 3 qubits
for target in ["000", "101", "111"]:
    result = GroverAlgorithm(text_mode="plain").run(n=3, target=target)
    print(f"Target={target}  Result={result['Result']}  Prob={result['Amplified target-state probability']:.4f}")

# Demonstrate quadratic speedup: 5-qubit search (N=32)
result = GroverAlgorithm(text_mode="plain").run(n=5, target="11010")
print(f"5-qubit search: {result['Result']}  Prob={result['Amplified target-state probability']:.4f}")
```

## Minimal Manual Implementation

```python
from unitarylab.core import Circuit
import math

def grover_search(n: int, target: str, backend='torch'):
    """Minimal Grover search for a single target bit string."""
    p = 1 / (2 ** n)
    reps = max(1, round(math.pi / (4 * math.asin(math.sqrt(p))) - 0.5))
    ancilla = n

    qc = Circuit(n + 1, name="Grover")
    # Uniform superposition
    for q in range(n):
        qc.h(q)

    for _ in range(reps):
        # --- Oracle: mark target ---
        qc.x(ancilla); qc.h(ancilla)          # ancilla → |−⟩
        for q, bit in enumerate(reversed(target)):
            if bit == '0':
                qc.x(q)                        # flip to control on |1⟩
        if n == 1:
            qc.cx(0, ancilla)
        else:
            qc.mcx(list(range(n)), ancilla)
        for q, bit in enumerate(reversed(target)):
            if bit == '0':
                qc.x(q)                        # unflip
        qc.h(ancilla); qc.x(ancilla)           # ancilla → |0⟩

        # --- Diffuser: reflect about uniform superposition ---
        for q in range(n):
            qc.h(q)
        qc.x(ancilla); qc.h(ancilla)
        for q in range(n):
            qc.x(q)                            # all-zeros oracle
        if n == 1:
            qc.cx(0, ancilla)
        else:
            qc.mcx(list(range(n)), ancilla)
        for q in range(n):
            qc.x(q)
        qc.h(ancilla); qc.x(ancilla)
        for q in range(n):
            qc.h(q)

    result = qc.execute(backend=backend)
    probs = np.abs(result.state) ** 2
    return max(result.items(), key=lambda kv: kv[1])[0]
```

## Debugging Tips

1. **Target length mismatch**: `target` must be a binary string of exactly `n` characters. For example, `n=3`, `target="101"`, not `target=5` or `target="01"`.
2. **Target has wrong bits**: `target` should only contain `'0'` and `'1'` characters.
3. **Too many iterations**: Grover iterations beyond optimal $k$ rotate the state past the target, reducing success probability. The optimal count is auto-computed via `_get_optimal_iterations(p)`.
4. **Single target only**: This implementation supports exactly one marked state. Multiple targets change the search angle ($\sin\theta = \sqrt{M/N}$ rather than $\sqrt{1/N}$), requiring the general amplitude-amplification framework.
5. **Probability is ideal**: The returned `Amplified target-state probability` is from an ideal noiseless statevector simulation.
6. **Ancilla index**: The ancilla is always at index `n` (one beyond data register). Do not include it in target specification.
7. **Bit-ordering checks**: When debugging target-state mismatches, test non-palindromic targets such as `"110"` and `"001"`, not only `"101"`. Palindromic strings can hide reversed bit-ordering issues. Keep the user-facing `target` and returned `Result` in the same bit-string convention.

## Relationship to Amplitude Amplification

Grover search is not separate from amplitude amplification mathematically; it is the canonical special case. Keep the distinction practical:

Use `grover` only when **all** of the following are true:
- the initial state is the uniform superposition;
- there is exactly one marked computational-basis target;
- the target can be represented as a binary string of length `n`;
- the standard Grover diffuser is sufficient.

Use `amplitude-amplification` instead when the task needs arbitrary state preparation, multiple good states, an estimated success probability from another circuit, or a custom oracle/good-state condition.

## Reference Implementations

These examples are for conceptual comparison only. For UnitaryLab code generation, prefer the provided `GroverAlgorithm` implementation described above.

### Qiskit

```python
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.amplitude_amplifiers import AmplificationProblem, Grover

# Oracle marking |11⟩
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

problem = AmplificationProblem(oracle=oracle, is_good_state=["11"])
grover = Grover(iterations=1, sampler=StatevectorSampler())
result = grover.amplify(problem)
print("Top measurement:", result.top_measurement)
```

### PennyLane

```python
import pennylane as qml
import numpy as np

n = 3
U = qml.prod(qml.Hadamard(w) for w in range(n))
O = qml.FlipSign(5, wires=range(n))  # mark |101⟩

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    for w in range(n):
        qml.Hadamard(wires=w)
    qml.AmplitudeAmplification(U, O, iters=2)
    return qml.probs(wires=range(n))

print(np.round(circuit(), 3))
```
