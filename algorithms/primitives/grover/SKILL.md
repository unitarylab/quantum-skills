---
name: grover
description: Grover search algorithm for finding a marked computational-basis state in an unstructured search space with quadratic query speedup. Use this skill for the repository's standalone GroverAlgorithm implementation.
---

# Grover Search

## Purpose

Grover search finds a marked item in an unstructured search space. In this implementation, the search space is the $2^n$ computational-basis states of an $n$-qubit data register, and the marked item is a single target bit string.

Grover search is the uniform-superposition, fixed-target special case of amplitude amplification:

- state preparation is $U = H^{\otimes n}$,
- the initial success probability is $p = 1/2^n$ for one marked state,
- the oracle marks one computational-basis target,
- the diffuser reflects about the uniform superposition.

Use this skill when you need to:
- Search for a target bit string such as `'101'`.
- Demonstrate Grover's quadratic speedup for unstructured search.
- Use the standalone `GroverAlgorithm` class rather than the general amplitude-amplification interface.

## Overview

The algorithm proceeds as follows:

1. Prepare the uniform superposition $|s\rangle = H^{\otimes n}|0^n\rangle$.
2. Mark the target state with a phase oracle.
3. Apply the diffuser, a reflection about $|s\rangle$.
4. Repeat oracle plus diffuser for approximately $\frac{\pi}{4}\sqrt{2^n}$ iterations.
5. Simulate the circuit and return the most likely basis state.

For one target state, the initial success probability is
$$
p = \frac{1}{2^n}.
$$
Writing $\sin\theta = \sqrt{p}$, after $k$ Grover iterations the target-state probability is ideally
$$
\sin^2((2k+1)\theta).
$$

## Using The Provided Implementation

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

## Core Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | required | Number of data-register qubits. The search space has size $N=2^n$. |
| `target` | `str` | required | Target computational-basis state as a binary string. Its length should equal `n`. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device. |
| `dtype` | dtype | `np.complex128` | Numeric dtype for simulation. |

`GroverAlgorithm(text_mode="plain", algo_dir=None)` sets output formatting and save location.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | Return status from `_build_return_dict`. |
| `Amplified target-state probability` | `float` | Probability of the most likely state after Grover iterations. |
| `Result` | `str` | Most likely basis state found by the simulation. |
| `circuit_path` | `str` | Path to the saved circuit diagram. |
| `plot` | `list` | Saved output file metadata. |
| `circuit` | `Circuit` | The assembled Grover circuit. |

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

## Relationship To Amplitude Amplification

Grover search is not separate from amplitude amplification mathematically; it is the canonical special case. Keep the distinction practical:

- Use `grover` when the task is unstructured search over computational basis states with a target bit string.
- Use `amplitude-amplification` when the task starts from an arbitrary state-preparation circuit `U`, an estimated success probability `p`, or a custom good-subspace condition.

## Common Misunderstandings

- `target` should be a binary string of length `n`; for example, `n=3`, `target="101"`.
- The implementation is for a single marked target state. Multiple marked states are better described through the general amplitude-amplification framework.
- Too many Grover iterations can rotate past the target state and reduce success probability.
- The returned `Result` is the most likely state from statevector probabilities, not a finite-shot measurement sample.

## Example

```python
from unitarylab_algorithms import GroverAlgorithm

for target in ["000", "101", "111"]:
    result = GroverAlgorithm(text_mode="plain").run(n=3, target=target)
    print(target, result["Result"], result["Amplified target-state probability"])
```
