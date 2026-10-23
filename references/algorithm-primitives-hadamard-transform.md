# Hadamard Transform

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The Hadamard Transform ($H^{\otimes n}$) applies a Hadamard gate to every qubit simultaneously, mapping the computational basis to a uniform superposition and vice versa. It is a foundational building block in virtually all quantum algorithms.

This reference covers zero-state uniform-superposition preparation and an internally generated random-state self-inverse check.

Source pointers: [algorithm-library usage example](../scripts/hadamard_transform_algorithm.py) and [independent teaching implementation](../scripts/hadamard_transform_implementation.py). The teaching solver's extra diagnostics are not algorithm-library return fields.

Algorithm-library source: `HadamardTransformAlgorithm` (`unitarylab_algorithms/fundamental_algorithm/hadamard_transform/algorithm.py`).

## Overview

The algorithm supports two modes:
1. **`'superposition'`**: Applies $H^{\otimes n}$ to $|0\rangle^n$, producing the uniform superposition state. Verifies that all $2^n$ basis states have equal probability $1/2^n$.
2. **`'reflexive_test'`**: Initializes a random quantum state $|\psi\rangle$, applies $H^{\otimes n}$ twice, and verifies that the result equals the original state ($H^2 = I$).

## Prerequisites

- Basic single-qubit Hadamard gate: $H = \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$
- Python: `numpy`, `Circuit`.

## Reference Implementation Example

```python
from unitarylab_algorithms import HadamardTransformAlgorithm

algo = HadamardTransformAlgorithm()

# Mode 1: Generate uniform superposition
result = algo.run(n=3, mode='superposition', backend='torch')
print(result['status'])                    # 'ok' on normal completion
print(result['State vector'])              # Final state vector
print(result['Probability distribution'])  # Bitstring → probability over 2^3 basis states
print(result['circuit_path'])              # Path to SVG circuit diagram

# Mode 2: Verify H^2 = I
result2 = algo.run(n=3, mode='reflexive_test', backend='torch')
print(result2['status'])         # Completion status, not the H^2 check outcome
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | `3` | Number of qubits. Must be $\geq 1$. |
| `mode` | `str` | `'superposition'` | `'superposition'` or `'reflexive_test'`. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device passed to the backend. |
| `dtype` | `dtype` | `np.complex128` | Numeric dtype for the simulation. |
| `algo_dir` | `str` or `None` | `None` | Output directory (set in `__init__`, not `run`). |

**Common misunderstandings:**
- `'reflexive_test'` uses a random initial state generated internally; you cannot supply your own.
- `'superposition'` starts from $|0\rangle^n$ (the default zero state, no explicit preparation needed).

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on normal completion: `run()` passes `True` to the return builder even if its internal uniformity/reflexivity check fails. Exceptions propagate. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `plot` | `list` | List of `{"format": str, "filename": str}` dicts for each saved output file. |
| `circuit` | `Circuit` | The constructed `Circuit` object. |
| `State vector` | `np.ndarray` | Final state vector after the transform. |
| `Probability distribution` | `dict` | Bitstring → probability (populated for `'superposition'` mode; empty dict for `'reflexive_test'`). |
| `Computation time (s)` | `float` | Wall-clock simulation time in seconds. |

## Implementation Architecture

`HadamardTransformAlgorithm.run()` sequences the five stages directly and uses helpers for the H layer and state-array conversion.

**`run(n, mode, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Validation | Checks `n >= 1`, validates `mode` string | Guards against invalid inputs |
| 2 — Circuit Construction | Creates `Circuit(n, name=...)`; dispatches on mode: `'superposition'` calls `_apply_hadamard_layer` once; `'reflexive_test'` generates random state via `numpy` and calls `qc.initialize(psi, target)`, then calls `_apply_hadamard_layer` **twice** | Builds the transform circuit appropriate to each mode |
| 3 — Simulation | `qc.execute(backend=backend, device=device, dtype=dtype)` → `_as_statevector(raw_result.state)` | Runs statevector simulation; wraps result as `numpy` array |
| 4 — Post-Processing | `'superposition'`: uses `raw_result.probabilities` and checks uniformity; `'reflexive_test'`: computes `np.allclose(raw_result.state, original_state)` | Verifies algorithm correctness based on mode |
| 5 — Export | `self.save_circuit(qc)` and `self.save_txt()` | Saves SVG circuit diagram and text results; returns `_build_return_dict()` |

**Helper Methods:**

- **`_apply_hadamard_layer(qc, target_qubits)`** — Applies `qc.h(q)` to every qubit in `target_qubits`. The entire Hadamard transform is just this one loop.
- **`_as_statevector(res)`** — `np.asarray(res, dtype=complex)` — converts `execute()` output to a flat NumPy array.
- **`_probabilities(statevec, threshold)`** — Computes `|amp|²` for each basis state; returns a sorted dict of binary-string → float, filtering values below `threshold`.
- **`_build_return_dict(success, circuit_path, filepath, circuit)`** — Packages `plot` and the output fields. This `run()` always supplies `True`; the internal verification result is reflected in the text summary, not a public Boolean field.

**Key design note:** In `'reflexive_test'` mode, the random initial state is generated with `numpy` and loaded into the circuit via `qc.initialize(original_state, target=target_qubits)`. The state is stored in a local variable `original_state` for comparison after two H-layer applications.

**Data flow (superposition):** `n` → `Circuit` → `_apply_hadamard_layer` → `execute()` → `raw_result.probabilities` → uniformity check → `_build_return_dict()`.
**Data flow (reflexive_test):** random `psi` → `qc.initialize(psi)` → two `_apply_hadamard_layer()` calls → `execute()` → `np.allclose(raw_result.state, psi)` → `_build_return_dict()`.

## Understanding the Key Quantum Components

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}, \quad H|0\rangle = |+\rangle = \frac{|0\rangle+|1\rangle}{\sqrt{2}}, \quad H|1\rangle = |-\rangle = \frac{|0\rangle-|1\rangle}{\sqrt{2}}$$

### Tensor Product of $n$ Hadamard Gates
$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}(-1)^{x \cdot y}|y\rangle$$
where $x \cdot y = \sum_i x_i y_i \pmod{2}$ is the bitwise inner product.

### Superposition State
Starting from $|0\rangle^n$, the transform produces:
$$H^{\otimes n}|0\rangle^n = \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1}|y\rangle$$
All $2^n$ basis states have equal amplitude $1/\sqrt{2^n}$ and equal probability $1/2^n$.

### Self-Inverse Property
$H^\dagger = H$ and $H^2 = I$ exactly in the mathematical model. The source compares simulated amplitudes with `np.allclose(..., atol=1e-10)` using NumPy's default relative tolerance; this tolerance is not an achieved numerical-error guarantee.

### Relationship to QFT
When the input is $|0\rangle^n$, the Hadamard transform equals the Quantum Fourier Transform (QFT), as both produce the uniform superposition.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| $n$-qubit Hadamard transform $H^{\otimes n}$ | `_apply_hadamard_layer(qc, target_qubits)` — one `qc.h(q)` per qubit |
| Starting state $|0\rangle^n$ | Default: `Circuit` starts in $|0\rangle^n$ without explicit init |
| Arbitrary initial state (reflexive test) | `qc.initialize(original_state, target=target_qubits)` |
| Self-inverse property $H^2 = I$ | `'reflexive_test'` mode: `_apply_hadamard_layer` called twice; verified via `np.allclose()` |
| Uniform output probability $1/2^n$ | Checked in post-processing via `np.isclose(p, 1/2^n, atol=1e-5)` |
| Returned probability dict | `raw_result.probabilities` in superposition mode; the local `_probabilities()` helper is unused by `run()` |
| Internal verification / public status | Local `is_success` controls the summary; `_build_return_dict(True, ...)` produces public `'ok'` regardless of that check |

**Notes on encapsulation:** The transform itself is entirely realized by `qc.h(q)` calls inside `_apply_hadamard_layer`. There is no separate `_build_circuit` method; circuit construction happens inline in `run()`. The `_probabilities()` helper avoids near-zero states using a `threshold=1e-12` filter.

## Mathematical Deep Dive

For $|0\rangle^n$:
$$H^{\otimes n}|0\rangle^n = \frac{1}{\sqrt{2^n}}\sum_{y \in \{0,1\}^n}|y\rangle$$

Norm preservation: $\|H^{\otimes n}|x\rangle\|^2 = \sum_{y}|(-1)^{x\cdot y}/\sqrt{2^n}|^2 = 2^n \cdot 1/2^n = 1$.

## Hands-On Example

```python
from unitarylab_algorithms import HadamardTransformAlgorithm
import numpy as np

algo = HadamardTransformAlgorithm()

# Demonstrate 4-qubit uniform superposition
result = algo.run(n=4, mode='superposition', backend='torch')
print(f"Number of basis states: {len(result['Probability distribution'])}")  # 16
print(f"Each probability: {list(result['Probability distribution'].values())[0]:.6f}")  # ~0.0625 = 1/16
print(f"Status: {result['status']}")
print(f"Computation time: {result['Computation time (s)']:.4f} s")

# Verify H^2 = I numerically
result2 = algo.run(n=4, mode='reflexive_test', backend='torch')
print(f"State vector shape: {result2['State vector'].shape}")
```

## Minimal Manual Implementation

```python
from unitarylab.core import Circuit

def hadamard_transform(n: int, backend: str = 'torch') -> Circuit:
    """Apply H to all n qubits."""
    qc = Circuit(n, name=f"H^{n}")
    for q in range(n):
        qc.h(q)
    return qc
```

## Debugging Tips

1. **Probability not exactly $1/2^n$**: Inspect observed residuals. The uniformity check uses `np.isclose(..., atol=1e-5)` with default relative tolerance and requires exactly $2^n$ entries; it does not establish a fixed floating-point error for every backend/dtype.
2. **Mode typo**: Only `'superposition'` and `'reflexive_test'` are valid; they are case-sensitive.
3. **`'reflexive_test'` with $n=1$**: Works correctly; the random state is a complex unit vector on the Bloch sphere.
4. **Expected state includes imaginary parts**: $H^{\otimes n}|0\rangle^n$ has only real amplitudes, but the `'reflexive_test'` random state will be complex.
5. **Bit ordering**: Probabilities and state vector use little-endian convention (qubit 0 is least significant bit).
