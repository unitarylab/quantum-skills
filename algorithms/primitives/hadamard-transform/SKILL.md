---
name: hadamard-transform
description: A quantum algorithm for performing the Hadamard transform, which is a fundamental operation in quantum computing that creates superposition states. This skill includes efficient implementations and educational resources for understanding and utilizing the Hadamard transform in various quantum algorithms and applications.
--- 

# Hadamard Transform

## Purpose

The Hadamard Transform ($H^{\otimes n}$) applies a Hadamard gate to every qubit simultaneously, mapping the computational basis to a uniform superposition and vice versa. It is a foundational building block in virtually all quantum algorithms.

Use this skill when you need to:
- Prepare a uniform superposition $\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$ from $|0\rangle^n$.
- Verify the self-inverse property $H^2 = I$ numerically.
- Understand the relationship between QFT and Hadamard transforms.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

The algorithm supports two modes:
1. **`'superposition'`**: Applies $H^{\otimes n}$ to $|0\rangle^n$, producing the uniform superposition state. Verifies that all $2^n$ basis states have equal probability $1/2^n$.
2. **`'reflexive_test'`**: Initializes a random quantum state $|\psi\rangle$, applies $H^{\otimes n}$ twice, and verifies that the result equals the original state ($H^2 = I$).

## Prerequisites

- Basic single-qubit Hadamard gate: $H = \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$
- Python: `numpy`, `GateSequence`.

## Using the Provided Implementation

```python
from unitarylab.algorithms import HadamardTransformAlgorithm

algo = HadamardTransformAlgorithm()

# Mode 1: Generate uniform superposition
result = algo.run(n_qubits=3, mode='superposition', backend='torch')
print(result['status'])          # 'ok' or 'failed'
print(result['state_vector'])    # Final state vector
print(result['probabilities'])       # Probability distribution over 2^3 basis states
print(result['circuit_path'])    # SVG circuit diagram

# Mode 2: Verify H^2 = I
result2 = algo.run(n_qubits=3, mode='reflexive_test', backend='torch')
print(result2['status'])         # 'ok' if H^2 recovers original state
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_qubits` | `int` | `3` | Number of qubits. Must be $\geq 1$. |
| `mode` | `str` | `'superposition'` | `'superposition'` or `'reflexive_test'`. |
| `backend` | `str` | `'torch'` | Simulation backend. Forces `'torch'`. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `'reflexive_test'` uses a random initial state generated internally; you cannot supply your own.
- `'superposition'` starts from $|0\rangle^n$ (the default zero state, no explicit preparation needed).

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` otherwise. |
| `state_vector` | `np.ndarray` | Final state vector after the transform. |
| `probabilities` | `dict` | Bitstring → probability (for `'superposition'` mode). |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `message` | `str` | Human-readable result summary. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`HadamardTransformAlgorithm` in `algorithm.py` is intentionally minimal. The `run()` method sequences five stages directly, with two tiny helper functions for data conversion.

**`run(n_qubits, mode, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Validation | Checks `n_qubits >= 1`, validates `mode` string | Guards against invalid inputs |
| 2 — Circuit Construction | Creates `GateSequence(n_qubits)`; dispatches on mode: `'superposition'` calls `_apply_hadamard_layer` once; `'reflexive_test'` generates random state via `numpy` and calls `gs.initialize(psi, target)`, then calls `_apply_hadamard_layer` **twice** | Builds the transform circuit appropriate to each mode |
| 3 — Simulation | `gs.execute()` → `_as_statevector(raw_result)` | Runs statevector simulation; wraps result as `numpy` array |
| 4 — Post-Processing | `'superposition'`: calls `_probabilities(state_vector)` to compute bitstring→prob dict and checks uniformity; `'reflexive_test'`: computes `np.allclose(state_vector, original_state)` | Verifies algorithm correctness based on mode |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_apply_hadamard_layer(gs, target_qubits)`** — Applies `gs.h(q)` to every qubit in `target_qubits`. The entire Hadamard transform is just this one loop.
- **`_as_statevector(res)`** — `np.asarray(res, dtype=complex)` — converts `execute()` output to a flat NumPy array.
- **`_probabilities(statevec, threshold)`** — Computes `|amp|²` for each basis state; returns a sorted dict of binary-string → float, filtering values below `threshold`.
- **`_update_last_result` / `_build_return`** — Store runtime fields and package result dict.

**Key design note:** In `'reflexive_test'` mode, the random initial state is generated with `numpy` and loaded into the circuit via `gs.initialize(original_state, target=target_qubits)`. The state is stored in a local variable `original_state` for comparison after two H-layer applications.

**Data flow (superposition):** `n_qubits` → `GateSequence` → `_apply_hadamard_layer` → `execute()` → `_probabilities()` → uniformity check → `_build_return()`.  
**Data flow (reflexive_test):** random `psi` → `gs.initialize(psi)` → two `_apply_hadamard_layer()` calls → `execute()` → `np.allclose(result, psi)` → `_build_return()`.

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
$H^\dagger = H$ and $H^2 = I$. Applying $H^{\otimes n}$ twice recovers the original state exactly (up to floating-point precision $\sim 10^{-10}$).

### Relationship to QFT
When the input is $|0\rangle^n$, the Hadamard transform equals the Quantum Fourier Transform (QFT), as both produce the uniform superposition.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| $n$-qubit Hadamard transform $H^{\otimes n}$ | `_apply_hadamard_layer(gs, target_qubits)` — one `gs.h(q)` per qubit |
| Starting state $|0\rangle^n$ | Default: `GateSequence` starts in $|0\rangle^n$ without explicit init |
| Arbitrary initial state (reflexive test) | `gs.initialize(original_state, target=target_qubits)` |
| Self-inverse property $H^2 = I$ | `'reflexive_test'` mode: `_apply_hadamard_layer` called twice; verified via `np.allclose()` |
| Uniform output probability $1/2^n$ | Checked in post-processing via `np.isclose(p, 1/2^n, atol=1e-5)` |
| Bitstring probability dict | `_probabilities(state_vector)` — converts amplitudes to `{bitstring: prob}` |
| Status / success | `is_success` in `_build_return()`; `'ok'` if uniformity/reflexivity test passes |

**Notes on encapsulation:** This implementation is the simplest in the codebase. The transform itself is entirely realized by `gs.h(q)` calls inside `_apply_hadamard_layer`. There is no separate `_build_circuit` method; circuit construction happens inline in `run()`. The `_probabilities()` helper avoids near-zero states using a `threshold=1e-12` filter.

## Mathematical Deep Dive = \bigotimes_{j=1}^n \frac{|0\rangle + (-1)^{x_j}|1\rangle}{\sqrt{2}}$$

For $|0\rangle^n$:
$$H^{\otimes n}|0\rangle^n = \frac{1}{\sqrt{2^n}}\sum_{y \in \{0,1\}^n}|y\rangle$$

Norm preservation: $\|H^{\otimes n}|x\rangle\|^2 = \sum_{y}|(-1)^{x\cdot y}/\sqrt{2^n}|^2 = 2^n \cdot 1/2^n = 1$.

## Hands-On Example

```python
from unitarylab.algorithms import HadamardTransformAlgorithm
import numpy as np

algo = HadamardTransformAlgorithm()

# Demonstrate 4-qubit uniform superposition
result = algo.run(n_qubits=4, mode='superposition', backend='torch')
print(f"Number of basis states: {len(result['probabilities'])}")  # 16
print(f"Each probability: {list(result['probabilities'].values())[0]:.6f}")  # ~0.0625 = 1/16
print(f"Status: {result['status']}")

# Verify H^2 = I numerically
result2 = algo.run(n_qubits=4, mode='reflexive_test', backend='torch')
```

## Implementing Your Own Version

```python
from unitarylab.core import GateSequence

def hadamard_transform(n: int, backend: str = 'torch') -> GateSequence:
    """Apply H to all n qubits."""
    gs = GateSequence(n, name=f"H^{n}", backend=backend)
    for q in range(n):
        gs.h(q)
    return gs
```

## Debugging Tips

1. **Probability not exactly $1/2^n$**: Floating-point arithmetic causes small deviations ($\sim 10^{-15}$). The code uses `np.isclose` with `atol=1e-5`.
2. **Mode typo**: Only `'superposition'` and `'reflexive_test'` are valid; they are case-sensitive.
3. **`'reflexive_test'` with $n=1$**: Works correctly; the random state is a complex unit vector on the Bloch sphere.
4. **Expected state includes imaginary parts**: $H^{\otimes n}|0\rangle^n$ has only real amplitudes, but the `'reflexive_test'` random state will be complex.
5. **Bit ordering**: Probabilities and state vector use little-endian convention (qubit 0 is least significant bit).
