---
name: quantum-fourier-transform
description: >
  Skill for implementing and using the Quantum Fourier Transform (QFT) via the
  PennyLane `QFT` operation. Covers circuit construction, gate decomposition,
  matrix access, adjoint (inverse QFT), and integration into larger circuits
  such as QPE and Shor's algorithm.
---

# Quantum Fourier Transform (QFT)

## Algorithm Overview

| Property        | Value                                                                 |
|-----------------|-----------------------------------------------------------------------|
| Category        | Quantum circuit primitive                                             |
| Framework       | PennyLane (`pennylane`)                                               |
| Main class      | `QFT` (subclass of `pennylane.operation.Operation`)                   |
| Trainable params| 0                                                                     |
| Gradient method | None                                                                  |
| Used in         | Quantum Phase Estimation, Shor's algorithm, quantum arithmetic        |

The QFT maps an $N$-qubit computational basis state $|m\rangle$ to a uniform superposition over all basis states with phase factors encoding the Fourier coefficients. It is the quantum analogue of the Discrete Fourier Transform (DFT) and runs in $O(N^2)$ gates versus $O(N \cdot 2^N)$ classically.

---

## Mathematical Principles

**Core transformation** (acting on an $N$-qubit register):

$$
|m\rangle \;\xrightarrow{\text{QFT}}\; \frac{1}{\sqrt{2^N}} \sum_{n=0}^{2^N-1} \omega_N^{mn} \,|n\rangle, \qquad \omega_N = e^{\frac{2\pi i}{2^N}}
$$

**Gate decomposition** (for wire $i$, where $i = 0, 1, \ldots, N-1$):

1. Apply $H$ to wire $i$.
2. Apply `ControlledPhaseShift`$(2\pi / 2^k)$ from wire $i+k$ (control) to wire $i$ (target), for $k = 2, 3, \ldots, N - i$.
3. After all wires are processed, apply $\lfloor N/2 \rfloor$ `SWAP` gates to reverse the qubit order.

**Phase shift angles**:

$$
\phi_k = \frac{2\pi}{2^k}, \quad k = 2, 3, \ldots, N
$$

**Gate count** (from `_qft_decomposition_resources`):

| Gate                  | Count                    |
|-----------------------|--------------------------|
| `Hadamard`            | $N$                      |
| `ControlledPhaseShift`| $N(N-1)/2$               |
| `SWAP`                | $\lfloor N/2 \rfloor$    |

**Matrix representation** (computed via `compute_matrix`):

$$
QFT_{mn} = \frac{1}{\sqrt{2^N}}\, \omega_N^{mn}
$$

Implemented internally as `np.fft.ifft(np.eye(2**num_wires), norm="ortho")`.

---

## Core Parameters

| Parameter   | Type                          | Required | Default | Description                                      |
|-------------|-------------------------------|----------|---------|--------------------------------------------------|
| `wires`     | `int` or `Iterable[int/str]`  | Yes      | —       | Wires the QFT acts on; length determines $N$     |
| `id`        | `str` or `None`               | No       | `None`  | Optional label for the operation instance        |

**Derived hyperparameter** (set automatically):

| Key          | Value           | Description                        |
|--------------|-----------------|------------------------------------|
| `num_wires`  | `len(wires)`    | Number of qubits in the transform  |

---

## Inputs and Outputs

| Direction | Object              | Shape / Type                  | Description                                                     |
|-----------|---------------------|-------------------------------|-----------------------------------------------------------------|
| Input     | `wires`             | `Wires` of length $N$         | Qubit register in the computational basis                       |
| Output    | Quantum state       | $2^N$-dimensional complex     | State in the Fourier basis with phases $\omega_N^{mn}$          |
| Output    | `compute_matrix`    | `ndarray` of shape $(2^N, 2^N)$ | Unitary matrix of the QFT (parameterised by `num_wires`)      |
| Output    | `compute_decomposition` | `list[Operator]`          | Explicit gate list: `[H, CPS, ..., SWAP]`                       |

---

## Implementation Description

### Class Structure (`QFT`)

The `QFT` class lives in `scripts/qft.py` and subclasses `pennylane.operation.Operation`.

| Component                  | Responsibility                                                                                  |
|----------------------------|-------------------------------------------------------------------------------------------------|
| `__init__(wires, id)`      | Wraps `wires` in `Wires`, stores `num_wires` as a hyperparameter, calls `super().__init__`      |
| `compute_matrix(num_wires)`| Returns the $2^N \times 2^N$ unitary via `np.fft.ifft(np.eye(2**N), norm="ortho")`; LRU-cached |
| `compute_decomposition(wires)` | Builds and returns the explicit gate list (static method)                                  |
| `decomposition()`          | Instance wrapper that delegates to `compute_decomposition(wires=self.wires)`                    |
| `resource_params`          | Returns `{"num_wires": len(self.wires)}` for resource estimation                               |
| `_qft_decomposition`       | JAX/capture-compatible functional variant using `for_loop`; registered via `add_decomps`        |

### Execution Flow (`compute_decomposition`)

```python
# Simplified reconstruction — matches control flow of compute_decomposition
def compute_decomposition(wires):
    wires = Wires(wires)
    N = len(wires)
    shifts = [2 * np.pi * 2**-i for i in range(2, N + 1)]  # [π, π/2, π/4, ...]

    ops = []
    for i, wire in enumerate(wires):
        ops.append(Hadamard(wire))
        for shift, ctrl in zip(shifts[:N - 1 - i], wires[i + 1:]):
            ops.append(ControlledPhaseShift(shift, wires=[ctrl, wire]))

    # Reverse qubit order with SWAPs
    for w1, w2 in zip(wires[:N // 2], reversed(wires[-(N // 2):])):
        ops.append(SWAP(wires=[w1, w2]))

    return ops
```

> **Note**: `shifts` is indexed so that wire $i$ receives phase shifts from $k=2$ up to $k = N-i$, matching $\phi_k = 2\pi/2^k$. The trailing SWAP block restores the standard QFT output ordering.

### JAX-Compatible Variant (`_qft_decomposition`)

For PennyLane's program-capture (JAX) mode, `_qft_decomposition` reimplements the same logic using `for_loop` from `pennylane.control_flow`. It is registered with `add_decomps(QFT, _qft_decomposition)` and activated automatically when `pennylane.capture.enabled()` returns `True`. No user action is needed to select between the two paths.

---

## Key Quantum Components

| Component              | Gates Used              | Role in QFT                                                                  |
|------------------------|-------------------------|------------------------------------------------------------------------------|
| Hadamard ($H$)         | `Hadamard`              | Creates uniform superposition on each qubit; initialises Fourier phases      |
| Controlled Phase Shift | `ControlledPhaseShift`  | Encodes inter-qubit phase correlations $e^{2\pi i / 2^k}$                   |
| Bit-reversal SWAP      | `SWAP`                  | Corrects output bit order to match standard QFT convention                   |
| Inverse QFT            | `qml.adjoint(QFT)`      | Hermitian conjugate; used in QPE readout and arithmetic uncomputation         |

---

## Sample Code

### Minimal usage in a PennyLane circuit

```python
import pennylane as qml
import numpy as np

# Import local implementation (or use qml.QFT directly if pennylane >= 0.36)
from scripts.qft import QFT  # local copy

N = 3
dev = qml.device("default.qubit", wires=N)

@qml.qnode(dev)
def circuit(basis_state):
    qml.BasisState(basis_state, wires=range(N))
    QFT(wires=range(N))
    return qml.state()

state = circuit(np.array([1, 0, 0]))
print(state)
# [ 0.3536+0.j, -0.3536+0.j,  0.3536+0.j, -0.3536+0.j,
#   0.3536+0.j, -0.3536+0.j,  0.3536+0.j, -0.3536+0.j]
```

### Access the unitary matrix

```python
from scripts.qft import QFT

matrix = QFT.compute_matrix(num_wires=3)  # shape (8, 8), complex128
print(matrix.shape)  # (8, 8)
```

### Inspect the gate decomposition

```python
from scripts.qft import QFT
import pennylane as qml

ops = QFT.compute_decomposition(wires=(0, 1, 2))
for op in ops:
    print(op)
# H(0)
# ControlledPhaseShift(1.5708, wires=[1, 0])
# ControlledPhaseShift(0.7854, wires=[2, 0])
# H(1)
# ControlledPhaseShift(1.5708, wires=[2, 1])
# H(2)
# SWAP(wires=[0, 2])
```

### Inverse QFT (adjoint)

```python
import pennylane as qml
from scripts.qft import QFT

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def inverse_qft_circuit(state):
    qml.BasisState(state, wires=range(3))
    qml.adjoint(QFT)(wires=range(3))   # applies QFT†
    return qml.state()
```

---

## Debugging Tips

| Symptom                              | Likely Cause                                                                 | Fix                                                             |
|--------------------------------------|------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `ImportError: No module named pennylane` | PennyLane not installed                                                 | `pip install pennylane`                                         |
| Wrong output ordering                | QFT convention uses bit-reversal; SWAPs at the end fix this                 | Use `compute_decomposition` output directly; do not remove SWAPs |
| Phase mismatch in QPE                | Using QFT where inverse QFT is needed                                        | Wrap with `qml.adjoint(QFT)(wires=...)`                         |
| `lru_cache` stale matrix             | `compute_matrix` is cached by `num_wires`; different wire labels hit same cache | Use `num_wires` correctly; cache key is qubit count only     |
| JAX tracing errors                   | Static decomposition used inside JIT context                                | PennyLane's `capture` mode selects `_qft_decomposition` automatically |
| Gate count grows quadratically       | Expected: $O(N^2)$ two-qubit gates for $N$ qubits                          | For large $N$, consider approximate QFT (truncate small phases) |