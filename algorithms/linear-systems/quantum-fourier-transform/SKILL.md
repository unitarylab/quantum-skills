---
name: quantum-fourier-transform
description: "Use this skill when the user asks for Quantum Fourier Transform (QFT), inverse QFT (IQFT), Fourier-basis state conversion, QFT circuit construction, or implementing/running/debugging QFTAlgorithm in the UnitaryLab algorithm library. Prefer the UnitaryLab `QFTAlgorithm` and `unitarylab.core.Circuit` implementation; PennyLane material is reference-only. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Quantum Fourier Transform (QFT)

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Quantum Fourier Transform (QFT).

The Quantum Fourier Transform maps computational-basis amplitudes into the Fourier basis. It is the quantum analogue of the discrete Fourier transform and is a core subroutine in phase estimation, Shor-style period finding, quantum arithmetic, and several linear-algebra routines.

Use this skill when you need to:
- Build or run a QFT or inverse QFT circuit in UnitaryLab.
- Verify a QFT simulation against NumPy FFT/iFFT.
- Explain the `QFTAlgorithm` implementation in `unitarylab_algorithms.linear_algebra.qft.algorithm`.
- Diagnose phase-ordering, inverse-transform, or bit-reversal issues.

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Overview

For an $n$-qubit register, QFT acts on basis state $|x\rangle$ as:

$$
|x\rangle \mapsto \frac{1}{\sqrt{2^n}}\sum_{y=0}^{2^n-1} e^{2\pi ixy/2^n}|y\rangle
$$

The UnitaryLab implementation constructs the transform with:
- Hadamard gates `h(i)` to create local Fourier superpositions.
- Multi-controlled phase gates `mcp(theta, control, target)` to encode relative phases.
- Final `swap(i, n-1-i)` gates to restore output bit order.
- `qft.dagger()` for inverse QFT.

The algorithm then appends this QFT/IQFT block to an optional initialized state and verifies the simulator output against NumPy's FFT routines.

## Prerequisites

- Qubit indexing and computational-basis states.
- Hadamard, controlled phase rotation, and SWAP gates.
- Python: `numpy`, `numpy.fft`, `unitarylab.core.Circuit`, `unitarylab_algorithms`.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms.linear_algebra.qft.algorithm import QFTAlgorithm

algo = QFTAlgorithm(text_mode="plain")

state = np.zeros(8, dtype=complex)
state[1] = 1.0

result = algo.run(
    n=3,
    state=state,
    inverse=False,
    backend="torch",
    device="cpu",
)

print(result["status"])
print(result["Verification error"])
print(result["Final state"])
print(result["Expected state"])
print(result["circuit_path"])
```

For inverse QFT:

```python
result = algo.run(n=3, state=state, inverse=True, backend="torch")
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | `int` | required | Number of qubits. The state dimension is `2**n`. |
| `state` | `np.ndarray` or `None` | `None` | Optional initial state vector. If provided, it is normalized before initialization. If omitted, the circuit starts from `|0...0>`. |
| `inverse` | `bool` | `False` | If `False`, applies QFT. If `True`, constructs QFT and then applies `dagger()` to run IQFT. |
| `backend` | `str` | `'torch'` | Simulation backend passed to `Circuit.execute()`. |
| `device` | `str` | `'cpu'` | Compute device for simulation. |
| `dtype` | type | `np.complex128` | Complex dtype for statevector simulation. |

**Common misunderstandings:**
- `state` must have length `2**n`. The helper `test()` validates this before calling `run()`.
- The implementation normalizes `state` internally. Pass amplitudes, not probabilities.
- `inverse=True` does not rebuild a separate circuit by hand; it calls `qft.dagger()` and renames the circuit to `IQFT`.
- The NumPy check uses `ifft(state) * sqrt(2**n)` for QFT and `fft(state) / sqrt(2**n)` for IQFT, matching the implemented sign and normalization convention.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success. |
| `Final state` | `np.ndarray` | Statevector returned by the UnitaryLab simulator after QFT/IQFT. |
| `Expected state` | `np.ndarray` | NumPy FFT/iFFT reference state. |
| `Verification error` | `float` | L2 norm `||final_state - expected_state||`. |
| `Computation time (s)` | `float` | Simulator execution time. |
| `circuit_path` | `str` | Saved SVG circuit diagram path. |
| `plot` | `list` | Saved text output metadata. |
| `circuit` | `Circuit` | The assembled example circuit. |

## Implementation Architecture

`QFTAlgorithm` in `algorithm.py` extends `BaseAlgorithm` and follows three computational stages plus export.

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — QFT Circuit Construction | Creates `qft = Circuit(n, name="QFT")`; loops `i` from `n-1` down to `0`; applies `qft.h(i)` and `qft.mcp(np.pi/2**(i-j), j, i)` for lower-index controls; applies final swaps | Builds the QFT gate decomposition |
| 1b — Optional Inversion | If `inverse=True`, calls `qft = qft.dagger()`, then renames circuit and gate sequence to `IQFT` | Converts QFT into inverse QFT |
| 2 — Simulation | Creates `qc = Circuit(n, name="QFT Example")`; optionally initializes the normalized state; appends the QFT/IQFT circuit; calls `qc.execute(...)` | Runs statevector simulation |
| 3 — Classical Verification | Uses NumPy `ifft` for QFT and `fft` for IQFT with matching normalization; computes L2 error | Confirms the circuit implements the intended transform |
| 4 — Export | Calls `save_circuit(qc.decompose())` and `save_txt()` | Saves circuit diagram and text report |

**QFT construction loop:**

```python
qft = Circuit(n, name="QFT")
for i in range(n - 1, -1, -1):
    qft.h(i)
    for j in range(i - 1, -1, -1):
        qft.mcp(np.pi / 2 ** (i - j), j, i)

for i in range(n // 2):
    qft.swap(i, n - 1 - i)
```

For `n=3`, the forward circuit structure is:

```text
H(2)
MCP(pi/2, 1 -> 2)
MCP(pi/4, 0 -> 2)
H(1)
MCP(pi/2, 0 -> 1)
H(0)
SWAP(0, 2)
```

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Fourier basis transform | `QFTAlgorithm.run(n, state, inverse=False)` |
| Inverse transform | `qft.dagger()` when `inverse=True` |
| Hadamard layer | `qft.h(i)` inside descending loop |
| Controlled phase rotations | `qft.mcp(np.pi / 2**(i-j), j, i)` |
| Bit reversal | `qft.swap(i, n - 1 - i)` |
| Optional input state | `qc.initialize(state, range(n))` after normalization |
| Simulator execution | `qc.execute(backend=backend, device=device, dtype=dtype).state` |
| Classical QFT reference | `ifft(state) * np.sqrt(2**n)` |
| Classical IQFT reference | `fft(state) / np.sqrt(2**n)` |
| Error metric | `np.linalg.norm(final_state - expected_state)` |

## Minimal Manual Implementation

```python
import numpy as np
from unitarylab.core import Circuit

def build_qft(n: int, inverse: bool = False) -> Circuit:
    qft = Circuit(n, name="QFT")
    for i in range(n - 1, -1, -1):
        qft.h(i)
        for j in range(i - 1, -1, -1):
            qft.mcp(np.pi / 2 ** (i - j), j, i)
    for i in range(n // 2):
        qft.swap(i, n - 1 - i)

    if inverse:
        qft = qft.dagger()
        qft.update_name("IQFT")
        qft.gate_sequence.update_name("IQFT")
    return qft

def run_qft_state(state, inverse: bool = False, backend: str = "torch"):
    state = np.asarray(state, dtype=complex)
    n = int(np.log2(state.size))
    if state.size != 2**n:
        raise ValueError("State length must be a power of two.")

    state = state / np.linalg.norm(state)
    qc = Circuit(n, name="QFT Example")
    qc.initialize(state, range(n))
    qc.append(build_qft(n, inverse=inverse), range(n))
    final = qc.execute(backend=backend).state

    expected = np.fft.fft(state) / np.sqrt(2**n) if inverse else np.fft.ifft(state) * np.sqrt(2**n)
    return final, expected, np.linalg.norm(final - expected)
```

## Debugging Tips

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ValueError: Initial state vector must be a 1D array of size 2**n` | `state` length does not match `n` | Set `n = int(log2(len(state)))` or resize the state vector. |
| Large verification error | QFT convention mismatch, missing swaps, or using FFT where iFFT is expected | Match the implementation convention: QFT uses `ifft * sqrt(N)`, IQFT uses `fft / sqrt(N)`. |
| Output appears reversed | Bit-reversal swaps were removed or interpreted differently | Keep the final `swap(i, n-1-i)` block. |
| State amplitudes unexpectedly scaled | Input vector was not normalized before comparison | Normalize the state before both circuit initialization and NumPy verification. |
| Circuit name still says QFT for inverse run | Forgot to rename after `dagger()` | Use `update_name("IQFT")` and `gate_sequence.update_name("IQFT")` as in the implementation. |

## Understanding the Key Quantum Components

1. **Hadamard layer**: Each qubit $i$ receives a Hadamard gate `h(i)`, creating the local Fourier superposition $|0\rangle \to (|0\rangle + |1\rangle)/\sqrt{2}$.
2. **Controlled phase rotations**: Multi-controlled phase gates `mcp(pi/2**(i-j), j, i)` apply relative phase $e^{2\pi i / 2^{i-j+1}}$ between basis states. The angle decreases exponentially with qubit separation $i-j$.
3. **Bit-reversal swaps**: The natural QFT construction produces output in bit-reversed order. Final `swap(i, n-1-i)` gates restore the canonical ordering.
4. **Inverse QFT via dagger**: Calling `qft.dagger()` inverts all gates (Hadamards are self-inverse, phases negate, swaps reverse). The circuit is renamed to `IQFT`.
5. **State initialization**: An optional input statevector can be loaded via `qc.initialize(state, range(n))`. The state is normalized internally before initialization.

## Mathematical Deep Dive

The QFT on $n$ qubits maps computational basis state $|x\rangle$ to:

$$|x\rangle \mapsto \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n-1} e^{2\pi i x y / 2^n} |y\rangle$$

For a general state $|\psi\rangle = \sum_x \alpha_x |x\rangle$, the QFT produces $\sum_y \tilde{\alpha}_y |y\rangle$ where $\tilde{\alpha}_y = \frac{1}{\sqrt{2^n}} \sum_x \alpha_x e^{2\pi i x y / 2^n}$ — the discrete Fourier transform of the amplitudes.

**Gate decomposition** for qubit $i$ (descending from $n-1$ to $0$):

1. Hadamard on qubit $i$: $H_i$
2. Controlled phase rotations from lower-index control qubits: $R_k^{ji}$ with angle $\pi/2^k$ for $k = i-j$

The controlled phase used by the source applies an angle of $\pi/2^k$ through `qc.mcp(np.pi/2**k, ...)`:

$$R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi / 2^k} \end{pmatrix}$$

For $n=3$, the forward circuit structure (before swaps) is:
$$H_2 \cdot R_1^{12} \cdot R_2^{02} \cdot H_1 \cdot R_1^{01} \cdot H_0$$

**Verification convention**: The implementation uses:
- QFT: `ifft(state) * sqrt(2**n)` (NumPy convention matches the implemented phase signs)
- IQFT: `fft(state) / sqrt(2**n)`

The normalization factor $\sqrt{2^n}$ accounts for the difference between the quantum state normalization ($\sum |\alpha_x|^2 = 1$) and the classical DFT convention.

## Hands-On Example

Verify QFT and IQFT are exact inverses by applying both in sequence:

```python
import numpy as np
from unitarylab_algorithms.linear_algebra.qft.algorithm import QFTAlgorithm

algo = QFTAlgorithm(text_mode="plain")

# Create a non-trivial input state
state = np.array([0.3+0.1j, 0.5-0.2j, -0.4+0.3j, 0.2-0.6j,
                  0.1+0.4j, -0.3-0.1j, 0.6+0.2j, -0.1+0.5j], dtype=complex)

# Forward QFT
result_fwd = algo.run(n=3, state=state, inverse=False, backend="torch")
fwd_state = result_fwd["Final state"]

# Inverse QFT on the transformed state
result_inv = algo.run(n=3, state=fwd_state, inverse=True, backend="torch")
recovered = result_inv["Final state"]

# Check round-trip fidelity
fidelity = np.abs(np.vdot(recovered, state / np.linalg.norm(state)))**2
print(f"Round-trip fidelity: {fidelity:.10f}")
print(f"Verification error (fwd): {result_fwd['Verification error']:.2e}")
print(f"Verification error (inv): {result_inv['Verification error']:.2e}")
```

Expected behavior: round-trip fidelity should be $> 0.9999$; forward and inverse verification errors should be near machine precision.

## Reference Implementation

PennyLane's `QFT` operation is useful as a reference for matrix conventions, decomposition, and resource counts. In this repository, however, the primary implementation path is **UnitaryLab** through `QFTAlgorithm`, `Circuit`, and simulator verification against NumPy.

PennyLane's decomposition uses the same conceptual pieces:
- Hadamard gates.
- Controlled phase shifts with angles $2\pi / 2^k$.
- Final SWAP gates for bit order.
- `qml.adjoint(qml.QFT)` for inverse QFT.

Minimal PennyLane reference:

```python
import pennylane as qml
import numpy as np

n = 3
dev = qml.device("default.qubit", wires=n)

@qml.qnode(dev)
def circuit():
    qml.BasisState(np.array([1, 0, 0]), wires=range(n))
    qml.QFT(wires=range(n))
    return qml.state()

state = circuit()
```

Use this reference only for conceptual comparison or cross-framework checks. For this skill, examples, debugging, and production guidance should stay centered on `unitarylab_algorithms.linear_algebra.qft.algorithm.QFTAlgorithm`.
