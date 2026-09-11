---
name: unitarylab
description: Use UnitaryLab for runnable quantum-circuit construction, statevector or density-matrix simulation, noise, measurement, expectations, transpilation, hardware compilation, and chemistry. Consult the routed reference for version-sensitive APIs and use dedicated algorithm skills for algorithm-specific workflows.
---

# UnitaryLab

## Purpose

Use UnitaryLab for concrete circuit construction, local execution, result queries, circuit analysis,
serialization, compilation, and chemistry. Keep algorithm theory and algorithm-specific
workflows in the relevant quantum algorithm skill.

## Installation

UnitaryLab supports Python 3.10–3.12. Install only when execution or import verification is required:

```bash
python -m pip install unitarylab
python -m pip install "unitarylab[cuda]"       # optional CUDA 12 C++ backends
python -m pip install "unitarylab[chemistry]"  # optional molecular workflow
```

Check the active environment before installing:

```bash
python -c "from unitarylab import Circuit; print('OK')"
```

The main package supports Windows, Linux, and macOS. CUDA requires a compatible NVIDIA driver and
is supported on Linux x86_64/aarch64 server environments and Windows x86-64. Chemistry currently
supports Linux and macOS.

## Minimal Example

```python
from unitarylab import Circuit

circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
result = circuit.execute(backend="numpy")

print(result.state)
print(result.probabilities)
```

Probability bit strings are little-endian: qubit 0 is the rightmost bit.

## Routing

Read only the reference needed for the request:

| User need | Reference |
|---|---|
| Build, execute, measure, sample, optimize, transpile, serialize, or import/export OpenQASM 2/3 circuits | [references/circuit-building.md](references/circuit-building.md) |
| Look up a core signature, return field, gate, backend, or `unitarylab.library` API | [references/api-reference.md](references/api-reference.md) |
| Circuit depth, gate counts, layers, coupling map, parameters, or qubit history | [references/circuit-info.md](references/circuit-info.md) |
| Mixed states, density matrices, Kraus noise, or density CUDA | [references/density-matrix-and-noise.md](references/density-matrix-and-noise.md) |
| Molecular electronic structure or DMRG/CVD workflow | [references/chemistry.md](references/chemistry.md) |
| Local compilation, mapping, hardware providers, or submission | [references/compilation-and-hardware.md](references/compilation-and-hardware.md) |

For QFT, QPE, QSP, QSVT, LCU, linear solving, Hamiltonian simulation, or equation algorithms,
consult the matching algorithm skill in addition to the API reference.

## Backend Selection

| Need | Selection |
|---|---|
| Broad compatibility | `backend="torch", device="cpu"` |
| NumPy-native dense state | `backend="numpy", device="cpu"` |
| Faster native CPU execution | `backend="cpp", device="cpu"` when the native extension is available |
| NVIDIA CUDA statevector | `backend="cpp", device="gpu"` with `unitarylab[cuda]` |
| Larger low-entanglement circuit | `backend="tensornet", device="cpu"` |
| Mixed state or gate-local noise | `circuit.execute_density(...)` on CPU or CUDA |

Use `np.complex128` unless the device or memory budget requires `complex64`. Torch on Apple MPS
rejects complex128 and the current implementation limits that path to 16 qubits. TensorNet options
belong only to the TensorNet backend.

## Safety and Boundaries

- Execute locally only when requested or needed for verification.
- Before a real-hardware submission, obtain or confirm the provider, machine, credentials, shots,
  quota, and possible charges. Never invent credentials or submit as part of documentation testing.
- Treat chemistry as an optional dependency and allow its documented missing-extra error to surface.
- Density-matrix execution currently has no circuit measurement or shots/counts.
- Prefer targeted result queries over dense state materialization for large TensorNet jobs.
- Verify unfamiliar signatures against the implementation or focused tests; code and tests take
  precedence over README prose when they disagree.

## Common Failures

| Symptom | First action |
|---|---|
| Optional CUDA import fails | Install `unitarylab[cuda]` only when CUDA execution is required, or use CPU. |
| Chemistry import fails | Install `unitarylab[chemistry]` only when the molecular workflow is required. |
| Unsupported backend/device/dtype | Select a compatible row from Backend Selection. |
| Invalid shots or seed | Require a positive integer for shots and a non-negative integer or `None` for seed. |
| Result interpretation is ambiguous | Check little-endian bit strings and Pauli-to-qubit order. |
| Hardware request lacks target or credentials | Stop before submission and request the missing external details. |
