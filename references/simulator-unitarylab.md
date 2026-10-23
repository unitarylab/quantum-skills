
# UnitaryLab

## Purpose

This SDK reference covers circuit construction, execution, result queries, circuit analysis,
serialization, CUDA-Q interoperability, compilation, chemistry, native QML, and classical Ising optimization.
Algorithm principles and wrapper-specific workflows belong in the matching algorithm reference.
Follow [SKILL.md](../SKILL.md) for task routing, framework selection, and execution permission.

## Sources and Version

[UnitaryLab 1.5.1 on PyPI](https://pypi.org/project/unitarylab/1.5.1/) requires Python `>=3.10,<3.13`.

The release links to the [official simulator manual](https://docs.unitarylab.com/en/docs/unitarylab-simulator-user-manual/).

## Installation

For an authorized environment setup, install the selected release and only the optional extras the task needs:

```bash
python -m pip install "unitarylab==1.5.1"
python -m pip install "unitarylab[cuda]==1.5.1"       # optional CUDA 12 C++ backends
python -m pip install "unitarylab[chemistry]==1.5.1"  # optional molecular workflow
```

Inspect existing package metadata before changing an environment. If an import check is requested, the following is a runtime check, not a static version check:

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

For the statevector `ExecutionResult` used here, probability strings place qubit 0 on the right.

## Routing

Read only the reference needed for the request:

| User need | Reference |
|---|---|
| Build, choose gate storage, execute, measure/sample, optimize/transpile, or import/export OpenQASM/Python/CUDA-Q circuits | [references/circuit-building.md](simulator-unitarylab-references-circuitsbuild.md) |
| Look up a core signature, return field, gate, backend, or `unitarylab.library` API | [references/api-reference.md](simulator-unitarylab-references-api-reference.md) |
| Circuit depth, gate counts, layers, coupling map, parameters, or qubit history | [references/circuit-info.md](simulator-unitarylab-references-CircuitInfo.md) |
| Mixed states, density matrices, Kraus noise, or density CUDA | [references/density-matrix-and-noise.md](simulator-unitarylab-references-density-matrix-and-noise.md) |
| Molecular electronic structure or DMRG/CVD workflow | [references/chemistry.md](simulator-unitarylab-references-chemistry.md) |
| Local compilation, mapping, hardware providers, or submission | [references/compilation-and-hardware.md](simulator-unitarylab-references-compilation-and-hardware.md) |
| Native `unitarylab.library.qml` models, quantum kernels, training, gradients, or checkpoints | [references/qml.md](simulator-unitarylab-references-qml.md) |
| Classical Ising/QUBO heuristics through `unitarylab.library.qubocore` | [references/qubocore.md](simulator-unitarylab-references-qubocore.md) |

For QFT, QPE, QSP, QSVT, LCU, linear solving, Hamiltonian simulation, or equation algorithms,
consult the matching algorithm reference in addition to the API reference. `unitarylab.library` belongs to this SDK; `unitarylab_algorithms` is a separate package.

## Backend Selection

| Need | Selection |
|---|---|
| Broad compatibility | `backend="torch", device="cpu"` |
| NumPy-native dense state | `backend="numpy", device="cpu"` |
| Faster native CPU execution | `backend="cpp", device="cpu"` when the native extension is available |
| NVIDIA CUDA statevector | `backend="cpp", device="gpu"` with `unitarylab[cuda]` |
| Larger low-entanglement circuit | `backend="tensornet", device="cpu"` |
| Mixed state or gate-local noise | `circuit.execute_density(...)` on CPU or CUDA |

For the documented examples, use `np.complex128` unless the device or memory budget requires `complex64`. Torch on Apple MPS rejects complex128 and the current implementation limits that path to 16 qubits. TensorNet options belong only to the TensorNet backend.

## Safety and Boundaries

- Treat chemistry as an optional dependency and allow its documented missing-extra error to surface.
- Density-matrix execution supports circuit measurement and shots/counts; final state and classical snapshots describe the final shot.
- CUDA-Q source generation needs no CUDA-Q installation; running it or calling `to_cudaq()` does.
- Prefer targeted result queries over dense state materialization for large TensorNet jobs.
- Hardware submission returns a `HardwareJob`; `job.result()` queries once and returns a `HardwareResult`. See the [hardware reference](simulator-unitarylab-references-compilation-and-hardware.md) for the distinction.
- Example outputs are theoretical or documented expectations unless actual execution evidence is supplied.

## Common Failures

| Symptom | First action |
|---|---|
| Optional CUDA import fails | Install `unitarylab[cuda]` only when CUDA execution is required, or use CPU. |
| Chemistry import fails | Install `unitarylab[chemistry]` only when the molecular workflow is required. |
| Unsupported backend/device/dtype | Select a compatible row from Backend Selection. |
| Invalid circuit shots or seed | Require a positive integer for shots and a non-negative integer or `None` for seed. |
| Result interpretation is ambiguous | Check little-endian bit strings and Pauli-to-qubit order. |
| Hardware request lacks target or credentials | Stop before submission and request the missing external details. |
