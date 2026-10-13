# UnitaryLab Circuit Workflow

Use this reference for circuit construction and ordinary statevector execution. Read
[density-matrix-and-noise.md](density-matrix-and-noise.md) for mixed states and
[compilation-and-hardware.md](compilation-and-hardware.md) for compiler or hardware work.

## Build

```python
import numpy as np
from unitarylab import Circuit, ClassicalRegister, Register

circuit = Circuit(2)

q = Register("q", 2)
c = ClassicalRegister("c", 2)
measured = Circuit(q, c)

prepared = Circuit(np.array([1, 0, 0, 0], dtype=np.complex128))
```

`Circuit(statevector)` requires a normalized, one-dimensional NumPy array of power-of-two length;
a Python list is not accepted. Gate methods mutate the circuit and return `None`. Preserve argument
order such as `rx(angle, target)`, `cx(control, target)`, and `mcx(controls, target)`.

`append()` and `prepend()` mutate the receiver. `copy()`, `inverse()`, `dagger()`, `reverse()`,
`repeat()`, `control()`, `decompose()`, `optimize()`, and `transpile()` return circuits.

### Gate Storage

`Circuit(..., gate_storage="python")` is the default. `gate_storage="packed"` uses
the native C++ `GateSequencePacked` store and requires the packaged C++ extension:

```python
packed = Circuit(2, gate_storage="packed")
packed.h(0)
packed.cx(0, 1)
```

Storage is independent of the execution backend; packed storage does not imply CUDA execution.
`circuit.gate_storage` identifies the store and `circuit.data()` returns its gate sequence.
`Circuit.from_qasm(..., gate_storage="packed")` and `from_qasm_file()` accept the same option.

## Execute

```python
result = circuit.execute(
    initial_state=None,
    backend="torch",
    device="cpu",
    dtype=np.complex128,
    shots=1,
    seed=42,
    backend_options=None,
)
```

- `shots` is a positive integer; `seed` is a non-negative integer or `None`.
- Each shot starts from an independent copy of `initial_state`.
- `state`, `classical_results_map`, and `classical_registers` describe the final shot; `counts`
  aggregates all shots.
- `backend_options` is TensorNet-only and accepts `max_bond`, `cutoff`, and routing `auto`, `swap`,
  or `mpo`.

## Query Results

| Need | Member |
|---|---|
| Dense statevector | `result.state` |
| Backend-native state | `result.backend_state` |
| Full distribution | `result.probabilities` |
| One probability | `result.probability(bitstring, qubits=None)` |
| Marginal distribution | `result.marginal_probabilities(qubits=None, threshold=0.0)` |
| Non-collapsing samples | `result.sample(shots=1, qubits=None, seed=None)` |
| Projective measurement | `result.measure(qubits, seed=None)` |
| Normalized expectation | `result.expectation(observable, qubits=None)` |

Probability strings are little-endian. A Pauli string instead maps character-by-character to the
explicit `qubits` sequence, or to all qubits in ascending order when omitted.

```python
circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
result = circuit.execute(backend="numpy")

assert result.probability("00") == result.probability("11")
print(result.marginal_probabilities(qubits=[0]))
print(result.sample(shots=8, seed=7))
print(result.expectation("ZZ"))
```

Observable inputs may be Pauli strings, a finite Hermitian 2x2 matrix on one qubit, or a list of
weighted tuple/mapping terms.

## Circuit Measurement

```python
q = Register("q", 2)
c = ClassicalRegister("c", 2)
circuit = Circuit(q, c)
circuit.h(q[0])
circuit.cx(q[0], q[1])
circuit.measure(q[0:2], c[0:2])

result = circuit.execute(shots=1000, seed=42)
print(result.counts)
print(result.classical_results_map)
print(result.classical_registers)
```

Map equal numbers of qubits and classical bits and do not map either twice. Unmeasured classical
bits use `#` in count keys and `-1` in register snapshots. Use `sample()` when classical storage and
collapse are unnecessary.

## Backends

| Backend | Device | Native result | Notes |
|---|---|---|---|
| `torch` | `cpu`, `gpu` | Torch tensor | Apple MPS uses `device="gpu"`, requires complex64, and is limited to 16 qubits. |
| `numpy` | `cpu` | NumPy array | Dense and portable. |
| `cpp` | `cpu` | Native extension | Use when packaged native gates are available. |
| `cpp` | `gpu` | CUDA extension | Requires `unitarylab[cuda]` and compatible NVIDIA CUDA 12 environment. |
| `tensornet` | `cpu` | `TensorNetState` | MPS backend for low-entanglement circuits. |

For TensorNet inputs, site tensors use `(left_bond, 2, right_bond)` and the first tensor represents
qubit 0. Statevector backends do not implicitly contract an MPS; call `to_statevector()` first.
Avoid `result.state` and full `probabilities` when targeted queries suffice, because they may require
dense contraction.

## Transform, Analyze, and Serialize

```python
optimized = circuit.optimize(optimization_level=1)
transpiled = circuit.transpile(basis="default")
info = circuit.analyze(show=False)

qasm3 = circuit.to_qasm()
qasm2 = circuit.to_qasm2()
restored = Circuit.from_qasm(qasm3)

exportable = Circuit(2)
exportable.h(0)
exportable.cx(0, 1)
python_source = exportable.to_python(variable_name="generated")
cudaq_source = exportable.to_python(framework="cudaq", variable_name="generated")
```

- `to_qasm()` emits OpenQASM 3; `to_qasm2()` emits OpenQASM 2.
- `from_qasm()` detects the version. OpenQASM 2 parsing currently ignores measurements, so use
  OpenQASM 3 when classical mappings must survive a round trip.
- File variants are `to_qasm_file()`, `from_qasm_file()`, and `to_python_file()`.
- `to_python(framework="unitarylab")` preserves ordinary gates, registers, measurements,
  nested controlled blocks, and custom unitary matrices. Internal gates without a direct
  circuit method use a `QuantumGate` reconstruction. Malformed or unsupported input can still fail.
- `to_python()` and `to_python_file()` accept keyword-only `framework`, `circuit_name`,
  and `variable_name`; they do not accept QASM's `decompose`/`transpile` options.

### CUDA-Q

`to_python(framework="cudaq")` generates source without importing CUDA-Q.
`to_python_file("generated.py", framework="cudaq")` writes that source and returns a `Path`.
`to_cudaq()` instead imports the optional `cudaq` package and returns a non-parameterized
`cudaq.Kernel` with the circuit's current numerical parameters fixed.

The exporter lowers supported gates and nested controlled blocks, including zero-state controls.
Arbitrary unitary gates are limited to one or two target qubits. Measurements may be absent;
when present, they must be terminal and cover every qubit exactly in order
`q0 -> c0, q1 -> c1, ...`. Partial, reordered, controlled, or mid-circuit measurements raise
`NotImplementedError`. These CUDA-Q limits do not restrict UnitaryLab Python export.
Run generated code only when the user authorizes execution.

## Failure Guide

| Problem | Response |
|---|---|
| Invalid statevector | Check rank, length `2**n`, finite values, and unit norm. |
| Invalid custom unitary | Check square shape `2**len(target)`, finite values, and unitarity. |
| C++/CUDA extension missing | Use Torch/NumPy CPU or install the appropriate optional extra. |
| TensorNet options on another backend | Remove them or select TensorNet. |
| QASM rejects an operation | Decompose/transpile when supported; otherwise report it as unsupported. |
