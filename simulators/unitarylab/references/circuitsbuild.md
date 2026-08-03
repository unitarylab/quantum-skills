# UnitaryLab Quantum Circuit Agent Guide

Use this reference after routing a circuit task from [SKILL.md](../SKILL.md). Read [api-reference.md](api-reference.md) only for simulator signatures and parameter definitions.

## Workflow

1. Derive the required qubits, gates, measurements, backend constraints, and output.
2. Build the circuit in program order.
3. Choose statevector or TensorNet execution.
4. Validate execution parameters and run only when requested.
5. Query only the result data needed by the user.

## Create the Circuit

Use the simplest constructor that preserves the request:

    import numpy as np
    from unitarylab import Circuit, Register, ClassicalRegister

    circuit = Circuit(2)

    q = Register("q", 2)
    c = ClassicalRegister("c", 2)
    measured_circuit = Circuit(q, c)

    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    prepared_circuit = Circuit(state)

For Circuit(state_vector), pass a one-dimensional, normalized np.ndarray with length 2**n; the constructor does not accept a Python list as a statevector. Use explicit registers when measurement results need named classical storage.

Add gates in program order. Preserve argument conventions such as rx(angle, target), cx(control, target), and mcx(controls, target). Consult the API reference for other gates.

For subcircuits:

- append() and prepend() mutate the receiving circuit.
- copy(), inverse(), dagger(), reverse(), repeat(), control(), decompose(), and transpile() return new circuits.

## Execute

    result = circuit.execute(
        initial_state=None,
        backend="torch",
        device="cpu",
        dtype=np.complex128,
        shots=1,
        seed=42,
        backend_options=None,
    )

Only specify non-default arguments required by the task.

Validate before execution:

- shots: positive integer.
- seed: non-negative integer or None.
- initial_state: normalized dense statevector of the circuit dimension; the TensorNet backend also accepts TensorNetState or an open-boundary MPS tensor sequence.
- backend_options: TensorNet only.

## Handle ExecutionResult

| Need | Result member |
|---|---|
| Dense statevector | result.state |
| Native backend state | result.backend_state |
| Full distribution | result.probabilities |
| One probability | result.probability(bitstring, qubits=None) |
| Selected-qubit distribution | result.marginal_probabilities(qubits=None) |
| Non-collapsing samples | result.sample(shots, qubits=None, seed=None) |
| Projective measurement | result.measure(qubits, seed=None) |
| Observable expectation | result.expectation(observable, qubits=None) |
| Circuit measurement counts | result.counts |
| Final-shot classical values | result.classical_results_map and result.classical_registers |

Basis labels are little-endian: qubit 0 is least significant.

Avoid full state or probabilities for large TensorNet results when a targeted query is sufficient.

## Measurement and Sampling

Use circuit measurement when the user needs classical counts or register values:

    q = Register("q", 2)
    c = ClassicalRegister("c", 2)
    circuit = Circuit(q, c)
    circuit.h(q[0])
    circuit.cx(q[0], q[1])
    circuit.measure(q[0:2], c[0:2])

    result = circuit.execute(shots=1000, seed=42)
    print(result.counts)
    print(result.classical_registers)

Rules:

- Map equal numbers of qubits and classical bits.
- Do not map the same qubit or classical bit twice.
- counts aggregates all shots.
- classical_results_map and classical_registers describe the final shot.
- Unmeasured bits use # in count keys and -1 in register snapshots.
- Use sample() when collapse and classical storage are unnecessary.

## Expectation Values

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result = circuit.execute(backend="numpy")

    print(result.expectation("ZZ"))

    hamiltonian = [
        (0.5, "ZZ", (0, 1)),
        {"coeff": -0.25, "pauli": "X", "qubits": (0,)},
    ]
    print(result.expectation(hamiltonian))

Use Pauli strings, a one-qubit Hermitian 2x2 matrix, or weighted tuple/mapping terms. Pauli labels are I, X, Y, and Z.

## Choose a Backend

| Situation | Choice |
|---|---|
| Default local execution | backend="torch", device="cpu" |
| Dense NumPy statevector | backend="numpy", device="cpu" |
| Available native extension | backend="cpp", device="cpu" |
| MPS / low-entanglement circuit | backend="tensornet", device="cpu" |
| Compatible accelerator | Torch with device="gpu" |

Use np.complex128 by default. Use complex64 when required by the device or memory budget; do not rely on implicit promotion from integer, real, or bool dtypes. Apple MPS rejects complex128 and is limited to at most 16 qubits in the current implementation.

Do not select C++ unless the native cppgates extension is available.

## TensorNet

    from unitarylab.backend.tensornet import TensorNetState

    dense_state = np.array([1, 0, 0, 0], dtype=np.complex128)
    mps_state = TensorNetState.from_statevector(dense_state, max_bond=64)

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    result = circuit.execute(
        initial_state=mps_state,
        backend="tensornet",
        backend_options={
            "max_bond": 64,
            "cutoff": 1e-10,
            "routing": "auto",
        },
    )

TensorNet rules:

- initial_state may be a normalized dense statevector, TensorNetState, or an open-boundary MPS tensor sequence.
- Each MPS site tensor uses (left_bond, 2, right_bond) order, and the first tensor represents qubit 0.
- Routing is auto, swap, or mpo.
- Explicit execution options override max_bond and cutoff stored in the input state.
- Convert MPS inputs with to_statevector() before using a statevector backend.
- Prefer backend_state, targeted probability, sampling, and expectation over dense contraction.

## Transpile, QASM, and Serialization

Use transpilation for a target basis, circuit lowering, or export preparation:

    circuit = Circuit(3)
    circuit.h(0)
    circuit.mcx([0, 1], 2)
    transpiled = circuit.transpile(basis="default")

Use serialization according to the requested output:

    serializable = Circuit(2)
    serializable.rx(0.5, 0)
    serializable.cx(0, 1)

    qasm3 = serializable.to_qasm()
    qasm2 = serializable.to_qasm2()
    restored = Circuit.from_qasm(qasm3)
    python_source = serializable.to_python(variable_name="generated")

- to_qasm(): OpenQASM 3.0.
- to_qasm2(): OpenQASM 2.0.
- from_qasm(): detect QASM 2 or 3.
- OpenQASM 2 parsing currently ignores measurement statements; use OpenQASM 3 when measurement-to-classical-bit mappings must survive a round trip.
- File workflows: to_qasm_file(), from_qasm_file(), and to_python_file().
- For supported composite gates, retry QASM export with decompose=True and transpile=True.
- Python source generation supports only the native gates documented in the API reference.

## Common Errors

| Problem | Agent response |
|---|---|
| Invalid qubit or overlapping control/target | Validate indexes and keep controls separate from targets. |
| Invalid statevector | Require one dimension, unit norm, and length 2**n. |
| Non-unitary custom matrix | Validate dimension, finite values, and unitarity. |
| Control-state width mismatch | Match control_state width to the controls. |
| Invalid shots or seed | Apply the execute validation rules before running. |
| C++ backend import failure | Fall back to NumPy or Torch CPU. |
| TensorNet options on another backend | Remove them or select TensorNet. |
| MPS input on a statevector backend | Select TensorNet or convert to a statevector. |
| Large TensorNet result becomes slow | Avoid dense state and full probabilities. |
| QASM export rejects a gate | Try supported decomposition/transpilation; otherwise report the unsupported operation. |
