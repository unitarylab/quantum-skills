---
name: unitarylab
description: Use UnitaryLab for local quantum circuit construction, simulation, measurement, expectation values, transpilation, drawing, serialization, and algorithms provided by unitarylab.library. Trigger for runnable UnitaryLab workflows; consult bundled references for package APIs and dedicated algorithm skills for algorithm-specific workflows.
---

# UnitaryLab

## Purpose

Use UnitaryLab when the task needs:

- Quantum circuit construction.
- Local statevector or tensor-network simulation.
- Measurement, sampling, or expectation values.
- Circuit analysis, drawing, or transpilation.
- OpenQASM or Python serialization.

Keep this file as the simulator entry point. Use the references for implementation details and do not reproduce complete API or algorithm documentation here.

## Installation Rule

Install only when code will actually be executed.

- Install for run, execute, verification, or import-error requests.
- Skip installation for explanation, circuit design, code generation, and review.

Check the active environment first:

```bash
python -c "from unitarylab import Circuit; print('OK')"
```

If the import fails and installation is authorized:

```bash
python -m pip install unitarylab
```

Use an existing virtual environment when available. Do not create or replace environments unless execution requires it.

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

Expected Bell-state probabilities are approximately `{"00": 0.5, "11": 0.5}`.

## Simulator Entry

Follow this high-level flow:

1. Convert the request into qubits, gates, measurements, execution constraints, and requested output.
2. Create a `Circuit` and add operations in program order.
3. Choose the backend according to scale and output needs.
4. Execute only when requested.
5. Return the smallest useful result and explain the little-endian basis convention when relevant.
6. Consult the references before using unfamiliar signatures or advanced features.

When running a demonstration, print at least one validation artifact such as a statevector, probability distribution, counts, or expectation value.

## Agent Decision

| User need | Route |
|---|---|
| Build or simulate a circuit | Use the workflow in [references/circuitsbuild.md](references/circuitsbuild.md). |
| Measurement counts | Use its measurement workflow with `shots` and `seed`. |
| Sampling without collapse | Use its ExecutionResult sampling workflow. |
| Observable or Hamiltonian expectation | Use its expectation workflow. |
| Large low-entanglement simulation | Use TensorNet and avoid unnecessary dense results. |
| Circuit lowering or target basis | Use transpilation guidance. |
| QASM import/export or Python generation | Use serialization guidance. |
| Simulator signature, parameter, or return-type lookup | Read [references/api-reference.md](references/api-reference.md). |
| Circuit depth, gate counts, or structure analysis | Read [references/CircuitInfo.md](references/CircuitInfo.md). |
| QFT, QPE, QSP, QSVT, LCU, block encoding, linear solving, Hamiltonian simulation, or equation algorithms | Use the `unitarylab.library` API in [references/api-reference.md](references/api-reference.md); consult the relevant algorithm skill for the full workflow. |

## Common Pitfalls

| Problem | Response |
|---|---|
| `ModuleNotFoundError: unitarylab` | Install in the active environment only when execution is required. |
| Backend or device is unavailable | Select a supported local backend; fall back to NumPy or Torch CPU when appropriate. |
| TensorNet state is sent to a statevector backend | Use TensorNet or convert the input to a statevector. |
| TensorNet options are used with another backend | Remove them or select TensorNet. |
| Invalid `shots` | Require a positive integer. |
| Invalid `seed` | Require a non-negative integer or `None`. |
| Result interpretation is ambiguous | Check little-endian basis labels and use the targeted result query described in the Agent Guide. |

## References

- [references/api-reference.md](references/api-reference.md): primary simulator and `unitarylab.library` signatures, parameters, return values, and advanced capabilities.
- [references/circuitsbuild.md](references/circuitsbuild.md): Quantum Circuit Agent Guide for construction, execution, measurement, backend selection, TensorNet, transpilation, and serialization.
- [references/CircuitInfo.md](references/CircuitInfo.md): circuit analysis, depth, gate counts, layers, coupling maps, and qubit history.

Read only the reference needed for the current task.
