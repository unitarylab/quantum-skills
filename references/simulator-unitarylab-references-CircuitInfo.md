# Circuit Analysis Reference

Follow [SKILL.md](../SKILL.md) for task routing and execution permission; see the [SDK version and APIs](simulator-unitarylab.md#sources-and-version).

Use `Circuit.analyze()` for static circuit structure, not simulation results.

## Minimal Workflow

```python
from unitarylab import Circuit

circuit = Circuit(3)
circuit.h(0)
circuit.cx(0, 1)
circuit.rz(0.5, 2)

info = circuit.analyze(show=False)
print(info.get_summary())
```

`analyze()` always returns a `CircuitInfo`. Set `show=False` for programmatic use; `sections` and
`qubit` only control formatted output when `show=True`.

## Structured Queries

| Need | Call |
|---|---|
| Qubits, size, depth, histogram, usage, and coupling summary | `info.get_summary()` |
| Gate count | `info.size()` |
| Parallel depth | `info.depth()` |
| Gate histogram | `info.count_ops()` |
| Instruction records | `info.get_instructions()` |
| Parallel layers | `info.get_layers()` |
| Per-qubit usage | `info.get_qubit_usage()` |
| Undirected multi-qubit pairs | `info.get_coupling_map()` |
| Parameter values | `info.get_parameters()` |
| Parameter presence | `info.is_parameterized()` |
| One qubit's history | `info.get_qubit_history(qubit)` |
| Complete structured result | `info.to_dict()` |

Return the smallest structure that answers the request.

## Formatted Sections

```python
circuit.analyze(sections="overview")
circuit.analyze(sections=["instructions", "layers"])
circuit.analyze(sections="qubit_history", qubit=0)
```

Supported section names are `overview`/`summary`, `instructions`, `layers`, `qubit_usage`,
`coupling_map`, `parameters`, and `qubit_history`. `qubit_history` requires a valid `qubit`.
With `sections=None`, formatted output includes overview, instructions, and layers. Unknown sections
are ignored by the formatter rather than becoming structured-query errors.

## Interpretation

- Depth groups operations into earliest non-conflicting qubit layers.
- Gate width includes target and control qubits.
- Coupling entries are undirected pairs participating in multi-qubit operations.
- Instruction records preserve targets, controls, parameters, and control state.
- `get_summary()` includes aggregate statistics, usage, and coupling information.
- `to_dict()` additionally includes instructions, layers, and parameters.
