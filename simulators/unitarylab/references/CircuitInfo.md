# Circuit Analysis Reference

Use circuit analysis when the user asks for circuit depth, gate counts, instruction structure, parallel layers, qubit usage, coupling relationships, parameters, or one qubit's operation history.

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

Prefer `circuit.analyze(show=False)` when the result will be processed or returned programmatically. Use `show=True` only when formatted console output is requested.

## Select the Required Data

| User need | Call |
|---|---|
| Qubit count, gate count, depth, and gate histogram | `info.get_summary()` |
| Gate count | `info.size()` |
| Circuit depth | `info.depth()` |
| Gate histogram | `info.count_ops()` |
| Instruction records | `info.get_instructions()` |
| Parallel layers | `info.get_layers()` |
| Qubit usage counts | `info.get_qubit_usage()` |
| Coupling pairs | `info.get_coupling_map()` |
| Parameterized-gate values | `info.get_parameters()` |
| Whether parameters are present | `info.is_parameterized()` |
| One qubit's operation history | `info.get_qubit_history(qubit)` |
| All structured analysis data | `info.to_dict()` |

Return the smallest structure that answers the request. Do not print every analysis section by default.

## Formatted Output

Pass one section or a collection of sections to `Circuit.analyze()`:

```python
circuit.analyze(sections="overview")
circuit.analyze(sections=["instructions", "layers"])
circuit.analyze(sections="qubit_history", qubit=0)
```

Supported sections are:

- `overview` or `summary`
- `instructions`
- `layers`
- `qubit_usage`
- `coupling_map`
- `parameters`
- `qubit_history` with an explicit `qubit`

With `sections=None`, formatted output includes overview, instructions, and layers.

## Interpretation Rules

- Depth is computed by grouping gates into layers without qubit conflicts.
- Gate width includes target and control qubits.
- Coupling-map entries are undirected qubit pairs appearing in multi-qubit operations.
- Instruction records include targets, controls, parameters, and control state.
- `get_summary()` includes aggregate statistics, qubit usage, and the coupling map.
- `to_dict()` adds instructions, layers, and parameters.

## Common Errors

| Problem | Agent response |
|---|---|
| Full analysis is printed for a narrow question | Use `show=False` and query one method. |
| `qubit_history` is requested without a qubit | Supply a valid qubit index. |
| An invalid section name produces no output | Select a supported section. |
| Analysis is confused with simulation | Use analysis for static structure and `execute()` for state evolution or measurement. |

