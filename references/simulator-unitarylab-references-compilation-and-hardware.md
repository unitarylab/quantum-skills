# Compilation and Hardware

Follow [SKILL.md](../SKILL.md) for task routing and execution permission; see the [SDK version and APIs](simulator-unitarylab.md#sources-and-version).

This reference distinguishes circuit transforms, local compilation, external submission, and result queries.

## Public Entry Points

```python
from unitarylab import (
    Circuit,
    CompilationResult,
    CompileConfig,
    HardwareJob,
    HardwareResult,
    QCompiler,
)
```

The compiler/job/result types are lazy top-level exports. `unitarylab.qcompiler` additionally exports `RunResult`, `START_POINTS`, `STOP_POINTS`, and `run_pipeline` for staged workflows.

## Local Circuit Workflow

```python
circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)

optimized = circuit.optimize(optimization_level=1)
transpiled = circuit.transpile(basis="default")
compiled = circuit.compile(provider="local", machine=None, token=None)

print(compiled.circuit)
print(compiled.executable)
print(compiled.executable_format)
```

This snippet continues the imports above. `optimize()` and `transpile()` return circuits. `compile()` returns `CompilationResult`; it does not submit a hardware job. Provider-specific compilation may still require network access or credentials, so use `provider="local"` for a local-only compilation request.

`CompilationResult` contains `circuit` (also `mapped_circuit`), `executable`, `executable_format`, optional optimized circuit/machine/layouts, emitted QASM, and stage metadata. Do not treat it as a simulation result.

## Configuration

`CompileConfig.from_dict(...)`, `.from_json_file(...)`, and `.coerce(...)` validate configuration. Sections include `circuit`, `optimization`, `hardware`, `mapping`, `execution`, `measurement`, and `output`. Exactly one QASM/QCIS inline or file source is required unless `require_source=False`. Strategy is `rule_based` or `synthesis`; output QASM version is 2 or 3.

Native translation belongs under `mapping`, and mitigation under `execution.mitigation`. Top-level `native` and `mitigation` are not supported configuration sections.

## Submission and Result Queries

The following is an external-operation example. `provider_token` must come from the user's authorized credential configuration; the snippet does not define or embed a token.

```python
job = circuit.submit(
    provider="guodun",
    machine="gd_qc1",
    shots=1024,
    token=provider_token,
)
hardware_result = job.result()
```

In latest unitarylab SDK, `submit()` returns a `HardwareJob` without waiting for execution. Each `job.result()` queries once and returns a `HardwareResult`; unfinished results have empty distributions and a message. Do not describe the job as the completed result or assume `.result()` waits until completion.

Provider availability, machine selection, quota, costs, and returned data depend on the provider. Follow the real-hardware authorization requirements in the main skill. Handle network/authentication/queue failures according to their actual stage; a remote result does not share the local `ExecutionResult` contract.
