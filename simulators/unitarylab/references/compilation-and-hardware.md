# Compilation and Hardware

Use this reference for circuit optimization, local compilation, mapping, and provider submission.

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

The five compiler types are lazy top-level exports. `unitarylab.qcompiler` additionally exports
`RunResult`, `START_POINTS`, `STOP_POINTS`, and `run_pipeline` for advanced staged workflows.

## Circuit Facade

```python
circuit = Circuit(2)
circuit.h(0)
circuit.cx(0, 1)

optimized = circuit.optimize(optimization_level=1)
transpiled = circuit.transpile(basis="default")

compiled = circuit.compile(
    provider="local",
    machine=None,
    token=None,
)
print(compiled.circuit)
print(compiled.executable)
print(compiled.executable_format)
```

`CompilationResult` contains `circuit` (also available as `mapped_circuit`), `executable`,
`executable_format`, optional optimized circuit/machine/layouts, emitted QASM, and stage metadata.
Compilation does not itself submit a hardware task.

## Configuration

`CompileConfig.from_dict(...)`, `.from_json_file(...)`, and `.coerce(...)` validate configuration.
The major sections are `circuit`, `optimization`, `hardware`, `mapping`, `execution`, `measurement`,
and `output`. Exactly one QASM/QCIS inline or file source is required unless
`require_source=False`. Strategy is `rule_based` or `synthesis`; output QASM version is 2 or 3.

Native translation belongs under `mapping`, and mitigation belongs under `execution.mitigation`;
the removed top-level `native` and `mitigation` sections raise errors.

## Real Hardware Submission

```python
TOKEN = "your-provider-token"  # Replace with authorized credentials at runtime.

job = circuit.submit(
    provider="guodun",
    machine="gd_qc1",  # Guodun "Xiaohong-1" / 国盾量子“骁鸿一号”
    shots=1024,
    token=TOKEN,
)
hardware_result = job.result()  # Query once; call again later if unfinished.
```

Before calling `submit()`:

- Confirm that the user wants a real external submission.
- Confirm provider, exact machine, shots, credentials or documented credential environment
  variable, quota, and possible cost.
- Never log, embed, or invent credentials.
- Expect provider SDK, network, authentication, queue, and remote-job failures.

`Circuit.submit()` and `QCompiler.submit()` return a `HardwareJob` without waiting for execution.
Each `job.result()` call performs one provider query and returns a `HardwareResult`; it does not
poll until completion. An unfinished result contains empty distributions and a `message`.
Inspect `status` and `message`, then call `job.result()` again later when another query is needed.

`HardwareJob` exposes `provider`, `machine`, `job_ids`, `status`, and `metadata`.
`HardwareResult` provides `raw_counts`, `raw_probabilities`, `corrected_counts`,
`corrected_probabilities`, and provider/stage metadata. Provider-specific capabilities and result
availability vary; do not assume the local `ExecutionResult` shape or treat a pending empty
distribution as a completed measurement result.

Use local compilation or mocked provider tests for documentation verification; never submit a live
job merely to validate the skill.
