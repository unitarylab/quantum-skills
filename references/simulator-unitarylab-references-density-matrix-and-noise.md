# Density Matrix and Noise

Follow [SKILL.md](../SKILL.md) for task routing and execution permission; see the [SDK version and APIs](simulator-unitarylab.md#sources-and-version).

Use this reference for mixed-state simulation and gate-local single-qubit noise. The `Circuit.execute_density()` Python source accepts `shots` and `seed` and aggregates measurement counts.

## Public API

```python
from unitarylab import Circuit
from unitarylab.backend import (
    AmplitudeDamping,
    BitFlip,
    DensityMatrixExecutor,
    DensityMatrixResult,
    DensityMatrixState,
    DepolarizingNoise,
    NoiseModel,
    NoiseOperator,
    PhaseDamping,
    PhaseFlip,
)
```

`Circuit.execute_density(initial_state=None, device="cpu", dtype=np.complex128,
noise_model=None, shots=1, seed=42)` is the normal entry point. It returns `DensityMatrixResult`.

Initial state may be `None`, a normalized statevector, a valid density matrix, or a
`DensityMatrixState`. Supported dtypes are `np.complex64` and `np.complex128`. Device may be `cpu`,
`gpu`/`cuda` (aliases for `cuda:0`), or `cuda:<id>`. CUDA requires `unitarylab[cuda]`.

Circuit measurements, including mid-circuit collapse, are supported on CPU and CUDA.
`shots` is a positive integer; `seed` is a non-negative integer or `None`. Each shot executes
from the supplied initial state; `counts` aggregates measured classical outcomes. The returned
density matrix, `classical_results_map`, and `classical_registers` describe the final shot,
not a density matrix averaged across shots. Without circuit measurements, `counts` is empty.
`DensityMatrixResult` does not expose the statevector result's `sample()` or `measure()` methods.

## Result Queries

| Need | Member |
|---|---|
| Read-only dense matrix | `result.state` |
| Native wrapper | `result.backend_state` |
| Qubit count | `result.num_qubits` |
| Trace | `result.trace` (property) |
| Purity | `result.purity` (property) |
| Full probabilities | `result.probabilities` |
| One or marginal probability | `result.probability(...)`, `result.marginal_probabilities(...)` |
| Observable expectation | `result.expectation(observable, qubits=None)` |
| Number of executions | `result.shots` |
| Aggregated classical outcomes | `result.counts` |
| Final-shot classical-bit values | `result.classical_results_map` |
| Final-shot values by register | `result.classical_registers` |

## Result Ordering

In `DensityMatrixResult`, `probability(bitstring, qubits=...)` pairs the first character with the first requested qubit. `marginal_probabilities()` and `.probabilities` emit strings in that same requested-qubit order. With default ascending qubits, qubit 0 is therefore on the **left**. This differs from statevector `ExecutionResult` probability strings, which place qubit 0 on the right.

For example, an ideal two-qubit circuit applying X only to qubit 0 is expected to have statevector probability key `"01"`, but density-result query key `"10"` for `qubits=[0, 1]`. The raw density-matrix row/column indices still use the corresponding basis integer; do not reverse the matrix itself to reconcile display keys. Classical `counts` use the classical-bit convention below and must not be inferred from density-query key order.

## Measurement Workflow

```python
from unitarylab import Circuit, ClassicalRegister, Register

q = Register("q", 2)
c = ClassicalRegister("c", 2)
circuit = Circuit(q, c)
circuit.h(q[0])
circuit.cx(q[0], q[1])
circuit.measure(q[:], c[:])
result = circuit.execute_density(shots=1000, seed=42)
print(result.counts)
print(result.classical_registers)
```

Count keys place the highest classical-bit index on the left. Unmeasured classical bits
are `#` in keys and `-1` in register snapshots. Measurement collapses the state before
subsequent gates and noise are applied.

## Noise Workflow

```python
import numpy as np
from unitarylab import Circuit
from unitarylab.backend import BitFlip, NoiseModel, NoiseOperator

circuit = Circuit(1)
circuit.x(0)

noise = NoiseModel([
    NoiseOperator(BitFlip(0.25), gate_ids=("x",), qubits=(0,)),
])
result = circuit.execute_density(noise_model=noise)

np.testing.assert_allclose(result.state, np.diag([0.25, 0.75]))
```

Built-in channels are `BitFlip(p)`, `PhaseFlip(p)`, `DepolarizingNoise(p)`,
`AmplitudeDamping(gamma)`, and `PhaseDamping(gamma)`. Parameters must be real, finite values in
`[0, 1]`; booleans are rejected.

`NoiseOperator(channel, gate_ids=None, qubits=None)` applies its channel after matching primitive
gates. `gate_ids` filters by normalized gate identifier. `qubits` filters gate targets, not controls.
For a multi-target gate, a single-qubit channel is applied independently to each matched target.
Rules are resolved inside appended circuit blocks and multiple matching rules run in model order.

Use `NoiseModel.add(operator)` to extend a model. Only single-qubit Kraus channels are currently
supported by the public noise workflow.

## Validation Notes

- Density matrices are validated for square power-of-two dimension, finite values, Hermiticity,
  and unit trace. The current validator does not test positive semidefiniteness; validate that
  separately when accepting an untrusted matrix.
- Noise channels must be completely positive and trace preserving; built-ins expose read-only
  Kraus matrices.
- For a noiseless pure state, compare the result with `outer(state, state.conj())`.
- GPU density execution uses the optional CUDA extension and should be treated as conditionally
  available.
