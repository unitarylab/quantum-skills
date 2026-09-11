# Density Matrix and Noise

Use this reference for mixed-state simulation and gate-local single-qubit noise.

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
noise_model=None)` is the normal entry point. It returns `DensityMatrixResult`.

Initial state may be `None`, a normalized statevector, a valid density matrix, or a
`DensityMatrixState`. Supported dtypes are `np.complex64` and `np.complex128`. Device may be `cpu`,
`gpu`/`cuda` (aliases for `cuda:0`), or `cuda:<id>`. CUDA requires `unitarylab[cuda]`.

Circuit measurement, shots, counts, sampling, and projective result measurement are not available
in density mode.

## Result Queries

| Need | Member |
|---|---|
| Read-only dense matrix | `result.state` |
| Native wrapper | `result.backend_state` |
| Qubit count | `result.num_qubits` |
| Trace | `result.trace()` |
| Purity | `result.purity()` |
| Full probabilities | `result.probabilities` |
| One or marginal probability | `result.probability(...)`, `result.marginal_probabilities(...)` |
| Observable expectation | `result.expectation(observable, qubits=None)` |

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
