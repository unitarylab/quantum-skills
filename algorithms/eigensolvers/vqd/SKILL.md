---
name: vqd
description: Variational Quantum Deflation (VQD) eigensolver for computing the lowest excited states of a quantum operator with Qiskit primitives.
---

# Variational Quantum Deflation (VQD)

## Algorithm Overview

VQD is a hybrid variational eigensolver for computing multiple low-lying eigenvalues of a qubit operator.

- Purpose: compute the lowest k eigenvalues (ground + excited states).
- Category: variational eigensolver.
- Core idea: solve one state at a time with a variational ansatz, and add overlap penalties to enforce orthogonality with previously found states.

## Mathematical Principles

For a Hamiltonian H and ansatz state |psi(theta)>, VQD minimizes at step j:

$$
C_j(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle + \sum_{i=0}^{j-2} \beta_i\,\left|\langle \psi(\theta) | \psi_i \rangle\right|^2
$$

- First term is the energy expectation.
- Second term penalizes overlap with previously optimized states psi_i.
- beta_i weights control the orthogonality penalty strength.

Returned eigenpairs satisfy the standard eigenvalue relation:

$$
H|\psi_i\rangle = \lambda_i|\psi_i\rangle
$$

## Core Parameters

Constructor parameters of VQD:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| estimator | BaseEstimatorV2 | Yes | - | Primitive used for expectation estimation. |
| fidelity | BaseStateFidelity | Yes | - | Fidelity primitive used for overlap penalties. |
| ansatz | QuantumCircuit | Yes | - | Parameterized trial state circuit. |
| optimizer | Optimizer \| Minimizer \| Sequence[Optimizer \| Minimizer] | Yes | - | Optimizer for each step (single or per-eigenstate sequence). |
| k | int | No | 2 | Number of eigenvalues to compute. |
| betas | np.ndarray \| None | No | None | Overlap penalty weights. Must cover all penalty terms (typically at least k-1 values). |
| initial_point | np.ndarray \| list[np.ndarray] \| None | No | None | Initial parameters (single point or one per step). |
| callback | Callable[[int, np.ndarray, float, dict[str, Any], int], None] \| None | No | None | Per-evaluation callback: eval_count, params, value, metadata, step. |
| convergence_threshold | float \| None | No | None | Max allowed average weighted fidelity with prior states. |
| transpiler | Transpiler \| None | No | None | Optional transpiler with run(...) used on ansatz/circuits. |
| transpiler_options | dict[str, Any] \| None | No | None | Keyword options passed to transpiler.run. |

## Inputs and Outputs

Method: compute_eigenvalues(operator, aux_operators=None)

### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| operator | BaseOperator | Yes | Main qubit operator whose lowest eigenvalues are estimated. |
| aux_operators | ListOrDict[BaseOperator] \| None | No | Optional operators evaluated at each optimized state. |

### Outputs

Type: VQDResult

| Field | Type | Description |
|---|---|---|
| eigenvalues | np.ndarray | Estimated eigenvalues (complex dtype; physical values are typically real). |
| optimal_points | np.ndarray | Optimal parameter vectors per step. |
| optimal_parameters | list[dict] | Parameter maps (ansatz Parameter -> value) per step. |
| optimal_values | np.ndarray | Final objective values per step (energy + penalties). |
| cost_function_evals | np.ndarray | Number of objective evaluations per step. |
| optimizer_times | np.ndarray | Optimization wall-clock time per step. |
| optimizer_results | list[OptimizerResult] | Raw optimizer results per step. |
| optimal_circuits | list[QuantumCircuit] | Circuits associated with optimized states. |
| aux_operators_evaluated | list[ListOrDict[tuple[float, dict[str, Any]]]] \| None | Auxiliary expectation values, if requested. |

## Implementation Description

Required components:

- qiskit_algorithms.eigensolvers.VQD
- qiskit.primitives.BaseEstimatorV2-compatible estimator
- qiskit_algorithms.state_fidelities.BaseStateFidelity-compatible fidelity
- Parameterized qiskit.circuit.QuantumCircuit ansatz
- qiskit_algorithms.optimizers optimizer

Execution flow:

1. Validate operator/ansatz compatibility and parameter bounds.
2. Prepare beta values (user-provided or auto-evaluated from operator coefficients).
3. For each step from 1 to k:
4. build objective = energy + overlap penalties against prior states,
5. run classical optimization from initial point,
6. store optimal point/value/circuit and optional auxiliary operator estimates.
7. Return VQDResult with per-step optimization artifacts.

Engineering constraints:

- ansatz must be parameterized (num_parameters > 0).
- operator.num_qubits must match ansatz.num_qubits after optional transpilation/layout.
- fidelity overhead grows with the number of already found states.
- if convergence_threshold is set, high average overlap raises AlgorithmError.

## Sample Code

```python
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

from qiskit_algorithms.eigensolvers import VQD
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.state_fidelities import ComputeUncompute

# 2-qubit operator
operator = SparsePauliOp(["ZZ", "XI", "IX"], coeffs=[1.0, 0.3, 0.3])

# Parameterized trial circuit
ansatz = RealAmplitudes(num_qubits=2, reps=2)

# Primitives
estimator = StatevectorEstimator()
fidelity = ComputeUncompute(StatevectorSampler())

# Classical optimizer
optimizer = SLSQP(maxiter=100)

solver = VQD(
	estimator=estimator,
	fidelity=fidelity,
	ansatz=ansatz,
	optimizer=optimizer,
	k=2,
)

result = solver.compute_eigenvalues(operator)

print(result.eigenvalues)
print(result.optimal_points)
```