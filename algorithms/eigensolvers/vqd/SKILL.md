---
name: vqd
description: "Variational Quantum Deflation (VQD) eigensolver for computing the lowest excited states of a quantum operator with Qiskit primitives. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Variational Quantum Deflation (VQD)

## Overview

VQD is a hybrid variational eigensolver for computing multiple low-lying eigenvalues of a qubit operator.

- Purpose: compute the lowest k eigenvalues (ground + excited states).
- Category: variational eigensolver.
- Core idea: solve one state at a time with a variational ansatz, and add overlap penalties to enforce orthogonality with previously found states.

## Reference Implementation Example

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

## Core Parameters Explained

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

## Return Fields

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

## Implementation Architecture

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

## Mathematical Deep Dive

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

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Variational Quantum Deflation (VQD).

VQD is a hybrid variational eigensolver that computes multiple low-lying eigenvalues of a qubit operator by solving one state at a time with overlap penalties against previously found states. It wraps the standard VQE workflow for each eigenstate and adds orthogonality constraints.

Use this skill when:
- Computing the lowest $k$ eigenvalues (ground + excited states) of a Hamiltonian
- A variational quantum algorithm is preferred over exact classical diagonalization
- Overlap penalties are acceptable as a mechanism for enforcing state orthogonality

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Prerequisites

- Qiskit parameterized ansatz circuit (`QuantumCircuit` with `num_parameters > 0`)
- Estimator primitive (`BaseEstimatorV2`) for expectation evaluation
- Fidelity primitive (`BaseStateFidelity`) for overlap computation — typically `ComputeUncompute` with a `StatevectorSampler`
- Classical optimizer (`Optimizer` or `Minimizer`) — `SLSQP` or `COBYLA` are common choices
- Understanding of the variational principle and the deflation technique for excited states

## Understanding the Key Quantum Components

1. **Stepwise deflation**: VQD optimizes states sequentially — step 1 finds the ground state $|\psi_0\rangle$, step 2 finds $|\psi_1\rangle$ orthogonal to $|\psi_0\rangle$, step 3 finds $|\psi_2\rangle$ orthogonal to both, and so on.
2. **Overlap penalty**: The cost function at step $j$ is $C_j(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle + \sum_{i=0}^{j-2} \beta_i |\langle\psi(\theta)|\psi_i\rangle|^2$. The penalty term grows with each step as more prior states accumulate.
3. **Beta weights**: Control the strength of orthogonality enforcement. Auto-computed from operator coefficients if not user-provided, but manual tuning may be needed for optimal convergence.
4. **Fidelity primitive**: The `BaseStateFidelity` implementation (typically `ComputeUncompute`) computes $|\langle\psi(\theta)|\psi_i\rangle|^2$ for each prior state. This is the computational bottleneck as $k$ grows.
5. **Convergence threshold**: When `convergence_threshold` is set, the average weighted fidelity with prior states is monitored — the algorithm raises `AlgorithmError` if overlap remains too high after optimization.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Hamiltonian $H$ | `operator` — `BaseOperator`, e.g., `SparsePauliOp` |
| Ansatz $|\psi(\theta)\rangle$ | `ansatz` — `QuantumCircuit` with `Parameter` objects |
| Energy expectation $\langle H\rangle$ | Computed via `estimator` primitive internally |
| Overlap $|\langle\psi(\theta)|\psi_i\rangle|^2$ | Computed via `fidelity` primitive (`ComputeUncompute`) |
| Cost function $C_j(\theta)$ | Internal objective: energy + Σ β_i · overlap_i |
| Penalty weights $\beta_i$ | `betas` parameter; auto-computed if `None` |
| Optimizer | `optimizer` — single or per-step sequence |
| Number of eigenvalues $k$ | Constructor parameter `k`; default `2` |
| Result eigenvalues | `result.eigenvalues` — `np.ndarray` of length $k$ |

## Hands-On Example

Compute the 3 lowest eigenvalues with custom beta weights:

```python
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import VQD
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.state_fidelities import ComputeUncompute

operator = SparsePauliOp(["ZZZ", "XII", "IXI", "IIX"], coeffs=[1.0, 0.5, 0.5, 0.5])
ansatz = RealAmplitudes(num_qubits=3, reps=2)
estimator = StatevectorEstimator()
fidelity = ComputeUncompute(StatevectorSampler())
optimizer = COBYLA(maxiter=200)

solver = VQD(
    estimator=estimator,
    fidelity=fidelity,
    ansatz=ansatz,
    optimizer=optimizer,
    k=3,
    betas=np.array([0.5, 1.0]),  # 2 penalty weights for 3 states
)

result = solver.compute_eigenvalues(operator)
print("Eigenvalues:", result.eigenvalues)
print("Cost function evals:", result.cost_function_evals)
```

Expected behavior: `eigenvalues` is sorted ascending; `cost_function_evals` increases per step as the penalty term adds overhead.

## Minimal Manual Implementation

```python
import numpy as np
from scipy.optimize import minimize

def vqd_loop(energy_fn, fidelity_fn, ansatz_params_init, k, betas, optimizer_method="COBYLA"):
    """Simplified VQD loop — sequential deflation with overlap penalties.

    Args:
        energy_fn: callable(params) -> float (energy expectation)
        fidelity_fn: callable(params, prior_params) -> float (|overlap|^2)
        ansatz_params_init: list of initial parameter vectors for each step
        k: number of eigenvalues
        betas: overlap penalty weights (length >= k-1)

    Returns:
        eigenvalues, optimal_parameters
    """
    prior_params = []  # store optimized parameters for prior states
    eigenvalues = []

    for step in range(k):
        def objective(params):
            energy = energy_fn(params)
            penalty = sum(
                betas[i] * fidelity_fn(params, prior_params[i])
                for i in range(len(prior_params))
            )
            return energy + penalty

        res = minimize(objective, ansatz_params_init[step], method=optimizer_method)
        optimal_params = res.x
        prior_params.append(optimal_params)
        eigenvalues.append(res.fun)  # energy at optimum (penalty ≈ 0 ideally)

    return np.array(eigenvalues), prior_params
```

Note: This skeleton captures the stepwise deflation pattern. The actual Qiskit implementation handles estimator/fidelity primitive batching, convergence monitoring, and aux_operator evaluation.

## Debugging Tips

1. **Overlap penalty too weak**: If excited-state energies are close to the ground state, increase `betas` to strengthen orthogonality enforcement. Start with $\beta \approx 1.0$ and increase if states are not orthogonal.
2. **Overlap penalty too strong**: If optimization fails to converge (cost function oscillates), the penalty term may dominate the energy term. Reduce `betas` or use a more robust optimizer (`SLSQP` with bounds instead of `COBYLA`).
3. **Convergence threshold triggered**: If `convergence_threshold` is set and the algorithm raises `AlgorithmError`, the final average weighted fidelity exceeds the threshold — increase optimizer `maxiter`, adjust `betas`, or relax the threshold.
4. **Per-step optimizer scaling**: The cost function complexity grows with each step as more prior states accumulate. For $k > 3$, consider using separate optimizer instances per step with step-appropriate `maxiter` values.
5. **Fidelity primitive choice**: `ComputeUncompute` requires two circuit executions per overlap evaluation. For large $k$, the fidelity overhead dominates — verify the fidelity primitive is appropriate for your ansatz depth and qubit count.
