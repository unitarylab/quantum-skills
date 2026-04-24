---
name: numpy-minimum-eigensolver
description: |
  NumPyMinimumEigensolver skill for deterministic minimum-eigenvalue computation
  in qiskit_algorithms.minimum_eigensolvers.
---

# Algorithm Overview

`NumPyMinimumEigensolver` is a classical exact minimum-eigensolver in the
`qiskit_algorithms.minimum_eigensolvers` module.

- Category: classical reference minimum eigensolver.
- Purpose: compute the minimum eigenvalue (and eigenstate) of a qubit operator.
- Core idea: delegate full eigendecomposition to `NumPyEigensolver`, then keep the
  feasible eigenpair with the smallest eigenvalue (after optional filtering).

# Mathematical Principles

Given an operator $H$, the algorithm solves:

$$
E_0 = \min_{\lVert \psi \rVert = 1} \langle \psi | H | \psi \rangle
$$

and returns the corresponding eigenstate $|\psi_0\rangle$ such that:

$$
H|\psi_0\rangle = E_0|\psi_0\rangle
$$

If `filter_criterion` is provided, only feasible eigenpairs are considered:

$$
E_\star = \min_{(\lambda_i, |\psi_i\rangle) \in \mathcal{F}} \lambda_i
$$

where $\mathcal{F}$ is the set accepted by the filter callback.

# Core Parameters

## `NumPyMinimumEigensolver(...)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `filter_criterion` | `Callable[[Union[List, np.ndarray], float, Optional[ListOrDict[Tuple[float, Dict[str, float]]]]], bool] \| None` | No | Feasibility filter `filter(eigenstate, eigenvalue, aux_values) -> bool` used before selecting the minimum eigenvalue. |

## `compute_minimum_eigenvalue(...)`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `operator` | `BaseOperator` | Yes | Main operator whose minimum eigenvalue is computed. |
| `aux_operators` | `ListOrDict[BaseOperator] \| None` | No | Auxiliary operators evaluated on the returned minimum-eigenvalue state. |

# Inputs and Outputs

## Inputs

| Item | Type | Description |
|---|---|---|
| `operator` | `BaseOperator` | Qubit operator (for example `SparsePauliOp`). |
| `aux_operators` | `List[BaseOperator] \| Dict[str, BaseOperator] \| None` | Optional observables to evaluate on the selected eigenstate. |

## Outputs

`compute_minimum_eigenvalue(...)` returns `NumPyMinimumEigensolverResult`.

| Field | Type | Description |
|---|---|---|
| `eigenvalue` | `complex \| None` | Minimum eigenvalue found (or `None` if no feasible state). |
| `eigenstate` | `Statevector \| None` | Eigenstate associated with `eigenvalue`. |
| `aux_operators_evaluated` | `ListOrDict[tuple[complex, dict[str, Any]]] \| None` | Auxiliary expectation values for the selected eigenstate. |

# Implementation Description

1. Construct `NumPyMinimumEigensolver`, optionally with `filter_criterion`.
2. Call `compute_minimum_eigenvalue(operator, aux_operators)`.
3. Internally, the solver calls `NumPyEigensolver.compute_eigenvalues(...)`.
4. The first feasible eigenpair (lowest eigenvalue) is mapped to:
   `result.eigenvalue`, `result.eigenstate`, and optionally `result.aux_operators_evaluated`.

Implementation notes:

- This is an exact dense linear-algebra approach; resource usage grows rapidly with qubit count.
- Use it as a deterministic baseline/reference and for small to medium problem sizes.
- `supports_aux_operators()` returns whether auxiliary operator evaluation is available.
- If all eigenpairs are filtered out, result fields can remain `None`.

# Sample Code

```python
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

# H = ZI + IZ + 0.5 * XX
operator = SparsePauliOp.from_list([
    ("ZI", 1.0),
    ("IZ", 1.0),
    ("XX", 0.5),
])

aux_ops = {
    "magnetization": SparsePauliOp.from_list([("ZZ", 1.0)]),
}

solver = NumPyMinimumEigensolver()
result = solver.compute_minimum_eigenvalue(operator, aux_operators=aux_ops)

print("minimum eigenvalue:", result.eigenvalue)
print("eigenstate available:", result.eigenstate is not None)
print("aux values:", result.aux_operators_evaluated)
```