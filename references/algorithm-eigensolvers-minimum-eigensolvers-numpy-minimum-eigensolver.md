# NumPy Minimum Eigensolver

## Scope and Reference Use

`NumPyMinimumEigensolver` delegates to `NumPyEigensolver` with `k=1` and returns the first selected eigenpair. A filter requests the full spectrum internally; without a filter, sparse methods can compute only the requested low eigenpair. It is a classical baseline, with numerical tolerances and no quantum speedup.

> **Note:** This is a classical NumPy-based eigensolver. No quantum circuit construction or execution is involved. The `## Minimal Manual Implementation` section is intentionally omitted — there is no quantum circuit to manually implement. Use this solver as a reference baseline when validating quantum eigensolvers like VQE.

Scope:
- An exact classical reference for the ground-state energy is needed
- Benchmarking quantum eigensolvers (VQE, VQD) against the true minimum eigenvalue
- Applying a feasibility filter to select the minimum eigenvalue from a constrained subset

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

API source: [Qiskit Algorithms 0.4.0](https://github.com/qiskit-community/qiskit-algorithms/blob/0.4.0/qiskit_algorithms/minimum_eigensolvers/numpy_minimum_eigensolver.py); these classes belong to `qiskit_algorithms`, not `unitarylab_algorithms` or `unitarylab.library`. The accompanying UnitaryLab teaching implementation has a separate interface. Examples target the V2 primitives available in Qiskit 2.6.1.

Reference scripts: [numpy_minimum_eigensolver_algorithm.py](../scripts/numpy_minimum_eigensolver_algorithm.py) (API usage example); [numpy_minimum_eigensolver_implementation.py](../scripts/numpy_minimum_eigensolver_implementation.py) (independent teaching implementation).
- **Classical baseline:** Treat this solver as an exact classical reference. Do not generate a quantum circuit or claim quantum speedup.
- **Numerical validation:** Verify operator Hermiticity, eigenvalue ordering, matrix conversion, and `filter_criterion` behavior.
- **Delegation contract:** Preserve delegation to `NumPyEigensolver`, then select the minimum feasible eigenpair after applying the filter criterion.
- **Scale limits:** Report the exponential memory and runtime growth with qubit count for dense or sparse matrix representations.

## Overview
`NumPyMinimumEigensolver` is a classical exact minimum-eigensolver in the
`qiskit_algorithms.minimum_eigensolvers` module.

- Category: classical reference minimum eigensolver.
- Purpose: compute the minimum eigenvalue (and eigenstate) of a qubit operator.
- Core idea: delegate to `NumPyEigensolver(k=1)`, then keep the
  feasible eigenpair with the smallest eigenvalue (after optional filtering).

## Mathematical Deep Dive
For a Hermitian operator $H$, the variational characterization of the minimum eigenvalue is:

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

## Core Parameters Explained

### `NumPyMinimumEigensolver(...)`
| Parameter | Type | Required | Description |
|---|---|---|---|
| `filter_criterion` | `Callable[[Union[List, np.ndarray], float, Optional[ListOrDict[Tuple[float, Dict[str, float]]]]], bool] \| None` | No | Feasibility filter `filter(eigenstate, eigenvalue, aux_values) -> bool` used before selecting the minimum eigenvalue. |

### `compute_minimum_eigenvalue(...)`
| Parameter | Type | Required | Description |
|---|---|---|---|
| `operator` | `BaseOperator` | Yes | Main operator whose minimum eigenvalue is computed. |
| `aux_operators` | `ListOrDict[BaseOperator] \| None` | No | Auxiliary operators evaluated on the returned minimum-eigenvalue state. |

## Inputs and Outputs

### Inputs
| Item | Type | Description |
|---|---|---|
| `operator` | `BaseOperator` | Qubit operator (for example `SparsePauliOp`). |
| `aux_operators` | `List[BaseOperator] \| Dict[str, BaseOperator] \| None` | Optional observables to evaluate on the selected eigenstate. |

### Outputs
`compute_minimum_eigenvalue(...)` returns `NumPyMinimumEigensolverResult`.

| Field | Type | Description |
|---|---|---|
| `eigenvalue` | `complex \| None` | Minimum eigenvalue found (or `None` if no feasible state). |
| `eigenstate` | `Statevector \| None` | Eigenstate associated with `eigenvalue`. |
| `aux_operators_evaluated` | `ListOrDict[tuple[complex, dict[str, Any]]] \| None` | Auxiliary expectation values for the selected eigenstate. |

## Implementation Architecture
1. Construct `NumPyMinimumEigensolver`, optionally with `filter_criterion`.
2. Call `compute_minimum_eigenvalue(operator, aux_operators)`.
3. Internally, the solver calls `NumPyEigensolver.compute_eigenvalues(...)`.
4. The first feasible eigenpair (lowest eigenvalue) is mapped to:
   `result.eigenvalue`, `result.eigenstate`, and optionally `result.aux_operators_evaluated`.

Implementation notes:

- This is a classical numerical eigensolver using the underlying sparse/dense selection; resource usage grows with operator dimension and requested/filter work.
- Use it as a deterministic baseline/reference and for small to medium problem sizes.
- `supports_aux_operators()` returns whether auxiliary operator evaluation is available.
- If all eigenpairs are filtered out, result fields can remain `None`.

## Reference Implementation Example
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

## Prerequisites

- NumPy, SciPy
- Qiskit `BaseOperator` objects (`SparsePauliOp`, etc.)
- Understanding of the minimum eigenvalue problem: $E_0 = \min_{\|\psi\|=1} \langle\psi|H|\psi\rangle$
- Familiarity with `NumPyEigensolver` as the underlying computation engine

## Understanding the Key Components

1. **Delegation to NumPyEigensolver**: This solver constructs `NumPyEigensolver(k=1)`, delegates computation, and maps the first returned pair. Full-spectrum work is required when filtering, not for every input.
2. **Feasibility filter**: `filter_criterion(eigenstate, eigenvalue, aux_values) -> bool` selects which eigenpairs are eligible. The minimum is taken only over eigenpairs that pass the filter. If no eigenpair passes, `result.eigenvalue` is `None`.
3. **Single result vs array**: Unlike `NumPyEigensolver` which returns a list of eigenvalues, this solver returns a single `eigenvalue` (the minimum) and a single `eigenstate`.
4. **Auxiliary operator evaluation**: The result exposes only the selected state's auxiliary values. With a filter, the delegated solver can evaluate auxiliary operators on all computed states before filtering; only the first accepted entry is retained here.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Minimum eigenvalue $E_0$ | `result.eigenvalue` — `complex` or `None` (if no feasible state) |
| Eigenstate $\lvert\psi_0\rangle$ | `result.eigenstate` — `Statevector` or `None` |
| Delegated numerical solve | Internal `NumPyEigensolver(k=1).compute_eigenvalues()` |
| Feasibility set $\mathcal{F}$ | `filter_criterion` predicate |
| Constrained minimum $E_\star = \min_{(\lambda_i, \psi_i) \in \mathcal{F}} \lambda_i$ | Post-filtered minimum selection in `compute_minimum_eigenvalue()` |
| Auxiliary expectations | `result.aux_operators_evaluated` on the selected eigenstate only |

## Hands-On Example

Use filter_criterion to find the minimum eigenvalue among eigenstates with specific properties:

```python
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

operator = SparsePauliOp(["ZZ", "XI", "IX"], coeffs=[2.0, 1.0, -0.5])

def symmetric_filter(eigenstate, eigenvalue, aux_values):
    """Accept eigenstates whose ZZ expectation exceeds the chosen threshold."""
    if eigenstate is None:
        return False
    sv = Statevector(eigenstate)
    zz_exp = np.real(sv.expectation_value(SparsePauliOp("ZZ")))
    return zz_exp > 0.5  # approximately even parity

solver = NumPyMinimumEigensolver(filter_criterion=symmetric_filter)
result = solver.compute_minimum_eigenvalue(operator)

print("Minimum eigenvalue passing ZZ threshold:", result.eigenvalue)
print("Eigenstate available:", result.eigenstate is not None)
```

Expected behavior: the returned eigenvalue is the minimum among eigenstates with $\langle ZZ \rangle > 0.5$. If no state passes the filter, `result.eigenvalue` is `None`.

## Debugging Tips

1. **All eigenpairs filtered out**: If `filter_criterion` rejects all eigenpairs, `result.eigenvalue` and `result.eigenstate` are `None`. Always check for `None` before using results. Loosen the filter or set `filter_criterion=None` to debug.
2. **Hermiticity**: A physical ground-state minimum requires a Hermitian operator. Check `np.allclose(matrix, matrix.conj().T)` on a suitable small reference. `NumPyEigensolver` 0.4.0 has a defective dense Hermiticity test, so non-Hermitian results are not a reliable baseline.
3. **Memory scaling**: Sparse or dense computation is selected by `NumPyEigensolver`; filtering can force a full-spectrum dense fallback. A dense complex128 matrix alone uses `16*4**n` bytes, with additional solver and eigenvector storage.
4. **Filter side effects**: The `filter_criterion` is called for every eigenpair found by `NumPyEigensolver`. Expensive filter logic (e.g., statevector tomography) will multiply runtime — keep filters lightweight (simple expectation values or coefficient checks).
5. **Aux operator evaluation**: Only the selected state's auxiliary values are exposed, though filtering can cause more states to be evaluated internally. Use `NumPyEigensolver` directly to obtain auxiliary results for multiple selected eigenstates.
