---
name: numpy-minimum-eigensolver
description: "NumPyMinimumEigensolver skill for deterministic minimum-eigenvalue computation in qiskit_algorithms.minimum_eigensolvers. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# NumPy Minimum Eigensolver

## Overview
`NumPyMinimumEigensolver` is a classical exact minimum-eigensolver in the
`qiskit_algorithms.minimum_eigensolvers` module.

- Category: classical reference minimum eigensolver.
- Purpose: compute the minimum eigenvalue (and eigenstate) of a qubit operator.
- Core idea: delegate full eigendecomposition to `NumPyEigensolver`, then keep the
  feasible eigenpair with the smallest eigenvalue (after optional filtering).

## Mathematical Deep Dive
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

- This is an exact dense linear-algebra approach; resource usage grows rapidly with qubit count.
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

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement NumPy Minimum Eigensolver.

`NumPyMinimumEigensolver` is a classical exact minimum-eigensolver that delegates full eigendecomposition to `NumPyEigensolver`, then selects the feasible eigenpair with the smallest eigenvalue. It serves as a deterministic baseline for quantum minimum-eigensolver benchmarks.

> **Note:** This is a classical NumPy-based eigensolver. No quantum circuit construction or execution is involved. The `## Minimal Manual Implementation` section is intentionally omitted — there is no quantum circuit to manually implement. Use this solver as a reference baseline when validating quantum eigensolvers like VQE.

Use this skill when:
- An exact classical reference for the ground-state energy is needed
- Benchmarking quantum eigensolvers (VQE, VQD) against the true minimum eigenvalue
- Applying a feasibility filter to select the minimum eigenvalue from a constrained subset

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.
- **Classical baseline:** Treat this solver as an exact classical reference. Do not generate a quantum circuit or claim quantum speedup.
- **Numerical validation:** Verify operator Hermiticity, eigenvalue ordering, matrix conversion, and `filter_criterion` behavior.
- **Delegation contract:** Preserve delegation to `NumPyEigensolver`, then select the minimum feasible eigenpair after applying the filter criterion.
- **Scale limits:** Report the exponential memory and runtime growth with qubit count for dense or sparse matrix representations.

## Prerequisites

- NumPy, SciPy
- Qiskit `BaseOperator` objects (`SparsePauliOp`, etc.)
- Understanding of the minimum eigenvalue problem: $E_0 = \min_{\|\psi\|=1} \langle\psi|H|\psi\rangle$
- Familiarity with `NumPyEigensolver` as the underlying computation engine

## Understanding the Key Components

1. **Delegation to NumPyEigensolver**: This solver does not perform its own diagonalization. It constructs a `NumPyEigensolver` internally, calls `compute_eigenvalues()` to obtain all eigenpairs, then selects the smallest eigenvalue after optional filtering.
2. **Feasibility filter**: `filter_criterion(eigenstate, eigenvalue, aux_values) -> bool` selects which eigenpairs are eligible. The minimum is taken only over eigenpairs that pass the filter. If no eigenpair passes, `result.eigenvalue` is `None`.
3. **Single result vs array**: Unlike `NumPyEigensolver` which returns a list of eigenvalues, this solver returns a single `eigenvalue` (the minimum) and a single `eigenstate`.
4. **Auxiliary operator evaluation**: Auxiliary operators are evaluated only on the selected minimum-eigenvalue state — not on all computed eigenstates. This is more efficient when only the ground state is of interest.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Minimum eigenvalue $E_0$ | `result.eigenvalue` — `complex` or `None` (if no feasible state) |
| Eigenstate $|\psi_0\rangle$ | `result.eigenstate` — `Statevector` or `None` |
| Full eigendecomposition | Internal `NumPyEigensolver.compute_eigenvalues()` |
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
    """Accept only eigenstates with even parity (ZZ expectation ≈ +1)."""
    if eigenstate is None:
        return False
    sv = Statevector(eigenstate)
    zz_exp = np.real(sv.expectation_value(SparsePauliOp("ZZ")))
    return zz_exp > 0.5  # approximately even parity

solver = NumPyMinimumEigensolver(filter_criterion=symmetric_filter)
result = solver.compute_minimum_eigenvalue(operator)

print("Minimum eigenvalue (even parity):", result.eigenvalue)
print("Eigenstate available:", result.eigenstate is not None)
```

Expected behavior: the returned eigenvalue is the minimum among eigenstates with $\langle ZZ \rangle > 0.5$. If no state passes the filter, `result.eigenvalue` is `None`.

## Debugging Tips

1. **All eigenpairs filtered out**: If `filter_criterion` rejects all eigenpairs, `result.eigenvalue` and `result.eigenstate` are `None`. Always check for `None` before using results. Loosen the filter or set `filter_criterion=None` to debug.
2. **Complex eigenvalues**: Non-Hermitian operators may yield complex eigenvalues. The "minimum" is determined by comparing complex numbers (by real part, then imaginary). For physical ground-state problems, verify the operator is Hermitian with `operator.is_hermitian()`.
3. **Memory scaling**: This solver internally calls `NumPyEigensolver`, which converts the operator to a dense matrix of size $2^n \times 2^n$. Same exponential scaling limits apply — practical for $n \leq 10$–$12$ qubits.
4. **Filter side effects**: The `filter_criterion` is called for every eigenpair found by `NumPyEigensolver`. Expensive filter logic (e.g., statevector tomography) will multiply runtime — keep filters lightweight (simple expectation values or coefficient checks).
5. **Aux operator evaluation**: Auxiliary operators are evaluated only on the selected minimum-eigenvalue eigenstate. If you need auxiliary values for all eigenstates, use `NumPyEigensolver` directly instead.
```
