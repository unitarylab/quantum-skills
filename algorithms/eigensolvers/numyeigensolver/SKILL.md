---
name: numyeigensolver
description: "Exact classical eigensolver for quantum operators using NumPy and SciPy backends, with optional auxiliary operator evaluation and eigenpair filtering. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# NumPy Eigensolver

## Overview

NumPyEigensolver is a classical eigensolver in Qiskit Algorithms for computing eigenvalues and eigenstates of a quantum operator.

- Purpose: compute the lowest k eigenvalues (and corresponding eigenstates) of a BaseOperator.
- Category: classical exact eigensolver.
- Core idea: convert the operator to sparse or dense matrix form, choose a numerical eigensolver based on matrix structure, then sort and return the first k solutions.

## Reference Implementation Example

```python
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver

operator = SparsePauliOp(["ZZ", "XI", "IX"], coeffs=[1.0, 0.5, 0.3])

solver = NumPyEigensolver(k=2)
result = solver.compute_eigenvalues(operator)

print(result.eigenvalues)
print(result.eigenstates)
```

```python
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver

operator = SparsePauliOp(["ZZ", "XI"], coeffs=[1.0, 0.2])
aux_ops = {"mag_z0": SparsePauliOp(["ZI"], coeffs=[1.0])}

solver = NumPyEigensolver(k=2)
result = solver.compute_eigenvalues(operator, aux_operators=aux_ops)

print(result.eigenvalues)
print(result.aux_operators_evaluated)
```

```python
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver

def keep_negative(eigenstate, eigenvalue, aux_values):
    return eigenvalue < 0.0

operator = SparsePauliOp(["Z", "X"], coeffs=[2.0, -1.0])

solver = NumPyEigensolver(k=2, filter_criterion=keep_negative)
result = solver.compute_eigenvalues(operator)

print(result.eigenvalues)
```

## Core Parameters Explained

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| k | int | No | 1 | Number of requested eigenvalues. Must be >= 1. Internally capped by operator dimension 2^n. |
| filter_criterion | Callable or None | No | None | Optional predicate for post-selection. Signature: filter_criterion(eigenstate, eigenvalue, aux_values) -> bool. |

## Inputs and Outputs

### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| operator | BaseOperator | Yes | Main operator for eigendecomposition. Must have valid num_qubits and matrix conversion support. |
| aux_operators | ListOrDict[BaseOperator] or None | No | Auxiliary operators to evaluate on each returned eigenstate. |

### Outputs

Type: NumPyEigensolverResult

| Field | Type | Description |
|---|---|---|
| eigenvalues | numpy.ndarray | Computed eigenvalues (sorted ascending, possibly filtered). |
| eigenstates | list[Statevector] | Eigenstates corresponding to eigenvalues. |
| aux_operators_evaluated | list or dict or None | Auxiliary expectation values. Each value is a tuple: (mean, {"variance": 0.0}). |

## Implementation Architecture

Required components:
- qiskit_algorithms.eigensolvers.NumPyEigensolver
- qiskit.quantum_info operators (for example SparsePauliOp)
- NumPy/SciPy linear algebra backend through Qiskit implementation

Execution flow:
1. Validate operator and effective k.
2. Convert operator to sparse matrix when supported; otherwise dense matrix.
3. Select solver:
4. diagonal sparse shortcut if applicable.
5. sparse eigsh or eigs for sparse matrices.
6. dense eigh or eig for dense matrices.
7. Sort eigenpairs and keep first k.
8. Evaluate aux_operators on each eigenstate if provided.
9. Apply filter_criterion if set (may return fewer than k results).
10. Package NumPyEigensolverResult.

Engineering constraints:
- Matrix size scales as 2^n with qubit count n.
- Dense conversion is memory-intensive for larger n.
- Non-Hermitian operators can yield complex eigenvalues.
- With filter_criterion, internal computation may evaluate more states than final returned count.

## Mathematical Deep Dive

The solver computes eigenpairs of:

$$
\hat{H}\lvert \psi_i\rangle = \lambda_i \lvert \psi_i\rangle
$$

For Hermitian operators:
- dense path uses numpy.linalg.eigh
- sparse path uses scipy.sparse.linalg.eigsh

For non-Hermitian operators:
- dense path uses numpy.linalg.eig
- sparse path uses scipy.sparse.linalg.eigs

Returned eigenvalues are ordered ascending and truncated to k.

Auxiliary expectation values (if provided) are computed per eigenstate as:

$$
\langle \psi_i \rvert \hat{A} \lvert \psi_i \rangle
$$

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement NumPy Eigensolver.

NumPyEigensolver is a classical exact eigensolver that computes eigenvalues and eigenstates of a qubit operator using NumPy and SciPy backends. It converts the operator to sparse or dense matrix form, selects a numerical eigensolver based on matrix structure, and returns the lowest $k$ eigenvalues.

> **Note:** This is a classical NumPy-based eigensolver. No quantum circuit construction or execution is involved. The `## Minimal Manual Implementation` section is intentionally omitted — there is no quantum circuit to manually implement. All remaining sections document the classical API and serve as a reference for comparison with quantum eigensolvers.

Use this skill when:
- An exact classical reference solution is needed for benchmarking quantum eigensolvers
- The operator dimension is small enough for dense or sparse classical diagonalization ($n \leq 10$–$12$ qubits)
- Auxiliary operator expectation values are required on eigenstates

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.
- **Classical baseline:** Treat this solver as an exact classical reference. Do not generate a quantum circuit or claim quantum speedup.
- **Numerical validation:** Verify operator Hermiticity, eigenvalue ordering, matrix conversion, solver selection, and filter behavior.
- **Scale limits:** Report the exponential memory and runtime growth with qubit count for dense or sparse matrix representations.

## Prerequisites

- NumPy, SciPy
- Qiskit `BaseOperator` objects (`SparsePauliOp`, etc.)
- Understanding of classical eigenvalue decomposition: `numpy.linalg.eigh`, `scipy.sparse.linalg.eigsh`
- Basic familiarity with the quantum operator formalism (operators act on $2^n$-dimensional Hilbert space)

## Understanding the Key Components

1. **Operator-to-matrix conversion**: The input `BaseOperator` is converted to a sparse matrix when possible (via `to_sparse_matrix()`), otherwise to a dense matrix (`to_matrix()`). The conversion path determines which numerical solver is used.
2. **Solver selection strategy**: For diagonal sparse matrices, a fast shortcut extracts eigenvalues directly. For Hermitian sparse matrices, `scipy.sparse.linalg.eigsh` is used. For Hermitian dense matrices, `numpy.linalg.eigh`. For non-Hermitian matrices, `scipy.sparse.linalg.eigs` or `numpy.linalg.eig`.
3. **Eigenvalue sorting and truncation**: All computed eigenvalues are sorted ascending, and the first $k$ (after optional filtering) are returned. Internal computation may evaluate more eigenpairs than the final result count.
4. **Filter criterion**: An optional predicate `filter_criterion(eigenstate, eigenvalue, aux_values) -> bool` post-selects eigenpairs. The first $k$ surviving pairs are returned — if fewer than $k$ survive, the result has fewer entries.
5. **Auxiliary operator evaluation**: If `aux_operators` are provided, each is evaluated on every returned eigenstate as $\langle\psi_i|\hat{A}|\psi_i\rangle$, returned as `(mean, {"variance": 0.0})` tuples.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Operator $\hat{H}$ | `operator` — `BaseOperator` with valid `num_qubits` |
| Eigenvalue equation $\hat{H}|\psi_i\rangle = \lambda_i|\psi_i\rangle$ | `compute_eigenvalues(operator)` |
| Dense Hermitian solver | `numpy.linalg.eigh` (when matrix is dense and Hermitian) |
| Sparse Hermitian solver | `scipy.sparse.linalg.eigsh` (when matrix is sparse and Hermitian) |
| Eigenvalue ordering | Sorted ascending; truncated to first $k$ |
| Filter callable | `filter_criterion` — signature `(eigenstate, eigenvalue, aux_values) -> bool` |
| Auxiliary expectations | `result.aux_operators_evaluated` — list/dict of `(mean, metadata)` tuples |
| Result object | `NumPyEigensolverResult` with `.eigenvalues`, `.eigenstates`, `.aux_operators_evaluated` |

## Hands-On Example

Compare filtered vs unfiltered eigenvalue selection:

```python
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver

operator = SparsePauliOp(["ZZ", "XI", "IX"], coeffs=[1.5, 0.8, 0.3])

# Unfiltered — returns lowest k eigenvalues
solver_unfiltered = NumPyEigensolver(k=4)
result_all = solver_unfiltered.compute_eigenvalues(operator)
print("All eigenvalues:", result_all.eigenvalues)

# Filtered — only return negative eigenvalues
def keep_negative(eigenstate, eigenvalue, aux_values):
    return np.real(eigenvalue) < 0.0

solver_filtered = NumPyEigensolver(k=4, filter_criterion=keep_negative)
result_filtered = solver_filtered.compute_eigenvalues(operator)
print("Negative eigenvalues:", result_filtered.eigenvalues)
print("Returned count:", len(result_filtered.eigenvalues))  # may be < 4
```

Expected behavior: the filtered result may return fewer than `k` eigenvalues if insufficient eigenpairs pass the filter.

## Debugging Tips

1. **Exponential memory scaling**: The dense matrix has shape $2^n \times 2^n$. For $n=10$, that's $1024 \times 1024$ (∼8 MB complex128). For $n=12$, $4096 \times 4096$ (∼128 MB). For $n > 12$, the dense path may exhaust memory — verify qubit count before calling.
2. **Non-Hermitian eigenvalues**: If the input operator is not Hermitian, eigenvalues may be complex. The solver uses `numpy.linalg.eig` / `scipy.sparse.linalg.eigs` in this case, and sorting is by real part.
3. **Filter criterion signature**: The filter receives `eigenstate` (list or `np.ndarray`), `eigenvalue` (complex), and `aux_values` (tuple or None). Ensure your callable matches this signature — `TypeError` otherwise.
4. **k exceeds dimension**: Internally, $k$ is capped at the operator dimension $2^n$. If you request `k=10` on a 2-qubit operator (dimension 4), only 4 eigenvalues are returned with no warning.
5. **Dense fallback**: Some `SparsePauliOp` objects may not support `to_sparse_matrix()` and fall back to dense conversion. For large operators, this can cause unexpected memory spikes — check `operator.to_sparse_matrix()` availability before construction.
