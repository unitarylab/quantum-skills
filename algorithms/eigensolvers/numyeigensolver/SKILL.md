---
name: numyeigensolver
description: Exact classical eigensolver for quantum operators using NumPy and SciPy backends, with optional auxiliary operator evaluation and eigenpair filtering.
---

# NumPy Eigensolver

## Algorithm Overview

NumPyEigensolver is a classical eigensolver in Qiskit Algorithms for computing eigenvalues and eigenstates of a quantum operator.

- Purpose: compute the lowest k eigenvalues (and corresponding eigenstates) of a BaseOperator.
- Category: classical exact eigensolver.
- Core idea: convert the operator to sparse or dense matrix form, choose a numerical eigensolver based on matrix structure, then sort and return the first k solutions.

## Mathematical Principles

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

## Core Parameters

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

## Implementation Description

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

## Sample Code

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
