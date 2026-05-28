"""NumPyEigensolver demo for exact classical eigendecomposition."""

from __future__ import annotations

from typing import Any

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import NumPyEigensolver


def _print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def example_basic_k_eigenpairs() -> None:
    """Compute the lowest k eigenpairs for a Hermitian operator."""
    operator = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.5),
        ("IX", 0.3),
    ])

    solver = NumPyEigensolver(k=2)
    result = solver.compute_eigenvalues(operator)

    _print_header("NumPyEigensolver: Basic k-Eigenpairs")
    print(f"  Requested k           : {solver.k}")
    print(f"  Supports aux operators: {solver.supports_aux_operators()}")
    print(f"  Eigenvalues           : {result.eigenvalues}")
    print(f"  Number of eigenstates : {len(result.eigenstates)}")


def example_with_aux_operators() -> None:
    """Evaluate auxiliary operators on each returned eigenstate."""
    operator = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.2),
    ])

    aux_ops = {
        "mag_z0": SparsePauliOp.from_list([("ZI", 1.0)]),
    }

    solver = NumPyEigensolver(k=2)
    result = solver.compute_eigenvalues(operator, aux_operators=aux_ops)

    _print_header("NumPyEigensolver: Auxiliary Operators")
    print(f"  Eigenvalues           : {result.eigenvalues}")
    print(f"  Aux values            : {result.aux_operators_evaluated}")


def _keep_negative(
    eigenstate: Any,
    eigenvalue: complex,
    aux_values: Any,
) -> bool:
    """Keep eigenpairs with negative real eigenvalue."""
    del eigenstate, aux_values
    return float(eigenvalue.real) < 0.0


def example_with_filter() -> None:
    """Filter eigenpairs using filter_criterion after eigendecomposition."""
    operator = SparsePauliOp.from_list([
        ("Z", 2.0),
        ("X", -1.0),
    ])

    solver = NumPyEigensolver(k=2, filter_criterion=_keep_negative)
    result = solver.compute_eigenvalues(operator)

    _print_header("NumPyEigensolver: Filter Criterion")
    print("  Filter criterion      : eigenvalue.real < 0")
    print(f"  Filtered eigenvalues  : {result.eigenvalues}")
    print(f"  Number returned       : {len(result.eigenvalues)}")


def main() -> None:
    example_basic_k_eigenpairs()
    example_with_aux_operators()
    example_with_filter()


if __name__ == "__main__":
    main()
