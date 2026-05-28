"""NumPyMinimumEigensolver demo for deterministic minimum-eigenvalue computation."""

from __future__ import annotations

from typing import Any

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver


def example_basic() -> None:
    """Compute minimum eigenvalue/eigenstate and evaluate auxiliary operators."""
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

    print("=" * 60)
    print("NumPyMinimumEigensolver: Basic Example")
    print("=" * 60)
    print(f"  Supports aux operators: {solver.supports_aux_operators()}")
    print(f"  Minimum eigenvalue    : {result.eigenvalue}")
    print(f"  Eigenstate available  : {result.eigenstate is not None}")
    print(f"  Aux values            : {result.aux_operators_evaluated}")


def _keep_negative_states(
    eigenstate: Any,
    eigenvalue: complex,
    aux_values: Any,
) -> bool:
    """Keep only eigenpairs whose real part of eigenvalue is negative."""
    del eigenstate, aux_values
    return float(eigenvalue.real) < 0.0


def example_with_filter() -> None:
    """Apply filter_criterion before selecting the minimum feasible eigenpair."""
    operator = SparsePauliOp.from_list([
        ("ZI", 0.5),
        ("IZ", -1.2),
        ("XX", 0.2),
    ])

    aux_ops = {
        "zz": SparsePauliOp.from_list([("ZZ", 1.0)]),
    }

    solver = NumPyMinimumEigensolver(filter_criterion=_keep_negative_states)
    result = solver.compute_minimum_eigenvalue(operator, aux_operators=aux_ops)

    print("=" * 60)
    print("NumPyMinimumEigensolver: Filtered Example")
    print("=" * 60)
    print("  Filter criterion      : eigenvalue.real < 0")
    print(f"  Feasible eigenvalue   : {result.eigenvalue}")
    print(f"  Eigenstate available  : {result.eigenstate is not None}")
    print(f"  Aux values            : {result.aux_operators_evaluated}")


def main() -> None:
    example_basic()
    example_with_filter()


if __name__ == "__main__":
    main()
