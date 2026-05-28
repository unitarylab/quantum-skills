"""Quantum Fisher Information (QFI) demos using a LinCombQGT backend."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.gradients import LinCombQGT, QFI


def _print_header(title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)


def example_qfi_basic() -> None:
    """Compute the QFI matrix for all parameters of one circuit."""
    theta = ParameterVector("theta", 2)

    # Keep to LinCombQGT supported gates.
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)

    parameter_values = [0.5, 1.0]

    estimator = StatevectorEstimator()
    qgt = LinCombQGT(estimator=estimator)
    qfi = QFI(qgt=qgt)

    result = qfi.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()

    matrix = np.asarray(result.qfis[0])
    _print_header("QFI: all parameters")
    print(f"parameter values: {parameter_values}")
    print(f"shape: {matrix.shape}")
    print(f"qfi matrix:\n{matrix}")
    print(f"precision: {result.precision}")
    print(f"metadata: {result.metadata}")


def example_qfi_parameter_subset() -> None:
    """Compute QFI for a selected subset of circuit parameters."""
    theta = ParameterVector("theta", 3)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 1)
    qc.ry(theta[2], 1)
    qc.cx(0, 1)

    parameter_values = [0.3, -0.2, 0.9]
    selected_parameters = [theta[0], theta[2]]

    estimator = StatevectorEstimator()
    qfi = QFI(qgt=LinCombQGT(estimator=estimator))

    result = qfi.run(
        circuits=[qc],
        parameter_values=[parameter_values],
        parameters=[selected_parameters],
    ).result()

    matrix = np.asarray(result.qfis[0])
    _print_header("QFI: selected parameters")
    print(f"parameter values: {parameter_values}")
    print("selected parameters: [theta[0], theta[2]]")
    print(f"shape: {matrix.shape}")
    print(f"qfi matrix:\n{matrix}")
    print(f"metadata: {result.metadata}")


def example_qfi_precision_override() -> None:
    """Show constructor and per-run precision override behavior."""
    theta = ParameterVector("theta", 2)

    qc = QuantumCircuit(2)
    qc.rx(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)

    parameter_values = [0.4, -0.7]
    estimator = StatevectorEstimator()

    # Base precision is set on QFI and can be overridden per run.
    qfi = QFI(
        qgt=LinCombQGT(estimator=estimator),
        precision=1e-5,
    )

    base_result = qfi.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()
    overridden_result = qfi.run(
        circuits=[qc],
        parameter_values=[parameter_values],
        precision=1e-8,
    ).result()

    _print_header("QFI: precision behavior")
    print(f"parameter values: {parameter_values}")
    print(f"base precision result: {base_result.precision}")
    print(f"overridden precision result: {overridden_result.precision}")
    print(f"base qfi:\n{np.asarray(base_result.qfis[0])}")
    print(f"overridden qfi:\n{np.asarray(overridden_result.qfis[0])}")


def main() -> None:
    example_qfi_basic()
    example_qfi_parameter_subset()
    example_qfi_precision_override()


if __name__ == "__main__":
    main()