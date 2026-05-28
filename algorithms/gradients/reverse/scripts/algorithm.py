"""Reverse-mode gradient and QGT demos using statevector backpropagation."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ReverseQGT
from qiskit_algorithms.gradients.utils import DerivativeType


def _print_header(title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)


def example_reverse_estimator_gradient_real() -> None:
    """Compute expectation gradients with REAL projection."""
    theta = ParameterVector("theta", 3)

    # Keep parameterized gates to rx/ry/rz for broad version compatibility.
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 1)
    qc.cx(0, 1)
    qc.rx(theta[2], 1)

    observable = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.2),
    ])
    parameter_values = [0.5, -0.35, 1.1]

    gradient = ReverseEstimatorGradient(
        derivative_type=DerivativeType.REAL,
    )

    result = gradient.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[parameter_values],
    ).result()

    _print_header("ReverseEstimatorGradient: REAL / all parameters")
    print(f"parameter values: {parameter_values}")
    print(f"gradient: {np.asarray(result.gradients[0])}")
    print(f"precision: {result.precision}")
    print(f"metadata: {result.metadata}")


def example_reverse_estimator_gradient_subset_complex() -> None:
    """Differentiate only a selected parameter subset with COMPLEX output."""
    theta = ParameterVector("theta", 4)

    qc = QuantumCircuit(2)
    qc.rx(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.rz(theta[2], 1)
    qc.rx(theta[3], 0)

    observable = SparsePauliOp("ZZ")
    parameter_values = [0.2, 0.7, -0.5, 1.3]
    selected_parameters: list[Parameter] = [theta[1], theta[3]]

    gradient = ReverseEstimatorGradient(
        derivative_type=DerivativeType.COMPLEX,
    )

    result = gradient.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[parameter_values],
        parameters=[selected_parameters],
    ).result()

    _print_header("ReverseEstimatorGradient: COMPLEX / selected parameters")
    print(f"parameter values: {parameter_values}")
    print("selected parameters: [theta[1], theta[3]]")
    print(f"gradient shape: {np.asarray(result.gradients[0]).shape}")
    print(f"gradient: {np.asarray(result.gradients[0])}")
    print(f"metadata: {result.metadata}")


def example_reverse_qgt_phase_fix_comparison() -> None:
    """Compare QGT with and without phase-fix subtraction."""
    theta = ParameterVector("theta", 3)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 1)
    qc.cx(0, 1)
    qc.rx(theta[2], 1)

    parameter_values = [0.45, -0.8, 0.6]

    qgt_phase_fix = ReverseQGT(
        phase_fix=True,
        derivative_type=DerivativeType.COMPLEX,
    )
    qgt_no_phase_fix = ReverseQGT(
        phase_fix=False,
        derivative_type=DerivativeType.COMPLEX,
    )

    phase_fix_result = qgt_phase_fix.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()
    no_phase_fix_result = qgt_no_phase_fix.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()

    phase_fix_matrix = np.asarray(phase_fix_result.qgts[0])
    no_phase_fix_matrix = np.asarray(no_phase_fix_result.qgts[0])

    _print_header("ReverseQGT: phase-fix comparison")
    print(f"parameter values: {parameter_values}")
    print(f"phase_fix=True, derivative_type={phase_fix_result.derivative_type}")
    print(f"QGT (phase fix):\n{phase_fix_matrix}")
    print(f"phase_fix=False, derivative_type={no_phase_fix_result.derivative_type}")
    print(f"QGT (no phase fix):\n{no_phase_fix_matrix}")
    print(f"metadata (phase fix): {phase_fix_result.metadata}")
    print(f"metadata (no phase fix): {no_phase_fix_result.metadata}")


def example_reverse_qgt_parameter_subset_real() -> None:
    """Compute a REAL-projected QGT on a chosen parameter subset."""
    theta = ParameterVector("theta", 4)

    qc = QuantumCircuit(2)
    qc.rx(theta[0], 0)
    qc.ry(theta[1], 0)
    qc.cx(0, 1)
    qc.rz(theta[2], 1)
    qc.rz(theta[3], 1)

    parameter_values = [0.15, -0.35, 0.72, 1.05]
    selected_parameters: list[Parameter] = [theta[0], theta[2], theta[3]]

    qgt = ReverseQGT(
        phase_fix=True,
        derivative_type=DerivativeType.REAL,
    )

    result = qgt.run(
        circuits=[qc],
        parameter_values=[parameter_values],
        parameters=[selected_parameters],
    ).result()

    matrix = np.asarray(result.qgts[0])
    _print_header("ReverseQGT: REAL / selected parameters")
    print(f"parameter values: {parameter_values}")
    print("selected parameters: [theta[0], theta[2], theta[3]]")
    print(f"shape: {matrix.shape}")
    print(f"qgt:\n{matrix}")
    print(f"derivative type: {result.derivative_type}")
    print(f"precision: {result.precision}")
    print(f"metadata: {result.metadata}")


def main() -> None:
    example_reverse_estimator_gradient_real()
    example_reverse_estimator_gradient_subset_complex()
    example_reverse_qgt_phase_fix_comparison()
    example_reverse_qgt_parameter_subset_real()


if __name__ == "__main__":
    main()
