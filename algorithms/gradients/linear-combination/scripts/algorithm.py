"""Linear-combination gradient demos for estimator, sampler, and QGT."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import (
    LinCombEstimatorGradient,
    LinCombQGT,
    LinCombSamplerGradient,
)
from qiskit_algorithms.gradients.utils import DerivativeType


def _print_header(title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)


def _print_estimator_gradients(
    label: str,
    gradients: Iterable[np.ndarray],
) -> None:
    print(label)
    for idx, grad in enumerate(gradients):
        print(f"  circuit[{idx}] gradient: {np.asarray(grad)}")


def _format_distribution_grad(
    grad_dict: dict[int, float],
    num_qubits: int,
) -> dict[str, float]:
    return {
        format(state, f"0{num_qubits}b"): value
        for state, value in sorted(grad_dict.items())
    }


def example_estimator_derivative_types() -> None:
    """Run LCU estimator gradients for REAL/IMAG/COMPLEX derivative types."""
    theta = ParameterVector("theta", 2)

    # Use only gates listed in SUPPORTED_GATES.
    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 1)
    qc.cx(0, 1)

    observable = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.25),
    ])
    parameter_values = [0.5, 1.0]

    estimator = StatevectorEstimator()

    _print_header("LinCombEstimatorGradient: derivative types")
    print(f"parameter values: {parameter_values}")

    for derivative_type in (
        DerivativeType.REAL,
        DerivativeType.IMAG,
        DerivativeType.COMPLEX,
    ):
        gradient = LinCombEstimatorGradient(
            estimator=estimator,
            derivative_type=derivative_type,
        )
        result = gradient.run(
            circuits=[qc],
            observables=[observable],
            parameter_values=[parameter_values],
        ).result()

        _print_estimator_gradients(
            f"derivative_type={derivative_type.name}",
            result.gradients,
        )
        print(f"  metadata: {result.metadata}")


def example_sampler_distribution_gradient() -> None:
    """Compute LCU sampler gradients of output probabilities."""
    theta = ParameterVector("theta", 1)

    qc = QuantumCircuit(1)
    qc.ry(theta[0], 0)
    qc.measure_all()

    parameter_values = [0.5]

    sampler = StatevectorSampler()
    gradient = LinCombSamplerGradient(sampler=sampler)

    result = gradient.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()

    _print_header("LinCombSamplerGradient: distribution gradients")
    print(f"parameter values: {parameter_values}")
    for circuit_index, per_parameter_grads in enumerate(result.gradients):
        print(f"circuit[{circuit_index}]:")
        for param_index, grad_dict in enumerate(per_parameter_grads):
            pretty = _format_distribution_grad(grad_dict, num_qubits=1)
            print(f"  dP/dtheta[{param_index}] = {pretty}")
    print(f"metadata: {result.metadata}")


def example_qgt_with_phase_fix() -> None:
    """Compute LCU QGT with and without phase-fix subtraction."""
    theta = ParameterVector("theta", 2)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)

    parameter_values = [0.5, 1.0]
    estimator = StatevectorEstimator()

    _print_header("LinCombQGT: phase_fix comparison")
    print(f"parameter values: {parameter_values}")

    for phase_fix in (True, False):
        qgt = LinCombQGT(
            estimator=estimator,
            phase_fix=phase_fix,
            derivative_type=DerivativeType.COMPLEX,
        )
        result = qgt.run(
            circuits=[qc],
            parameter_values=[parameter_values],
        ).result()

        print(f"phase_fix={phase_fix}")
        print(f"  qgt shape: {np.asarray(result.qgts[0]).shape}")
        print(f"  qgt matrix:\n{np.asarray(result.qgts[0])}")
        print(f"  metadata: {result.metadata}")


def main() -> None:
    example_estimator_derivative_types()
    example_sampler_distribution_gradient()
    example_qgt_with_phase_fix()


if __name__ == "__main__":
    main()