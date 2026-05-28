"""Parameter-shift gradient demos for estimator and sampler primitives."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import (
    ParamShiftEstimatorGradient,
    ParamShiftSamplerGradient,
)


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


def example_estimator_gradient() -> None:
    """Compute exact expectation-value gradients with parameter shift."""
    theta = ParameterVector("theta", 2)

    # Keep to parameter-shift supported gates.
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
    gradient = ParamShiftEstimatorGradient(estimator=estimator)

    result = gradient.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[parameter_values],
    ).result()

    _print_header("ParamShiftEstimatorGradient: all parameters")
    print(f"parameter values: {parameter_values}")
    _print_estimator_gradients("exact analytic gradient", result.gradients)
    print(f"precision: {result.precision}")
    print(f"metadata: {result.metadata}")


def example_estimator_parameter_subset() -> None:
    """Differentiate only a selected subset of circuit parameters."""
    theta = ParameterVector("theta", 3)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rx(theta[1], 1)
    qc.rz(theta[2], 1)
    qc.cx(0, 1)

    observable = SparsePauliOp("ZZ")
    parameter_values = [0.2, -0.4, 0.7]

    estimator = StatevectorEstimator()
    gradient = ParamShiftEstimatorGradient(estimator=estimator)

    selected_parameters: list[Parameter] = [theta[0], theta[2]]
    result = gradient.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[parameter_values],
        parameters=[selected_parameters],
    ).result()

    _print_header("ParamShiftEstimatorGradient: selected parameters")
    print(f"parameter values: {parameter_values}")
    print("selected parameters: [theta[0], theta[2]]")
    print(f"gradient shape: {np.asarray(result.gradients[0]).shape}")
    print(f"gradient values: {np.asarray(result.gradients[0])}")
    print(f"metadata: {result.metadata}")


def example_sampler_gradient() -> None:
    """Compute probability-distribution gradients with parameter shift."""
    theta = ParameterVector("theta", 2)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rxx(theta[1], 0, 1)
    qc.measure_all()

    parameter_values = [0.35, -0.2]

    sampler = StatevectorSampler()
    gradient = ParamShiftSamplerGradient(sampler=sampler)

    result = gradient.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()

    _print_header("ParamShiftSamplerGradient: distribution gradients")
    print(f"parameter values: {parameter_values}")
    for circuit_index, per_parameter_grads in enumerate(result.gradients):
        print(f"circuit[{circuit_index}]:")
        for param_index, grad_dict in enumerate(per_parameter_grads):
            pretty = _format_distribution_grad(grad_dict, num_qubits=2)
            print(f"  dP/dtheta[{param_index}] = {pretty}")
    print(f"shots: {result.shots}")
    print(f"metadata: {result.metadata}")


def main() -> None:
    example_estimator_gradient()
    example_estimator_parameter_subset()
    example_sampler_gradient()


if __name__ == "__main__":
    main()
