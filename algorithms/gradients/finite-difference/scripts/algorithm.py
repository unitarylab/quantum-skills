"""Finite difference gradient demos for estimator and sampler primitives."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import (
    FiniteDiffEstimatorGradient,
    FiniteDiffSamplerGradient,
)


def _print_header(title: str) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)


def _format_distribution_grad(
    grad_dict: dict[int, float],
    num_qubits: int,
) -> dict[str, float]:
    return {
        format(state, f"0{num_qubits}b"): value
        for state, value in sorted(grad_dict.items())
    }


def _print_estimator_gradients(
    name: str,
    gradients: Iterable[np.ndarray],
) -> None:
    print(name)
    for idx, grad in enumerate(gradients):
        print(f"  circuit[{idx}] gradient: {np.asarray(grad)}")


def example_estimator_all_methods() -> None:
    """Compare central/forward/backward finite-difference estimator gradients."""
    theta = ParameterVector("theta", 2)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 1)
    qc.cx(0, 1)

    observable = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.25),
    ])
    parameter_values = [0.31, -0.58]

    estimator = StatevectorEstimator()
    epsilon = 1e-3

    _print_header("FiniteDiffEstimatorGradient: method comparison")
    print(f"epsilon: {epsilon}")
    print(f"parameter values: {parameter_values}")

    for method in ("central", "forward", "backward"):
        grad = FiniteDiffEstimatorGradient(
            estimator=estimator,
            epsilon=epsilon,
            method=method,
        )
        result = grad.run(
            circuits=[qc],
            observables=[observable],
            parameter_values=[parameter_values],
        ).result()
        _print_estimator_gradients(f"method={method}", result.gradients)
        print(f"  metadata: {result.metadata}")


def example_estimator_parameter_subset() -> None:
    """Differentiate only a subset of circuit parameters."""
    theta = ParameterVector("theta", 3)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.rz(theta[2], 1)
    qc.cx(0, 1)

    observable = SparsePauliOp("ZZ")
    parameter_values = [0.2, -0.1, 0.9]

    estimator = StatevectorEstimator()
    grad = FiniteDiffEstimatorGradient(
        estimator=estimator,
        epsilon=1e-3,
        method="central",
    )

    result = grad.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[parameter_values],
        parameters=[[theta[0], theta[2]]],
    ).result()

    _print_header("FiniteDiffEstimatorGradient: selected parameters")
    print("selected parameters: [theta[0], theta[2]]")
    print(f"gradient shape: {np.asarray(result.gradients[0]).shape}")
    print(f"gradient values: {np.asarray(result.gradients[0])}")
    print(f"metadata: {result.metadata}")


def example_sampler_distribution_gradient() -> None:
    """Compute gradients of output probability distributions."""
    theta = ParameterVector("theta", 2)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rx(theta[1], 1)
    qc.cx(0, 1)
    qc.measure_all()

    parameter_values = [0.41, -0.27]

    sampler = StatevectorSampler()
    grad = FiniteDiffSamplerGradient(
        sampler=sampler,
        epsilon=1e-3,
        method="central",
    )

    result = grad.run(
        circuits=[qc],
        parameter_values=[parameter_values],
    ).result()

    _print_header("FiniteDiffSamplerGradient: distribution gradients")
    print(f"parameter values: {parameter_values}")
    for circuit_index, per_parameter_grads in enumerate(result.gradients):
        print(f"circuit[{circuit_index}]:")
        for param_index, grad_dict in enumerate(per_parameter_grads):
            pretty = _format_distribution_grad(grad_dict, num_qubits=2)
            print(f"  dP/dtheta[{param_index}] = {pretty}")
    print(f"metadata: {result.metadata}")


def main() -> None:
    example_estimator_all_methods()
    example_estimator_parameter_subset()
    example_sampler_distribution_gradient()


if __name__ == "__main__":
    main()
