"""SPSA gradient demos for estimator and sampler primitives."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import SPSAEstimatorGradient, SPSASamplerGradient


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


def example_spsa_estimator_gradient() -> None:
    """Estimate expectation-value gradients with SPSA."""
    epsilon = 0.01
    batch_size = 4
    seed = 123

    theta = ParameterVector("theta", 3)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.rz(theta[1], 1)
    qc.cx(0, 1)
    qc.rx(theta[2], 1)

    observable = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.2),
    ])
    parameter_values = [0.3, -0.2, 0.9]

    estimator = StatevectorEstimator()
    grad = SPSAEstimatorGradient(
        estimator=estimator,
        epsilon=epsilon,
        batch_size=batch_size,
        seed=seed,
    )

    result = grad.run(
        circuits=[qc],
        observables=[observable],
        parameter_values=[parameter_values],
        parameters=[[theta[0], theta[2]]],
        precision=None,
    ).result()

    _print_header("SPSAEstimatorGradient: selected parameters")
    print(f"epsilon: {epsilon}, batch_size: {batch_size}, seed: {seed}")
    print(f"parameter values: {parameter_values}")
    print("selected parameters: [theta[0], theta[2]]")
    print(f"gradient shape: {np.asarray(result.gradients[0]).shape}")
    print(f"gradient: {np.asarray(result.gradients[0])}")
    print(f"precision: {result.precision}")
    print(f"metadata: {result.metadata}")


def example_spsa_sampler_gradient() -> None:
    """Estimate output-distribution gradients with SPSA."""
    epsilon = 0.01
    batch_size = 4
    seed = 123

    theta = ParameterVector("theta", 2)

    qc = QuantumCircuit(2)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.measure_all()

    parameter_values = [0.3, -0.2]

    sampler = StatevectorSampler()
    grad = SPSASamplerGradient(
        sampler=sampler,
        epsilon=epsilon,
        batch_size=batch_size,
        seed=seed,
    )

    result = grad.run(
        circuits=[qc],
        parameter_values=[parameter_values],
        parameters=[[theta[0], theta[1]]],
        shots=None,
    ).result()

    _print_header("SPSASamplerGradient: distribution gradients")
    print(f"epsilon: {epsilon}, batch_size: {batch_size}, seed: {seed}")
    print(f"parameter values: {parameter_values}")
    for circuit_index, per_parameter_grads in enumerate(result.gradients):
        print(f"circuit[{circuit_index}]:")
        for param_index, grad_dict in enumerate(per_parameter_grads):
            pretty = _format_distribution_grad(grad_dict, num_qubits=2)
            print(f"  dP/dtheta[{param_index}] = {pretty}")
    print(f"shots: {getattr(result, 'shots', None)}")
    print(f"metadata: {result.metadata}")


def main() -> None:
    example_spsa_estimator_gradient()
    example_spsa_sampler_gradient()


if __name__ == "__main__":
    main()
