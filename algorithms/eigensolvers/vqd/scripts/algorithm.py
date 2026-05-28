"""Variational Quantum Deflation (VQD) demo for low-lying eigenvalues."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.eigensolvers import VQD
from qiskit_algorithms.optimizers import SLSQP
from qiskit_algorithms.state_fidelities import ComputeUncompute


def _print_header(title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)


def example_basic_vqd() -> None:
    """Compute the ground and first-excited energies with default penalties."""
    operator = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.3),
        ("IX", 0.3),
    ])

    ansatz = RealAmplitudes(num_qubits=2, reps=2)
    estimator = StatevectorEstimator()
    fidelity = ComputeUncompute(StatevectorSampler())
    optimizer = SLSQP(maxiter=100)

    solver = VQD(
        estimator=estimator,
        fidelity=fidelity,
        ansatz=ansatz,
        optimizer=optimizer,
        k=2,
    )

    result = solver.compute_eigenvalues(operator)

    _print_header("VQD: Basic Ground + First Excited State")
    print(f"  k requested            : {solver.k}")
    print(f"  Eigenvalues            : {result.eigenvalues}")
    print(f"  Optimal values         : {result.optimal_values}")
    print(f"  Cost function evals    : {result.cost_function_evals}")


def example_with_aux_operators() -> None:
    """Evaluate auxiliary observables on each optimized eigenstate."""
    operator = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XX", 0.2),
        ("ZI", -0.1),
    ])
    aux_ops = {
        "mag_z0": SparsePauliOp.from_list([("ZI", 1.0)]),
        "mag_z1": SparsePauliOp.from_list([("IZ", 1.0)]),
    }

    ansatz = RealAmplitudes(num_qubits=2, reps=2)
    estimator = StatevectorEstimator()
    fidelity = ComputeUncompute(StatevectorSampler())
    optimizer = SLSQP(maxiter=100)

    solver = VQD(
        estimator=estimator,
        fidelity=fidelity,
        ansatz=ansatz,
        optimizer=optimizer,
        k=2,
    )

    result = solver.compute_eigenvalues(operator, aux_operators=aux_ops)

    _print_header("VQD: Auxiliary Operator Evaluation")
    print(f"  Eigenvalues            : {result.eigenvalues}")
    print(f"  Aux evaluated          : {result.aux_operators_evaluated}")


def example_with_callback_and_options() -> None:
    """Use explicit betas, initial point, callback, and convergence threshold."""
    operator = SparsePauliOp.from_list([
        ("ZZ", 1.0),
        ("XI", 0.25),
        ("IX", 0.25),
    ])

    ansatz = RealAmplitudes(num_qubits=2, reps=2)
    estimator = StatevectorEstimator()
    fidelity = ComputeUncompute(StatevectorSampler())
    optimizer = SLSQP(maxiter=80)

    evaluation_log: list[tuple[int, int, float]] = []

    def callback(
        eval_count: int,
        params: np.ndarray,
        value: float,
        metadata: dict[str, Any],
        step: int,
    ) -> None:
        del params, metadata
        evaluation_log.append((step, eval_count, value))

    initial_point = np.zeros(ansatz.num_parameters, dtype=float)

    solver = VQD(
        estimator=estimator,
        fidelity=fidelity,
        ansatz=ansatz,
        optimizer=optimizer,
        k=2,
        betas=np.array([5.0]),
        initial_point=initial_point,
        callback=callback,
        convergence_threshold=1e-6,
    )

    result = solver.compute_eigenvalues(operator)

    _print_header("VQD: Callback + Convergence Controls")
    print(f"  Eigenvalues            : {result.eigenvalues}")
    print(f"  Optimal points shape   : {result.optimal_points.shape}")
    print(f"  Optimizer times (s)    : {result.optimizer_times}")

    if evaluation_log:
        print("  Last callback entries  :")
        for step, eval_count, value in evaluation_log[-5:]:
            print(
                "    "
                f"step={step} eval={eval_count} objective={value:.10f}"
            )


def main() -> None:
    example_basic_vqd()
    example_with_aux_operators()
    example_with_callback_and_options()


if __name__ == "__main__":
    main()
