"""Examples for Quantum Phase Estimation (QPE) using QPEAlgorithm."""

import numpy as np
from engine import GateSequence
from engine.algorithms.fundamental_algorithm import QPEAlgorithm


def run_t_gate_example() -> None:
    """Estimate phase for T gate eigenstate |1>, expected phase=1/8."""
    U = GateSequence(1)
    U.p(np.pi / 4, 0)

    prepare_target = GateSequence(1)
    prepare_target.x(0)

    algo = QPEAlgorithm()
    result = algo.run(
        U=U,
        d=5,
        prepare_target=prepare_target,
        backend="torch",
        algo_dir="./qpe_results",
    )

    print("=" * 60)
    print("QPE Example: T Gate")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Estimated phase: {result.get('estimated_phase')}")
    print(f"Expected phase: {1.0/8.0:.6f}")
    print(f"Confidence: {result.get('confidence_probability')}")
    print(f"Circuit path: {result.get('circuit_path')}")


def run_s_gate_example() -> None:
    """Estimate phase for S gate eigenstate |1>, expected phase=1/4."""
    U = GateSequence(1)
    U.p(np.pi / 2, 0)

    prepare_target = GateSequence(1)
    prepare_target.x(0)

    algo = QPEAlgorithm()
    result = algo.run(
        U=U,
        d=5,
        prepare_target=prepare_target,
        backend="torch",
        algo_dir="./qpe_results",
    )

    print("=" * 60)
    print("QPE Example: S Gate")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Estimated phase: {result.get('estimated_phase')}")
    print(f"Expected phase: {1.0/4.0:.6f}")
    print(f"Confidence: {result.get('confidence_probability')}")


if __name__ == "__main__":
    run_t_gate_example()
    run_s_gate_example()
