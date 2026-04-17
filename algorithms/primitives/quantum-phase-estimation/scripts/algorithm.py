"""Quantum Phase Estimation (QPE) — estimate eigenphase of a unitary operator."""

import numpy as np
from engine.algorithms import QPEAlgorithm
from engine.core import GateSequence


def example_s_gate():
    """Estimate phase of S gate eigenstate |1>.  Expected phi = 0.25."""
    U = GateSequence(1, name="S_gate", backend="torch")
    U.s(0)

    prepare_target = GateSequence(1, name="prep_1", backend="torch")
    prepare_target.x(0)

    algo = QPEAlgorithm()
    result = algo.run(
        U=U,
        d=4,
        prepare_target=prepare_target,
        backend="torch",
    )

    print("=" * 50)
    print("QPE Example: S Gate  (expected phi = 0.25)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Estimated phase : {result['estimated_phase']}")
    print(f"  Best probability: {result['confidence_probability']:.4f}")
    print(f"  Circuit path    : {result.get('circuit_path')}")


def example_t_gate():
    """Estimate phase of T gate eigenstate |1>.  Expected phi = 0.125."""
    U = GateSequence(1, name="T_gate", backend="torch")
    U.p(np.pi / 4, 0)

    prepare_target = GateSequence(1, name="prep_1", backend="torch")
    prepare_target.x(0)

    algo = QPEAlgorithm()
    result = algo.run(
        U=U,
        d=5,
        prepare_target=prepare_target,
        backend="torch",
    )

    print("=" * 50)
    print("QPE Example: T Gate  (expected phi = 0.125)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Estimated phase : {result['estimated_phase']}")
    print(f"  Best probability: {result['confidence_probability']:.4f}")


if __name__ == "__main__":
    example_s_gate()
    example_t_gate()
