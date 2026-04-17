"""LCU — apply a linear combination of unitaries M = sum(alpha_j * U_j)."""

import numpy as np
from engine.algorithms import LCUAlgorithm
from engine.core import GateSequence


def example_h_plus_x():
    """Apply M = 0.6*H + 0.4*X to a single qubit starting in |0>."""
    n_sys = 1

    U0 = GateSequence(n_sys, backend="torch")
    U0.h(0)

    U1 = GateSequence(n_sys, backend="torch")
    U1.x(0)

    algo = LCUAlgorithm()
    result = algo.run(
        alphas=[0.6, 0.4],
        unitaries=[U0, U1],
        n_sys=n_sys,
        initial_state=None,
        backend="torch",
    )

    print("=" * 50)
    print("LCU Example: M = 0.6*H + 0.4*X")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Success prob : {result['lcu_success_probability']:.4f}")
    print(f"  Circuit path : {result.get('circuit_path')}")


def example_pauli_mix():
    """Apply M = 0.5*I + 0.3*Z + 0.2*X to |+>."""
    n_sys = 1

    U_I = GateSequence(n_sys, backend="torch")  # Identity (no gates)
    U_Z = GateSequence(n_sys, backend="torch")
    U_Z.z(0)
    U_X = GateSequence(n_sys, backend="torch")
    U_X.x(0)

    prep = GateSequence(n_sys, backend="torch")
    prep.h(0)

    algo = LCUAlgorithm()
    result = algo.run(
        alphas=[0.5, 0.3, 0.2],
        unitaries=[U_I, U_Z, U_X],
        n_sys=n_sys,
        initial_state=prep,
        backend="torch",
    )

    print("=" * 50)
    print("LCU Example: M = 0.5*I + 0.3*Z + 0.2*X on |+>")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Success prob : {result['lcu_success_probability']:.4f}")


if __name__ == "__main__":
    example_h_plus_x()
    example_pauli_mix()
