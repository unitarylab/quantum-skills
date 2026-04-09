"""
Beginner UnitaryLab example: create and run a 1-qubit Hadamard circuit.

Circuit:
1) Start in |0>
2) Apply H on qubit 0
Expected measurement probabilities:
- P(|0>) = 0.5
- P(|1>) = 0.5
"""

import numpy as np

try:
    from engine import GateSequence
except ModuleNotFoundError as exc:
    raise SystemExit(
        "UnitaryLab engine is not installed in the active Python environment.\n"
        "Install it first, for example:\n"
        "  python -m pip install <path-to-unitarylab_engine-*.whl>"
    ) from exc


def beginner_hadamard_example() -> dict:
    """Build and execute a 1-qubit H circuit, then validate probabilities."""
    circuit = GateSequence(1, name="beginner_hadamard")
    circuit.h(0)

    # |0> state for one qubit.
    initial_state = np.array([1.0, 0.0], dtype=complex)
    final_state = circuit.execute(initial_state.copy())

    probabilities = np.abs(final_state) ** 2
    expected_probabilities = np.array([0.5, 0.5], dtype=float)
    matches_theory = np.allclose(probabilities, expected_probabilities, atol=1e-6)

    print("Beginner Quantum Circuit Example")
    print("=" * 33)
    print("Circuit: H(0)")
    print("Initial state: |0>")
    print("\nFinal statevector:")
    print(np.array2string(final_state, precision=6, suppress_small=True))

    print("\nMeasurement probabilities:")
    print(f"P(|0>) = {probabilities[0]:.6f}")
    print(f"P(|1>) = {probabilities[1]:.6f}")

    print("\nExpected: P(|0>) = 0.5, P(|1>) = 0.5")
    print(f"Validation passed: {matches_theory}")

    return {
        "final_state": final_state,
        "probabilities": probabilities,
        "matches_theory": matches_theory,
    }


if __name__ == "__main__":
    beginner_hadamard_example()
