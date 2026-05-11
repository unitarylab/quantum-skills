"""Amplitude Amplification — boost probability of target states."""

from unitarylab-algorithms import AmplitudeAmplificationAlgorithm
from unitarylab.core import Circuit


def example_2qubit():
    """Amplify |00> from a 2-qubit uniform superposition (initial p=0.25)."""
    U = Circuit(2, name="PrepU", backend="torch")
    U.h(0)
    U.h(1)

    algo = AmplitudeAmplificationAlgorithm()
    result = algo.run(
        U=U,
        good_zero_qubits=[0, 1],
        p=0.25,
        backend="torch",
    )

    print("=" * 50)
    print("Amplitude Amplification: 2-Qubit (p=0.25)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status        : {result['status']}")
    print(f"  Amplified prob: {result['amplified_prob']:.4f}")
    print(f"  Circuit path  : {result.get('circuit_path')}")


def example_3qubit():
    """Amplify |000> from a 3-qubit uniform superposition (initial p=0.125)."""
    U = Circuit(3, name="PrepU", backend="torch")
    for q in range(3):
        U.h(q)

    algo = AmplitudeAmplificationAlgorithm()
    result = algo.run(
        U=U,
        good_zero_qubits=[0, 1, 2],
        p=1.0 / 8.0,
        backend="torch",
    )

    print("=" * 50)
    print("Amplitude Amplification: 3-Qubit (p=0.125)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status        : {result['status']}")
    print(f"  Amplified prob: {result['amplified_prob']:.4f}")


if __name__ == "__main__":
    example_2qubit()
    example_3qubit()
