"""Quantum Amplitude Estimation (QAE) — estimate success probability of a circuit."""

from unitarylab.algorithms import AmplitudeEstimationAlgorithm
from unitarylab.core import Circuit


def example_uniform():
    """Estimate p(|00>) in a 2-qubit uniform state.  Expected p ~ 0.25."""
    U = Circuit(2, name="PrepU", backend="torch")
    U.h(0)
    U.h(1)

    algo = AmplitudeEstimationAlgorithm()
    result = algo.run(
        U=U,
        good_zero_qubits=[0, 1],
        d=6,
        backend="torch",
    )

    print("=" * 50)
    print("QAE Example: 2-Qubit Uniform  (expected p ~ 0.25)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Estimated amplitude: {result['estimated_amplitude']:.4f}")
    print(f"  Estimated phi      : {result['phi']:.4f}")
    print(f"  Circuit path       : {result.get('circuit_path')}")


def example_biased():
    """Estimate p for a biased single-qubit Ry preparation."""
    U = Circuit(2, name="PrepU", backend="torch")
    U.ry(1.1, 0)
    U.cx(0, 1)

    algo = AmplitudeEstimationAlgorithm()
    result = algo.run(
        U=U,
        good_zero_qubits=[0],
        d=6,
        backend="torch",
    )

    print("=" * 50)
    print("QAE Example: Biased Preparation")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Estimated amplitude: {result['estimated_amplitude']:.4f}")
    print(f"  Estimated phi      : {result['phi']:.4f}")


if __name__ == "__main__":
    example_uniform()
    example_biased()
