"""Examples for Quantum Amplitude Estimation using the documented API."""

from engine import GateSequence
from engine.algorithms import AmplitudeEstimationAlgorithm


def run_uniform_2qubit_example() -> None:
    """Estimate p for |00> in a 2-qubit uniform superposition (true p=0.25)."""
    U = GateSequence(2)
    U.h(0)
    U.h(1)

    algo = AmplitudeEstimationAlgorithm()
    result = algo.run(
        U=U,
        good_zero_qubits=[0, 1],
        d=6,
        backend="torch",
        algo_dir="./qae_results",
    )

    print("=" * 60)
    print("Amplitude Estimation Example: 2-Qubit Uniform State")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Status: {result.get('status')}")
    print(f"Estimated p: {result.get('estimated_amplitude')}")
    print(f"Expected p: {0.25:.6f}")
    print(f"Estimated phi: {result.get('phi')}")
    print(f"Circuit path: {result.get('circuit_path')}")


def run_biased_state_example() -> None:
    """Estimate p for a biased single-qubit preparation embedded in 2 qubits."""
    U = GateSequence(2)
    U.ry(0.8, 0)
    U.h(1)

    algo = AmplitudeEstimationAlgorithm()
    result = algo.run(
        U=U,
        good_zero_qubits=[0],
        d=5,
        backend="torch",
        algo_dir="./qae_results",
    )

    print("=" * 60)
    print("Amplitude Estimation Example: Biased Preparation")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Status: {result.get('status')}")
    print(f"Estimated p: {result.get('estimated_amplitude')}")
    print(f"Estimated phi: {result.get('phi')}")


if __name__ == "__main__":
    run_uniform_2qubit_example()
    run_biased_state_example()
