"""Examples for Amplitude Amplification using UnitaryLab engine APIs."""

from engine import GateSequence
from engine.algorithms.fundamental_algorithm import AmplitudeAmplificationAlgorithm


def run_uniform_2qubit_example() -> None:
    """Amplify |00> from a 2-qubit uniform superposition."""
    U_prep = GateSequence(2)
    U_prep.h(0)
    U_prep.h(1)

    algo = AmplitudeAmplificationAlgorithm()
    result = algo.run(
        U=U_prep,
        good_zero_qubits=[0, 1],
        p=0.25,
        backend="torch",
        algo_dir="./aa_results",
    )

    print("=" * 60)
    print("Amplitude Amplification Example: 2-Qubit Uniform State")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Status: {result.get('status')}")
    print(f"Initial p: {0.25:.6f}")
    print(f"Amplified p: {result.get('amplified_prob')}")
    print(f"Circuit path: {result.get('circuit_path')}")


def run_3qubit_example() -> None:
    """Amplify |000> from a 3-qubit uniform superposition."""
    U_prep = GateSequence(3)
    for q in range(3):
        U_prep.h(q)

    initial_p = 1.0 / 8.0
    algo = AmplitudeAmplificationAlgorithm()
    result = algo.run(
        U=U_prep,
        good_zero_qubits=[0, 1, 2],
        p=initial_p,
        backend="torch",
        algo_dir="./aa_results",
    )

    print("=" * 60)
    print("Amplitude Amplification Example: 3-Qubit Uniform State")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Status: {result.get('status')}")
    print(f"Initial p: {initial_p:.6f}")
    print(f"Amplified p: {result.get('amplified_prob')}")
    print(f"Circuit path: {result.get('circuit_path')}")


if __name__ == "__main__":
    run_uniform_2qubit_example()
    run_3qubit_example()
