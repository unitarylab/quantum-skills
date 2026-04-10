"""Examples for Hadamard Transform algorithm modes."""

from engine.algorithms import HadamardTransformAlgorithm


def run_superposition_example() -> None:
    """Create uniform superposition with n=3 qubits."""
    algo = HadamardTransformAlgorithm()
    result = algo.run(
        n_qubits=3,
        mode="superposition",
        backend="torch",
        algo_dir="./hadamard_results",
    )

    print("=" * 60)
    print("Hadamard Transform Example: Superposition Mode")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Status: {result.get('status')}")
    print(f"Probabilities: {result.get('probabilities')}")
    print(f"Circuit path: {result.get('circuit_path')}")


def run_reflexive_test_example() -> None:
    """Verify H^2 = I with reflexive test mode."""
    algo = HadamardTransformAlgorithm()
    result = algo.run(
        n_qubits=2,
        mode="reflexive_test",
        backend="torch",
        algo_dir="./hadamard_results",
    )

    print("=" * 60)
    print("Hadamard Transform Example: Reflexive Test")
    print("=" * 60)
    print(result.get("plot", "(no plot)"))
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('message')}")


if __name__ == "__main__":
    run_superposition_example()
    run_reflexive_test_example()
