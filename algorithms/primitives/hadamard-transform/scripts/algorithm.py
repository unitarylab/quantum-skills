"""Hadamard Transform — uniform superposition and self-inverse verification."""

from unitarylab.algorithms import HadamardTransformAlgorithm


def example_superposition():
    """Apply H^{otimes 3} to |000> and verify uniform distribution."""
    algo = HadamardTransformAlgorithm()
    result = algo.run(n_qubits=3, mode="superposition", backend="torch")

    print("=" * 50)
    print("Hadamard Transform: Superposition (3 qubits)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status       : {result['status']}")
    print(f"  Prob dist    : {result['probabilities']}")
    print(f"  Circuit path : {result.get('circuit_path')}")


def example_reflexive():
    """Verify H^2 = I by applying the transform twice."""
    algo = HadamardTransformAlgorithm()
    result = algo.run(n_qubits=3, mode="reflexive_test", backend="torch")

    print("=" * 50)
    print("Hadamard Transform: Reflexive Test (H^2 = I)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status        : {result['status']}")
    print(f"  Message       : {result['message']}")


if __name__ == "__main__":
    example_superposition()
    example_reflexive()
