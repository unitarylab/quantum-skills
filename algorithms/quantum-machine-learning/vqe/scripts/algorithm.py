"""VQE — Variational Quantum Eigensolver for 2-qubit Ising Hamiltonian."""

from unitarylab.algorithms import VQEAlgorithm


def main():
    algo = VQEAlgorithm(seed=42)
    result = algo.run(
        n_qubits=2,
        n_layers=3,
        max_iter=150,
        backend="torch",
    )

    print("=" * 50)
    print("VQE: 2-Qubit Ising Ground State")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status         : {result['status']}")
    print(f"  VQE energy     : {result['energy']:.4f}")


if __name__ == "__main__":
    main()
