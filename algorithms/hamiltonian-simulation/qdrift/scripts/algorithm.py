"""QDrift Hamiltonian Simulation — randomized Pauli sampling for e^{-iHt}."""

from unitarylab_algorithms import QDriftAlgorithm


def main():
    # 2-qubit Hamiltonian as (pauli_string, coefficient) list
    H_list = [
        ("ZI", 0.45),
        ("IZ", 0.45),
        ("XX", 0.1),
    ]
    t = 1.0
    n_qubits = 2

    print("=" * 50)
    print("QDrift Hamiltonian Simulation")
    print("=" * 50)

    for epsilon in [0.1, 0.01]:
        algo = QDriftAlgorithm(seed=1234)
        result = algo.run(
            H_list=H_list,
            t=t,
            epsilon=epsilon,
            n_qubits=n_qubits,
            backend="torch",
        )

        print(f"\n  Epsilon {epsilon}:")
        print(f"    status : {result['status']}")
        print(f"    error  : {result['error']:.6e}")
        print(result.get("plot", ""))


if __name__ == "__main__":
    main()
