"""Trotter-Suzuki Hamiltonian Simulation — approximate e^{-iHt} via product formulas."""

from unitarylab.algorithms import SuzukiTrotterAlgorithm


def main():
    # 2-qubit Heisenberg-like Hamiltonian as grouped Pauli terms
    # Each group is a list of (pauli_string, coefficient) tuples
    grouped_paulis = [
        [("ZI", 0.5), ("IZ", 0.5)],
        [("XX", 0.3), ("YY", 0.3)],
    ]
    total_time = 1.0

    print("=" * 50)
    print("Trotter-Suzuki Hamiltonian Simulation")
    print("=" * 50)

    for order in [1, 2, 4]:
        algo = SuzukiTrotterAlgorithm(order=order, reps=1)
        result = algo.run(
            grouped_paulis=grouped_paulis,
            total_time=total_time,
            backend="torch",
        )

        print(f"\n  Order {order}:")
        print(f"    status : {result['status']}")
        print(result.get("plot", ""))


if __name__ == "__main__":
    main()
