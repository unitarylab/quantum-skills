"""QAOA — Quantum Approximate Optimization for Max-Cut."""

from unitarylab.algorithms import QAOAAlgorithm


def main():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
    n_qubits = 6

    algo = QAOAAlgorithm(seed=42)
    result = algo.run(
        edges=edges,
        n_qubits=n_qubits,
        p_layers=4,
        max_iter=100,
        backend="torch",
    )

    print("=" * 50)
    print("QAOA: Max-Cut on 6-node graph")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status         : {result['status']}")
    print(f"  Best cut value : {result['maxcut']}")


if __name__ == "__main__":
    main()
