"""QAOA — Quantum Approximate Optimization for Max-Cut."""

from unitarylab_algorithms import QAOAAlgorithm


def main():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)]
    n = 6

    algo = QAOAAlgorithm(text_mode="plain")
    result = algo.run(
        edges=edges,
        n=n,
        layers=4,
        max_iter=100,
        backend="torch",
    )

    print("=" * 50)
    print("QAOA: Max-Cut on 6-node graph")
    print("=" * 50)
    for plot_file in result.get("plot", []):
        print(f"  Plot           : {plot_file['filename']}")
    print(f"  Status         : {result['status']}")
    print(f"  Best cut value : {result['Max-Cut Value']}")
    print(f"  Optimal bits   : {result['Optimal bitstring']}")
    print(f"  Optimized energy: {result['Optimized Energy']:.6f}")


if __name__ == "__main__":
    main()
