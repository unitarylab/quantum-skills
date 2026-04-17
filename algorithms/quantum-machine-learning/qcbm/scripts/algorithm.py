"""QCBM — Quantum Circuit Born Machine for learning Bars-and-Stripes distribution."""

from engine.algorithms import QCBMAlgorithm


def main():
    algo = QCBMAlgorithm(seed=42)
    result = algo.run(
        n_qubits=4,
        n_layers=4,
        epochs=40,
        lr=0.1,
        backend="torch",
    )

    print("=" * 50)
    print("QCBM: 2x2 Bars-and-Stripes")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status          : {result['status']}")
    print(f"  Final KL div    : {result['loss']:.4f}")


if __name__ == "__main__":
    main()
