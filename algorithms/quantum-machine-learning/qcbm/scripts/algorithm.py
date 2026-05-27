"""QCBM — Quantum Circuit Born Machine for learning Bars-and-Stripes distribution."""

from unitarylab_algorithms import QCBMAlgorithm


def main():
    algo = QCBMAlgorithm(text_mode="plain")
    result = algo.run(
        n=4,
        layers=4,
        epochs=40,
        lr=0.1,
        backend="torch",
    )

    print("=" * 50)
    print("QCBM: 2x2 Bars-and-Stripes")
    print("=" * 50)
    for f in result.get("plot", []):
        print(f"  Output file     : {f['filename']} ({f['format']})")
    print(f"  Status          : {result['status']}")
    print(f"  Final KL div    : {result['Final KL Loss']:.4f}")


if __name__ == "__main__":
    main()
