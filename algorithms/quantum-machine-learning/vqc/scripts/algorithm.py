"""VQC — Variational Quantum Classifier on the Iris dataset."""

from unitarylab_algorithms import VQCAlgorithm


def main():
    algo = VQCAlgorithm(seed=42)
    result = algo.run(
        n_layers=3,
        epochs=20,
        lr=0.05,
        batch_size=16,
        backend="torch",
    )

    print("=" * 50)
    print("VQC: Iris Classification (4 features, 3 classes)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status         : {result['status']}")
    print(f"  Test accuracy  : {result['accuracy']:.2%}")


if __name__ == "__main__":
    main()
