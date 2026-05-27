"""VQC — Variational Quantum Classifier on the Iris dataset."""

from unitarylab_algorithms import VQCAlgorithm


def main():
    algo = VQCAlgorithm(text_mode="plain")
    result = algo.run(
        layers=3,
        epochs=20,
        lr=0.05,
        batch_size=16,
        backend="torch",
    )

    print("=" * 50)
    print("VQC: Iris Classification (4 features, 3 classes)")
    print("=" * 50)
    for plot_file in result.get("plot", []):
        print(f"  Plot           : {plot_file['filename']}")
    print(f"  Status         : {result['status']}")
    print(f"  Test accuracy  : {result['Final Accuracy']:.2%}")
    print(f"  Final loss     : {result['Final Loss']:.4f}")
    print(f"  Quantum time   : {result['Quantal Computation Time (s)']:.4f}s")


if __name__ == "__main__":
    main()
