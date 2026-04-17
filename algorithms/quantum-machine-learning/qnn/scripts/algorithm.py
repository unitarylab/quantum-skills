"""QNN — Quantum Neural Network for binary classification."""

import numpy as np
from engine.algorithms import QNNAlgorithm


def main():
    np.random.seed(0)
    X_train = np.random.randn(40, 2)
    y_train = (X_train[:, 0] > 0).astype(int)

    algo = QNNAlgorithm(seed=42)
    result = algo.run(
        x_train=X_train,
        y_train=y_train,
        n_qubits=2,
        layers=2,
        epochs=20,
        learning_rate=0.1,
        backend="torch",
    )

    print("=" * 50)
    print("QNN: Binary Classification (2 features)")
    print("=" * 50)
    print(result.get("plot", ""))
    print(f"  Status        : {result['status']}")
    print(f"  Final accuracy: {result['accuracy']:.2%}")


if __name__ == "__main__":
    main()
