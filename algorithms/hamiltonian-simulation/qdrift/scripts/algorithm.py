"""QDrift Hamiltonian Simulation — randomized Pauli sampling for e^{-iHt}."""

import numpy as np
from unitarylab_algorithms import QDriftAlgorithm


def main():
    # 2-qubit Heisenberg-like Hamiltonian (4×4 matrix)
    XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
    ZZ = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=float)
    H = XX + ZZ
    t = 1.0

    print("=" * 50)
    print("QDrift Hamiltonian Simulation")
    print("=" * 50)

    for steps in [1000, 5000]:
        np.random.seed(42)
        algo = QDriftAlgorithm()
        result = algo.run(
            H=H,
            t=t,
            error=1e-8,
            steps=steps,
            backend="torch",
        )

        print(f"\n  steps={steps}:")
        print(f"    status                : {result['status']}")
        print(f"    Frobenius norm of error: {result['Frobenius norm of error']:.6e}")
        print(f"    circuit_path          : {result['circuit_path']}")
        for f in result["plot"]:
            print(f"    saved {f['format']} file: {f['filename']}")


if __name__ == "__main__":
    main()
