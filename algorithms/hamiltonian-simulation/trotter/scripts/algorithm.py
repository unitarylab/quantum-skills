"""Trotter-Suzuki Hamiltonian Simulation — approximate e^{-iHt} via product formulas."""

import numpy as np
from unitarylab_algorithms.hamiltonian_simulation.trotter.algorithm import TrotterAlgorithm


def main():
    # 2-qubit Heisenberg-like Hamiltonian as a numpy matrix
    H = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    t = 1.0
    error = 1e-8

    print("=" * 50)
    print("Trotter-Suzuki Hamiltonian Simulation")
    print("=" * 50)

    for order in [1, 2, 4]:
        algo = TrotterAlgorithm()
        result = algo.run(H=H, t=t, error=error, order=order, steps=1000, backend="torch")

        print(f"\n  Order {order}:")
        print(f"    status          : {result['status']}")
        print(f"    Frobenius error : {result['Frobenius norm of error']:.2e}")
        print(f"    circuit_path    : {result['circuit_path']}")
        print(f"    saved files     : {result['plot']}")


if __name__ == "__main__":
    main()
