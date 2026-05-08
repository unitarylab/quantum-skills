"""Taylor Series Hamiltonian Simulation — approximate e^{-iHt} via LCU of Taylor-expanded Pauli terms."""

import numpy as np
from unitarylab.algorithms import TaylorAlgorithm


def main():
    # 2×2 Hermitian Hamiltonian
    H = np.array([[2, 1],
                  [1, 3]], dtype=complex)

    t = 1.0
    error = 1e-8

    print("=" * 55)
    print("Taylor Series Hamiltonian Simulation")
    print("=" * 55)
    print(f"\nHamiltonian H =\n{H}")
    print(f"Evolution time t = {t}")
    print(f"Target error    = {error}\n")

    # --- Minimal example ---
    algo = TaylorAlgorithm(text_mode="plain")
    result = algo.run(H=H, t=t, error=error, degree=15)

    print("status      :", result["status"])
    print("circuit_path:", result["circuit_path"])
    print("file_path   :", result["file_path"])
    print("Frobenius error:", algo.output["Frobenius norm of error"])

    # --- Accuracy sweep: degree vs. t ---
    print("\n" + "=" * 55)
    print("Accuracy Sweep — degree vs. t")
    print("=" * 55)
    print(f"{'t':>5}  {'degree':>6}  {'Frobenius error':>16}  {'status':>6}")
    print("-" * 40)

    for t_val in [1.0, 3.0, 5.0]:
        for deg in [5, 10, 15]:
            sweep_algo = TaylorAlgorithm(text_mode="plain")
            sweep_result = sweep_algo.run(H=H, t=t_val, error=1e-8, degree=deg)
            frob_err = sweep_algo.output["Frobenius norm of error"]
            print(f"{t_val:>5.1f}  {deg:>6d}  {frob_err:>16.2e}  {sweep_result['status']:>6}")


if __name__ == "__main__":
    main()
