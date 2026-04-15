"""
Trotter-Suzuki Hamiltonian Simulation Example
==============================================
Demonstrates time-evolution approximation via product formulas
using the Trotter class from UnitaryLab.
"""

import numpy as np
from engine.algorithms.trotter import Trotter


def main():
    # 2x2 Hermitian Hamiltonian
    H = np.array([
        [1.0, 0.3],
        [0.3, -1.0],
    ], dtype=complex)

    t = 1.0

    for order in [1, 2, 4]:
        sim = Trotter(H=H, t=t, target_error=1e-3, order=order, steps=80)

        print(f"Order {order}:")
        print(f"  method         : {sim.method}")
        print(f"  target_qubits  : {sim.target_qubits}")
        print(f"  effective_steps: {sim.steps}")
        print(f"  spectral error : {sim.total_error:.6e}")
        print()


if __name__ == "__main__":
    main()
