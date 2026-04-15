"""
QDrift Hamiltonian Simulation Example
=====================================
Demonstrates randomized Hamiltonian simulation via stochastic
Pauli sampling using the QDrift class from UnitaryLab.
"""

import numpy as np
from engine.algorithms.qdrift import QDrift


def main():
    np.random.seed(1234)

    H = np.array([
        [0.9, 0.1],
        [0.1, -0.9],
    ], dtype=complex)

    t = 1.0

    for steps in [1000, 4000]:
        sim = QDrift(H=H, t=t, target_error=1e-3, steps=steps)

        print(f"Steps {steps}:")
        print(f"  method        : {sim.method}")
        print(f"  target_qubits : {sim.target_qubits}")
        print(f"  spectral error: {sim.total_error:.6e}")
        print()


if __name__ == "__main__":
    main()
