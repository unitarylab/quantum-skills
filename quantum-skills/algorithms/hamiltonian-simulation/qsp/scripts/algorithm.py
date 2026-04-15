"""
QSP Hamiltonian Simulation Example
===================================
Demonstrates block-encoding + Quantum Signal Processing
time evolution using the QSP class from UnitaryLab.
"""

import numpy as np
from engine.algorithms.qsp import QSP


def main():
    H = np.array([
        [0.8, 0.2],
        [0.2, -0.8],
    ], dtype=complex)

    t = 1.2

    sim = QSP(H=H, t=t, target_error=1e-4, degree=15, beta=0.7)

    print(f"method : {sim.method}")
    print(f"degree : {sim.degree}")
    print(f"alpha  : {sim.alpha:.4f}")
    print(f"beta   : {sim.beta}")
    print(f"error  : {sim.total_error:.6e}")


if __name__ == "__main__":
    main()
