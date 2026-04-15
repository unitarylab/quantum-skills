"""
Taylor-Series Hamiltonian Simulation Example
============================================
Demonstrates truncated-Taylor LCU-based time evolution
using the Taylor class from UnitaryLab.
"""

import numpy as np
from engine.algorithms.taylor import Taylor


def main():
    H = np.array([
        [1.0, 0.4],
        [0.4, -0.5],
    ], dtype=complex)

    t = 0.8

    for degree in [5, 10]:
        sim = Taylor(H=H, t=t, target_error=1e-4, degree=degree)

        print(f"Degree {degree}:")
        print(f"  method        : {sim.method}")
        print(f"  effective deg  : {sim.degree}")
        print(f"  lambda=||H||*t: {sim.lam:.4f}")
        print(f"  spectral error: {sim.total_error:.6e}")
        print()


if __name__ == "__main__":
    main()
