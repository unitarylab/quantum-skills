"""Generic Schrodingerization demo for a linear ODE system.

This file demonstrates the core workflow from the method skill:
1. Build a non-Hermitian linear system du/dt = A u + b.
2. Solve with direct matrix exponential as reference.
3. Solve with schro_classical and compare errors.
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import matrix_exponential


def main() -> None:
    n = 8
    T = 0.02
    na = 6

    # Non-Hermitian upper-triangular drift plus weak damping.
    A = np.diag(-0.2 * np.ones(n)) + 0.5 * np.diag(np.ones(n - 1), k=1)
    A = A.astype(complex)
    b = np.linspace(0.0, 0.2, n, dtype=complex)
    u0 = np.exp(-10.0 * (np.linspace(0.0, 1.0, n) - 0.4) ** 2).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=na, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    err = float(np.linalg.norm(u_sch - u_ref))
    print("Generic Schrodingerization Method Demo")
    print("=" * 40)
    print(f"System size: {n}, T: {T}, na: {na}")
    print(f"L2 error ||u_sch - u_ref||_2 = {err:.6e}")


if __name__ == "__main__":
    main()

