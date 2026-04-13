"""1D Maxwell first-order system via Schrodingerization.

Model (normalized units):
    E_t = -H_x
    H_t = -E_x
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    first_order_derivative,
    matrix_exponential,
)


def main() -> None:
    n = 8
    T = 0.01
    dx = 1.0 / n

    D1, _ = first_order_derivative(N=n, dx=dx, boundary_condition="periodic", scheme="central")
    D = D1.toarray()
    Z = np.zeros((n, n))
    A = np.block([[Z, -D], [-D, Z]]).astype(complex)

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    E0 = np.sin(2.0 * np.pi * x)
    H0 = np.zeros_like(E0)
    u0 = np.concatenate([E0, H0]).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("1D Maxwell Schrodingerization")
    print(f"System size: {2 * n}")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

