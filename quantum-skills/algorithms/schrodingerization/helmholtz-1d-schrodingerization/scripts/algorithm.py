"""1D Helmholtz equation via pseudo-time Schrodingerization workflow.

Steady equation:
    -u_xx - k^2 u = f
We use pseudo-time evolution:
    u_t = (u_xx + k^2 u + f)
until short final time for demonstration.
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main() -> None:
    n = 8
    k = 3.0
    T = 0.01
    dx = 1.0 / n

    D2, bvec = second_order_derivative(N=n, dx=dx, boundary_condition="dirichlet")
    A = (D2.toarray() + (k * k) * np.eye(n)).astype(complex)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    f = np.sin(np.pi * x)
    b = np.asarray(bvec + f, dtype=complex)
    u0 = np.zeros(n, dtype=complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    print("1D Helmholtz (Pseudo-Time) Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

