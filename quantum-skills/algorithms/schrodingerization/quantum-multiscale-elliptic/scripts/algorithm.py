"""Multiscale elliptic equation via pseudo-time Schrodingerization.

Steady model:
    -d/dx(A(x) du/dx) = f
We use a simplified operator A(x) * u_xx as a small runnable surrogate.
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main() -> None:
    n = 8
    T = 0.01
    dx = 1.0 / n

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    Acoeff = 1.2 + 0.3 * np.sin(2.0 * np.pi * x / 0.2)

    D2, bvec = second_order_derivative(N=n, dx=dx, boundary_condition="dirichlet")
    A = (np.diag(Acoeff) @ D2.toarray()).astype(complex)
    f = np.sin(np.pi * x)
    b = np.asarray(bvec + f, dtype=complex)
    u0 = np.zeros(n, dtype=complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    print("Multiscale Elliptic Surrogate Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

