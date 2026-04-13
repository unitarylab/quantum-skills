"""General 1D linear PDE Schrodingerization demo.

Model:
    u_t = a2 * u_xx + a1 * u_x + a0 * u + f(x)
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    first_order_derivative,
    matrix_exponential,
    second_order_derivative,
)


def main() -> None:
    n = 8
    a2 = 0.08
    a1 = -0.15
    a0 = -0.05
    T = 0.02
    dx = 1.0 / n

    D1, b1 = first_order_derivative(N=n, dx=dx, boundary_condition="dirichlet", scheme="upwind")
    D2, b2 = second_order_derivative(N=n, dx=dx, boundary_condition="dirichlet")

    A = (a2 * D2.toarray() + a1 * D1.toarray() + a0 * np.eye(n)).astype(complex)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    f = 0.1 * np.cos(2.0 * np.pi * x)
    b = np.asarray(a2 * b2 + a1 * b1 + f, dtype=complex)
    u0 = np.sin(np.pi * x).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    print("General Linear 1D Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

