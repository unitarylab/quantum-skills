"""1D traffic flow (LWR) linearized surrogate via Schrodingerization.

Nonlinear model:
    rho_t + (rho * (1-rho))_x = 0
Linearization around rho_bar gives:
    rho_t + c(x) * rho_x = 0,   c(x) = 1 - 2 rho_bar(x)
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

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    rho_bar = 0.35 + 0.1 * np.sin(2.0 * np.pi * x)
    c = 1.0 - 2.0 * rho_bar

    D1, _ = first_order_derivative(N=n, dx=dx, boundary_condition="periodic", scheme="central")
    A = (-(np.diag(c) @ D1.toarray())).astype(complex)
    u0 = (0.02 * np.exp(-90.0 * (x - 0.5) ** 2)).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("1D Traffic Flow (Linearized) Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

