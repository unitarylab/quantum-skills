"""1D viscous Burgers equation (linearized) via Schrodingerization.

Target nonlinear PDE:
    u_t + u * u_x = nu * u_xx
This demo linearizes convection around a frozen profile u_bar(x).
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
    nu = 0.02
    T = 0.01
    dx = 1.0 / n

    D1, _ = first_order_derivative(N=n, dx=dx, boundary_condition="periodic", scheme="central")
    D2, _ = second_order_derivative(N=n, dx=dx, boundary_condition="periodic")

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    u_bar = 0.5 + 0.2 * np.sin(2.0 * np.pi * x)
    A = (-np.diag(u_bar) @ D1.toarray() + nu * D2.toarray()).astype(complex)
    u0 = (0.4 + 0.2 * np.sin(2.0 * np.pi * x)).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("1D Burgers (Linearized) Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

