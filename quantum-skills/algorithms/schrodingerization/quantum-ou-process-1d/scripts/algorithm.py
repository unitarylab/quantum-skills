"""Ornstein-Uhlenbeck Fokker-Planck surrogate via Schrodingerization.

Density evolution model:
    p_t = -mu * d(x p)/dx + 0.5 * sigma^2 * p_xx
Here we approximate the drift as -mu * x * p_x for a compact demo.
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
    mu = 0.7
    sigma = 0.4
    T = 0.01
    dx = 1.0 / n

    x = np.linspace(-1.0, 1.0, n, endpoint=False)
    D1, _ = first_order_derivative(N=n, dx=dx, boundary_condition="periodic", scheme="central")
    D2, _ = second_order_derivative(N=n, dx=dx, boundary_condition="periodic")

    A = (-mu * np.diag(x) @ D1.toarray() + 0.5 * sigma * sigma * D2.toarray()).astype(complex)
    u0 = np.exp(-6.0 * x * x).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("1D Ornstein-Uhlenbeck Surrogate Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

