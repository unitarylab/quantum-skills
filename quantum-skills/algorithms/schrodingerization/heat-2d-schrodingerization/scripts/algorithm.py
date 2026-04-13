"""2D heat equation via Schrodingerization.

Model:
    u_t = ax * u_xx + ay * u_yy + f(x, y)
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main() -> None:
    nx = 4
    ny = 4
    ax = 0.08
    ay = 0.12
    T = 0.02

    D2x, _ = second_order_derivative(N=nx, dx=1.0 / nx, boundary_condition="periodic")
    D2y, _ = second_order_derivative(N=ny, dx=1.0 / ny, boundary_condition="periodic")
    Ix = np.eye(nx)
    Iy = np.eye(ny)

    A = (ax * np.kron(D2x.toarray(), Iy) + ay * np.kron(Ix, D2y.toarray())).astype(complex)

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    fxy = 0.05 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    b = fxy.reshape(-1).astype(complex)
    u0 = np.exp(-35.0 * ((X - 0.45) ** 2 + (Y - 0.45) ** 2)).reshape(-1).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    print("2D Heat Schrodingerization")
    print(f"State size: {nx * ny}")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

