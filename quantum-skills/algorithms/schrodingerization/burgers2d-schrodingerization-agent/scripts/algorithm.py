"""2D Burgers-type transport surrogate via Schrodingerization.

Target nonlinear PDE:
    u_t + u u_x + v u_y = 0
This demo uses frozen velocities to build a linear transport operator.
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    first_order_derivative,
    matrix_exponential,
)


def main() -> None:
    nx = 4
    ny = 4
    T = 0.01

    D1x, _ = first_order_derivative(N=nx, dx=1.0 / nx, boundary_condition="periodic", scheme="central")
    D1y, _ = first_order_derivative(N=ny, dx=1.0 / ny, boundary_condition="periodic", scheme="central")
    Ix = np.eye(nx)
    Iy = np.eye(ny)

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ux = 0.4 + 0.1 * np.sin(2.0 * np.pi * X)
    uy = -0.3 + 0.1 * np.cos(2.0 * np.pi * Y)

    Dx2d = np.kron(D1x.toarray(), Iy)
    Dy2d = np.kron(Ix, D1y.toarray())
    A = (-(np.diag(ux.reshape(-1)) @ Dx2d) - (np.diag(uy.reshape(-1)) @ Dy2d)).astype(complex)

    u0 = np.exp(-25.0 * ((X - 0.35) ** 2 + (Y - 0.55) ** 2)).reshape(-1).astype(complex)
    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("2D Burgers Surrogate Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

