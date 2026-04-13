"""2D Schrodinger equation with absorbing boundary via Schrodingerization.

Model:
    psi_t = -i * Laplacian(psi) - W(x,y) * psi
The absorbing potential W makes the generator non-skew-Hermitian.
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
    T = 0.01

    D2x, _ = second_order_derivative(N=nx, dx=1.0 / nx, boundary_condition="periodic")
    D2y, _ = second_order_derivative(N=ny, dx=1.0 / ny, boundary_condition="periodic")
    Ix = np.eye(nx)
    Iy = np.eye(ny)
    Lap = np.kron(D2x.toarray(), Iy) + np.kron(Ix, D2y.toarray())

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    edge = np.minimum.reduce([X, 1.0 - X, Y, 1.0 - Y])
    W = 4.0 * np.clip(0.2 - edge, 0.0, None) ** 2

    A = (-1j * Lap - np.diag(W.reshape(-1))).astype(complex)
    psi0 = np.exp(-40.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)).reshape(-1).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, psi0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, psi0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("2D Schrodinger + ABC Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

