"""1D multiscale transport surrogate via Schrodingerization.

We discretize a velocity-quadrature kinetic model:
    f_t + v * f_x = sigma_s (avg_v(f) - f)
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    first_order_derivative,
    matrix_exponential,
)


def main() -> None:
    nx = 8
    velocities = np.array([-1.0, -0.3, 0.3, 1.0])
    nv = len(velocities)
    sigma_s = 1.2
    T = 0.01
    dx = 1.0 / nx

    D1, _ = first_order_derivative(N=nx, dx=dx, boundary_condition="periodic", scheme="central")
    D = D1.toarray()
    I = np.eye(nx)

    blocks = []
    for v in velocities:
        row = []
        for _ in velocities:
            row.append((sigma_s / nv) * I)
        blocks.append(row)
    A = np.block(blocks)
    for i, v in enumerate(velocities):
        ii = slice(i * nx, (i + 1) * nx)
        A[ii, ii] += (-v * D - sigma_s * I)
    A = A.astype(complex)

    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    f0 = [np.exp(-40.0 * (x - 0.3 - 0.05 * j) ** 2) for j in range(nv)]
    u0 = np.concatenate(f0).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("1D Multiscale Transport Schrodingerization")
    print(f"State size: {nx * nv}")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

