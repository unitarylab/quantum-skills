"""1D backward heat equation via Schrodingerization.

Model:
    u_t = -kappa * u_xx
Backward diffusion is unstable in classical forward simulation, but this linear
semi-discrete system is still suitable for the Schrodingerization workflow.
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main() -> None:
    n = 8
    kappa = 0.15
    T = 0.01
    dx = 1.0 / n

    D2, bvec = second_order_derivative(N=n, dx=dx, boundary_condition="dirichlet")
    A = (-kappa * D2).toarray().astype(complex)
    b = np.asarray(-kappa * bvec, dtype=complex)
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    u0 = np.sin(np.pi * x).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    print("1D Backward Heat Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

