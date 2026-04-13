"""1D variable-coefficient heat equation via Schrodingerization.

Model:
    u_t = a(x) * u_xx,   a(x) = 1 + 0.4 cos(2*pi*x)
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main() -> None:
    n = 8
    T = 0.02
    dx = 1.0 / n

    D2, _ = second_order_derivative(N=n, dx=dx, boundary_condition="periodic")
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    a = 1.0 + 0.4 * np.cos(2.0 * np.pi * x)
    A = (np.diag(a) @ D2.toarray()).astype(complex)
    u0 = (0.6 + 0.2 * np.sin(2.0 * np.pi * x)).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("1D Variable-Coefficient Heat Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

