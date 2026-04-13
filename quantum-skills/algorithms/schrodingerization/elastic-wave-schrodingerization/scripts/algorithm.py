"""1D elastic wave first-order system via Schrodingerization.

Velocity-stress form:
    v_t = (1/rho) * sigma_x
    sigma_t = K * v_x
"""

import numpy as np

from engine.library import schro_classical
from engine.library.differential_operator.classical_matrices import (
    first_order_derivative,
    matrix_exponential,
)


def main() -> None:
    n = 8
    rho = 1.0
    K = 1.5
    T = 0.01
    dx = 1.0 / n

    D1, _ = first_order_derivative(N=n, dx=dx, boundary_condition="periodic", scheme="central")
    D = D1.toarray()
    Z = np.zeros((n, n))

    A = np.block([[Z, (1.0 / rho) * D], [K * D, Z]]).astype(complex)

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    v0 = np.exp(-60.0 * (x - 0.4) ** 2)
    s0 = np.zeros_like(v0)
    u0 = np.concatenate([v0, s0]).astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.0005), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1),
        dtype=complex,
    )

    print("Elastic Wave Schrodingerization")
    print(f"System size: {2 * n}")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

