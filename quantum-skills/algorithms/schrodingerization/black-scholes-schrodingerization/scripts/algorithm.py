"""Black-Scholes PDE in log-price coordinates via Schrodingerization.

Model in x = log(S):
    u_t = mu * u_x + 0.5 * sigma^2 * u_xx - r * u
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
    sigma = 0.25
    r = 0.03
    mu = r - 0.5 * sigma * sigma
    T = 0.02
    dx = 1.0 / n

    D1, b1 = first_order_derivative(N=n, dx=dx, boundary_condition="dirichlet", scheme="upwind")
    D2, b2 = second_order_derivative(N=n, dx=dx, boundary_condition="dirichlet")
    A = (
        mu * D1.toarray() + 0.5 * sigma * sigma * D2.toarray() - r * np.eye(n)
    ).astype(complex)
    b = np.asarray(mu * b1 + 0.5 * sigma * sigma * b2, dtype=complex)

    x = np.linspace(-0.5, 0.5, n, endpoint=False)
    K = 1.0
    payoff = np.maximum(K - np.exp(x), 0.0)
    u0 = payoff.astype(complex)

    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)
    u_sch = np.asarray(
        schro_classical(A, u0, T=T, na=6, R=4.0, order=2, point=1, b=b),
        dtype=complex,
    )

    print("Black-Scholes Schrodingerization")
    print(f"L2 error: {np.linalg.norm(u_sch - u_ref):.6e}")


if __name__ == "__main__":
    main()

