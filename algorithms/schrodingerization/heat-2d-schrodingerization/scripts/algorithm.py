"""2D Heat Equation via Schrodingerization — u_t = a1*u_xx + a2*u_yy + f(x,y).

Compares classical Schrodingerization solver with Trotter quantum circuit.
"""

import numpy as np
from engine.core import GateSequence
from engine.library import schro_classical, schro_trotter
from engine.library.differential_operator import TDiff
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main():
    # --- Parameters ---
    nx_q = 2              # qubits per spatial dimension
    Nx = 2 ** nx_q        # grid points per dim = 4
    L = 1.0
    ax, ay = 0.08, 0.12
    T = 0.02
    dx = L / Nx
    na = 6
    R = 4.0
    order = 2
    point = 1
    Nt = 10
    bd = "periodic"

    # --- Grid & initial condition ---
    x = np.linspace(0.0, L, Nx, endpoint=False)
    y = np.linspace(0.0, L, Nx, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    fxy = 0.05 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    b = fxy.reshape(-1).astype(complex)
    u0 = np.exp(-35 * ((X - 0.45) ** 2 + (Y - 0.45) ** 2)).reshape(-1).astype(complex)

    # --- Classical operators ---
    D2x, _ = second_order_derivative(N=Nx, dx=dx, boundary_condition=bd)
    D2y, _ = second_order_derivative(N=Nx, dx=dx, boundary_condition=bd)
    Ix = np.eye(Nx)
    A = (ax * np.kron(D2x.toarray(), Ix)
         + ay * np.kron(Ix, D2y.toarray())).astype(complex)

    # --- Reference ---
    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)

    # --- Method 1: Classical Schrodingerization ---
    u_cls = np.asarray(
        schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point, b=b),
        dtype=complex,
    )

    # --- Method 2: Trotter quantum circuit ---
    dt_trotter = T / Nt
    func1_x = TDiff(nx_q, dx, order=2, boundary=bd).data()[0]
    func1_y = TDiff(nx_q, dx, order=2, boundary=bd).data()[0]

    H1 = GateSequence(2 * nx_q)
    H1.append(func1_x(ax * dt_trotter / R), list(range(nx_q)))
    H1.append(func1_y(ay * dt_trotter / R), list(range(nx_q, 2 * nx_q)))

    u_trot, qc = schro_trotter(
        u0=u0, H1=H1, H2=None, Nt=Nt, na=na,
    )
    u_trot = np.asarray(u_trot, dtype=complex)

    # --- Report ---
    print("=" * 55)
    print("2D Heat Schrodingerization")
    print("=" * 55)
    print(f"  State size        : {Nx * Nx}")
    print(f"  Trotter steps     : {Nt}")
    print(f"  Classical L2 err  : {np.linalg.norm(u_cls - u_ref):.6e}")
    print(f"  Trotter   L2 err  : {np.linalg.norm(u_trot - u_ref):.6e}")
    print(f"  Circuit qubits    : {qc.get_num_qubits()}")


if __name__ == "__main__":
    main()
