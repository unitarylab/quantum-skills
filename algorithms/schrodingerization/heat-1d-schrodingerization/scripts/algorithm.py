"""1D Heat Equation via Schrodingerization — u_t = a * u_xx + f(x).

Compares classical Schrodingerization solver with Trotter quantum circuit.
"""

import numpy as np
from unitarylab.library import schro_classical, schro_trotter
from unitarylab.library.differential_operator import TDiff
from unitarylab.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


def main():
    # --- Parameters ---
    nx = 3                # spatial qubits
    Nx = 2 ** nx          # grid points = 8
    L = 1.0
    a = 0.1               # diffusion coefficient
    T = 0.02
    dx = L / Nx
    na = 6
    R = 4.0
    order = 2
    point = 1
    Nt = 10

    # --- Grid & initial condition ---
    x = np.linspace(0.0, L, Nx, endpoint=False)
    u0 = np.sin(np.pi * x).astype(complex)
    f = 0.1 * np.sin(2.0 * np.pi * x)

    # --- Classical operators ---
    D2, bvec = second_order_derivative(N=Nx, dx=dx, boundary_condition="dirichlet")
    A = (a * D2).toarray().astype(complex)
    b = np.asarray(a * bvec + f, dtype=complex)

    # --- Reference ---
    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)

    # --- Method 1: Classical Schrodingerization ---
    u_cls = np.asarray(
        schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point, b=b),
        dtype=complex,
    )

    # --- Method 2: Trotter quantum circuit ---
    dt_trotter = T / Nt
    tdiff = a * TDiff(nx, dx, order=2, scheme="central", boundary="dirichlet")
    func1, func2 = tdiff.data()
    H1 = func1(dt_trotter / R)
    H2 = func2(dt_trotter)

    u_trot, qc = schro_trotter(
        u0=u0, H1=H1, H2=H2, Nt=Nt, na=na, R=R, order=order, point=point, b=b,
        theta=dt_trotter,
    )
    u_trot = np.asarray(u_trot, dtype=complex)

    # --- Report ---
    print("=" * 55)
    print("1D Heat Schrodingerization")
    print("=" * 55)
    print(f"  Grid points       : {Nx}")
    print(f"  Trotter steps     : {Nt}")
    print(f"  Classical L2 err  : {np.linalg.norm(u_cls - u_ref):.6e}")
    print(f"  Trotter   L2 err  : {np.linalg.norm(u_trot - u_ref):.6e}")
    print(f"  Circuit qubits    : {qc.get_num_qubits()}")


if __name__ == "__main__":
    main()
