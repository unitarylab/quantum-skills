"""1D Heat Equation via Schrodingerization — u_t = a * u_xx + f(x).

Compares classical Schrodingerization solver with Trotter quantum circuit.
"""

import numpy as np
from unitarylab.library.equation.schrodingerization import schro_classical, schro_trotter, circuit_classical
from unitarylab.library.equation.differential_operator import CDiff, TDiff
from unitarylab.library.equation.differential_operator.classical_matrices import matrix_exponential


def main():
    # --- Parameters ---
    nx = 3                # spatial qubits
    Nx = 2 ** nx          # grid points = 8
    L = 1.0
    a = 0.1               # diffusion coefficient
    T = 0.02
    na = 6
    R = 4.0
    order = 2
    point = 1
    Nt = 10
    bd = "dirichlet"
    scheme = "central"

    # --- Grid & initial condition (Dirichlet) ---
    dx = L / (Nx + 1)
    x  = np.arange(dx, L, dx)
    u0 = np.sin(np.pi * x).astype(complex)
    f  = 0.1 * np.sin(2.0 * np.pi * x)

    # --- Classical operators ---
    A_raw = a * CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()
    A = (A_raw.toarray() if hasattr(A_raw, 'toarray') else np.asarray(A_raw)).astype(complex)
    b = np.asarray(f, dtype=complex)   # zero Dirichlet BCs → no boundary correction

    # --- Reference ---
    u_ref = np.asarray(matrix_exponential(A, u0, T=T, dt=0.001), dtype=complex)

    # --- Method 1: Classical Schrodingerization ---
    u_cls = np.asarray(
        schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point, b=b),
        dtype=complex,
    )
    qc_cls = circuit_classical(nx, na)

    # --- Method 2: Trotter quantum circuit ---
    dt_trotter = T / Nt
    func1, func2 = (a * TDiff(nx, dx, 2, scheme=scheme, boundary=bd)).data()
    H1 = func1(dt_trotter / R)
    H2 = func2(dt_trotter)

    u_trot, qc = schro_trotter(
        u0=u0, H1=H1, H2=H2, Nt=Nt, na=na, R=R, order=order, point=point
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
    print(f"  Classical circuit : {qc_cls.get_num_qubits()} qubits")
    print(f"  Trotter   circuit : {qc.get_num_qubits()} qubits")


if __name__ == "__main__":
    main()
