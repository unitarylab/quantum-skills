"""2D Heat Equation via Schrodingerization — u_t = a1*u_xx + a2*u_yy + f(x,y).

Compares classical Schrodingerization solver with Trotter quantum circuit.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from unitarylab import Circuit
from unitarylab.library.equation.schrodingerization import (
    schro_classical,
    schro_trotter,
    circuit_classical,
)
from unitarylab.library.equation.differential_operator import CDiff, TDiff


def main():
    # --- Parameters ---
    nx = 2              # qubits per spatial dimension
    Nx = 2 ** nx        # grid points per dim = 4
    L = 1.0
    a1, a2 = 0.08, 0.12
    T = 0.02
    na = 6
    R = 4.0
    order = 2
    point = 1
    Nt = 10
    bd = "periodic"
    scheme = "central"
    device = "cpu"

    # --- Grid (periodic BC: dx = L / Nx) ---
    dx = L / Nx
    x = np.arange(0, L, dx)
    y = np.arange(0, L, dx)

    # --- Initial condition & source term ---
    X, Y = np.meshgrid(x, y, indexing="ij")
    u0 = np.exp(-35 * ((X - 0.45) ** 2 + (Y - 0.45) ** 2)).reshape(-1).astype(complex)
    b_src = 0.05 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    b = b_src.reshape(-1).astype(complex)

    # --- Assemble 2D Laplacian via Kronecker product (Step 3) ---
    A0 = CDiff(N=Nx, dx=dx, order=2, scheme=scheme, boundary=bd).get_matrix()
    A = a1 * sp.kron(A0, sp.eye(Nx)) + a2 * sp.kron(sp.eye(Nx), A0)

    # --- Reference: dense matrix exponential ---
    A_dense = A.toarray().astype(complex)
    u_ref = (expm(A_dense * T) @ u0).astype(complex)

    # --- Method 1: Classical Schrodingerization (Steps 5) ---
    u_cls = np.asarray(
        schro_classical(A, u0, T=T, na=na, R=R, order=order, point=point, b=b),
        dtype=complex,
    )
    u_cls_2d = u_cls.reshape((Nx, Nx))
    qc_cls = circuit_classical(nx, na, dim=2)

    # --- Method 2: Trotter quantum circuit (Steps 6-7) ---
    dt = T / Nt
    func1 = TDiff(nx, dx, 2, scheme=scheme, boundary=bd).data()[0]
    D1 = lambda a: func1(a * dt / R)

    H1 = Circuit(2 * nx)
    H1.append(D1(a1), range(nx))
    H1.append(D1(a2), range(nx, 2 * nx))
    H2 = None

    u_trot, qc = schro_trotter(
        u0=u0, H1=H1, H2=H2,
        Nt=Nt, na=na, R=R,
        order=order, point=point,
        device=device,
    )
    u_trot = np.asarray(u_trot, dtype=complex)
    u_trot_2d = u_trot.reshape((Nx, Nx))

    # --- Report ---
    print("=" * 55)
    print("2D Heat Schrodingerization")
    print("=" * 55)
    print(f"  State size        : {Nx * Nx}")
    print(f"  Trotter steps     : {Nt}")
    print(f"  Classical L2 err  : {np.linalg.norm(u_cls - u_ref):.6e}")
    print(f"  Trotter   L2 err  : {np.linalg.norm(u_trot - u_ref):.6e}")
    print(f"  Classical circuit : {qc_cls.get_num_qubits()} qubits")
    print(f"  Trotter circuit   : {qc.get_num_qubits()} qubits")


if __name__ == "__main__":
    main()
