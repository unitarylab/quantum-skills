"""
1D Advection Equation Solver via Schrödingerization (UnitaryLab)
================================================================
Equation:
    du/dt = a * du/dx        on x in [0, L],  t in [0, T]

Two computational paths are demonstrated:

Path A — Classical Schrödingerization (non-unitary system)
    Discretize with an upwind finite-difference scheme and Dirichlet boundary.
    The resulting semi-discrete operator A is NOT skew-Hermitian, so the full
    Schrödingerization transformation is applied:
        A = H1 + i*H2   (Hermitian decomposition)
        H = eta*H1 + H2  (lifted Hamiltonian)
    Solved with ``schro_classical``.

Path B — Quantum Trotter simulation (unitary system)
    Discretize with a central finite-difference scheme and periodic boundary.
    A central + periodic discretization yields a skew-Hermitian operator, so
    the evolution is already unitary — no Schrödingerization lifting needed.
    Solved with ``schro_trotter`` via a GateSequence block from ``TDiff``.

Theory recap
------------
The advection equation  u_t = a*u_x  admits transport solutions
u(x,t) = u0(x + a*t).  After spatial discretization we obtain

    du/dt = A u + b

where A encodes the chosen finite-difference stencil and b the boundary
forcing.  When A is non-Hermitian (upwind scheme) we apply Schrödingerization:

   1.  Decompose A = H1 + i*H2  with H1=(A+A†)/2, H2=(A-A†)/(2i).
   2.  Introduce auxiliary variable p and lift the state:
           v(t,x,p) = exp(-p) u(t,x),  p > 0
   3.  Fourier-transform in p (frequency η):
           d/dt v̂ = i(η H1 + H2) v̂  =: -i Ĥ v̂
   4.  Simulate the resulting Schrödinger equation.

References
----------
Jin, Liu, Yu (2023) "Quantum simulation of partial differential equations
via Schrödingerization", arXiv:2212.13969.
"""

from __future__ import annotations

import os

import numpy as np

from engine.library import schro_classical, schro_trotter
from engine.library.differential_operator import TDiff
from engine.library.differential_operator.classical_matrices import (
    first_order_derivative,
    matrix_exponential,
)

# ---------------------------------------------------------------------------
# Problem parameters (Path A — upwind / Dirichlet)
# ---------------------------------------------------------------------------
N_QUBITS_X: int = 3          # spatial qubits;  Nx = 2^N_QUBITS_X grid points
DOMAIN_LENGTH: float = 1.0   # x in [0, L]
VELOCITY: float = 1.0        # advection speed  a
FINAL_TIME: float = 0.02     # evolution time   T

# Schrödingerization lifting parameters
NA_QUBITS: int = 6           # auxiliary p-direction qubits
R_DOMAIN: float = 4.0        # p-domain half-width (grid: [-pi*R, pi*R])
SMOOTH_ORDER: int = 2        # smoothness order of g(p) lifting function
RECOVERY_POINT: int = 1      # recovery point index in the p-grid

# ---------------------------------------------------------------------------
# Problem parameters (Path B — central / periodic Trotter demo)
# ---------------------------------------------------------------------------
TROTTER_N_QUBITS_X: int = 3
TROTTER_STEPS: int = 8
TROTTER_NA: int = 2
TROTTER_R: float = 3.0

# ---------------------------------------------------------------------------
# Output directory (sits alongside this script)
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_separator(title: str = "") -> None:
    width = 60
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


def build_advection_system_upwind(n_grid: int, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the semi-discrete advection system for the upwind + Dirichlet case.

    Returns
    -------
    A : (n_grid, n_grid) ndarray
        System matrix.  u_t = A u + b
    b : (n_grid,) ndarray
        Boundary forcing vector.
    """
    D, bvec = first_order_derivative(
        N=n_grid,
        dx=dx,
        boundary_condition="dirichlet",
        scheme="upwind",
    )
    A = (-VELOCITY * D).toarray().astype(complex)
    b = np.asarray(-VELOCITY * bvec, dtype=complex)
    return A, b


def initial_condition_dirichlet(x_grid: np.ndarray) -> np.ndarray:
    """Gaussian pulse centered at x=0.25; compatible with Dirichlet BCs."""
    return np.exp(-80.0 * (x_grid - 0.25) ** 2).astype(complex)


def initial_condition_periodic(x_grid: np.ndarray) -> np.ndarray:
    """Sinusoidal initial condition; satisfies periodic BCs exactly."""
    return np.sin(2.0 * np.pi * x_grid).astype(complex)


# ---------------------------------------------------------------------------
# Path A — Classical Schrödingerization (upwind / Dirichlet)
# ---------------------------------------------------------------------------

def run_path_a_schrodingerization() -> dict:
    """
    Solve  u_t = a u_x  on [0,L] with upwind discretization.

    1. Build the (non-unitary) system matrix A via upwind differences.
    2. Compute reference solution with direct matrix-exponential time-stepping.
    3. Solve with schro_classical (Schrödingerization-lifted expm).
    4. Report L2 error.
    """
    n_grid = 2 ** N_QUBITS_X
    dx = DOMAIN_LENGTH / n_grid
    x_grid = np.linspace(0.0, DOMAIN_LENGTH, n_grid, endpoint=False)

    A, b = build_advection_system_upwind(n_grid, dx)
    u0 = initial_condition_dirichlet(x_grid)

    # --- Reference: direct matrix exponential ---
    u_ref = np.asarray(
        matrix_exponential(A, u0, T=FINAL_TIME, dt=0.002), dtype=complex
    )

    # --- Schrödingerization classical solver ---
    u_schro = np.asarray(
        schro_classical(
            A,
            u0,
            T=FINAL_TIME,
            na=NA_QUBITS,
            R=R_DOMAIN,
            order=SMOOTH_ORDER,
            point=RECOVERY_POINT,
            b=b,
        ),
        dtype=complex,
    )

    error = float(np.linalg.norm(u_schro - u_ref))

    return {
        "x_grid": x_grid,
        "u0": u0,
        "A": A,
        "b": b,
        "u_ref": u_ref,
        "u_schro": u_schro,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Path B — Quantum Trotter simulation (central / periodic)
# ---------------------------------------------------------------------------

def run_path_b_quantum_trotter() -> dict:
    """
    Solve  u_t = a u_x  on [0,L] with central-difference + periodic BCs.

    A central + periodic scheme yields a skew-Hermitian matrix (H1 = 0),
    so the Schrödinger equation reduces to a pure H2 block and the
    Schrödingerization lifting is skipped on the spatial side.

    The TDiff operator provides a ready-made GateSequence block for
    schro_trotter: H1=None, H2=h2_circuit.
    """
    n_grid = 2 ** TROTTER_N_QUBITS_X
    dx = DOMAIN_LENGTH / n_grid
    x_grid = np.linspace(0.0, DOMAIN_LENGTH, n_grid, endpoint=False)

    u0 = initial_condition_periodic(x_grid)

    # Reference: direct matrix exponential with central + periodic A
    D_central, _ = first_order_derivative(
        N=n_grid,
        dx=dx,
        boundary_condition="periodic",
        scheme="central",
    )
    A_central = (-VELOCITY * D_central).toarray().astype(complex)
    u_ref = np.asarray(
        matrix_exponential(A_central, u0, T=FINAL_TIME, dt=0.001), dtype=complex
    )

    # Build Trotter operator for the first-order central derivative
    trotter_op = TDiff(
        n=TROTTER_N_QUBITS_X,
        dx=dx,
        order=1,
        scheme="central",
        boundary="periodic",
        target=list(range(TROTTER_N_QUBITS_X)),
    )
    _, h2_factory = trotter_op.data()

    # Per-step angle:  theta = -a * T / Nt
    theta_step = -VELOCITY * FINAL_TIME / TROTTER_STEPS
    h2_circuit = h2_factory(theta_step)

    # Quantum Trotter solve
    u_quantum, circuit = schro_trotter(
        u0=u0,
        H1=None,
        H2=h2_circuit,
        Nt=TROTTER_STEPS,
        na=TROTTER_NA,
        R=TROTTER_R,
        order=1,
        point=RECOVERY_POINT,
    )
    u_quantum = np.asarray(u_quantum, dtype=complex)

    error = float(np.linalg.norm(u_quantum - u_ref))

    return {
        "x_grid": x_grid,
        "u0": u0,
        "A_central": A_central,
        "u_ref": u_ref,
        "u_quantum": u_quantum,
        "error": error,
        "theta_step": theta_step,
        "n_steps": TROTTER_STEPS,
        "circuit": circuit,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_path_a(res: dict) -> None:
    n_grid = len(res["x_grid"])
    dx = DOMAIN_LENGTH / n_grid

    print_separator("Path A — Classical Schrödingerization (upwind + Dirichlet)")
    print(f"  Equation    : du/dt = {VELOCITY} * du/dx")
    print(f"  Scheme      : upwind, Dirichlet BC  (non-unitary A)")
    print(f"  Grid        : Nx={n_grid}, dx={dx:.4f}, T={FINAL_TIME}")
    print(f"  Schröd. params : na={NA_QUBITS}, R={R_DOMAIN}, "
          f"order={SMOOTH_ORDER}, point={RECOVERY_POINT}")

    print("\n  Initial condition u0:")
    print("  ", np.round(res["u0"].real, 6))

    print("\n  System matrix A (upwind):")
    print(np.array2string(res["A"].real, precision=4, suppress_small=True,
                          prefix="  "))

    print("\n  Reference solution (matrix-expm):")
    print("  ", np.round(res["u_ref"].real, 6))

    print("\n  Schrödingerization solution:")
    print("  ", np.round(res["u_schro"].real, 6))

    print(f"\n  L2 error  ‖u_schro − u_ref‖₂ = {res['error']:.6e}")


def report_path_b(res: dict) -> None:
    n_grid = len(res["x_grid"])
    dx = DOMAIN_LENGTH / n_grid

    print_separator("Path B — Quantum Trotter Simulation (central + periodic)")
    print(f"  Equation    : du/dt = {VELOCITY} * du/dx")
    print(f"  Scheme      : central difference, periodic BC  (unitary A)")
    print(f"  Grid        : Nx={n_grid}, dx={dx:.4f}, T={FINAL_TIME}")
    print(f"  Trotter     : {res['n_steps']} steps, "
          f"theta_step={res['theta_step']:.6f}")
    print(f"  Schröd. params : na={TROTTER_NA}, R={TROTTER_R}")

    print("\n  Initial condition u0:")
    print("  ", np.round(res["u0"].real, 6))

    print("\n  Reference solution (matrix-expm):")
    print("  ", np.round(res["u_ref"].real, 6))

    print("\n  Quantum Trotter solution:")
    print("  ", np.round(res["u_quantum"].real, 6))

    print(f"\n  L2 error  ‖u_quantum − u_ref‖₂ = {res['error']:.6e}")
    print(f"\n  Circuit type : {type(res['circuit']).__name__}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("1D Advection Equation — Schrödingerization Solver (UnitaryLab)")
    print("=" * 60)
    print("  du/dt = a * du/dx,   a =", VELOCITY, ",  T =", FINAL_TIME)

    # Path A: non-unitary (upwind + Dirichlet) → schro_classical
    res_a = run_path_a_schrodingerization()
    report_path_a(res_a)

    # Path B: unitary (central + periodic) → schro_trotter
    res_b = run_path_b_quantum_trotter()
    report_path_b(res_b)

    print_separator("Summary")
    print(f"  Path A (Schrödingerization classical) error : {res_a['error']:.6e}")
    print(f"  Path B (Quantum Trotter)              error : {res_b['error']:.6e}")


if __name__ == "__main__":
    main()
