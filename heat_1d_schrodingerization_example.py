"""
1D Heat equation solver using the Schrodingerization workflow in UnitaryLab.

Equation:
    u_t = a * u_xx + f(x)

This example includes two paths:
1) General non-unitary semi-discrete system (Dirichlet) solved with schro_classical.
2) Periodic central-difference quantum Trotter demo using GateSequence blocks from TDiff.
"""

import numpy as np
from engine.library import schro_classical, schro_trotter
from engine.library.differential_operator import TDiff
from engine.library.differential_operator.classical_matrices import (
    matrix_exponential,
    second_order_derivative,
)


# Heat equation setup (general Dirichlet case)
N_QUBITS_X = 3
N_GRID = 2 ** N_QUBITS_X
DOMAIN_LENGTH = 1.0
DIFFUSIVITY = 0.1
FINAL_TIME = 0.02
NA_QUBITS = 6
R_DOMAIN = 4
SMOOTH_ORDER = 2
RECOVERY_POINT = 1


# Quantum Trotter demo setup (periodic/source-free supported path)
TROTTER_STEPS = 8


def source_term(x_grid):
    """Source term f(x). Set to zero for a clean reference comparison."""
    return np.zeros_like(x_grid)


def initial_condition_dirichlet(x_grid):
    """Smooth initial condition compatible with homogeneous Dirichlet boundary."""
    return np.sin(np.pi * x_grid)


def initial_condition_periodic(x_grid):
    """Periodic initial condition for the gate-based Trotter path."""
    return np.sin(2.0 * np.pi * x_grid)


def build_heat_system(boundary):
    """Build the semi-discrete system u_t = A u + b for the 1D heat equation."""
    dx = DOMAIN_LENGTH / N_GRID
    x_grid = np.linspace(0.0, DOMAIN_LENGTH, N_GRID, endpoint=False)

    laplacian, boundary_vector = second_order_derivative(
        N=N_GRID,
        dx=dx,
        boundary_condition=boundary,
    )

    A = (DIFFUSIVITY * laplacian).toarray()
    b = np.asarray(DIFFUSIVITY * boundary_vector + source_term(x_grid), dtype=complex)
    return x_grid, dx, A, b


def solve_classical_reference(A, u0):
    """Reference evolution from direct matrix exponential for u_t = A u."""
    return np.asarray(matrix_exponential(A, u0, T=FINAL_TIME, dt=0.001), dtype=complex)


def solve_with_schrodingerization(A, b, u0):
    """Schrodingerization-based classical solver for non-unitary evolution."""
    return np.asarray(
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


def run_periodic_quantum_trotter_demo():
    """
    Run the gate-based quantum simulation path for the 1D heat equation.

    In the current engine, this path is available for periodic central-difference
    operators via TDiff that returns GateSequence factories.
    """
    x_grid, dx, A, _ = build_heat_system(boundary="periodic")
    u0 = initial_condition_periodic(x_grid)

    reference_solution = solve_classical_reference(A, u0)

    trotter_operator = TDiff(
        n=N_QUBITS_X,
        dx=dx,
        order=2,
        scheme="central",
        boundary="periodic",
        target=list(range(N_QUBITS_X)),
    )
    _, h2_factory = trotter_operator.data()

    theta_step = FINAL_TIME / TROTTER_STEPS
    quantum_solution, circuit = schro_trotter(
        u0=u0,
        H1=None,
        H2=h2_factory(theta_step),
        Nt=TROTTER_STEPS,
        na=2,
        R=3,
        order=1,
        point=RECOVERY_POINT,
    )

    quantum_solution = np.asarray(quantum_solution, dtype=complex)
    error = np.linalg.norm(quantum_solution - reference_solution)

    return {
        "x_grid": x_grid,
        "u0": u0,
        "reference_solution": reference_solution,
        "quantum_solution": quantum_solution,
        "error": error,
        "theta_step": theta_step,
        "circuit": circuit,
    }


def main():
    x_grid, dx, A, b = build_heat_system(boundary="dirichlet")
    u0 = initial_condition_dirichlet(x_grid)

    reference_solution = solve_classical_reference(A, u0)
    schro_solution = solve_with_schrodingerization(A, b, u0)
    error = np.linalg.norm(schro_solution - reference_solution)

    print("1D Heat Equation via Schrodingerization")
    print("=" * 44)
    print("Equation: u_t = a * u_xx + f(x)")
    print("Boundary: Dirichlet in the general Schrodingerization path")
    print(f"Grid points: {N_GRID}, dx = {dx:.4f}, diffusivity = {DIFFUSIVITY}, final time = {FINAL_TIME}")
    print(f"Schrodingerization parameters: na = {NA_QUBITS}, R = {R_DOMAIN}, order = {SMOOTH_ORDER}, point = {RECOVERY_POINT}")

    print("\nInitial condition u(x, 0):")
    print(np.round(u0, 6))

    print("\nSemi-discrete matrix A (a * D2):")
    print(np.array2string(A, precision=4, suppress_small=True))

    print("\nReference solution exp(A T) u0:")
    print(np.round(reference_solution.real, 6))

    print("\nSchrodingerization solution:")
    print(np.round(schro_solution.real, 6))

    print(f"\nValidation (Dirichlet) L2 error ||u_schro - u_ref||_2 = {error:.6e}")

    quantum_demo = run_periodic_quantum_trotter_demo()
    print("\nQuantum Trotter demo (periodic central-difference path):")
    print(f"Trotter steps: {TROTTER_STEPS}, theta_step = {quantum_demo['theta_step']:.6f}")
    print("Initial periodic state:")
    print(np.round(quantum_demo["u0"], 6))
    print("Reference periodic solution:")
    print(np.round(quantum_demo["reference_solution"].real, 6))
    print("Quantum Trotter solution:")
    print(np.round(quantum_demo["quantum_solution"].real, 6))
    print(f"Validation (periodic) L2 error = {quantum_demo['error']:.6e}")
    print(f"Circuit object type: {type(quantum_demo['circuit']).__name__}")


if __name__ == "__main__":
    main()
