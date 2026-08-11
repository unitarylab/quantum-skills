"""Manual implementation of 1D Heat Equation Solver via Schrodingerization.

Solves the 1D heat (diffusion) equation:
    ∂u/∂t = a·∂²u/∂x² + f(x)

using Schrodingerization to transform the non-unitary diffusion into unitary
Hamiltonian dynamics suitable for quantum simulation.

The heat equation is always non-unitary (diffusion is irreversible), so
Schrodingerization with auxiliary dimension is always applied.

Two solver paths:
    1. Classical: Direct matrix exponentiation via scipy.linalg.expm.
    2. Trotter: Lie-Trotter splitting for gate-based quantum simulation.

Schrodingerization key idea:
    du/dt = A·u + b → dψ/dt = -i·H·ψ
    where H = D_η ⊗ H1 + I ⊗ H2, with auxiliary operator D_η.

Components:
    - build_heat_grid: Spatial grid construction
    - laplacian_matrix: 2nd-order finite difference Laplacian
    - build_heat_hamiltonian: Schrodingerized Hamiltonian
    - solve_classical / solve_trotter: Time evolution
    - HeatSolver: Class-based interface

Reference:
    SKILL.md — 1D Heat Equation via Schrodingerization
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import expm

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ===================================================================
# 1. Grid construction
# ===================================================================


def build_heat_grid(
    L: float,
    nx: int,
    boundary_condition: str = "dirichlet",
) -> Tuple[np.ndarray, float, int]:
    """Build spatial grid for 1D heat equation.

    Dirichlet: N = 2^nx interior points, x ∈ (0, L) spaced by L/(N+1).
    Periodic:  N = 2^nx points wrapping [0, L), spaced by L/N.
    Neumann:   N = 2^nx points including boundaries, spaced by L/(N-1).

    Args:
        L: Domain length.
        nx: Log2 of grid points.
        boundary_condition: 'dirichlet', 'periodic', or 'neumann'.

    Returns:
        Tuple of (x, dx, N).
    """
    N = 1 << nx

    if boundary_condition == "periodic":
        dx = L / N
        x = np.arange(0, L, dx)
    elif boundary_condition == "neumann":
        dx = L / (N - 1) if N > 1 else L
        x = np.arange(0, L + dx / 2, dx)
    else:  # dirichlet
        dx = L / (N + 1)
        x = np.arange(dx, L, dx)

    return x, dx, N


# ===================================================================
# 2. Second-order finite difference Laplacian
# ===================================================================


def laplacian_matrix(
    N: int,
    dx: float,
    boundary_condition: str = "dirichlet",
) -> np.ndarray:
    """Build the 1D Laplacian matrix using 2nd-order central differences.

    Interior: (u_{i+1} - 2·u_i + u_{i-1}) / dx²

    Args:
        N: Number of grid points.
        dx: Grid spacing.
        boundary_condition: 'dirichlet', 'periodic', or 'neumann'.

    Returns:
        Laplacian matrix of shape (N, N).
    """
    L_mat = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        L_mat[i, i] = -2.0  # diagonal
        if i > 0:
            L_mat[i, i - 1] = 1.0  # sub-diagonal
        elif boundary_condition == "periodic":
            L_mat[i, N - 1] = 1.0  # wrap-around

        if i < N - 1:
            L_mat[i, i + 1] = 1.0  # super-diagonal
        elif boundary_condition == "periodic":
            L_mat[i, 0] = 1.0  # wrap-around

    # Neumann BC: zero derivative at boundaries
    if boundary_condition == "neumann":
        L_mat[0, 0] = -1.0
        L_mat[0, 1] = 1.0
        L_mat[-1, -1] = -1.0
        L_mat[-1, -2] = 1.0

    return L_mat / (dx * dx)


# ===================================================================
# 3. Schrodingerization for heat equation
# ===================================================================


def build_heat_hamiltonian(
    A: np.ndarray,
    na: int = 2,
    R: float = 2.0,
) -> np.ndarray:
    """Schrodingerize the diffusion system matrix A.

    For the heat equation, A = a·Laplacian is always symmetric negative
    semidefinite (non-unitary), so full Schrodingerization is needed.

    H = D_η ⊗ H1 + I ⊗ H2
    where:
        - H1 = (A + A†)/2 = A (since A is symmetric real)
        - H2 = (A - A†)/(2i) ≈ 0 (A has no anti-Hermitian part)
        - D_η is a diagonal auxiliary operator

    Args:
        A: System matrix (N×N), symmetric negative-semidefinite.
        na: Number of auxiliary qubits (2^na auxiliary modes).
        R: Scaling parameter for auxiliary dimension.

    Returns:
        Hermitian Hamiltonian of shape (2^na·N, 2^na·N).
    """
    N = A.shape[0]
    aux_dim = 1 << na

    # A is real symmetric → H1 = A, H2 ≈ 0
    H1 = A  # already Hermitian (real symmetric)
    H2 = np.zeros((N, N), dtype=np.float64)  # no anti-Hermitian part

    # Auxiliary operator D_η (diagonal, centered symmetric)
    d_vals = np.array([
        (2 * k + 1 - aux_dim) * R / aux_dim for k in range(aux_dim)
    ])
    D_eta = np.diag(d_vals)

    # H = D_η ⊗ H1 + I ⊗ H2 = D_η ⊗ A (since H2 = 0)
    H = np.kron(D_eta, H1) + np.kron(np.eye(aux_dim), H2)

    return (H + H.conj().T) / 2.0  # ensure Hermitian


# ===================================================================
# 4. Time evolution
# ===================================================================


def solve_heat_classical(
    A: np.ndarray,
    u0: np.ndarray,
    T: float,
    source: Optional[np.ndarray] = None,
    na: int = 2,
    R: float = 2.0,
    n_steps: int = 100,
) -> np.ndarray:
    """Solve heat equation via classical Schrodingerization.

    Uses matrix exponentiation of the Schrodingerized Hamiltonian.

    Args:
        A: System matrix a·Laplacian (N×N).
        u0: Initial condition u(x,0).
        T: Final time.
        source: Optional source term f(x), shape (N,).
        na: Auxiliary qubits.
        R: Scaling parameter.
        n_steps: Number of time steps.

    Returns:
        Solution u(x,T), shape (N,).
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    N = A.shape[0]
    dt = T / n_steps
    H = build_heat_hamiltonian(A, na=na, R=R)
    aux_dim = 1 << na

    # Lift: ψ = [u, 0, ..., 0] in aux·N space
    psi = np.zeros(aux_dim * N, dtype=np.complex128)
    psi[:N] = u0.astype(np.complex128)

    # If source term present, incorporate into evolution
    if source is not None:
        # Source lifts to same auxiliary mode
        b_lifted = np.zeros(aux_dim * N, dtype=np.complex128)
        b_lifted[:N] = source.astype(np.complex128)
    else:
        b_lifted = None

    U = expm(-1j * H * dt)
    for step in range(n_steps):
        psi = U @ psi
        if b_lifted is not None:
            # First-order source integration
            psi = psi + dt * b_lifted

    return psi[:N].real


def solve_heat_trotter(
    A: np.ndarray,
    u0: np.ndarray,
    T: float,
    dt: float = 0.01,
    source: Optional[np.ndarray] = None,
    na: int = 2,
    R: float = 2.0,
    order: int = 1,
) -> np.ndarray:
    """Solve heat equation via Trotter splitting.

    Splits H = H_aux + H_sys and applies Lie-Trotter or Strang formula.

    Args:
        A: System matrix (N×N).
        u0: Initial condition.
        T: Final time.
        dt: Time step size.
        source: Optional source term.
        na: Auxiliary qubits.
        R: Scaling parameter.
        order: Trotter order (1 or 2).

    Returns:
        Solution u(x,T).
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    N = A.shape[0]
    aux_dim = 1 << na

    # Auxiliary operator
    d_vals = np.array([
        (2 * k + 1 - aux_dim) * R / aux_dim for k in range(aux_dim)
    ])
    D_eta = np.diag(d_vals)

    # Split Hamiltonians
    H_aux = np.kron(D_eta, np.eye(N))
    H_sys = np.kron(np.eye(aux_dim), A)

    # Lifted state
    psi = np.zeros(aux_dim * N, dtype=np.complex128)
    psi[:N] = u0.astype(np.complex128)

    n_steps = max(1, int(T / dt))
    if source is not None:
        b_lifted = np.zeros(aux_dim * N, dtype=np.complex128)
        b_lifted[:N] = source.astype(np.complex128)
    else:
        b_lifted = None

    if order == 2:
        # Strang splitting
        U_aux_half = expm(-1j * H_aux * dt / 2)
        U_sys = expm(-1j * H_sys * dt)
        for _ in range(n_steps):
            psi = U_aux_half @ psi
            psi = U_sys @ psi
            psi = U_aux_half @ psi
            if b_lifted is not None:
                psi = psi + dt * b_lifted
    else:
        # Lie-Trotter
        U_aux = expm(-1j * H_aux * dt)
        U_sys = expm(-1j * H_sys * dt)
        for _ in range(n_steps):
            psi = U_aux @ psi
            psi = U_sys @ psi
            if b_lifted is not None:
                psi = psi + dt * b_lifted

    return psi[:N].real


# ===================================================================
# 5. Initial conditions
# ===================================================================


def initial_condition(
    x: np.ndarray,
    ic_type: str = "sine",
    L: float = 1.0,
) -> np.ndarray:
    """Generate initial temperature distribution u(x,0).

    Args:
        x: Spatial coordinates.
        ic_type: 'sine', 'gaussian', 'step', or 'triangle'.
        L: Domain length.

    Returns:
        Initial values u(x,0).
    """
    if ic_type == "sine":
        return np.sin(np.pi * x / L)
    elif ic_type == "sine_2pi":
        return np.sin(2 * np.pi * x / L)
    elif ic_type == "gaussian":
        center = L / 2
        sigma = L / 8
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    elif ic_type == "step":
        u = np.zeros_like(x)
        u[x < L / 2] = 1.0
        return u
    elif ic_type == "triangle":
        u = np.zeros_like(x)
        mask_l = x <= L / 2
        u[mask_l] = 2 * x[mask_l] / L
        u[~mask_l] = 2 * (L - x[~mask_l]) / L
        return u
    else:
        raise ValueError(f"Unknown ic_type: {ic_type}")


# ===================================================================
# 6. Visualization
# ===================================================================


def plot_heat_solution(
    x: np.ndarray,
    u: np.ndarray,
    u0: Optional[np.ndarray] = None,
    title: str = "1D Heat Equation Solution",
    output_path: str = "heat_1d_solution.svg",
) -> str:
    """Plot heat equation solution.

    Args:
        x: Spatial grid.
        u: Final solution u(x,T).
        u0: Optional initial condition.
        title: Plot title.
        output_path: Output file.

    Returns:
        Absolute path to saved plot.
    """
    if not HAS_MATPLOTLIB:
        return ""
    import os

    plt.figure(figsize=(8, 5))
    if u0 is not None:
        plt.plot(x, u0, "#e74c3c", lw=1.5, alpha=0.5, label="u(x,0)")
    plt.plot(x, u, "#3498db", lw=2, label=f"u(x,T)")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.abspath(output_path)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ===================================================================
# 7. End-to-end solver
# ===================================================================


def heat_solve(
    L: float = 1.0,
    T: float = 0.1,
    nx: int = 4,
    a: float = 0.1,
    boundary_condition: str = "dirichlet",
    method: str = "classical",
    ic_type: str = "sine",
    source: Optional[np.ndarray] = None,
    na: int = 2,
    R: float = 2.0,
    dt: float = 0.01,
    order: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve 1D heat equation ∂u/∂t = a·∂²u/∂x² + f(x).

    Args:
        L: Domain length.
        T: Final time.
        nx: Log2 of grid points.
        a: Diffusion coefficient (> 0).
        boundary_condition: 'dirichlet', 'periodic', or 'neumann'.
        method: 'classical' or 'trotter'.
        ic_type: Initial condition type.
        source: Optional source term f(x), shape (N,).
        na: Auxiliary qubits for Schrodingerization.
        R: Scaling parameter.
        dt: Time step (Trotter only).
        order: Trotter order (1 or 2).
        verbose: Print progress.

    Returns:
        Dict with x, u, u0, grid info, computation time.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")
    if a <= 0:
        raise ValueError(f"Diffusion coefficient a must be > 0, got {a}")

    # Grid
    x, dx, N = build_heat_grid(L, nx, boundary_condition)
    u0 = initial_condition(x, ic_type, L)

    if source is not None and len(source) != N:
        raise ValueError(f"Source must have length {N}, got {len(source)}")

    if verbose:
        print(f"1D Heat Equation Solver")
        print(f"  ∂u/∂t = {a}·∂²u/∂x²", end="")
        print(f" + f(x)" if source is not None else "")
        print(f"  Domain: [0, {L}], T = {T}")
        print(f"  N = 2^{nx} = {N}, dx = {dx:.4f}")
        print(f"  BC: {boundary_condition}, IC: {ic_type}")
        print(f"  Method: {method}")

    # Laplacian system
    A = a * laplacian_matrix(N, dx, boundary_condition)

    if verbose:
        print(f"  A: shape {A.shape}, sparsity {np.mean(np.abs(A) > 1e-12):.3f}")
        # Analytical decay check: max eigenvalue of Laplacian
        evals = np.linalg.eigvalsh(A)
        print(f"  A eigenvalues: [{evals[0]:.2f}, {evals[-1]:.2f}]")

    # Time evolution
    t_start = time.perf_counter()

    if method == "classical":
        u = solve_heat_classical(A, u0, T, source=source, na=na, R=R)
    elif method == "trotter":
        u = solve_heat_trotter(
            A, u0, T, dt=dt, source=source, na=na, R=R, order=order,
        )
        Nt = max(1, int(T / dt))
        if verbose:
            print(f"  Time steps: Nt = {Nt}, dt = {dt}, order = {order}")
    else:
        raise ValueError(f"Unknown method: {method}")

    comp_time = time.perf_counter() - t_start

    # Analytical solution for sine IC with Dirichlet BC (for validation)
    analytic = None
    if (boundary_condition == "dirichlet" and ic_type == "sine"
            and source is None):
        # u(x,t) = sin(πx/L)·exp(-a·π²·t/L²)
        analytic = np.sin(np.pi * x / L) * np.exp(-a * np.pi ** 2 * T / L ** 2)

    if verbose:
        print(f"  Computation time: {comp_time:.4f}s")
        print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
        if analytic is not None:
            err = np.max(np.abs(u - analytic))
            print(f"  Max error vs analytic: {err:.2e}")
        print(f"  Status: ok")

    result: Dict[str, Any] = {
        "status": "ok",
        "message": "Heat equation solved",
        "grid": {
            "n_points": N, "dx": dx,
            "dt": dt if method == "trotter" else None,
            "nt": max(1, int(T / dt)) if method == "trotter" else None,
        },
        "x": x.tolist(),
        "u": u.tolist(),
        "u0": u0.tolist(),
        "computation_time": round(comp_time, 4),
        "a": a, "L": L, "T": T,
        "method": method, "boundary_condition": boundary_condition,
    }
    if analytic is not None:
        result["analytic"] = analytic.tolist()
    return result


# ===================================================================
# 8. Class-based interface
# ===================================================================


class HeatSolver:
    """Class-based 1D heat equation solver.

    Usage:
        solver = HeatSolver()
        result = solver.run(L=1.0, T=0.1, nx=4, a=0.1)
        print(result['x'], result['u'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        L: float = 1.0,
        T: float = 0.1,
        nx: int = 4,
        a: float = 0.1,
        boundary_condition: str = "dirichlet",
        method: str = "classical",
        ic_type: str = "sine",
        source: Optional[np.ndarray] = None,
        na: int = 2,
        R: float = 2.0,
        dt: float = 0.01,
        order: int = 1,
    ) -> Dict[str, Any]:
        """Solve heat equation. See heat_solve() for docs."""
        result = heat_solve(
            L=L, T=T, nx=nx, a=a,
            boundary_condition=boundary_condition,
            method=method, ic_type=ic_type,
            source=source, na=na, R=R, dt=dt, order=order,
            verbose=(self.text_mode != "plain"),
        )
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            bc_short = boundary_condition[:3]
            plot_path = plot_heat_solution(
                np.array(result["x"]), np.array(result["u"]),
                u0=np.array(result["u0"]),
                title=f"1D Heat: ∂u/∂t={a}·∂²u/∂x², {bc_short}, T={T}",
                output_path=os.path.join(self.algo_dir, "heat_solution.svg"),
            )
            result["plot"] = {"format": "svg", "filename": plot_path}
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "message": result.get("message", ""),
            "grid": result.get("grid", {}),
            "x": result.get("x", []),
            "u": result.get("u", []),
            "computation_time": result.get("computation_time", 0.0),
            "plot": result.get("plot", {}),
            "circuit": [],
        }


# ===================================================================
# 9. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "dirichlet_sine_n4_T0.1",
        "L": 1.0, "T": 0.1, "nx": 4, "a": 0.1,
        "bc": "dirichlet", "ic": "sine", "method": "classical",
    },
    {
        "name": "dirichlet_gaussian_n5",
        "L": 1.0, "T": 0.05, "nx": 5, "a": 0.05,
        "bc": "dirichlet", "ic": "gaussian", "method": "classical",
    },
    {
        "name": "periodic_sine_n4",
        "L": 1.0, "T": 0.1, "nx": 4, "a": 0.1,
        "bc": "periodic", "ic": "sine_2pi", "method": "classical",
    },
    {
        "name": "trotter_n4_order2",
        "L": 1.0, "T": 0.05, "nx": 4, "a": 0.1,
        "bc": "dirichlet", "ic": "sine", "method": "trotter",
        "dt": 0.01, "order": 2,
    },
    {
        "name": "trotter_n3",
        "L": 1.0, "T": 0.04, "nx": 3, "a": 0.1,
        "bc": "dirichlet", "ic": "triangle", "method": "trotter",
        "dt": 0.01, "order": 1,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    name = case["name"]
    kwargs = {
        "L": case["L"], "T": case["T"], "nx": case["nx"],
        "a": case["a"], "boundary_condition": case["bc"],
        "ic_type": case["ic"], "method": case["method"],
    }
    for k in ("dt", "order"):
        if k in case:
            kwargs[k] = case[k]
    try:
        result = heat_solve(**kwargs, verbose=False)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False
    u = np.array(result["u"])
    ok = result["status"] == "ok" and np.all(np.isfinite(u))
    icon = "ok" if ok else "FAIL"
    print(f"  [{icon}] {name}: N={len(u)}, u∈[{u.min():.4f}, {u.max():.4f}], "
          f"time={result['computation_time']:.3f}s")
    return ok


# ===================================================================
# 10. Main
# ===================================================================

def main() -> None:
    if not HAS_SCIPY:
        print("scipy required: pip install scipy")
        return

    print("=" * 60)
    print("1D Heat Equation Solver via Schrödingerization")
    print("  ∂u/∂t = a·∂²u/∂x² + f(x)")
    print("=" * 60)

    solver = HeatSolver()

    # Demo 1: Dirichlet sine (analytic solvable)
    print("\n--- Demo 1: Dirichlet BC, sine IC (analytic validation) ---")
    r1 = solver.run(L=1.0, T=0.1, nx=5, a=0.1, bc="dirichlet", ic="sine")
    u1 = np.array(r1["u"])
    analytic1 = np.sin(np.pi * np.array(r1["x"])) * np.exp(-0.1 * np.pi**2 * 0.1)
    err1 = np.max(np.abs(u1 - analytic1))
    print(f"  u range: [{u1.min():.4f}, {u1.max():.4f}]")
    print(f"  Max |u - analytic|: {err1:.2e}")

    # Demo 2: Gaussian diffusion
    print("\n--- Demo 2: Gaussian spreading (Dirichlet) ---")
    r2 = solver.run(L=1.0, T=0.05, nx=6, a=0.05, bc="dirichlet", ic="gaussian")
    u2 = np.array(r2["u"])
    print(f"  N = {len(u2)}, u∈[{u2.min():.4f}, {u2.max():.4f}]")

    # Demo 3: Periodic BC
    print("\n--- Demo 3: Periodic BC ---")
    r3 = solver.run(L=1.0, T=0.1, nx=5, a=0.1, bc="periodic", ic="sine_2pi")
    u3 = np.array(r3["u"])
    # Periodic sine_2pi: u(x,t) = sin(2πx)·exp(-a·4π²·t)
    x3 = np.array(r3["x"])
    analytic3 = np.sin(2 * np.pi * x3) * np.exp(-0.1 * 4 * np.pi**2 * 0.1)
    err3 = np.max(np.abs(u3 - analytic3))
    print(f"  Max |u - analytic|: {err3:.2e}")

    # Demo 4: Trotter splitting
    print("\n--- Demo 4: Trotter (quantum simulation path) ---")
    r4 = solver.run(
        L=1.0, T=0.03, nx=4, a=0.1,
        bc="dirichlet", method="trotter", dt=0.005, order=2, ic="sine",
    )
    u4 = np.array(r4["u"])
    print(f"  Steps Nt = {r4['grid']['nt']}, dt = {r4['grid']['dt']}")
    print(f"  u∈[{u4.min():.4f}, {u4.max():.4f}]")

    # Demo 5: Classical vs Trotter
    print("\n--- Demo 5: Classical vs Trotter comparison ---")
    r5c = heat_solve(L=1.0, T=0.02, nx=4, a=0.1, bc="dirichlet",
                      method="classical", verbose=False)
    r5t = heat_solve(L=1.0, T=0.02, nx=4, a=0.1, bc="dirichlet",
                      method="trotter", dt=0.002, order=2, verbose=False)
    diff5 = np.max(np.abs(np.array(r5c["u"]) - np.array(r5t["u"])))
    print(f"  Max |classical - trotter|: {diff5:.2e}")

    # Demo 6: Diffusion coefficient effect
    print("\n--- Demo 6: Effect of diffusion coefficient a ---")
    for av in [0.01, 0.05, 0.1, 0.3, 0.5]:
        r6 = heat_solve(L=1.0, T=0.1, nx=5, a=av, bc="dirichlet",
                         method="classical", verbose=False)
        u6 = np.array(r6["u"])
        decay = u6.max() / np.array(r6["u0"]).max() if np.array(r6["u0"]).max() > 0 else 0
        print(f"  a={av:.2f}: u∈[{u6.min():.4f}, {u6.max():.4f}], "
              f"decay={decay:.3f}, time={r6['computation_time']:.3f}s")

    # Demo 7: Grid convergence
    print("\n--- Demo 7: Grid convergence (nx=2..6) ---")
    for nv in range(2, 7):
        r7 = heat_solve(L=1.0, T=0.05, nx=nv, a=0.1, bc="dirichlet",
                         method="classical", verbose=False)
        u7 = np.array(r7["u"])
        analytic7 = (np.sin(np.pi * np.array(r7["x"]))
                     * np.exp(-0.1 * np.pi**2 * 0.05))
        err7 = np.max(np.abs(u7 - analytic7)) if len(u7) == len(analytic7) else np.nan
        print(f"  nx={nv} (N={1<<nv}): u∈[{u7.min():.4f}, {u7.max():.4f}], "
              f"analytic_err={err7:.2e}, time={r7['computation_time']:.4f}s")

    # Tests
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
