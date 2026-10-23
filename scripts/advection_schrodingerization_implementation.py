"""Manual implementation of 1D Advection Equation Solver via Schrodingerization.

Solves the 1D advection equation:
    ∂u/∂t + a·∂u/∂x = 0

using Schrodingerization — a framework that transforms non-unitary PDE evolution
into unitary (quantum-simulatable) Hamiltonian dynamics.

Two solver paths:
    1. Classical: Direct matrix exponentiation in the Schrodingerized space.
    2. Trotter (Quantum): Lie-Trotter splitting of the Hamiltonian for
       gate-based quantum simulation.

Workflow:
    1. Parse parameters (domain, grid, boundary condition, scheme, velocity).
    2. Build spatial grid and first-order finite difference operator.
    3. Detect unitary structure:
       - Central difference + periodic BC → already unitary (skip Schrod.)
       - Upwind / Dirichlet → apply Schrodingerization.
    4. Construct Hamiltonian H from system matrix A.
    5. Time evolution (classical expm or Trotter steps).
    6. Recover solution u(x,T) and visualize.

Schrodingerization key idea:
    du/dt = A·u + b  →  dψ/dt = -i·H·ψ
    where H is constructed from the Hermitian/anti-Hermitian parts of A.

Components:
    - build_grid: Construct spatial grid and coordinate array
    - first_order_derivative_matrix: Finite difference operator
    - schrodingerize: Transform non-unitary A into Hamiltonian H
    - solve_classical: Direct matrix exponential evolution
    - solve_trotter: Lie-Trotter splitting evolution
    - AdvectionSolver: Class-based interface

Reference:
    SKILL.md — Advection Equation Solver via Schrodingerization
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

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


def build_grid(
    L: float,
    nx: int,
    boundary_condition: str = "periodic",
) -> Tuple[np.ndarray, float, int]:
    """Build the spatial grid for 1D advection.

    For periodic BC: N = 2^nx points, x ∈ [0, L), dx = L/N.
    For Dirichlet BC: N = 2^nx interior points, x ∈ (0, L), dx = L/(N+1).

    Args:
        L: Domain length.
        nx: Log2 of grid points (N = 2^nx).
        boundary_condition: 'periodic' or 'dirichlet'.

    Returns:
        Tuple of (x_coords, dx, N).
            - x_coords: Array of spatial coordinates.
            - dx: Grid spacing.
            - N: Number of grid points.
    """
    N = 1 << nx  # 2^nx

    if boundary_condition == "periodic":
        dx = L / N
        x = np.arange(0, L, dx)
    else:
        dx = L / (N + 1)
        x = np.arange(dx, L, dx)

    return x, dx, N


# ===================================================================
# 2. First-order finite difference operator
# ===================================================================


def first_order_derivative_matrix(
    N: int,
    dx: float,
    boundary_condition: str = "periodic",
    scheme: str = "central",
    a: float = 1.0,
) -> np.ndarray:
    """Build the first-order spatial derivative matrix A = a·D.

    Supports:
        - central:  2nd-order central difference (skew-symmetric)
        - forward:  1st-order forward difference (upwind for a < 0)
        - backward: 1st-order backward difference (upwind for a > 0)

    For central + periodic BC, A is skew-Hermitian → unitary evolution.

    Args:
        N: Number of grid points.
        dx: Grid spacing.
        boundary_condition: 'periodic' or 'dirichlet'.
        scheme: 'central', 'forward', or 'backward'.
        a: Advection velocity.

    Returns:
        Derivative matrix A of shape (N, N).
    """
    D = np.zeros((N, N), dtype=np.float64)

    if scheme == "central":
        # Central difference: (u_{i+1} - u_{i-1}) / (2*dx)
        for i in range(N):
            D[i, (i + 1) % N] = 0.5
            D[i, (i - 1) % N] = -0.5
        if boundary_condition == "dirichlet":
            # Fix boundaries for Dirichlet
            D[0, :] = 0
            D[0, 0] = -1.5
            D[0, 1] = 2.0
            D[0, 2] = -0.5
            D[-1, :] = 0
            D[-1, -1] = 1.5
            D[-1, -2] = -2.0
            D[-1, -3] = 0.5

    elif scheme == "forward":
        # Forward difference: (u_{i+1} - u_i) / dx
        for i in range(N):
            D[i, i] = -1.0
            if i < N - 1:
                D[i, i + 1] = 1.0
            elif boundary_condition == "periodic":
                D[i, 0] = 1.0
        if boundary_condition == "dirichlet":
            D[-1, -1] = 0  # last row zero for Dirichlet

    elif scheme == "backward":
        # Backward difference: (u_i - u_{i-1}) / dx
        for i in range(N):
            D[i, i] = 1.0
            if i > 0:
                D[i, i - 1] = -1.0
            elif boundary_condition == "periodic":
                D[i, -1] = -1.0
        if boundary_condition == "dirichlet":
            D[0, 0] = 0  # first row zero for Dirichlet

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return (a / dx) * D


# ===================================================================
# 3. Schrodingerization
# ===================================================================


def is_unitary_system(A: np.ndarray) -> bool:
    """Check if system matrix A is skew-Hermitian (→ unitary evolution).

    A is skew-Hermitian if A† = −A, meaning iA is Hermitian.
    Central difference with periodic BC produces skew-Hermitian matrices.

    Args:
        A: System matrix.

    Returns:
        True if A is approximately skew-Hermitian.
    """
    return bool(np.allclose(A, -A.conj().T, atol=1e-12))


def decompose_matrix(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split A into Hermitian and anti-Hermitian parts.

    H1 = (A + A†)/2  (Hermitian part)
    H2 = (A − A†)/(2i)  (Hermitian part from anti-Hermitian of A)

    Then: A = H1 + i·H2, and H = H1 ⊗ ... for Schrodingerization.

    Args:
        A: System matrix.

    Returns:
        Tuple of (H1, H2) — both Hermitian.
    """
    H1 = (A + A.conj().T) / 2.0
    H2 = (A - A.conj().T) / (2.0j)
    # Ensure Hermitian (remove tiny imaginary noise)
    H1 = (H1 + H1.conj().T) / 2.0
    H2 = (H2 + H2.conj().T) / 2.0
    return H1, H2


def build_schrodinger_hamiltonian(
    A: np.ndarray,
    na: int = 2,
    R: float = 2.0,
) -> np.ndarray:
    """Construct the Schrodingerized Hamiltonian from system matrix A.

    The auxiliary dimension (size 2^na) is used to lift the non-unitary
    dynamics into a higher-dimensional unitary evolution.

    H = D ⊗ H1 + I ⊗ H2
    where D is a diagonal matrix in auxiliary space and H1, H2 are the
    Hermitian/anti-Hermitian parts of A.

    For the simplified case (already unitary, or single auxiliary mode):
        If A is skew-Hermitian: H = i·A (direct Hamiltonian)
        Otherwise: H is built from H1, H2 with auxiliary dimension.

    Args:
        A: System matrix (N×N).
        na: Number of auxiliary qubits (aux dim = 2^na).
        R: Scaling parameter for the auxiliary dimension.

    Returns:
        Schrodingerized Hamiltonian H of shape (2^na·N, 2^na·N).
    """
    if is_unitary_system(A):
        # Already unitary: H = i·A (i·skew-Hermitian = Hermitian)
        H = 1j * A
        return (H + H.conj().T) / 2.0  # ensure Hermitian

    # Non-unitary case: full Schrodingerization
    H1, H2 = decompose_matrix(A)
    aux_dim = 1 << na
    N = A.shape[0]

    # Auxiliary space: diagonal matrix D
    # D[k] = (2k+1 - aux_dim) * R / aux_dim  (centered symmetric)
    d_vals = np.array([
        (2 * k + 1 - aux_dim) * R / aux_dim for k in range(aux_dim)
    ])
    D = np.diag(d_vals)

    # H = D ⊗ H1 + I_aux ⊗ H2
    H = np.kron(D, H1) + np.kron(np.eye(aux_dim), H2)

    # Ensure Hermitian
    return (H + H.conj().T) / 2.0


# ===================================================================
# 4. Time evolution
# ===================================================================


def solve_classical(
    A: np.ndarray,
    u0: np.ndarray,
    T: float,
    na: int = 2,
    R: float = 2.0,
    n_steps: int = 100,
) -> np.ndarray:
    """Solve advection via classical Schrodingerization.

    Uses scipy.linalg.expm for exact matrix exponentiation.

    Two paths:
        - If A is already unitary (skew-Hermitian): evolve directly
          in the original N-dimensional space via expm(A·T).
        - Otherwise: Schrodingerize into aux_dim·N space and evolve
          via expm(−i·H·t), then recover u from the first aux mode.

    Args:
        A: System matrix (N×N).
        u0: Initial condition u(x,0), shape (N,).
        T: Final time.
        na: Auxiliary qubits (only used for non-unitary path).
        R: Scaling parameter (only used for non-unitary path).
        n_steps: Number of time steps.

    Returns:
        Solution u(x,T), shape (N,).
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required. Install with: pip install scipy")

    N = A.shape[0]
    dt = T / n_steps

    if is_unitary_system(A):
        # Already unitary: evolve directly in original space
        # du/dt = A·u,  A is skew-Hermitian → u(t) = expm(A·t)·u(0)
        U = expm(A * dt)
        psi = u0.astype(np.complex128)
        for _ in range(n_steps):
            psi = U @ psi
        return psi.real

    # Non-unitary: full Schrodingerization with auxiliary dimension
    H = build_schrodinger_hamiltonian(A, na=na, R=R)
    aux_dim = 1 << na

    # Lifted initial state: ψ0 = [u0, 0, ..., 0] (first auxiliary mode)
    psi = np.zeros(aux_dim * N, dtype=np.complex128)
    psi[:N] = u0.astype(np.complex128)

    # Time evolution via matrix exponential
    U = expm(-1j * H * dt)
    for _ in range(n_steps):
        psi = U @ psi

    # Recover u from first auxiliary mode
    u_final = psi[:N].real
    return u_final


def solve_trotter(
    A: np.ndarray,
    u0: np.ndarray,
    T: float,
    dt: float = 0.01,
    na: int = 2,
    R: float = 2.0,
    order: int = 1,
) -> np.ndarray:
    """Solve advection via Lie-Trotter splitting (quantum simulation path).

    Uses first-order Trotter: e^{-i(H1+H2)dt} ≈ e^{-iH1·dt}·e^{-iH2·dt}

    Args:
        A: System matrix (N×N).
        u0: Initial condition u(x,0).
        T: Final time.
        dt: Time step size.
        na: Auxiliary qubits.
        R: Scaling parameter.
        order: Trotter order (1 or 2).

    Returns:
        Solution u(x,T).
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required. Install with: pip install scipy")

    H1, H2 = decompose_matrix(A)
    aux_dim = 1 << na
    N = A.shape[0]

    # If already unitary (H1 ≈ 0), evolve directly in original space
    if np.allclose(H1, 0, atol=1e-12):
        # A is skew-Hermitian: du/dt = A·u → u(t+dt) = expm(A·dt)·u(t)
        U = expm(A * dt)
        psi = u0.astype(np.complex128)
        n_steps = int(T / dt)
        if n_steps < 1:
            n_steps = 1
        for _ in range(n_steps):
            psi = U @ psi
        return psi.real

    # Non-unitary: Schrodingerized Trotter
    d_vals = np.array([
        (2 * k + 1 - aux_dim) * R / aux_dim for k in range(aux_dim)
    ])
    D = np.diag(d_vals)

    # Build split Hamiltonians
    H_aux = np.kron(D, np.eye(N))  # D ⊗ I
    H_sys_1 = np.kron(np.eye(aux_dim), H1)  # I ⊗ H1
    H_sys_2 = np.kron(np.eye(aux_dim), H2)  # I ⊗ H2

    # Lifted initial state
    psi = np.zeros(aux_dim * N, dtype=np.complex128)
    psi[:N] = u0.astype(np.complex128)

    n_steps = int(T / dt)
    if n_steps < 1:
        n_steps = 1

    if order == 1:
        U1 = expm(-1j * H_aux * dt)
        U2 = expm(-1j * (H_sys_1 + H_sys_2) * dt)
        for _ in range(n_steps):
            psi = U2 @ (U1 @ psi)
    else:
        # Strang splitting (2nd order)
        U1_half = expm(-1j * H_aux * dt / 2)
        U2_full = expm(-1j * (H_sys_1 + H_sys_2) * dt)
        for _ in range(n_steps):
            psi = U1_half @ psi
            psi = U2_full @ psi
            psi = U1_half @ psi

    u_final = psi[:N].real
    return u_final


# ===================================================================
# 5. Initial conditions
# ===================================================================


def initial_condition(x: np.ndarray, ic_type: str = "sine", L: float = 1.0) -> np.ndarray:
    """Generate initial condition u(x, 0).

    Args:
        x: Spatial coordinates.
        ic_type: 'sine', 'gaussian', or 'step'.
        L: Domain length.

    Returns:
        Initial values u(x,0).
    """
    if ic_type == "sine":
        return np.sin(2 * np.pi * x / L)
    elif ic_type == "gaussian":
        center = L / 2
        sigma = L / 10
        return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    elif ic_type == "step":
        u = np.zeros_like(x)
        u[x < L / 2] = 1.0
        return u
    else:
        raise ValueError(f"Unknown initial condition: {ic_type}")


# ===================================================================
# 6. Visualization
# ===================================================================


def plot_solution(
    x: np.ndarray,
    u: np.ndarray,
    u0: Optional[np.ndarray] = None,
    title: str = "Advection Equation Solution",
    output_path: str = "advection_solution.svg",
) -> str:
    """Plot the advection solution.

    Args:
        x: Spatial coordinates.
        u: Final solution u(x,T).
        u0: Optional initial condition for comparison.
        title: Plot title.
        output_path: Output file path.

    Returns:
        Absolute path to saved plot.
    """
    if not HAS_MATPLOTLIB:
        return ""

    import os

    plt.figure(figsize=(8, 5))
    if u0 is not None:
        plt.plot(x, u0, "b--", lw=1.5, alpha=0.6, label="u(x,0) Initial")
    plt.plot(x, u, "r-", lw=2, label="u(x,T) Final")
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


def advection_solve(
    L: float = 1.0,
    T: float = 0.5,
    nx: int = 3,
    a: float = 0.5,
    boundary_condition: str = "periodic",
    scheme: str = "central",
    method: str = "classical",
    ic_type: str = "sine",
    na: int = 2,
    R: float = 2.0,
    dt: float = 0.01,
    order: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the 1D advection equation ∂u/∂t + a·∂u/∂x = 0.

    Pipeline:
        1. Build spatial grid.
        2. Construct finite difference derivative matrix A.
        3. Detect unitary structure.
        4. Time evolution (classical expm or Trotter).
        5. Generate plot.

    Args:
        L: Domain length [0, L].
        T: Final time.
        nx: Log2 of grid points (N = 2^nx).
        a: Advection velocity.
        boundary_condition: 'periodic' or 'dirichlet'.
        scheme: 'central', 'forward', or 'backward'.
        method: 'classical' or 'trotter'.
        ic_type: 'sine', 'gaussian', or 'step'.
        na: Auxiliary qubits for Schrodingerization.
        R: Scaling parameter.
        dt: Time step (Trotter only).
        order: Trotter order (1 or 2).
        verbose: Print progress.

    Returns:
        Dict with: status, x, u, u0, grid info, computation_time.

    Raises:
        ImportError: If scipy is missing.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required. Install with: pip install scipy")

    # --- Stage 1: Grid ---
    x, dx, N = build_grid(L, nx, boundary_condition)

    # Auto-select scheme for stability
    if scheme == "upwind":
        scheme = "forward" if a >= 0 else "backward"

    u0 = initial_condition(x, ic_type, L)

    if verbose:
        print(f"1D Advection Equation Solver")
        print(f"  ∂u/∂t + {a}·∂u/∂x = 0")
        print(f"  Domain: [0, {L}], T = {T}")
        print(f"  Grid: N = 2^{nx} = {N}, dx = {dx:.4f}")
        print(f"  BC: {boundary_condition}, scheme: {scheme}")
        print(f"  Method: {method}, aux qubits na = {na}")

    # --- Stage 2: Derivative matrix ---
    A = first_order_derivative_matrix(N, dx, boundary_condition, scheme, a)
    _unitary = is_unitary_system(A)

    if verbose:
        print(f"  System matrix A: shape={A.shape}")
        print(f"  Already unitary: {_unitary}")

    # --- Stage 3: Time evolution ---
    t_start = time.perf_counter()

    if method == "classical":
        u = solve_classical(A, u0, T, na=na, R=R)
    elif method == "trotter":
        u = solve_trotter(A, u0, T, dt=dt, na=na, R=R, order=order)
        Nt = int(T / dt)
        if verbose:
            print(f"  Time steps: Nt = {Nt}, dt = {dt}")
    else:
        raise ValueError(f"Unknown method: {method}")

    comp_time = time.perf_counter() - t_start

    if verbose:
        print(f"  Computation time: {comp_time:.4f}s")
        print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
        print(f"  Status: ok")

    return {
        "status": "ok",
        "message": "Advection equation solved",
        "grid": {
            "n_points": N,
            "dx": dx,
            "dt": dt if method == "trotter" else None,
            "nt": int(T / dt) if method == "trotter" else None,
        },
        "x": x.tolist(),
        "u": u.tolist(),
        "u0": u0.tolist(),
        "computation_time": round(comp_time, 4),
        "is_unitary": _unitary,
        "a": a,
        "L": L,
        "T": T,
        "method": method,
        "scheme": scheme,
    }


# ===================================================================
# 8. Class-based interface
# ===================================================================


class AdvectionSolver:
    """Class-based solver for 1D advection via Schrodingerization.

    Usage:
        solver = AdvectionSolver()
        result = solver.run(L=1.0, T=0.5, nx=3, a=0.5)
        print(result['x'], result['u'])

        # Trotter (quantum simulation) path
        result2 = solver.run(method='trotter', dt=0.02)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        L: float = 1.0,
        T: float = 0.5,
        nx: int = 3,
        a: float = 0.5,
        boundary_condition: str = "periodic",
        scheme: str = "central",
        method: str = "classical",
        ic_type: str = "sine",
        na: int = 2,
        R: float = 2.0,
        dt: float = 0.01,
        order: int = 1,
    ) -> Dict[str, Any]:
        """Solve advection equation. See advection_solve() for docs."""
        result = advection_solve(
            L=L, T=T, nx=nx, a=a,
            boundary_condition=boundary_condition,
            scheme=scheme, method=method, ic_type=ic_type,
            na=na, R=R, dt=dt, order=order,
            verbose=(self.text_mode != "plain"),
        )

        # Generate plot
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            title = (
                f"1D Advection: ∂u/∂t + {a}·∂u/∂x = 0\n"
                f"nx={nx}, {method}, {scheme}, T={T}"
            )
            plot_path = plot_solution(
                np.array(result["x"]), np.array(result["u"]),
                u0=np.array(result["u0"]),
                title=title,
                output_path=os.path.join(self.algo_dir, "advection_solution.svg"),
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
            "is_unitary": result.get("is_unitary", False),
            "plot": result.get("plot", {}),
            "circuit": [],
        }


# ===================================================================
# 9. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "periodic_central_n3_T0.5",
        "L": 1.0, "T": 0.5, "nx": 3, "a": 0.5,
        "bc": "periodic", "scheme": "central",
        "method": "classical", "ic": "sine",
    },
    {
        "name": "periodic_central_n4_T1.0",
        "L": 1.0, "T": 1.0, "nx": 4, "a": 0.3,
        "bc": "periodic", "scheme": "central",
        "method": "classical", "ic": "sine",
    },
    {
        "name": "periodic_central_gaussian",
        "L": 2.0, "T": 0.5, "nx": 4, "a": 0.5,
        "bc": "periodic", "scheme": "central",
        "method": "classical", "ic": "gaussian",
    },
    {
        "name": "periodic_trotter_n3",
        "L": 1.0, "T": 0.2, "nx": 3, "a": 0.5,
        "bc": "periodic", "scheme": "central",
        "method": "trotter", "ic": "sine",
        "dt": 0.02, "order": 1,
    },
    {
        "name": "periodic_trotter_n4_order2",
        "L": 1.0, "T": 0.3, "nx": 4, "a": 0.3,
        "bc": "periodic", "scheme": "central",
        "method": "trotter", "ic": "sine",
        "dt": 0.02, "order": 2,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]

    kwargs = {
        "L": case["L"], "T": case["T"], "nx": case["nx"],
        "a": case["a"], "boundary_condition": case["bc"],
        "scheme": case["scheme"], "method": case["method"],
        "ic_type": case["ic"],
    }
    if "dt" in case:
        kwargs["dt"] = case["dt"]
    if "order" in case:
        kwargs["order"] = case["order"]

    try:
        result = advection_solve(**kwargs, verbose=False)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False

    u = np.array(result["u"])
    x = np.array(result["x"])

    # Basic sanity checks
    ok = (
        result["status"] == "ok"
        and u.shape == x.shape
        and np.all(np.isfinite(u))
        and len(u) > 0
    )

    icon = "ok" if ok else "FAIL"
    print(f"  [{icon}] {name}: N={len(u)}, "
          f"u∈[{u.min():.3f}, {u.max():.3f}], "
          f"time={result['computation_time']:.3f}s, "
          f"unitary={result['is_unitary']}")
    return ok


# ===================================================================
# 10. Main
# ===================================================================

def main() -> None:
    """Run advection equation demonstration pipeline."""
    if not HAS_SCIPY:
        print("scipy is not installed. Install with: pip install scipy")
        return

    print("=" * 60)
    print("1D Advection Equation Solver via Schrödingerization")
    print("  ∂u/∂t + a·∂u/∂x = 0")
    print("=" * 60)

    solver = AdvectionSolver()

    # --- Demo 1: Classical, periodic + central (already unitary) ---
    print("\n--- Demo 1: Classical, periodic+central (skew-Hermitian) ---")
    result1 = solver.run(
        L=1.0, T=0.5, nx=4, a=0.5,
        boundary_condition="periodic", scheme="central", method="classical", ic_type="sine",
    )
    u1 = np.array(result1["u"])
    print(f"  Grid N = {len(u1)}, dx = {result1['grid']['dx']:.4f}")
    print(f"  Is unitary: {result1['is_unitary']}")
    print(f"  u range: [{u1.min():.4f}, {u1.max():.4f}]")

    # --- Demo 2: Gaussian pulse advection ---
    print("\n--- Demo 2: Gaussian pulse advection ---")
    result2 = solver.run(
        L=2.0, T=0.3, nx=5, a=0.8,
        boundary_condition="periodic", scheme="central", method="classical", ic_type="gaussian",
    )
    u2 = np.array(result2["u"])
    print(f"  Grid N = {len(u2)}")
    print(f"  u range: [{u2.min():.4f}, {u2.max():.4f}]")

    # --- Demo 3: Trotter (quantum simulation path) ---
    print("\n--- Demo 3: Trotter splitting (quantum simulation) ---")
    result3 = solver.run(
        L=1.0, T=0.2, nx=3, a=0.5,
        boundary_condition="periodic", scheme="central", method="trotter",
        dt=0.02, order=2, ic_type="sine",
    )
    u3 = np.array(result3["u"])
    print(f"  Steps Nt = {result3['grid']['nt']}, dt = {result3['grid']['dt']}")
    print(f"  u range: [{u3.min():.4f}, {u3.max():.4f}]")

    # --- Demo 4: Compare classical vs Trotter ---
    print("\n--- Demo 4: Classical vs Trotter comparison ---")
    result4c = advection_solve(
        L=1.0, T=0.1, nx=4, a=0.5,
        boundary_condition="periodic", scheme="central", method="classical",
        verbose=False,
    )
    result4t = advection_solve(
        L=1.0, T=0.1, nx=4, a=0.5,
        boundary_condition="periodic", scheme="central", method="trotter",
        dt=0.01, order=2, verbose=False,
    )
    u_c = np.array(result4c["u"])
    u_t = np.array(result4t["u"])
    diff = np.max(np.abs(u_c - u_t))
    print(f"  Classical:  u∈[{u_c.min():.4f}, {u_c.max():.4f}]")
    print(f"  Trotter:    u∈[{u_t.min():.4f}, {u_t.max():.4f}]")
    print(f"  Max |diff|: {diff:.2e}")

    # --- Demo 5: Scheme comparison ---
    print("\n--- Demo 5: Central vs Upwind schemes ---")
    for sc in ["central", "forward"]:
        result5 = advection_solve(
            L=1.0, T=0.2, nx=4, a=0.5,
            boundary_condition="periodic", scheme=sc, method="classical",
            verbose=False,
        )
        u5 = np.array(result5["u"])
        print(f"  {sc:>8}: u∈[{u5.min():.4f}, {u5.max():.4f}], "
              f"unitary={result5['is_unitary']}")

    # --- Demo 6: Grid convergence ---
    print("\n--- Demo 6: Grid convergence (nx = 2..6) ---")
    for nv in range(2, 7):
        result6 = advection_solve(
            L=1.0, T=0.3, nx=nv, a=0.5,
            boundary_condition="periodic", scheme="central", method="classical",
            verbose=False,
        )
        u6 = np.array(result6["u"])
        print(f"  nx={nv} (N={1<<nv}): u∈[{u6.min():.4f}, {u6.max():.4f}], "
              f"time={result6['computation_time']:.4f}s")

    # --- Demo 7: Matrix structure analysis ---
    print("\n--- Demo 7: System matrix structure (nx=3, central+periodic) ---")
    A7 = first_order_derivative_matrix(8, 1.0/8, "periodic", "central", 0.5)
    H1, H2 = decompose_matrix(A7)
    print(f"  A shape: {A7.shape}")
    print(f"  Skew-Hermitian: {is_unitary_system(A7)}")
    print(f"  ‖H1‖ (Hermitian part): {np.linalg.norm(H1, 'fro'):.2e}")
    print(f"  ‖H2‖ (anti-Herm. part): {np.linalg.norm(H2, 'fro'):.4f}")
    print(f"  Eigenvalues of A (top 3): "
          f"{np.sort(np.linalg.eigvals(A7).real)[:3]}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
