"""Manual implementation of 2D Heat Equation Solver via Schrodingerization.

Solves the 2D heat (diffusion) equation:
    ∂u/∂t = a1·∂²u/∂x² + a2·∂²u/∂y² + f(x,y)

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

2D Laplacian via Kronecker product:
    A = a1·(L_x ⊗ I) + a2·(I ⊗ L_y)

Components:
    - build_heat2d_grid: 2D spatial grid construction
    - laplacian_matrix_2d: 2D Laplacian via Kronecker product
    - build_heat2d_hamiltonian: Schrodingerized Hamiltonian
    - solve_classical / solve_trotter: Time evolution
    - Heat2DSolver: Class-based interface

Reference:
    SKILL.md — 2D Heat Equation via Schrodingerization
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


def build_heat2d_grid(
    L: float,
    nx: int,
    boundary_condition: str = "dirichlet",
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Build 2D spatial grid for the heat equation.

    Dirichlet: N = 2^nx interior points per dimension, x,y ∈ (0,L) spaced by L/(N+1).
    Periodic:  N = 2^nx points wrapping [0,L), spaced by L/N.
    Neumann:   N = 2^nx points including boundaries, spaced by L/(N-1).

    Args:
        L: Domain length (same in x and y).
        nx: Log2 of grid points per dimension.
        boundary_condition: 'dirichlet', 'periodic', or 'neumann'.

    Returns:
        Tuple of (x, y, dx, N).
    """
    N = 1 << nx

    if boundary_condition == "periodic":
        dx = L / N
        x = np.arange(0, L, dx)
        y = np.arange(0, L, dx)
    elif boundary_condition == "neumann":
        dx = L / (N - 1) if N > 1 else L
        x = np.arange(0, L + dx / 2, dx)
        y = np.arange(0, L + dx / 2, dx)
    else:  # dirichlet
        dx = L / (N + 1)
        x = np.arange(dx, L, dx)
        y = np.arange(dx, L, dx)

    return x, y, dx, N


# ===================================================================
# 2. Second-order finite difference Laplacian (1D, reused for 2D)
# ===================================================================


def laplacian_matrix_1d(
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
        L_mat[i, i] = -2.0
        if i > 0:
            L_mat[i, i - 1] = 1.0
        elif boundary_condition == "periodic":
            L_mat[i, N - 1] = 1.0

        if i < N - 1:
            L_mat[i, i + 1] = 1.0
        elif boundary_condition == "periodic":
            L_mat[i, 0] = 1.0

    if boundary_condition == "neumann":
        L_mat[0, 0] = -1.0
        L_mat[0, 1] = 1.0
        L_mat[-1, -1] = -1.0
        L_mat[-1, -2] = 1.0

    return L_mat / (dx * dx)


# ===================================================================
# 3. 2D Laplacian via Kronecker product
# ===================================================================


def laplacian_matrix_2d(
    N: int,
    dx: float,
    a1: float = 1.0,
    a2: float = 1.0,
    boundary_condition: str = "dirichlet",
) -> np.ndarray:
    """Build the 2D anisotropic Laplacian matrix.

    A = a1·(L_x ⊗ I) + a2·(I ⊗ L_y)

    where L_x and L_y are identical 1D Laplacians for a square domain.

    Args:
        N: Number of grid points per dimension.
        dx: Grid spacing.
        a1: Diffusion coefficient in x-direction.
        a2: Diffusion coefficient in y-direction.
        boundary_condition: 'dirichlet', 'periodic', or 'neumann'.

    Returns:
        2D Laplacian matrix of shape (N², N²).
    """
    A0 = laplacian_matrix_1d(N, dx, boundary_condition)
    A = a1 * np.kron(A0, np.eye(N)) + a2 * np.kron(np.eye(N), A0)
    return A


# ===================================================================
# 4. Source term assembly (2D)
# ===================================================================


def build_source_2d(
    x: np.ndarray,
    y: np.ndarray,
    N: int,
    a1: float = 1.0,
    a2: float = 1.0,
    source_func: Optional[callable] = None,
) -> Optional[np.ndarray]:
    """Build 2D source term vector.

    If source_func(x, y) is provided, evaluates it on the 2D grid and flattens.
    Otherwise returns None.

    Args:
        x: 1D x-coordinate array.
        y: 1D y-coordinate array.
        N: Number of grid points per dimension.
        a1, a2: Diffusion coefficients.
        source_func: Callable f(x, y) → scalar, or None.

    Returns:
        Flattened source vector of shape (N²,) or None.
    """
    if source_func is None:
        return None

    X, Y = np.meshgrid(x, y, indexing="ij")
    f_vals = source_func(X, Y)
    return f_vals.flatten().astype(np.float64)


# ===================================================================
# 5. Schrodingerization for 2D heat equation
# ===================================================================


def build_heat2d_hamiltonian(
    A: np.ndarray,
    na: int = 2,
    R: float = 2.0,
) -> np.ndarray:
    """Schrodingerize the 2D diffusion system matrix A.

    For the heat equation, A = a1·kron(L,I) + a2·kron(I,L) is always
    symmetric negative semidefinite (non-unitary), so full Schrodingerization
    is needed.

    H = D_η ⊗ H1 + I ⊗ H2
    where:
        - H1 = (A + A†)/2 = A (since A is symmetric real)
        - H2 = (A - A†)/(2i) ≈ 0 (A has no anti-Hermitian part)
        - D_η is a diagonal auxiliary operator

    Args:
        A: System matrix (N²×N²), symmetric negative-semidefinite.
        na: Number of auxiliary qubits (2^na auxiliary modes).
        R: Scaling parameter for auxiliary dimension.

    Returns:
        Hermitian Hamiltonian of shape (2^na·N², 2^na·N²).
    """
    N2 = A.shape[0]
    aux_dim = 1 << na

    # A is real symmetric → H1 = A, H2 ≈ 0
    H1 = A
    H2 = np.zeros((N2, N2), dtype=np.float64)

    # Auxiliary operator D_η (diagonal, centered symmetric)
    d_vals = np.array([
        (2 * k + 1 - aux_dim) * R / aux_dim for k in range(aux_dim)
    ])
    D_eta = np.diag(d_vals)

    # H = D_η ⊗ H1 + I ⊗ H2 = D_η ⊗ A (since H2 = 0)
    H = np.kron(D_eta, H1) + np.kron(np.eye(aux_dim), H2)

    return (H + H.conj().T) / 2.0  # ensure Hermitian


# ===================================================================
# 6. Time evolution
# ===================================================================


def solve_heat2d_classical(
    A: np.ndarray,
    u0: np.ndarray,
    T: float,
    source: Optional[np.ndarray] = None,
    na: int = 2,
    R: float = 2.0,
    n_steps: int = 100,
) -> np.ndarray:
    """Solve 2D heat equation via classical Schrodingerization.

    Uses matrix exponentiation of the Schrodingerized Hamiltonian.

    Args:
        A: 2D system matrix a1·kron(L,I)+a2·kron(I,L) of shape (N²,N²).
        u0: Flattened 2D initial condition, shape (N²,).
        T: Final time.
        source: Optional flattened source term, shape (N²,).
        na: Auxiliary qubits.
        R: Scaling parameter.
        n_steps: Number of time steps.

    Returns:
        Flattened solution u(x,y,T), shape (N²,).
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    N2 = A.shape[0]
    dt = T / n_steps
    H = build_heat2d_hamiltonian(A, na=na, R=R)
    aux_dim = 1 << na

    # Lift: ψ = [u, 0, ..., 0] in aux·N² space
    psi = np.zeros(aux_dim * N2, dtype=np.complex128)
    psi[:N2] = u0.astype(np.complex128)

    if source is not None:
        b_lifted = np.zeros(aux_dim * N2, dtype=np.complex128)
        b_lifted[:N2] = source.astype(np.complex128)
    else:
        b_lifted = None

    U = expm(-1j * H * dt)
    for _ in range(n_steps):
        psi = U @ psi
        if b_lifted is not None:
            psi = psi + dt * b_lifted

    return psi[:N2].real


def solve_heat2d_trotter(
    A: np.ndarray,
    u0: np.ndarray,
    T: float,
    dt: float = 0.01,
    source: Optional[np.ndarray] = None,
    na: int = 2,
    R: float = 2.0,
    order: int = 1,
) -> np.ndarray:
    """Solve 2D heat equation via Trotter splitting.

    Splits H into x-direction and y-direction blocks for gate-based
    quantum simulation.

    Args:
        A: 2D system matrix (N²×N²).
        u0: Flattened 2D initial condition.
        T: Final time.
        dt: Time step size.
        source: Optional flattened source term.
        na: Auxiliary qubits.
        R: Scaling parameter.
        order: Trotter order (1 or 2).

    Returns:
        Flattened solution u(x,y,T), shape (N²,).
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    N2 = A.shape[0]
    aux_dim = 1 << na

    # Auxiliary operator
    d_vals = np.array([
        (2 * k + 1 - aux_dim) * R / aux_dim for k in range(aux_dim)
    ])
    D_eta = np.diag(d_vals)

    # Split Hamiltonians
    H_aux = np.kron(D_eta, np.eye(N2))
    H_sys = np.kron(np.eye(aux_dim), A)

    # Lifted state
    psi = np.zeros(aux_dim * N2, dtype=np.complex128)
    psi[:N2] = u0.astype(np.complex128)

    n_steps = max(1, int(T / dt))
    if source is not None:
        b_lifted = np.zeros(aux_dim * N2, dtype=np.complex128)
        b_lifted[:N2] = source.astype(np.complex128)
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

    return psi[:N2].real


# ===================================================================
# 7. Initial conditions (2D)
# ===================================================================


def initial_condition_2d(
    x: np.ndarray,
    y: np.ndarray,
    ic_type: str = "sine",
    L: float = 1.0,
) -> np.ndarray:
    """Generate 2D initial temperature distribution u(x,y,0).

    Args:
        x: 1D x-coordinate array.
        y: 1D y-coordinate array.
        ic_type: 'sine', 'gaussian', 'step', or 'custom'.
        L: Domain length.

    Returns:
        2D array u(x,y,0) of shape (len(x), len(y)).
    """
    X, Y = np.meshgrid(x, y, indexing="ij")

    if ic_type == "sine":
        # u(x,y,0) = sin(πx/L)·sin(πy/L)
        return np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    elif ic_type == "sine_2d_multi":
        # u(x,y,0) = sin(πx/L)·sin(2πy/L)
        return np.sin(np.pi * X / L) * np.sin(2 * np.pi * Y / L)
    elif ic_type == "gaussian":
        center = L / 2
        sigma = L / 4
        return np.exp(-((X - center) ** 2 + (Y - center) ** 2) / (2 * sigma ** 2))
    elif ic_type == "step":
        u = np.zeros_like(X)
        mask = (X < L / 2) & (Y < L / 2)
        u[mask] = 1.0
        return u
    elif ic_type == "double_sine":
        # u(x,y,0) = sin(2πx/L)·sin(2πy/L)
        return np.sin(2 * np.pi * X / L) * np.sin(2 * np.pi * Y / L)
    else:
        raise ValueError(f"Unknown ic_type: {ic_type}")


# ===================================================================
# 8. Visualization
# ===================================================================


def plot_heat2d_solution(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    u0: Optional[np.ndarray] = None,
    title: str = "2D Heat Equation Solution",
    output_path: str = "heat_2d_solution.svg",
    plot_type: str = "surface",
) -> str:
    """Plot 2D heat equation solution.

    Args:
        x: 1D x-coordinate array.
        y: 1D y-coordinate array.
        u: 2D solution u(x,y,T).
        u0: Optional 2D initial condition.
        title: Plot title.
        output_path: Output file.
        plot_type: 'surface' (3D) or 'contour' (2D contour).

    Returns:
        Absolute path to saved plot.
    """
    if not HAS_MATPLOTLIB:
        return ""
    import os

    X, Y = np.meshgrid(x, y, indexing="ij")

    if plot_type == "contour":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if u0 is not None:
            c1 = ax1.contourf(X, Y, u0, levels=20, cmap="viridis")
            fig.colorbar(c1, ax=ax1)
            ax1.set_title("u(x,y,0)")
        else:
            ax1.set_visible(False)

        c2 = ax2.contourf(X, Y, u, levels=20, cmap="viridis")
        fig.colorbar(c2, ax=ax2)
        ax2.set_title(f"u(x,y,T)")
        for ax in (ax1, ax2):
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

    else:  # surface (3D)
        fig = plt.figure(figsize=(12, 5))

        if u0 is not None:
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.plot_surface(X, Y, u0, cmap="coolwarm", edgecolor="none",
                             alpha=0.9)
            ax1.set_title("u(x,y,0)")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("u")

            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            ax2.plot_surface(X, Y, u, cmap="coolwarm", edgecolor="none",
                             alpha=0.9)
            ax2.set_title(f"u(x,y,T)")
        else:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.plot_surface(X, Y, u, cmap="coolwarm", edgecolor="none",
                            alpha=0.9)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    out = os.path.abspath(output_path)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ===================================================================
# 9. End-to-end solver
# ===================================================================


def heat2d_solve(
    L: float = 1.0,
    T: float = 0.1,
    nx: int = 4,
    a1: float = 0.1,
    a2: float = 0.1,
    boundary_condition: str = "dirichlet",
    method: str = "classical",
    ic_type: str = "sine",
    source_func: Optional[callable] = None,
    na: int = 2,
    R: float = 2.0,
    dt: float = 0.01,
    order: int = 1,
    n_steps: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve 2D heat equation ∂u/∂t = a1·∂²u/∂x² + a2·∂²u/∂y² + f(x,y).

    Args:
        L: Domain length.
        T: Final time.
        nx: Log2 of grid points per dimension.
        a1: Diffusion coefficient in x-direction (> 0).
        a2: Diffusion coefficient in y-direction (> 0).
        boundary_condition: 'dirichlet', 'periodic', or 'neumann'.
        method: 'classical' or 'trotter'.
        ic_type: Initial condition type.
        source_func: Optional source term f(x,y), callable.
        na: Auxiliary qubits for Schrodingerization.
        R: Scaling parameter.
        dt: Time step (Trotter only).
        order: Trotter order (1 or 2).
        n_steps: Number of time steps (classical only).
        verbose: Print progress.

    Returns:
        Dict with x, y, u, u0, grid info, computation time.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")
    if a1 <= 0 or a2 <= 0:
        raise ValueError(
            f"Diffusion coefficients must be > 0, got a1={a1}, a2={a2}"
        )

    # Grid
    x, y, dx, N = build_heat2d_grid(L, nx, boundary_condition)

    # Initial condition (2D)
    u0_2d = initial_condition_2d(x, y, ic_type, L)
    u0 = u0_2d.flatten()

    # Source term
    source = build_source_2d(x, y, N, a1, a2, source_func)

    if verbose:
        print(f"2D Heat Equation Solver via Schrödingerization")
        print(f"  ∂u/∂t = {a1}·∂²u/∂x² + {a2}·∂²u/∂y²", end="")
        print(f" + f(x,y)" if source_func is not None else "")
        print(f"  Domain: [0,{L}]×[0,{L}], T = {T}")
        print(f"  N = 2^{nx} = {N} per dim, total = {N*N}")
        print(f"  dx = {dx:.4f}")
        print(f"  BC: {boundary_condition}, IC: {ic_type}")
        print(f"  Method: {method}")

    # 2D Laplacian
    A = laplacian_matrix_2d(N, dx, a1, a2, boundary_condition)

    if verbose:
        print(f"  A: shape {A.shape}, "
              f"sparsity {np.mean(np.abs(A) > 1e-12):.3f}")
        evals = np.linalg.eigvalsh(A)
        print(f"  A eigenvalues: [{evals[0]:.2f}, {evals[-1]:.2f}]")

    # Time evolution
    t_start = time.perf_counter()

    if method == "classical":
        u_flat = solve_heat2d_classical(
            A, u0, T, source=source, na=na, R=R, n_steps=n_steps,
        )
    elif method == "trotter":
        u_flat = solve_heat2d_trotter(
            A, u0, T, dt=dt, source=source, na=na, R=R, order=order,
        )
        Nt = max(1, int(T / dt))
        if verbose:
            print(f"  Time steps: Nt = {Nt}, dt = {dt}, order = {order}")
    else:
        raise ValueError(f"Unknown method: {method}")

    u = u_flat.reshape((N, N))
    comp_time = time.perf_counter() - t_start

    # Analytical solution for sine IC with Dirichlet BC (for validation)
    analytic = None
    if (boundary_condition == "dirichlet" and ic_type == "sine"
            and source_func is None):
        # u(x,y,t) = sin(πx/L)·sin(πy/L)·exp(-(a1+a2)·π²·t/L²)
        X, Y = np.meshgrid(x, y, indexing="ij")
        analytic = (np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
                    * np.exp(-(a1 + a2) * np.pi ** 2 * T / L ** 2))

    if verbose:
        print(f"  Computation time: {comp_time:.4f}s")
        print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
        if analytic is not None:
            err = np.max(np.abs(u - analytic))
            print(f"  Max error vs analytic: {err:.2e}")
        print(f"  Status: ok")

    result: Dict[str, Any] = {
        "status": "ok",
        "message": "2D Heat equation solved",
        "grid": {
            "n_points": N, "dx": dx, "total_points": N * N,
            "dt": dt if method == "trotter" else None,
            "nt": max(1, int(T / dt)) if method == "trotter" else None,
        },
        "x": x.tolist(),
        "y": y.tolist(),
        "u": u.tolist(),
        "u0": u0_2d.tolist(),
        "computation_time": round(comp_time, 4),
        "a1": a1, "a2": a2, "L": L, "T": T,
        "method": method, "boundary_condition": boundary_condition,
    }
    if analytic is not None:
        result["analytic"] = analytic.tolist()
    return result


# ===================================================================
# 10. Class-based interface
# ===================================================================


class Heat2DSolver:
    """Class-based 2D heat equation solver.

    Usage:
        solver = Heat2DSolver()
        result = solver.run(L=1.0, T=0.1, nx=4, a1=0.1, a2=0.1)
        print(result['x'], result['y'], result['u'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        L: float = 1.0,
        T: float = 0.1,
        nx: int = 4,
        a1: float = 0.1,
        a2: float = 0.1,
        boundary_condition: str = "dirichlet",
        method: str = "classical",
        ic_type: str = "sine",
        source_func: Optional[callable] = None,
        na: int = 2,
        R: float = 2.0,
        dt: float = 0.01,
        order: int = 1,
        n_steps: int = 100,
    ) -> Dict[str, Any]:
        """Solve 2D heat equation. See heat2d_solve() for docs."""
        verbose = self.text_mode != "plain"
        result = heat2d_solve(
            L=L, T=T, nx=nx, a1=a1, a2=a2,
            boundary_condition=boundary_condition,
            method=method, ic_type=ic_type,
            source_func=source_func, na=na, R=R,
            dt=dt, order=order, n_steps=n_steps,
            verbose=verbose,
        )
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            bc_short = boundary_condition[:3]
            plot_path = plot_heat2d_solution(
                np.array(result["x"]), np.array(result["y"]),
                np.array(result["u"]),
                u0=np.array(result["u0"]),
                title=f"2D Heat: ∂u/∂t={a1}∂²u/∂x²+{a2}∂²u/∂y², "
                      f"{bc_short}, T={T}",
                output_path=os.path.join(self.algo_dir, "heat_2d_solution.svg"),
            )
            result["plot"] = {"format": "svg", "filename": plot_path}
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "message": result.get("message", ""),
            "grid": result.get("grid", {}),
            "x": result.get("x", []),
            "y": result.get("y", []),
            "u": result.get("u", []),
            "computation_time": result.get("computation_time", 0.0),
            "plot": result.get("plot", {}),
            "circuit": [],
        }


# ===================================================================
# 11. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "dirichlet_sine_n4_T0.1",
        "L": 1.0, "T": 0.1, "nx": 4, "a1": 0.1, "a2": 0.1,
        "bc": "dirichlet", "ic": "sine", "method": "classical",
    },
    {
        "name": "dirichlet_gaussian_n5",
        "L": 1.0, "T": 0.05, "nx": 5, "a1": 0.05, "a2": 0.05,
        "bc": "dirichlet", "ic": "gaussian", "method": "classical",
    },
    {
        "name": "periodic_double_sine_n4",
        "L": 1.0, "T": 0.1, "nx": 4, "a1": 0.1, "a2": 0.1,
        "bc": "periodic", "ic": "double_sine", "method": "classical",
    },
    {
        "name": "trotter_n4_order2",
        "L": 1.0, "T": 0.05, "nx": 3, "a1": 0.1, "a2": 0.1,
        "bc": "dirichlet", "ic": "sine", "method": "trotter",
        "dt": 0.01, "order": 2,
    },
    {
        "name": "trotter_n3_order1",
        "L": 1.0, "T": 0.04, "nx": 3, "a1": 0.1, "a2": 0.1,
        "bc": "dirichlet", "ic": "sine", "method": "trotter",
        "dt": 0.01, "order": 1,
    },
    {
        "name": "anisotropic_n4",
        "L": 1.0, "T": 0.1, "nx": 4, "a1": 0.2, "a2": 0.05,
        "bc": "dirichlet", "ic": "sine", "method": "classical",
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case and report result."""
    name = case["name"]
    kwargs = {
        "L": case["L"], "T": case["T"], "nx": case["nx"],
        "a1": case["a1"], "a2": case["a2"],
        "boundary_condition": case["bc"],
        "ic_type": case["ic"], "method": case["method"],
    }
    for k in ("dt", "order"):
        if k in case:
            kwargs[k] = case[k]
    try:
        result = heat2d_solve(**kwargs, verbose=False)
    except Exception as e:
        print(f"  [ERROR] {name}: {e}")
        return False
    u = np.array(result["u"])
    ok = result["status"] == "ok" and np.all(np.isfinite(u))
    icon = "ok" if ok else "FAIL"
    print(f"  [{icon}] {name}: N²={len(u.flatten())}, "
          f"u∈[{u.min():.4f}, {u.max():.4f}], "
          f"time={result['computation_time']:.3f}s")
    return ok


# ===================================================================
# 12. Main
# ===================================================================


def main() -> None:
    if not HAS_SCIPY:
        print("scipy required: pip install scipy")
        return

    print("=" * 60)
    print("2D Heat Equation Solver via Schrödingerization")
    print("  ∂u/∂t = a1·∂²u/∂x² + a2·∂²u/∂y² + f(x,y)")
    print("=" * 60)

    solver = Heat2DSolver()

    # Demo 1: Dirichlet sine (analytic solvable)
    print("\n--- Demo 1: Dirichlet BC, sine IC (analytic validation) ---")
    r1 = solver.run(L=1.0, T=0.1, nx=4, a1=0.1, a2=0.1,
                     boundary_condition="dirichlet", ic_type="sine")
    u1 = np.array(r1["u"])
    x1 = np.array(r1["x"]); y1 = np.array(r1["y"])
    X1, Y1 = np.meshgrid(x1, y1, indexing="ij")
    analytic1 = (np.sin(np.pi * X1) * np.sin(np.pi * Y1)
                 * np.exp(-0.2 * np.pi**2 * 0.1))
    err1 = np.max(np.abs(u1 - analytic1))
    print(f"  u range: [{u1.min():.4f}, {u1.max():.4f}]")
    print(f"  Max |u - analytic|: {err1:.2e}")

    # Demo 2: Gaussian diffusion
    print("\n--- Demo 2: Gaussian spreading (Dirichlet) ---")
    r2 = solver.run(L=1.0, T=0.05, nx=4, a1=0.05, a2=0.05,
                     boundary_condition="dirichlet", ic_type="gaussian")
    u2 = np.array(r2["u"])
    print(f"  N² = {u2.size}, u∈[{u2.min():.4f}, {u2.max():.4f}]")

    # Demo 3: Periodic BC
    print("\n--- Demo 3: Periodic BC ---")
    r3 = solver.run(L=1.0, T=0.1, nx=4, a1=0.1, a2=0.1,
                     boundary_condition="periodic", ic_type="double_sine")
    u3 = np.array(r3["u"])
    x3 = np.array(r3["x"]); y3 = np.array(r3["y"])
    X3, Y3 = np.meshgrid(x3, y3, indexing="ij")
    analytic3 = (np.sin(2*np.pi*X3) * np.sin(2*np.pi*Y3)
                 * np.exp(-0.2 * 4 * np.pi**2 * 0.1))
    err3 = np.max(np.abs(u3 - analytic3))
    print(f"  Max |u - analytic|: {err3:.2e}")

    # Demo 4: Trotter splitting
    print("\n--- Demo 4: Trotter (quantum simulation path) ---")
    r4 = solver.run(
        L=1.0, T=0.03, nx=3, a1=0.1, a2=0.1,
        boundary_condition="dirichlet", method="trotter", dt=0.005, order=2, ic_type="sine",
    )
    u4 = np.array(r4["u"])
    print(f"  Steps Nt = {r4['grid']['nt']}, dt = {r4['grid']['dt']}")
    print(f"  u∈[{u4.min():.4f}, {u4.max():.4f}]")

    # Demo 5: Classical vs Trotter comparison
    print("\n--- Demo 5: Classical vs Trotter comparison ---")
    r5c = heat2d_solve(L=1.0, T=0.02, nx=3, a1=0.1, a2=0.1,
                         boundary_condition="dirichlet", method="classical", verbose=False)
    r5t = heat2d_solve(L=1.0, T=0.02, nx=3, a1=0.1, a2=0.1,
                         boundary_condition="dirichlet", method="trotter",
                         dt=0.002, order=2, verbose=False)
    diff5 = np.max(np.abs(np.array(r5c["u"]) - np.array(r5t["u"])))
    print(f"  Max |classical - trotter|: {diff5:.2e}")

    # Demo 6: Anisotropic diffusion
    print("\n--- Demo 6: Anisotropic diffusion (a1=0.2, a2=0.05) ---")
    r6 = solver.run(L=1.0, T=0.1, nx=4, a1=0.2, a2=0.05,
                     boundary_condition="dirichlet", ic_type="sine")
    u6 = np.array(r6["u"])
    x6 = np.array(r6["x"]); y6 = np.array(r6["y"])
    X6, Y6 = np.meshgrid(x6, y6, indexing="ij")
    analytic6 = (np.sin(np.pi * X6) * np.sin(np.pi * Y6)
                 * np.exp(-0.25 * np.pi**2 * 0.1))
    err6 = np.max(np.abs(u6 - analytic6))
    print(f"  u∈[{u6.min():.4f}, {u6.max():.4f}], "
          f"analytic err={err6:.2e}")

    # Demo 7: Grid convergence
    print("\n--- Demo 7: Grid convergence (nx=2..5) ---")
    for nv in range(2, 6):
        r7 = heat2d_solve(L=1.0, T=0.05, nx=nv, a1=0.1, a2=0.1,
                           boundary_condition="dirichlet", method="classical", verbose=False)
        u7 = np.array(r7["u"])
        x7 = np.array(r7["x"]); y7 = np.array(r7["y"])
        X7, Y7 = np.meshgrid(x7, y7, indexing="ij")
        analytic7 = (np.sin(np.pi * X7) * np.sin(np.pi * Y7)
                     * np.exp(-0.2 * np.pi**2 * 0.05))
        err7 = np.max(np.abs(u7 - analytic7))
        print(f"  nx={nv} (N={1<<nv}, N²={(1<<nv)**2}): "
              f"u∈[{u7.min():.4f}, {u7.max():.4f}], "
              f"analytic_err={err7:.2e}, time={r7['computation_time']:.4f}s")

    # Tests
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
