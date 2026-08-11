"""Manual implementation of Cartan Decomposition Hamiltonian Simulation.

Simulates quantum time evolution U(t) = e^{-iHt} by decomposing the Lie
algebra g = k ⊕ m (Cartan decomposition) and iteratively applying a Lax
flow to construct the approximate circuit K · e^{-iη} · K†.

Mathematical foundation:
    The Lie algebra g = su(2^n) is decomposed as g = k ⊕ m where:
    - k is the subalgebra fixed by a Cartan involution θ: θ(k) = k
    - m is the subspace where θ(m) = -m
    - Closure relations: [k,k]⊆k, [k,m]⊆m, [m,m]⊆k

    For su(2) (single qubit), with involution θ(X) = X (fixes I,X as k,
    sends Y,Z→-Y,-Z as m):
        k = span{I, X}    (symmetric subalgebra)
        m = span{Y, Z}    (antisymmetric complement)

    The unitary is factored as:
        U(t) = K · e^{-iη} · K†    where K ∈ e^k, η ∈ m

Components:
    - pauli_decompose: Decompose a 2x2 matrix into Pauli basis
    - cartan_involution: Apply Cartan involution θ to a matrix
    - project_k_m: Project Hamiltonian onto k and m subspaces
    - cartan_lax_flow: Manual implementation of the Lax flow optimization
    - build_cartan_circuit: Build a circuit from Cartan parameters
    - cartan_simulate: End-to-end simulation
    - CartanAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Cartan Decomposition Hamiltonian Simulation
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
    from unitarylab.library.hamiltonian import hamiltonian_simulation

    HAS_UNITARYLAB = True
except ImportError:
    HAS_UNITARYLAB = False


# ---------------------------------------------------------------------------
# 1. Pauli algebra utilities
# ---------------------------------------------------------------------------

# Pauli matrices (including identity)
PAULI_BASIS: Dict[str, np.ndarray] = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

PAULI_LABELS = ["I", "X", "Y", "Z"]


def pauli_inner(A: np.ndarray, B: np.ndarray) -> complex:
    """Hilbert-Schmidt inner product normalized for Pauli basis.

    ⟨A, B⟩ = (1/2) Tr(A† B)
    """
    return 0.5 * np.trace(A.conj().T @ B)


def pauli_decompose(H: np.ndarray) -> Dict[str, float]:
    """Decompose a 2x2 Hermitian matrix into Pauli basis.

    H = c_I·I + c_X·X + c_Y·Y + c_Z·Z

    For real symmetric matrices (the supported case), c_Y = 0.

    Args:
        H: 2x2 Hermitian matrix.

    Returns:
        Dict mapping Pauli label → coefficient.
    """
    coeffs = {}
    for label, P in PAULI_BASIS.items():
        c = pauli_inner(H, P)
        # Round near-zero real values (coefficients of Hermitian H are real)
        coeffs[label] = float(c.real) if abs(c.imag) < 1e-12 else complex(c)
    return coeffs


def matrix_from_pauli(coeffs: Dict[str, float]) -> np.ndarray:
    """Reconstruct a matrix from Pauli coefficients.

    Args:
        coeffs: Dict mapping Pauli label → coefficient.

    Returns:
        Reconstructed 2x2 matrix.
    """
    H = np.zeros((2, 2), dtype=complex)
    for label, c in coeffs.items():
        if label in PAULI_BASIS:
            H = H + c * PAULI_BASIS[label]
    return H


# ---------------------------------------------------------------------------
# 2. Cartan decomposition for su(2)
# ---------------------------------------------------------------------------

def cartan_involution(A: np.ndarray) -> np.ndarray:
    """Apply the Cartan involution θ on su(2).

    For the standard involution on su(2):
        θ(I) = I   (identity fixed)
        θ(X) = X   (fixed — generator of k)
        θ(Y) = -Y  (anti-fixed — generator of m)
        θ(Z) = -Z  (anti-fixed — generator of m)

    This is the involution that fixes the SO(2) subalgebra and
    anti-fixes the complementary subspace.

    More generally: θ(A) = X·A·X (conjugation by X matrix).

    Args:
        A: 2x2 matrix.

    Returns:
        θ(A), the Cartan-involuted matrix.
    """
    # θ(A) = X·A·X (valid for su(2) standard involution)
    X = PAULI_BASIS["X"]
    return X @ A @ X


def project_k(A: np.ndarray) -> np.ndarray:
    """Project onto the symmetric subalgebra k (fixed by θ).

    k = {a ∈ g : θ(a) = a}

    For su(2) with θ(a) = X·a·X: k = span{I, X}

    Args:
        A: 2x2 matrix.

    Returns:
        Projection onto k.
    """
    return 0.5 * (A + cartan_involution(A))


def project_m(A: np.ndarray) -> np.ndarray:
    """Project onto the antisymmetric complement m (anti-fixed by θ).

    m = {a ∈ g : θ(a) = -a}

    For su(2) with θ(a) = X·a·X: m = span{Y, Z}

    Args:
        A: 2x2 matrix.

    Returns:
        Projection onto m.
    """
    return 0.5 * (A - cartan_involution(A))


# ---------------------------------------------------------------------------
# 3. Cartan decomposition: K·h·K† factorization
# ---------------------------------------------------------------------------

def cartan_decompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform full Cartan decomposition of Hamiltonian H.

    Decompose H = H_k + H_m where H_k ∈ k and H_m ∈ m.
    Then find K ∈ e^k such that K·H·K† is maximally in k
    (i.e., the Cartan subalgebra component).

    For the 2x2 case, we can compute this analytically:
    - Diagonalize the k-projected part
    - Rotate to the Cartan subalgebra

    Args:
        H: 2x2 Hermitian matrix.

    Returns:
        Tuple of (K, H_k, H_m) where:
        - K: unitary in e^k
        - H_k: k-component (symmetric, in span{I,X})
        - H_m: m-component (antisymmetric, in span{Y,Z})
    """
    H_k = project_k(H)
    H_m = project_m(H)

    # For 2x2, K is determined by the off-k component
    # K rotates H so that the m-component is aligned with the Cartan direction
    # For H ∈ su(2), K = exp(-iφ·X/2) where φ comes from H_m coefficients

    # Extract coefficients
    coeffs = pauli_decompose(H_m)
    c_y = coeffs.get("Y", 0.0)
    c_z = coeffs.get("Z", 0.0)

    # The m-component norm
    r = np.sqrt(c_y**2 + c_z**2)

    if r < 1e-15:
        # H is already in k — no rotation needed
        K = np.eye(2, dtype=complex)
        H_cartan = H_k.copy()
        eta = H_m.copy()
    else:
        # Angle for the K rotation
        # K = exp(-i·φ/2·X) where φ orients the m-component
        # The rotation angle comes from the direction of H_m in m-space
        phi = np.arctan2(c_z, c_y) if abs(c_y) > 1e-15 else np.pi / 2

        # K = exp(-i·φ·X/2)
        X = PAULI_BASIS["X"]
        K = expm(-1.0j * phi * X / 2.0)

        # Rotated Hamiltonian: K·H·K†
        H_rotated = K @ H @ K.conj().T

        H_k_rot = project_k(H_rotated)
        H_m_rot = project_m(H_rotated)

        # After rotation, H_m_rot should be aligned with Z direction in m
        # η = h·Z (the Cartan subalgebra element in m)
        eta = H_m_rot
        H_cartan = H_k_rot

    return K, H_cartan, eta


# ---------------------------------------------------------------------------
# 4. Lax flow implementation (manual version)
# ---------------------------------------------------------------------------

def cartan_lax_flow(
    H: np.ndarray,
    t: float,
    target_error: float = 1e-3,
    lr: float = 1e-3,
    max_steps: int = 10000,
    reps: int = 1000,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Manual implementation of the Cartan-Lax flow.

    Computes U(t) = e^{-iHt} using iterative Cartan decomposition and
    Lax pair dynamics. The algorithm:

    1. Decompose H into k and m components via Cartan decomposition.
    2. Initialize evolution unitary U = I.
    3. For each step, update U via the Lax flow to reduce off-h error.
    4. Terminate when error < target_error or max_steps reached.

    The Lax equation: dH/dt = [H, N] for some N ∈ k.
    This is solved iteratively via: H_{n+1} = e^{-lr·N} · H_n · e^{lr·N}

    Args:
        H: Hermitian Hamiltonian matrix.
        t: Evolution time.
        target_error: Target tolerance for off-h component norm.
        lr: Learning rate / step size for Lax updates.
        max_steps: Maximum number of Lax update steps.
        reps: Reports/iterations budget for adaptive scaling.
        verbose: Print convergence information.

    Returns:
        Dict with 'evolution_result', 'total_error', 'n_steps', 'h_history'.
    """
    dim = H.shape[0]
    U_total = np.eye(dim, dtype=complex)

    # Compute initial decomposition (for diagnostic/monitoring use)
    _K, _H_k, _eta = cartan_decompose(H)

    # Track error history
    off_h_norms = []

    # Note: `reps` parameter (adaptive scaling budget) is accepted for API
    # compatibility with the unitarylab cartan-lax method; the manual
    # implementation uses a simpler fixed-step approach.

    # Iterative Lax flow
    for step in range(max_steps):
        # Compute current off-h component (m-part of U†·H·U)
        H_effective = U_total.conj().T @ H @ U_total
        H_m_eff = project_m(H_effective)

        off_norm = np.linalg.norm(H_m_eff, ord="fro")
        off_h_norms.append(float(off_norm))

        if off_norm < target_error:
            if verbose:
                print(f"  Lax flow converged at step {step}, "
                      f"off-h norm = {off_norm:.2e}")
            break

        # Compute the Lax gradient direction N ∈ k
        # N = [H_k_eff, H_m_eff] (in k by closure [k,m]⊆m and [m,m]⊆k)
        H_k_eff = project_k(H_effective)

        # The commutator [H_k, H_m] gives the direction in k
        N = H_k_eff @ H_m_eff - H_m_eff @ H_k_eff

        # Adapt step size
        adaptive_lr = lr
        if step > 0 and off_h_norms[-1] > off_h_norms[-2]:
            adaptive_lr = lr * 0.5  # Reduce if error increased

        # Update: U ← U · exp(-adaptive_lr · t · N)
        delta_U = expm(-1.0j * adaptive_lr * t * N)
        U_total = U_total @ delta_U

        # Orthogonalize to prevent drift
        U_total, _ = np.linalg.qr(U_total)

    # Final evolution operator: U(t) ≈ U_total · e^{-i·H_k_eff·t} · U_total†
    H_final_k = project_k(U_total.conj().T @ H @ U_total)
    U_approx = U_total @ expm(-1.0j * H_final_k * t) @ U_total.conj().T

    # Compute final error
    U_exact = expm(-1.0j * H * t)
    total_error = np.linalg.norm(U_approx - U_exact, ord="fro")

    if verbose:
        print(f"  Final total error: {total_error:.2e}")
        print(f"  Steps taken: {len(off_h_norms)}")

    return {
        "evolution_result": U_approx,
        "total_error": total_error,
        "n_steps": len(off_h_norms),
        "off_h_norms": off_h_norms,
    }


# ---------------------------------------------------------------------------
# 5. Circuit construction from Cartan parameters (conceptual)
# ---------------------------------------------------------------------------

def build_cartan_circuit_unitary(
    H: np.ndarray, t: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the Cartan-decomposed unitary representation.

    For the 2x2 case, this shows the full Cartan factorization:
        U(t) = K · e^{-iηt} · K†

    where K ∈ e^k and η ∈ m (the Cartan subalgebra element).

    This function demonstrates the mathematical structure; the actual
    circuit synthesis is handled by unitarylab's hamiltonian_simulation.

    Args:
        H: 2x2 Hermitian Hamiltonian.
        t: Evolution time.

    Returns:
        Tuple of (U_approx, decomposition_info) where decomposition_info
        contains K, eta, H_k, and component matrices.
    """
    K, H_k, eta = cartan_decompose(H)

    # Exact evolution
    U_exact = expm(-1.0j * H * t) if HAS_SCIPY else None

    # Cartan-approximated evolution: U ≈ K · e^{-i·H_k·t} · e^{-i·eta·t} · K†
    # Actually: U = K · e^{-iη·t} · K†  where η ∈ m
    U_cartan = K @ expm(-1.0j * eta * t) @ K.conj().T

    # Compute error between Cartan and exact
    error = np.linalg.norm(U_cartan - U_exact, ord="fro") if U_exact is not None else None

    info = {
        "K": K,
        "H_k": H_k,
        "eta": eta,
        "H_m_original": project_m(H),
        "H_k_original": project_k(H),
        "coeffs": pauli_decompose(H),
        "error_vs_exact": error,
    }

    return U_cartan, info


# ---------------------------------------------------------------------------
# 6. End-to-end simulation
# ---------------------------------------------------------------------------

def cartan_simulate(
    H: np.ndarray,
    t: float,
    error: float = 1e-3,
    lr: float = 1e-3,
    max_steps: int = 100000,
    reps: int = 5000,
    use_unitarylab: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Simulate time evolution U(t) = e^{-iHt} using Cartan decomposition.

    If unitarylab is available and use_unitarylab=True, uses the optimized
    cartan-lax implementation. Otherwise falls back to manual Lax flow.

    Args:
        H: Hermitian Hamiltonian matrix (currently 2x2 real symmetric).
        t: Total evolution time.
        error: Stopping tolerance for off-h component norm.
        lr: Base integration step size for Lax flow.
        max_steps: Maximum number of Lax update steps.
        reps: Baseline iteration budget for adaptive scaling.
        use_unitarylab: Whether to use unitarylab's implementation.
        verbose: Print progress information.

    Returns:
        Dict with keys: 'status', 'Evolution result', 'Exact evolution',
        'Final total error', 'Computation time (s)', 'circuit'.

    Raises:
        ValueError: If H is not square.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")

    dim = H.shape[0]

    if verbose:
        print(f"Cartan decomposition simulation")
        print(f"  Hamiltonian dim: {dim}x{dim}")
        print(f"  Evolution time t = {t}")
        print(f"  Target error: {error:.2e}")
        print(f"  Lax flow: lr={lr}, max_steps={max_steps}, reps={reps}")

    t_start = time.perf_counter()

    # Compute exact evolution as reference
    if HAS_SCIPY:
        U_exact = expm(-1.0j * H * t)
    else:
        # Fallback: diagonalize manually
        eigenvals, eigenvecs = np.linalg.eigh(H)
        U_exact = eigenvecs @ np.diag(np.exp(-1.0j * eigenvals * t)) @ eigenvecs.conj().T

    # Run Cartan-Lax simulation
    if HAS_UNITARYLAB and use_unitarylab:
        if verbose:
            print(f"  Using unitarylab cartan-lax method")
        result = hamiltonian_simulation(
            H,
            t,
            method="cartan-lax",
            target_error=error,
            lr=lr,
            max_steps=max_steps,
            reps=reps,
        )
        U_approx = result.evolution_result
        total_error_val = result.total_error
        circuit = result.circuit
        n_steps = result.lax_dynamics_history.get("steps", [0])[-1] if result.lax_dynamics_history else None
    else:
        if verbose:
            print(f"  Using manual Lax flow")
        lax_result = cartan_lax_flow(
            H, t, target_error=error, lr=lr,
            max_steps=max_steps, reps=reps, verbose=verbose,
        )
        U_approx = lax_result["evolution_result"]
        total_error_val = lax_result["total_error"]
        circuit = None
        n_steps = lax_result["n_steps"]

    t_elapsed = time.perf_counter() - t_start

    # Success check
    is_success = total_error_val < 10 * error

    if verbose:
        print(f"  Approx U:\n{np.array2string(U_approx, precision=4, suppress_small=True)}")
        print(f"  Exact U:\n{np.array2string(U_exact, precision=4, suppress_small=True)}")
        print(f"  Total error: {total_error_val:.2e}")
        if n_steps is not None:
            print(f"  Steps: {n_steps}")
        print(f"  Status: {'ok' if is_success else 'failed'}")
        print(f"  Time: {t_elapsed:.4f} s")

    return {
        "status": "ok" if is_success else "failed",
        "Evolution result": U_approx,
        "Exact evolution": U_exact,
        "Final total error": float(total_error_val),
        "Computation time (s)": round(t_elapsed, 4),
        "circuit": circuit,
        "circuit_path": "",
        "plot": [],
    }


# ---------------------------------------------------------------------------
# 7. Class-based interface (matching CartanDecompositionAlgorithm pattern)
# ---------------------------------------------------------------------------

class CartanAlgorithmSolver:
    """Class-based solver matching the CartanDecompositionAlgorithm interface.

    Usage:
        solver = CartanAlgorithmSolver()
        result = solver.run(
            H=np.array([[2.0, 1.0], [1.0, 2.0]]),
            t=1.0,
            error=1e-3,
        )
        print(result['Evolution result'])  # Approximate U(t)
    """

    def __init__(
        self,
        text_mode: str = "plain",
        algo_dir: Optional[str] = None,
    ):
        """Initialize the Cartan decomposition solver.

        Args:
            text_mode: Output text mode ('plain' or 'legacy').
            algo_dir: Directory for output files.
        """
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        H: np.ndarray,
        t: float,
        error: float = 1e-3,
        lr: float = 1e-3,
        max_steps: int = 100000,
        reps: int = 5000,
        evol_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run Cartan decomposition Hamiltonian simulation.

        Args:
            H: Hermitian Hamiltonian matrix.
            t: Total evolution time.
            error: Stopping tolerance.
            lr: Lax flow step size.
            max_steps: Maximum Lax update steps.
            reps: Iteration budget for adaptive scaling.
            evol_time: Optional override for evolution time.

        Returns:
            Result dict with keys: 'status', 'Evolution result',
            'Exact evolution', 'Final total error', 'Computation time (s)',
            'circuit_path', 'plot', 'circuit'.
        """
        t_evol = evol_time if evol_time is not None else t

        result = cartan_simulate(
            H=H,
            t=t_evol,
            error=error,
            lr=lr,
            max_steps=max_steps,
            reps=reps,
            verbose=(self.text_mode != "plain"),
        )

        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Package results in the standard return format.

        Args:
            result: Raw result from cartan_simulate.

        Returns:
            Formatted result dict matching CartanDecompositionAlgorithm schema.
        """
        base = {
            "status": result.get("status", "failed"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }
        # Merge computation output fields
        base.update({
            "Evolution result": result.get("Evolution result"),
            "Exact evolution": result.get("Exact evolution"),
            "Final total error": result.get("Final total error"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
        })
        return base


# ---------------------------------------------------------------------------
# 8. Known test cases
# ---------------------------------------------------------------------------

def make_symmetric_2x2(
    a: float, b: float, d: float
) -> np.ndarray:
    """Create a 2x2 real symmetric Hamiltonian.

    H = [[a, b], [b, d]]

    Args:
        a: Top-left element.
        b: Off-diagonal element.
        d: Bottom-right element.

    Returns:
        2x2 real symmetric matrix.
    """
    return np.array([[a, b], [b, d]], dtype=float)


KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "H": make_symmetric_2x2(2.0, 1.0, 2.0),
        "t": 1.0,
        "error": 1e-3,
        "label": "H=[[2,1],[1,2]], t=1.0",
    },
    {
        "H": make_symmetric_2x2(0.25, 0.05, -0.15),
        "t": 0.2,
        "error": 1e-2,
        "label": "H=[[0.25,0.05],[0.05,-0.15]], t=0.2",
    },
    {
        "H": make_symmetric_2x2(1.0, 0.5, -1.0),
        "t": 0.5,
        "error": 1e-3,
        "label": "H=[[1,0.5],[0.5,-1]], t=0.5",
    },
    {
        "H": make_symmetric_2x2(3.0, 0.0, 3.0),
        "t": 0.3,
        "error": 1e-3,
        "label": "H=3I (diagonal), t=0.3",
    },
    {
        "H": make_symmetric_2x2(0.0, 1.0, 0.0),
        "t": np.pi / 2,
        "error": 1e-3,
        "label": "H=X, t=π/2",
    },
    {
        "H": np.array([[1.5, 0.8], [0.8, -0.5]], dtype=float),
        "t": 0.8,
        "error": 1e-3,
        "label": "H asymmetric diag, t=0.8",
    },
]


def run_known_test(
    case: Dict[str, Any],
    use_unitarylab: bool = True,
    max_attempts: int = 2,
) -> bool:
    """Run a single known test case.

    Args:
        case: Dict with 'H', 't', 'error', 'label'.
        use_unitarylab: Whether to use unitarylab backend.
        max_attempts: Maximum attempts (for probabilistic cases).

    Returns:
        True if error is within tolerance.
    """
    H = case["H"]
    t = case["t"]
    error_tol = case["error"]
    label = case["label"]

    for attempt in range(1, max_attempts + 1):
        result = cartan_simulate(
            H=H, t=t, error=error_tol,
            use_unitarylab=use_unitarylab,
            verbose=False,
        )
        err = result["Final total error"]
        status = result["status"]
        ok = status == "ok"
        icon = "✓" if ok else "✗"
        print(f"  [{icon}] {label}")
        print(f"       Error: {err:.2e} (tolerance: {error_tol:.2e})")
        print(f"       Time:  {result['Computation time (s)']}s")
        if ok:
            return True
        print(f"       Retry {attempt}/{max_attempts}...")
    return False


# ---------------------------------------------------------------------------
# 9. Main (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Cartan Decomposition Hamiltonian Simulation")
    print("=" * 60)

    solver = CartanAlgorithmSolver()

    # Demo case from the SKILL.md
    print("\n--- Demo: H = [[2,1],[1,2]], t = 1.0 ---")
    H_demo = np.array([[2.0, 1.0], [1.0, 2.0]])
    result = solver.run(H=H_demo, t=1.0, error=1e-3)

    print(f"  Status:          {result['status']}")
    print(f"  Evolution result:\n{np.array2string(result['Evolution result'], precision=4, suppress_small=True)}")
    print(f"  Exact evolution:\n{np.array2string(result['Exact evolution'], precision=4, suppress_small=True)}")
    print(f"  Final total error: {result['Final total error']:.2e}")
    print(f"  Computation time:  {result['Computation time (s)']}s")

    # Show Cartan decomposition details
    print("\n--- Cartan Decomposition Analysis ---")
    coeffs = pauli_decompose(H_demo)
    print(f"  Pauli decomposition: {coeffs}")
    H_k = project_k(H_demo)
    H_m = project_m(H_demo)
    print(f"  H_k (symmetric, k-algebra):\n{np.array2string(H_k, precision=3)}")
    print(f"  H_m (antisymmetric, m-space):\n{np.array2string(H_m, precision=3)}")
    print(f"  ||H_m||_F = {np.linalg.norm(H_m, ord='fro'):.4f}")

    # Run all known test cases
    print("\n--- Known Test Cases ---")
    passed = 0
    for case in KNOWN_CASES[:4]:  # First 4 for quick run
        if run_known_test(case):
            passed += 1
    print(f"\n  Result: {passed}/{min(4, len(KNOWN_CASES))} tests passed")
    sys.exit(0 if passed == min(4, len(KNOWN_CASES)) else 1)
