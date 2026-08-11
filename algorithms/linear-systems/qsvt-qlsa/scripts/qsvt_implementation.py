"""QSVT-based Linear System Solver (QLSA).

Solves Ax = b by transforming matrix inversion into a singular-value
transformation: if A can be block-encoded into a unitary, QSVT applies
a polynomial approximation of f(x) = 1/x to the singular values of A,
producing a quantum state proportional to A^{-1}|b⟩.

Mathematical foundation:
    1. SVD: A = U Σ V† with singular values σ_j ≥ 0.
    2. Target polynomial: p(σ) ≈ 1/σ on [σ_min, σ_max].
    3. Block encoding: ⟨0|U_A|0⟩ = A/α (normalized).
    4. QSVT applies p(σ) via interleaved phase rotations.
    5. Result state: |x⟩ ∝ A^{-1}|b⟩.

Components:
    - qsvt_solve: End-to-end solver using unitarylab's QSVTSolver
    - qsvt_manual_solve: Educational reference using classical SVD
    - QSVTLinearSolverAlgorithm: Class-based interface

Reference:
    SKILL.md — QSVT QLSA
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from unitarylab.library.linear_solver import QSVTSolver

    HAS_QSVT = True
except ImportError:
    HAS_QSVT = False


# ---------------------------------------------------------------------------
# 1. Manual QSVT-like solution (educational SVD-based)
# ---------------------------------------------------------------------------

def qsvt_manual_solve(
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Solve Ax = b using SVD-based reciprocal singular-value transformation.

    This is the classical analog of what QSVT achieves quantumly:
        1. Compute SVD: A = U Σ V†.
        2. For each singular value σ_j, apply f(σ_j) = 1/σ_j (with cutoff).
        3. Reconstruct: x = V Σ^{-1} U† b.

    The quantum QSVT algorithm achieves this via block encoding and
    Chebyshev polynomial approximation of f(x) = 1/x, avoiding the
    explicit O(N³) classical SVD cost for large systems.

    Args:
        A: Square matrix (should be Hermitian for QSVT block encoding).
        b: Right-hand side vector.
        epsilon: Accuracy parameter (controls singular-value cutoff).

    Returns:
        Dict with 'solution', 'scaling_factor', 'singular_values',
        'condition_number', 'truncated_singular_values'.
    """
    # SVD: A = U Σ V†
    U, s, Vh = np.linalg.svd(A)

    # Condition number
    kappa = float(s[0] / s[-1]) if s[-1] > 1e-15 else float("inf")

    # Reciprocal singular values with cutoff
    s_inv = np.zeros_like(s)
    cutoff = epsilon * s[0]
    truncated = 0
    for j, sigma in enumerate(s):
        if sigma > cutoff:
            s_inv[j] = 1.0 / sigma
        else:
            truncated += 1

    # x = V Σ^{-1} U† b = Vh† @ diag(s_inv) @ U† @ b
    x = Vh.conj().T @ (s_inv * (U.conj().T @ b))

    # Scaling factor (1-norm of the inverse singular values)
    scaling = float(np.sum(s_inv))

    return {
        "solution": x,
        "scaling_factor": scaling,
        "singular_values": s,
        "condition_number": kappa,
        "truncated_singular_values": truncated,
    }


# ---------------------------------------------------------------------------
# 2. End-to-end QSVT solver
# ---------------------------------------------------------------------------

def qsvt_solve(
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
    algo_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Solve Ax = b using QSVT-based quantum linear solver.

    Delegates the heavy lifting to unitarylab's QSVTSolver, which:
        1. Block-encodes A.
        2. Designs phase factors for the 1/x polynomial.
        3. Applies QSVT to transform singular values.
        4. Returns (circuit, solution, scaling_factor).

    Requires unitarylab's QSVTSolver; no classical fallback is reported as a
    successful QSVT execution.

    Args:
        A: Coefficient matrix.
        b: Right-hand side vector.
        epsilon: Target accuracy.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress.

    Returns:
        Dict with: 'status', 'Solution vector', 'Scaling factor applied',
        'Simulation time (s)', 'circuit', 'circuit_path', 'plot',
        'condition_number', 'singular_values'.
    """
    if verbose:
        print(f"QSVT Linear Solver")
        print(f"  A: {A.shape[0]}x{A.shape[0]}, epsilon={epsilon:.2e}")

    t_start = time.perf_counter()

    # Validate
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"b length {b.shape[0]} != A dim {A.shape[0]}")

    # Classical check: condition number
    s = np.linalg.svd(A, compute_uv=False)
    kappa = float(s[0] / max(s[-1], 1e-15))

    if verbose:
        print(f"  Condition number κ = {kappa:.2f}")

    if not HAS_QSVT:
        raise ImportError("unitarylab QSVTSolver is required for qsvt_solve")
    if verbose:
        print(f"  Running QSVTSolver...")
    circuit, x_qsvt, scaling = QSVTSolver(
        A, b, epsilon, backend=backend, device=device, dtype=dtype,
    )

    t_elapsed = time.perf_counter() - t_start

    # Classical reference
    x_classical = np.linalg.solve(A, b)
    l2_error = float(np.linalg.norm(x_qsvt - x_classical))

    is_success = np.isfinite(l2_error)

    if verbose:
        print(f"\n  Results:")
        print(f"  QSVT solution:     {np.array2string(np.asarray(x_qsvt).real, precision=4)}")
        print(f"  Classical solution: {np.array2string(x_classical.real, precision=4)}")
        print(f"  L2 error:          {l2_error:.6e}")
        print(f"  Scaling factor:    {scaling:.6f}")
        print(f"  Status:            {'ok' if is_success else 'failed'}")
        print(f"  Time:              {t_elapsed:.4f}s")

    if algo_dir is None:
        algo_dir = os.path.join(
            os.getcwd(), "results", "linear-systems", "qsvt-qlsa"
        )
    os.makedirs(algo_dir, exist_ok=True)
    circuit_path = os.path.abspath(
        os.path.join(algo_dir, "qsvt_linear_solver_circuit.svg")
    )
    circuit.draw(filename=circuit_path, title="QSVT Linear Solver Circuit")

    return {
        "status": "ok" if is_success else "failed",
        "Solution vector": np.asarray(x_qsvt),
        "Scaling factor applied": float(scaling),
        "Simulation time (s)": round(t_elapsed, 4),
        "circuit": circuit,
        "circuit_path": circuit_path,
        "plot": [],
        "condition_number": kappa,
        "singular_values": s,
        "L2 error": l2_error,
        "epsilon": epsilon,
    }


# ---------------------------------------------------------------------------
# 3. Class-based interface
# ---------------------------------------------------------------------------

class QSVTLinearSolverAlgorithm:
    """Class-based solver matching QSVTLinearSolverAlgorithm interface.

    Usage:
        solver = QSVTLinearSolverAlgorithm(text_mode='plain')
        result = solver.run(
            A=np.array([[0.8, 0.0], [0.0, 0.4]]),
            b=np.array([1.0, 2.0]),
            epsilon=0.0001,
        )
        print(result['Solution vector'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir or os.path.join(
            os.getcwd(), "results", "linear-systems", "qsvt-qlsa"
        )
        os.makedirs(self.algo_dir, exist_ok=True)
        self.output: Dict[str, Any] = {}

    def run(
        self,
        A: np.ndarray,
        b: np.ndarray,
        epsilon: float,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        result = qsvt_solve(
            A=A, b=b, epsilon=epsilon,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"), algo_dir=self.algo_dir,
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Solution vector": result.get("Solution vector"),
            "Scaling factor applied": result.get("Scaling factor applied"),
            "Simulation time (s)": result.get("Simulation time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 4. Test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"A": np.array([[0.8, 0.0], [0.0, 0.4]], dtype=float), "b": np.array([1.0, 2.0]), "epsilon": 1e-4, "label": "diag(0.8,0.4), b=[1,2]"},
    {"A": np.array([[1.5, 0.5], [0.5, 1.5]], dtype=float), "b": np.array([1.0, 0.0]), "epsilon": 1e-3, "label": "symmetric 2x2"},
    {"A": np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float), "b": np.array([1.0, 1.0]), "epsilon": 1e-4, "label": "diag(2,1), b=[1,1]"},
    {"A": np.array([[3.0, 0.0], [0.0, 0.5]], dtype=float), "b": np.array([0.0, 1.0]), "epsilon": 1e-3, "label": "diag(3,0.5), κ=6"},
    {"A": np.eye(4, dtype=float) + 0.3 * np.eye(4)[::-1], "b": np.array([1.0, 0.5, 0.3, 0.1]), "epsilon": 1e-3, "label": "4x4 sparse"},
    {"A": 0.5 * np.array([[3.0, 1.0], [1.0, 3.0]], dtype=float), "b": np.array([1.0, 2.0]), "epsilon": 1e-4, "label": "scaled symmetric"},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    A, b, eps, label = case["A"], case["b"], case["epsilon"], case["label"]
    result = qsvt_solve(A=A, b=b, epsilon=eps, verbose=False)
    err = result["L2 error"]
    ok = result["status"] == "ok" and np.isfinite(err) and err < 1.0
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] {label}")
    print(f"       L2 error: {err:.6e}  κ={result['condition_number']:.2f}  time={result['Simulation time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("QSVT Linear Solver (QLSA) — Implementation")
    print("=" * 60)

    solver = QSVTLinearSolverAlgorithm()

    print("\n--- Demo: A=diag(0.8,0.4), b=[1,2], ε=1e-4 ---")
    A_demo = np.array([[0.8, 0.0], [0.0, 0.4]], dtype=float)
    b_demo = np.array([1.0, 2.0])
    result = solver.run(A=A_demo, b=b_demo, epsilon=1e-4)
    print(f"  Solution:     {np.array2string(np.asarray(result['Solution vector']).real, precision=4)}")
    print(f"  Scaling:      {result['Scaling factor applied']:.4f}")
    print(f"  Status:       {result['status']}")

    # Compare SVD approach
    print("\n--- SVD-Based Solution (for comparison) ---")
    manual = qsvt_manual_solve(A_demo, b_demo, 1e-4)
    print(f"  Singular values: {np.round(manual['singular_values'], 4)}")
    print(f"  SVD solution:    {np.round(manual['solution'].real, 4)}")
    print(f"  κ(A) = {manual['condition_number']:.2f}")

    # Epsilon study
    print("\n--- Epsilon Study ---")
    for eps in [1e-2, 1e-3, 1e-4, 1e-6]:
        r = qsvt_solve(A=A_demo, b=b_demo, epsilon=eps, verbose=False)
        print(f"  ε={eps:.0e}: L2 error={r['L2 error']:.6e}")

    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    n = len(KNOWN_CASES)
    print(f"\n  Result: {passed}/{n} tests passed")
    sys.exit(0 if passed == n else 1)
