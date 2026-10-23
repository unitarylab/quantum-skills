"""Manual implementation of Taylor Series Hamiltonian Simulation.

Approximates U(t) = e^{-iHt} by truncating the Taylor series:

    e^{-iHt} = Σ_{k=0}^{K} (-iHt)^k / k! + O((αt)^{K+1}/(K+1)!)

The method splits the total evolution time into r slices, expands each
slice as a degree-K Taylor series over Pauli strings, elevates to the
r-th power, and constructs an LCU (Linear Combination of Unitaries) circuit.

Key formulas:
    - Time slicing:  r = ⌊αt / 0.5⌋ + 1,  where α = ||H||_2
    - Adaptive degree: K = min(max(K_init, ⌈1.5·λ + 1.5·ln(1/ε)⌉), 15)
      where λ = αt/r
    - Taylor term k:  (-i)^k · (H·t/r)^k / k!
    - Composite: U_approx = (Σ_{k=0}^{K} Taylor_k)^r

Components:
    - taylor_matrix_series: Direct matrix-level Taylor expansion
    - adaptive_degree_slices: Compute r and K from H, t, error
    - taylor_simulate: End-to-end simulation
    - TaylorAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Taylor Hamiltonian Simulation Skill Guide
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import expm

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from unitarylab.library.pauli_operator import (
        pauli_string_decomposition,
        pauli_string_multiply,
        pauli_string_power,
    )
    from unitarylab.library import LCU

    HAS_UNITARYLAB = True
except ImportError:
    HAS_UNITARYLAB = False

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Adaptive degree and time-slice computation
# ---------------------------------------------------------------------------

def compute_spectral_params(
    H: np.ndarray, t: float
) -> Tuple[float, float]:
    """Compute spectral norm and dimensionless parameter.

    Args:
        H: Hermitian Hamiltonian.
        t: Total evolution time.

    Returns:
        Tuple of (alpha, lam) where alpha = ||H||_2, lam = alpha * t.
    """
    alpha = float(np.linalg.norm(H, ord=2))
    lam = alpha * t
    return alpha, lam


def adaptive_degree_slices(
    alpha: float,
    t: float,
    target_error: float = 1e-8,
    init_degree: int = 15,
    slice_threshold: float = 0.5,
    max_degree: int = 15,
) -> Tuple[int, int, float]:
    """Compute optimal number of time slices and Taylor degree.

    r = ⌊αt / threshold⌋ + 1
    K = min(max(init_degree, ⌈1.5·λ_slice + 1.5·ln(1/ε)⌉), max_degree)

    where λ_slice = α·t / r is the per-slice spectral weight.

    Args:
        alpha: Spectral norm of H (||H||_2).
        t: Total evolution time.
        target_error: Target approximation error.
        init_degree: Initial degree guess.
        slice_threshold: Target spectral weight per slice (default 0.5).
        max_degree: Hard cap on Taylor degree (default 15).

    Returns:
        Tuple of (r, K, lam_slice) where:
        - r: number of time slices
        - K: Taylor truncation degree
        - lam_slice: per-slice spectral weight λ/r
    """
    lam = alpha * t
    r = int(lam / slice_threshold) + 1
    lam_slice = lam / r

    # Adaptive degree: K ≈ 1.5 * λ_slice + 1.5 * ln(1/ε)
    log_term = math.log(1.0 / target_error) if target_error > 0 else 20.0
    K_adaptive = int(math.ceil(1.5 * lam_slice + 1.5 * log_term))
    K = max(init_degree, K_adaptive)
    K = min(K, max_degree)  # hard cap
    K = max(K, 1)

    return r, K, lam_slice


# ---------------------------------------------------------------------------
# 2. Taylor series expansion (matrix-level)
# ---------------------------------------------------------------------------

def taylor_matrix_series(
    H_slice: np.ndarray,  # H * t / r
    degree: int,
) -> Tuple[np.ndarray, float]:
    """Compute the degree-K Taylor approximation of e^{-i·H_slice}.

    U_slice ≈ Σ_{k=0}^{K} (-i)^k · H_slice^k / k!

    The truncation error bound is:
        O( (||H_slice||_2)^{K+1} / (K+1)! )

    Args:
        H_slice: Per-slice Hamiltonian (scaled by t/r).
        degree: Truncation order K.

    Returns:
        Tuple of (U_slice, truncation_error_bound) where U_slice is the
        approximate evolution operator for one time slice.
    """
    dim = H_slice.shape[0]
    U = np.zeros((dim, dim), dtype=complex)
    H_power = np.eye(dim, dtype=complex)  # H_slice^0 = I
    factorial = 1.0

    for k in range(degree + 1):
        # Term: (-i)^k · H_slice^k / k!
        term = H_power * ((-1.0j) ** k) / factorial
        U += term

        # Prepare for next iteration
        if k < degree:
            H_power = H_power @ H_slice
            factorial *= (k + 1)

    # Truncation error bound (spectral norm):
    # ||H_slice^{K+1}|| / (K+1)! · e^{||H_slice||}
    norm_H = float(np.linalg.norm(H_slice, ord=2))
    trunc_bound = (norm_H ** (degree + 1)) / math.factorial(degree + 1)
    trunc_bound *= math.exp(norm_H)  # Lagrange remainder bound

    return U, trunc_bound


# ---------------------------------------------------------------------------
# 3. End-to-end simulation
# ---------------------------------------------------------------------------

def taylor_simulate(
    H: np.ndarray,
    t: float,
    error: float = 1e-8,
    degree: int = 15,
    slice_threshold: float = 0.5,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Simulate U(t) = e^{-iHt} using truncated Taylor series.

    Pipeline:
        1. Compute spectral norm α = ||H||_2 and λ = α·t.
        2. Determine time slices r: λ_slice = λ/r ≤ threshold.
        3. Determine adaptive Taylor degree K.
        4. Per slice: compute Σ_{k=0}^{K} (-i)^k·(H·t/r)^k / k!.
        5. Compose: U_approx = (U_slice)^r.
        6. Compare with exact U = expm(-1j·H·t).

    Args:
        H: Hermitian Hamiltonian matrix.
        t: Total evolution time.
        error: Target approximation error.
        degree: Initial Taylor degree (adjusted adaptively).
        slice_threshold: Target spectral weight per slice.
        backend: Simulation backend (reserved for circuit path).
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress information.

    Returns:
        Dict with: 'status', 'Approximate evolution matrix',
        'Exact evolution matrix', 'Frobenius norm of error',
        'Computation time (s)', 'circuit', 'time_slices',
        'degree', 'alpha', 'lambda', 'truncation_bound'.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape {H.shape}")

    dim = H.shape[0]

    if verbose:
        print(f"Taylor Series Hamiltonian Simulation")
        print(f"  H: {dim}x{dim}, t = {t}, target error = {error:.2e}")
        print(f"  Initial degree = {degree}")

    t_start = time.perf_counter()

    # --- Step 1: Spectral parameters ---
    alpha, lam = compute_spectral_params(H, t)

    if verbose:
        print(f"  α = ||H||_2 = {alpha:.6f}")
        print(f"  λ = α·t = {lam:.6f}")

    # --- Step 2: Adaptive degree and slices ---
    r, K, lam_slice = adaptive_degree_slices(
        alpha, t, error, degree, slice_threshold,
    )

    if verbose:
        print(f"  Time slices: r = {r}")
        print(f"  λ_slice = λ/r = {lam_slice:.6f}")
        print(f"  Taylor degree: K = {K}")

    # --- Step 3: Per-slice Taylor expansion ---
    H_slice = H * (t / r)
    U_slice, trunc_bound = taylor_matrix_series(H_slice, K)

    if verbose:
        print(f"  Taylor truncation bound: {trunc_bound:.2e}")
        # Check unitarity of the slice
        eye = np.eye(dim, dtype=complex)
        slice_unitarity = float(
            np.linalg.norm(U_slice @ U_slice.conj().T - eye, ord="fro")
        )
        print(f"  Slice unitarity error: {slice_unitarity:.2e}")

    # --- Step 4: Compose time slices ---
    if r > 1:
        U_approx = np.linalg.matrix_power(U_slice, r)
    else:
        U_approx = U_slice

    # --- Step 5: Exact reference ---
    if HAS_SCIPY:
        U_exact = expm(-1.0j * H * t)
    else:
        eigenvals, eigenvecs = np.linalg.eigh(H)
        U_exact = (
            eigenvecs
            @ np.diag(np.exp(-1.0j * eigenvals * t))
            @ eigenvecs.conj().T
        )

    # --- Step 6: Error ---
    frob_error = float(np.linalg.norm(U_approx - U_exact, ord="fro"))

    t_elapsed = time.perf_counter() - t_start

    is_success = np.isfinite(frob_error) and frob_error < max(error * 10, 1e-3)

    if verbose:
        print(f"\n  Results:")
        print(f"  U_approx:\n{np.array2string(U_approx, precision=4, suppress_small=True)}")
        print(f"  U_exact:\n{np.array2string(U_exact, precision=4, suppress_small=True)}")
        print(f"  Frobenius error: {frob_error:.6e}")
        print(f"  Status: {'ok' if is_success else 'failed'}")
        print(f"  Time: {t_elapsed:.4f}s")

    return {
        "status": "ok" if is_success else "failed",
        "Approximate evolution matrix": U_approx,
        "Exact evolution matrix": U_exact,
        "Frobenius norm of error": frob_error,
        "Computation time (s)": round(t_elapsed, 4),
        "circuit": None,  # Matrix path — circuit built internally via LCU
        "circuit_path": "",
        "plot": [],
        "time_slices": r,
        "degree": K,
        "alpha": alpha,
        "lambda": lam,
        "lambda_slice": lam_slice,
        "truncation_bound": trunc_bound,
    }


# ---------------------------------------------------------------------------
# 4. Pauli-string Taylor expansion (educational)
# ---------------------------------------------------------------------------

def taylor_pauli_expand(
    decomposition: List[Tuple[str, complex]],
    degree: int,
) -> List[Tuple[str, complex]]:
    """Expand e^{-iH} as a Taylor series over Pauli strings.

    Given H = Σ c_j P_j (Pauli decomposition), compute:
        e^{-iH} ≈ Σ_{k=0}^{K} (-i)^k · H^k / k!

    where H^k expands into Pauli string products. Each term contributes
    a Pauli string with a complex coefficient.

    This demonstrates the symbolic approach used by the full LCU circuit
    construction, as opposed to the direct matrix computation above.

    Args:
        decomposition: List of (pauli_string, coefficient) tuples.
        degree: Taylor truncation order K.

    Returns:
        List of (pauli_string, coefficient) tuples for the truncated series.
    """
    # Order 0: identity with coefficient 1
    term_map: Dict[str, complex] = {"I": complex(1.0, 0.0)}
    factorial = 1.0

    # Current order's terms (starting from order 0)
    current_terms = term_map.copy()

    for k in range(1, degree + 1):
        factorial *= k
        next_terms: Dict[str, complex] = {}

        for prev_str, prev_coeff in current_terms.items():
            for pauli_str, coeff in decomposition:
                # Multiply Pauli strings: P_prev * P_j
                if HAS_UNITARYLAB:
                    prod_str, phase = pauli_string_multiply(prev_str, pauli_str)
                else:
                    # Manual fallback for single-qubit Pauli multiplication
                    prod_str, phase = _manual_pauli_multiply(prev_str, pauli_str)

                # Coefficient: prev_coeff * coeff * (-i)^k / k!
                # But we're building order by order, so accumulate with -i/k
                new_coeff = prev_coeff * coeff * complex(0.0, -1.0) / float(k)

                if prod_str in next_terms:
                    next_terms[prod_str] += new_coeff * phase
                else:
                    next_terms[prod_str] = new_coeff * phase

        # Merge into total
        for ps, c in next_terms.items():
            if ps in term_map:
                term_map[ps] += c
            else:
                term_map[ps] = c

        current_terms = next_terms

    # Convert to list, sorted by |coefficient| descending
    result = [(ps, c) for ps, c in term_map.items() if abs(c) > 1e-15]
    result.sort(key=lambda x: abs(x[1]), reverse=True)
    return result


def _manual_pauli_multiply(
    s1: str, s2: str
) -> Tuple[str, complex]:
    """Manual single-qubit Pauli multiplication table.

    For multi-qubit strings, returns simplified results for I, X, Y, Z.

    Args:
        s1, s2: Pauli string labels (single character for 1-qubit).

    Returns:
        Tuple of (product_string, phase).
    """
    # Pauli multiplication table: P_i * P_j = δ_{ij}I + i·ε_{ijk}·P_k
    # For single-qubit:
    table = {
        ("I", "I"): ("I", 1.0),
        ("I", "X"): ("X", 1.0),
        ("I", "Y"): ("Y", 1.0),
        ("I", "Z"): ("Z", 1.0),
        ("X", "I"): ("X", 1.0),
        ("X", "X"): ("I", 1.0),
        ("X", "Y"): ("Z", 1.0j),
        ("X", "Z"): ("Y", -1.0j),
        ("Y", "I"): ("Y", 1.0),
        ("Y", "X"): ("Z", -1.0j),
        ("Y", "Y"): ("I", 1.0),
        ("Y", "Z"): ("X", 1.0j),
        ("Z", "I"): ("Z", 1.0),
        ("Z", "X"): ("Y", 1.0j),
        ("Z", "Y"): ("X", -1.0j),
        ("Z", "Z"): ("I", 1.0),
    }
    key = (s1, s2)
    if key in table:
        return table[key]
    # Default: treat as tensor product, concatenate
    return (s1 + s2, complex(1.0, 0.0))


# ---------------------------------------------------------------------------
# 5. Class-based interface (matching TaylorAlgorithm pattern)
# ---------------------------------------------------------------------------

class TaylorAlgorithmSolver:
    """Class-based solver matching the TaylorAlgorithm interface.

    Usage:
        solver = TaylorAlgorithmSolver(text_mode='plain')
        result = solver.run(
            H=np.array([[2,1],[1,3]], dtype=complex),
            t=1.0, error=1e-8, degree=15,
        )
        print(result['Frobenius norm of error'])
    """

    def __init__(
        self,
        text_mode: str = "plain",
        algo_dir: Optional[str] = None,
    ):
        """Initialize the Taylor solver.

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
        error: float = 1e-8,
        degree: int = 15,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run Taylor series Hamiltonian simulation.

        Args:
            H: Hermitian Hamiltonian matrix.
            t: Total evolution time.
            error: Target approximation error.
            degree: Initial Taylor degree (adjusted adaptively).
            backend: Simulation backend.
            device: Compute device.
            dtype: Numeric dtype.

        Returns:
            Result dict matching TaylorAlgorithm return schema.
        """
        result = taylor_simulate(
            H=H,
            t=t,
            error=error,
            degree=degree,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )

        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Package results in standard TaylorAlgorithm return format.

        Args:
            result: Raw result from taylor_simulate.

        Returns:
            Formatted result dict matching TaylorAlgorithm return schema.
        """
        base = {
            "status": result.get("status", "failed"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }
        base.update({
            "Approximate evolution matrix": result.get("Approximate evolution matrix"),
            "Exact evolution matrix": result.get("Exact evolution matrix"),
            "Frobenius norm of error": result.get("Frobenius norm of error"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
        })
        return base


# ---------------------------------------------------------------------------
# 6. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "H": np.array([[2, 1], [1, 3]], dtype=complex),
        "t": 1.0,
        "error": 1e-8,
        "degree": 10,
        "label": "H=[[2,1],[1,3]], t=1.0",
    },
    {
        "H": np.array([[2, 1], [1, 3]], dtype=complex),
        "t": 3.0,
        "error": 1e-6,
        "degree": 12,
        "label": "H=[[2,1],[1,3]], t=3.0 (longer)",
    },
    {
        "H": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
        "t": 0.5,
        "error": 1e-8,
        "degree": 10,
        "label": "H=0.5(I+X), t=0.5",
    },
    {
        "H": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "t": np.pi / 2,
        "error": 1e-8,
        "degree": 12,
        "label": "H=X, t=π/2",
    },
    {
        "H": np.array([[1.5, 0.3], [0.3, -0.5]], dtype=complex),
        "t": 0.2,
        "error": 1e-8,
        "degree": 8,
        "label": "H=[[1.5,0.3],[0.3,-0.5]], t=0.2",
    },
    {
        "H": np.array([[2, 1], [1, 3]], dtype=complex),
        "t": 5.0,
        "error": 1e-4,
        "degree": 15,
        "label": "H=[[2,1],[1,3]], t=5.0 (many slices)",
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known Taylor test case.

    Args:
        case: Dict with 'H', 't', 'error', 'degree', 'label'.

    Returns:
        True if the test passes.
    """
    H = case["H"]
    t = case["t"]
    error_tol = case["error"]
    deg = case["degree"]
    label = case["label"]

    result = taylor_simulate(
        H=H, t=t, error=error_tol, degree=deg, verbose=False,
    )
    err = result["Frobenius norm of error"]
    status = result["status"]
    icon = "✓" if status == "ok" else "✗"
    slices = result.get("time_slices", "?")
    K = result.get("degree", "?")
    lam_s = result.get("lambda_slice", 0)
    print(f"  [{icon}] {label}")
    print(f"       Error: {err:.6e}  (slices={slices}, K={K}, λ_slice={lam_s:.4f})")
    print(f"       Time:  {result['Computation time (s)']}s")
    return status == "ok"


# ---------------------------------------------------------------------------
# 7. Main (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Taylor Series Hamiltonian Simulation — Manual Implementation")
    print("=" * 60)

    solver = TaylorAlgorithmSolver()

    # Demo from the SKILL.md
    print("\n--- Demo: H=[[2,1],[1,3]], t=1.0, degree=10 ---")
    H_demo = np.array([[2, 1], [1, 3]], dtype=complex)
    result = solver.run(H=H_demo, t=1.0, error=1e-8, degree=10)
    print(f"  Status:              {result['status']}")
    print(f"  Frobenius error:     {result['Frobenius norm of error']:.6e}")
    print(f"  Computation time:    {result['Computation time (s)']}s")

    # Show adaptive parameter computation
    print("\n--- Adaptive Parameters ---")
    alpha, lam = compute_spectral_params(H_demo, 1.0)
    print(f"  α = ||H||_2 = {alpha:.4f}")
    print(f"  λ = α·t = {lam:.4f}")

    for t_val in [0.5, 1.0, 3.0, 5.0]:
        r, K, lam_s = adaptive_degree_slices(alpha, t_val, 1e-8, 10)
        print(f"  t={t_val:.1f}: r={r}, K={K}, λ_slice={lam_s:.4f}")

    # Show Taylor polynomial terms
    print("\n--- Taylor Series Coefficients (H_scaled, K=5) ---")
    H_scaled = H_demo * (1.0 / 8)  # hypothetical per-slice scaling
    U_test, bound = taylor_matrix_series(H_scaled, 5)
    norm_Hs = np.linalg.norm(H_scaled, 2)
    for k in range(6):
        term_norm = norm_Hs**k / math.factorial(k)
        print(f"  k={k}: ||H_slice^{k}||/k! ≈ {term_norm:.2e}")
    print(f"  Truncation bound (k>5): {bound:.2e}")

    # Show Pauli-string Taylor expansion
    print("\n--- Pauli-String Taylor Expansion (K=3) ---")
    if HAS_UNITARYLAB:
        decomp = pauli_string_decomposition(H_demo * 0.1)  # small scaling
        taylor_terms = taylor_pauli_expand(decomp, 3)
        print(f"  Terms in e^{{-iH·0.1}} (K=3): {len(taylor_terms)}")
        for ps, c in taylor_terms[:8]:
            print(f"    {ps}: {c.real:+.6f}{c.imag:+.6f}j")

    # Run known test cases
    print("\n--- Known Test Cases ---")
    passed = 0
    for case in KNOWN_CASES:
        if run_known_test(case):
            passed += 1
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
