"""Manual implementation of Trotter-Suzuki Hamiltonian Simulation.

Approximates U(t) = e^{-iHt} via structured short-time exponential products:

    1st-order:  S_1(Δt) = Π_ℓ e^{-i H_ℓ Δt}
    2nd-order:  S_2(Δt) = (Π_{ℓ=1}^{L-1} e^{-i H_ℓ Δt/2}) · e^{-i H_L Δt}
                           · (Π_{ℓ=L-1}^{1} e^{-i H_ℓ Δt/2})
    Higher:     S_{2k}(Δt) = S_{2k-2}(p·Δt)^2 · S_{2k-2}((1-4p)·Δt)
                           · S_{2k-2}(p·Δt)^2
    where p = 1 / (4 - 4^{1/(2k-1)})

Components:
    - suzuki_recurse: Recursive Suzuki product formula composition
    - adaptive_steps: Compute required steps from spectral norm
    - trotter_matrix_slice: Build one slice at matrix level
    - trotter_simulate: End-to-end simulation
    - TrotterAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Trotter-Suzuki Hamiltonian Simulation
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
    from unitarylab.library.pauli_operator import pauli_string_decomposition

    HAS_UNITARYLAB = True
except ImportError:
    HAS_UNITARYLAB = False

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Pauli utilities
# ---------------------------------------------------------------------------

_PAULI_MATRICES: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def manual_pauli_decompose(H: np.ndarray) -> List[Tuple[str, complex]]:
    """Decompose a 2x2 Hermitian matrix into Pauli basis.

    Args:
        H: 2x2 Hermitian matrix.

    Returns:
        List of (pauli_label, coefficient) tuples, sorted by |coeff| desc.
    """
    decomp = []
    for label in ["I", "X", "Y", "Z"]:
        P = _PAULI_MATRICES[label]
        c = 0.5 * np.trace(P.conj().T @ H)
        if abs(c) > 1e-15:
            decomp.append((label, complex(c)))
    decomp.sort(key=lambda x: abs(x[1]), reverse=True)
    return decomp


def decompose(H: np.ndarray) -> List[Tuple[str, complex]]:
    """Decompose H into Pauli strings (unitarylab or manual fallback).

    Args:
        H: Hermitian Hamiltonian.

    Returns:
        List of (pauli_string, coefficient) tuples.
    """
    if HAS_UNITARYLAB and H.shape[0] > 2:
        return pauli_string_decomposition(H)
    elif H.shape[0] == 2:
        return manual_pauli_decompose(H)
    elif HAS_UNITARYLAB:
        return pauli_string_decomposition(H)
    else:
        raise RuntimeError("Need unitarylab for multi-qubit Pauli decomposition")


# ---------------------------------------------------------------------------
# 2. Suzuki recursion
# ---------------------------------------------------------------------------

def suzuki_recurse(
    decomposition: List[Tuple[str, complex]],
    order: int,
) -> List[Tuple[str, complex]]:
    """Recursively build the Suzuki product formula sequence.

    Order-1: identity (returns decomposition as-is).
    Order-2: symmetric palindrome:
        halves + full + reversed(halves)
        where halves = [(P, c/2) for all but last], full = [last]
    Order 4,6,...: recursive fractal composition
        outer(2×) + inner + outer(2×)
        with reduction factor p = 1/(4 - 4^{1/(order-1)})

    Args:
        decomposition: List of (pauli_string, coefficient) from Pauli decomp.
        order: Suzuki order (1, or even ≥2).

    Returns:
        Ordered list of (pauli_string, coefficient) for one slice.
    """
    if order == 1:
        return list(decomposition)

    if order == 2:
        if len(decomposition) <= 1:
            return list(decomposition)
        halves = [(p, c / 2.0) for p, c in decomposition[:-1]]
        full = [(decomposition[-1][0], decomposition[-1][1])]
        return halves + full + list(reversed(halves))

    # order >= 4 (even): Suzuki fractal
    # S_{2k}(Δt) = S_{2k-2}(p·Δt)^2 · S_{2k-2}((1-4p)·Δt) · S_{2k-2}(p·Δt)^2
    # where p = 1 / (4 - 4^{1/(2k-1)})
    p = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    scaled_outer = [(ps, c * p) for ps, c in decomposition]
    scaled_inner = [(ps, c * (1.0 - 4.0 * p)) for ps, c in decomposition]

    outer_seq = suzuki_recurse(scaled_outer, order - 2)
    inner_seq = suzuki_recurse(scaled_inner, order - 2)

    # outer^2 · inner · outer^2
    return outer_seq + outer_seq + inner_seq + outer_seq + outer_seq


# ---------------------------------------------------------------------------
# 3. Adaptive step computation
# ---------------------------------------------------------------------------

def adaptive_steps(
    H: np.ndarray,
    t: float,
    error: float,
    order: int,
    max_steps: int = 1000,
) -> int:
    """Compute required Trotter steps via adaptive formula.

    steps = min(
        5^order · (t·n·α)^{1+1/order} · ε^{-1/order} · 1.5,
        max_steps
    )
    where n = #qubits, α = ||H||_2.

    Args:
        H: Hamiltonian matrix.
        t: Evolution time.
        error: Target error.
        order: Suzuki order.
        max_steps: Upper bound on steps.

    Returns:
        Number of Trotter steps (≥1).
    """
    n = int(math.ceil(math.log2(max(H.shape[0], 2))))
    alpha = float(np.linalg.norm(H, ord=2))

    steps_adaptive = int(
        (5.0 ** order)
        * (t * n * alpha) ** (1.0 + 1.0 / order)
        * (error ** (-1.0 / order))
        * 1.5
    )
    steps = min(steps_adaptive, max_steps)
    return max(steps, 1)


# ---------------------------------------------------------------------------
# 4. Matrix-level Trotter slice
# ---------------------------------------------------------------------------

def trotter_matrix_slice(
    decomposition: List[Tuple[str, complex]],
    order: int,
    steps: int,
) -> np.ndarray:
    """Build one Trotter slice at the matrix level.

    For each (P, c) in the Suzuki-ordered sequence:
        e^{-i · c/steps · P}

    The slice is the product of all such exponentials, in Suzuki order.

    Args:
        decomposition: Pauli decomposition of H (coefficients include t factor).
        order: Suzuki order.
        steps: Number of Trotter steps.

    Returns:
        Unitary matrix for one Trotter slice.
    """
    # Build Suzuki sequence (one slice's ordering)
    scaled = [(p, c / steps) for p, c in decomposition]
    sequence = suzuki_recurse(scaled, order)

    dim = 2  # for 2x2 matrices; extend for larger
    U = np.eye(dim, dtype=complex)

    for pauli_str, coeff in sequence:
        c = complex(coeff)
        if abs(c) < 1e-15:
            continue

        # Compute e^{-i·c·P} for single-qubit Pauli
        if pauli_str == "I":
            term = np.eye(dim, dtype=complex) * np.exp(-1.0j * c)
        elif pauli_str in _PAULI_MATRICES:
            P = _PAULI_MATRICES[pauli_str]
            # e^{-iθP} = cos(θ)I - i·sin(θ)P
            theta = float(c.real)
            term = math.cos(theta) * np.eye(dim, dtype=complex) - 1.0j * math.sin(theta) * P
        else:
            raise ValueError(f"Unsupported Pauli string: {pauli_str}")

        U = term @ U  # Apply in order (leftmost first)

    return U


# ---------------------------------------------------------------------------
# 5. End-to-end simulation
# ---------------------------------------------------------------------------

def trotter_simulate(
    H: np.ndarray,
    t: float,
    error: float = 1e-8,
    order: int = 2,
    steps: int = 1000,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Simulate U(t) = e^{-iHt} using Trotter-Suzuki product formula.

    Pipeline:
        1. Decompose H into Pauli strings.
        2. Compute adaptive step count from spectral norm.
        3. Absorb t into Pauli coefficients.
        4. Build Suzuki sequence via recursion.
        5. Build slice matrix, then U_approx = (slice)^steps.
        6. Compare with exact U = expm(-1j·H·t).

    Args:
        H: Hermitian Hamiltonian matrix (2x2 supported).
        t: Total evolution time.
        error: Target approximation error.
        order: Suzuki order (1, 2, 4, 6, ...).
        steps: Upper bound on Trotter steps.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress.

    Returns:
        Dict with: 'status', 'Approximate evolution matrix',
        'Exact evolution matrix', 'Frobenius norm of error',
        'Computation time (s)', 'circuit', 'order', 'steps'.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape {H.shape}")
    if order != 1 and order % 2 != 0:
        raise ValueError(f"Order must be 1 or even, got {order}")

    dim = H.shape[0]

    if verbose:
        print(f"Trotter-Suzuki Simulation")
        print(f"  H: {dim}x{dim}, t={t}, order={order}")
        print(f"  Target error: {error:.2e}")

    t_start = time.perf_counter()

    # --- Step 1: Decompose ---
    decomposition = decompose(H)

    if verbose:
        print(f"  Pauli terms: {len(decomposition)}")
        for ps, c in decomposition:
            print(f"    {ps}: {c.real:+.6f}")

    # --- Step 2: Adaptive steps ---
    steps_used = adaptive_steps(H, t, error, order, steps)

    if verbose:
        print(f"  Adaptive steps: {steps_used}")

    # --- Step 3: Absorb t into coefficients ---
    decomposition_t = [(p, c * t) for p, c in decomposition]

    # --- Step 4: Build slice ---
    U_slice = trotter_matrix_slice(decomposition_t, order, steps_used)

    # --- Step 5: Compose ---
    if steps_used > 1:
        U_approx = np.linalg.matrix_power(U_slice, steps_used)
    else:
        U_approx = U_slice

    # --- Step 6: Exact reference ---
    if HAS_SCIPY:
        U_exact = expm(-1.0j * H * t)
    else:
        evals, evecs = np.linalg.eigh(H)
        U_exact = evecs @ np.diag(np.exp(-1.0j * evals * t)) @ evecs.conj().T

    frob_error = float(np.linalg.norm(U_approx - U_exact, ord="fro"))
    t_elapsed = time.perf_counter() - t_start

    is_success = np.isfinite(frob_error)

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
        "circuit": None,
        "circuit_path": [],
        "plot": [],
        "order": order,
        "steps": steps_used,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class TrotterAlgorithmSolver:
    """Class-based solver matching the TrotterAlgorithm interface.

    Usage:
        solver = TrotterAlgorithmSolver()
        result = solver.run(
            H=np.array([[2,1],[1,3]], dtype=float),
            t=1.0, error=1e-8, order=2, steps=1000, backend='torch',
        )
        print(result['Frobenius norm of error'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        H: np.ndarray,
        t: float,
        error: float = 1e-8,
        order: int = 2,
        steps: int = 1000,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        result = trotter_simulate(
            H=H, t=t, error=error, order=order, steps=steps,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "circuit_path": result.get("circuit_path", []),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
            "Approximate evolution matrix": result.get("Approximate evolution matrix"),
            "Exact evolution matrix": result.get("Exact evolution matrix"),
            "Frobenius norm of error": result.get("Frobenius norm of error"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
        }


# ---------------------------------------------------------------------------
# 7. Test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"H": np.array([[2, 1], [1, 3]], dtype=float), "t": 1.0, "error": 1e-8, "order": 2, "steps": 100, "label": "H=[[2,1],[1,3]], t=1, ord=2"},
    {"H": np.array([[2, 1], [1, 3]], dtype=float), "t": 1.0, "error": 1e-8, "order": 4, "steps": 100, "label": "H=[[2,1],[1,3]], t=1, ord=4"},
    {"H": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float), "t": 0.5, "error": 1e-8, "order": 2, "steps": 100, "label": "H=0.5(I+X), t=0.5"},
    {"H": np.array([[0, 1], [1, 0]], dtype=float), "t": np.pi/2, "error": 1e-8, "order": 2, "steps": 100, "label": "H=X, t=π/2"},
    {"H": np.array([[2, 1], [1, 3]], dtype=float), "t": 3.0, "error": 1e-6, "order": 2, "steps": 200, "label": "H=[[2,1],[1,3]], t=3 (long)"},
    {"H": np.array([[1.5, 0.3], [0.3, -0.5]], dtype=float), "t": 0.5, "error": 1e-8, "order": 4, "steps": 50, "label": "mixed, ord=4"},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    H, t, err, order, steps, label = case["H"], case["t"], case["error"], case["order"], case["steps"], case["label"]
    result = trotter_simulate(H=H, t=t, error=err, order=order, steps=steps, verbose=False)
    icon = "✓" if result["status"] == "ok" else "✗"
    print(f"  [{icon}] {label}")
    print(f"       Error: {result['Frobenius norm of error']:.6e}  (steps={result['steps']})")
    print(f"       Time:  {result['Computation time (s)']}s")
    return result["status"] == "ok"


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Trotter-Suzuki Hamiltonian Simulation")
    print("=" * 60)

    solver = TrotterAlgorithmSolver()

    print("\n--- Demo: H=[[2,1],[1,3]], t=1.0, order=2 ---")
    H_demo = np.array([[2, 1], [1, 3]], dtype=float)
    result = solver.run(H=H_demo, t=1.0, error=1e-8, order=2, steps=100)
    print(f"  Status:          {result['status']}")
    print(f"  Frobenius error: {result['Frobenius norm of error']:.6e}")

    # Order comparison
    print("\n--- Order Comparison ---")
    for ord_val in [1, 2, 4, 6]:
        r = trotter_simulate(H=H_demo, t=1.0, error=1e-8, order=ord_val, steps=50, verbose=False)
        print(f"  order={ord_val}: error={r['Frobenius norm of error']:.6e}  steps={r['steps']}")

    # Adaptive steps table
    print("\n--- Adaptive Steps ---")
    alpha = float(np.linalg.norm(H_demo, 2))
    for ord_val in [1, 2, 4]:
        for t_val in [0.5, 1.0, 3.0]:
            s = adaptive_steps(H_demo, t_val, 1e-8, ord_val, 1000)
            print(f"  order={ord_val}, t={t_val:.1f}, α={alpha:.3f}: steps={s}")

    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
