"""Manual implementation of Quantum Signal Processing (QSP).

Implements a polynomial transformation P(x) of a signal x via interleaved
phase rotations and signal operators on a single qubit:

    U_Φ(x) = e^{iφ_0 Z} Π_{k=1}^d [W(x) · e^{iφ_k Z}]

where W(x) = R_x(2·arccos(x)) and the (0,0) matrix element equals P(x).

The phase sequence Φ = (φ_0, ..., φ_d) is optimized classically via L-BFGS-B
to approximate the target function cos(t·x) on [-1, 1].

Components:
    - qsp_matrix: Compute the 2×2 QSP product matrix
    - find_qsp_phases: Optimize phase sequence via L-BFGS-B
    - build_qsp_circuit: Construct the alternating Rz/Rx circuit
    - qsp_evaluate: End-to-end QSP approximation at a test point
    - QSPAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Quantum Signal Processing (QSP)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from unitarylab.core import Circuit, Register


# ---------------------------------------------------------------------------
# 1. QSP matrix product (classical simulation for optimization)
# ---------------------------------------------------------------------------

def qsp_matrix(x: float, phases: np.ndarray) -> np.ndarray:
    """Compute the 2×2 QSP circuit product matrix for signal x.

    Circuit: Rz(2φ₀) → Π_{k=1}^d [W(x) · Rz(2φ_k)]
    where W(x) = R_x(2·arccos(x)).

    Args:
        x: Signal value in [-1, 1].
        phases: Phase array of length d+1.

    Returns:
        2×2 complex matrix U_Φ(x).
    """
    theta = math.acos(max(-1.0, min(1.0, x)))  # clip to [-1, 1]

    # Signal operator W(x) = R_x(2·arccos(x))
    # In matrix form: [[x, i√(1-x²)], [i√(1-x²), x]]
    sqrt_term = math.sqrt(max(0.0, 1.0 - x * x))
    W = np.array(
        [[x, 1.0j * sqrt_term], [1.0j * sqrt_term, x]], dtype=complex
    )

    # Initial phase rotation R_z(2φ₀) = diag(e^{iφ₀}, e^{-iφ₀})
    mat = np.array(
        [
            [np.exp(1.0j * phases[0]), 0.0],
            [0.0, np.exp(-1.0j * phases[0])],
        ],
        dtype=complex,
    )

    # Alternating signal and phase rotations
    for k in range(1, len(phases)):
        # R_z(2φ_k) = diag(e^{iφ_k}, e^{-iφ_k})
        Rz_k = np.array(
            [
                [np.exp(1.0j * phases[k]), 0.0],
                [0.0, np.exp(-1.0j * phases[k])],
            ],
            dtype=complex,
        )
        mat = Rz_k @ W @ mat

    return mat


# ---------------------------------------------------------------------------
# 2. Phase optimization
# ---------------------------------------------------------------------------

def find_qsp_phases(
    t: float,
    d: int,
    seed: int = 42,
    maxiter: int = 400,
    verbose: bool = False,
) -> np.ndarray:
    """Find QSP phases that approximate cos(t·x) on [-1, 1].

    Minimizes the average squared error between ⟨0|U_Φ(x)|0⟩ and
    cos(t·x) over 2d+1 sample points.

    Args:
        t: Evolution time (target = cos(t·x)).
        d: Polynomial degree (d+1 phases).
        seed: Random seed for reproducibility.
        maxiter: Maximum L-BFGS-B iterations.
        verbose: Print optimization progress.

    Returns:
        Phase array of length d+1.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy.optimize is required for phase optimization")

    np.random.seed(seed)

    # Uniform sample grid on [-1, 1]
    x_samples = np.linspace(-1.0, 1.0, 2 * d + 1)
    target_vals = np.cos(t * x_samples)  # real target function

    def loss(phi: np.ndarray) -> float:
        """Mean squared error over sample points."""
        total = 0.0
        for xi, ti in zip(x_samples, target_vals):
            mat = qsp_matrix(xi, phi)
            # The (0,0) element should approximate cos(t·xi)
            err = abs(mat[0, 0].real - ti) ** 2
            total += err
        return total / len(x_samples)

    # Random initialization near zero
    x0 = np.random.randn(d + 1) * 0.1

    if verbose:
        print(f"  Optimizing QSP phases (d={d}, t={t}, samples={len(x_samples)})...")

    result = minimize(
        loss,
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    if verbose:
        print(f"  Loss: {result.fun:.6e}, nit={result.nit}")

    return result.x


# ---------------------------------------------------------------------------
# 3. Circuit construction
# ---------------------------------------------------------------------------

def build_qsp_circuit(phases: np.ndarray, x: float) -> Circuit:
    """Build the QSP circuit for given phases and signal value.

    Structure: Rz(2φ₀) → [Rx(2θ) → Rz(2φₖ)] × d
    where θ = arccos(clip(x, -1, 1)).

    Total gates: 2d + 1 (d Rx + d+1 Rz).

    Args:
        phases: Phase array of length d+1.
        x: Signal value in [-1, 1].

    Returns:
        Single-qubit QSP Circuit.
    """
    d = len(phases) - 1
    theta = float(math.acos(max(-1.0, min(1.0, x))))

    qc = Circuit(Register("q", 1), name="QSP")

    # Initial phase rotation
    qc.rz(2.0 * phases[0], 0)

    for k in range(1, d + 1):
        qc.rx(2.0 * theta, 0)  # signal operator W(x)
        qc.rz(2.0 * phases[k], 0)  # phase rotation

    return qc


# ---------------------------------------------------------------------------
# 4. End-to-end evaluation
# ---------------------------------------------------------------------------

def qsp_evaluate(
    t: float,
    d: int,
    x: float,
    seed: int = 42,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Approximate cos(t·x) using Quantum Signal Processing.

    Pipeline:
        1. Optimize phase sequence Φ via L-BFGS-B.
        2. Build QSP circuit: Rz → [Rx → Rz] × d.
        3. Execute circuit and extract ⟨0|U_Φ(x)|0⟩.
        4. Compare with ideal cos(t·x).

    Args:
        t: Evolution time (target = cos(t·x)).
        d: Polynomial degree.
        x: Test signal point ∈ [-1, 1].
        seed: Random seed for phase optimization.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress.

    Returns:
        Dict with: 'status', 'Estimated value', 'Ideal value',
        'Absolute error', 'Computation time (s)', 'circuit',
        'phases', 'circuit_path', 'plot'.
    """
    if x < -1.0 or x > 1.0:
        x_clipped = max(-1.0, min(1.0, x))
        if verbose:
            print(f"  x = {x} outside [-1,1], clipped to {x_clipped}")
        x = x_clipped

    if verbose:
        print(f"QSP: cos({t}·x) at x={x}, degree d={d}")

    t_start = time.perf_counter()

    # --- Step 1: Phase optimization ---
    phases = find_qsp_phases(t, d, seed=seed, verbose=verbose)

    # --- Step 2: Circuit construction ---
    qc = build_qsp_circuit(phases, x)

    # --- Step 3: Simulation ---
    result = qc.execute(backend=backend, device=device, dtype=dtype)
    final_state = np.asarray(result.state, dtype=complex)

    # The (0,0) matrix element = ⟨0|U_Φ(x)|0⟩ = final_state[0]
    # since we start from |0⟩
    qsp_val = complex(final_state[0])

    # --- Step 4: Comparison ---
    ideal_val = float(np.cos(t * x))
    abs_error = float(np.abs(qsp_val.real - ideal_val))

    t_elapsed = time.perf_counter() - t_start

    is_success = abs_error < 0.1  # generous threshold for optimization

    if verbose:
        print(f"\n  Results:")
        print(f"  QSP value (Re):  {qsp_val.real:.6f}")
        print(f"  cos({t}·{x}):     {ideal_val:.6f}")
        print(f"  Absolute error:  {abs_error:.6e}")
        print(f"  Status:          {'ok' if is_success else 'failed'}")
        print(f"  Time:            {t_elapsed:.4f}s")

    return {
        "status": "ok" if is_success else "failed",
        "Estimated value": qsp_val,
        "Ideal value": ideal_val,
        "Absolute error": abs_error,
        "Computation time (s)": round(t_elapsed, 4),
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
        "phases": phases,
        "d": d,
        "t": t,
        "x": x,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class QSPAlgorithmSolver:
    """Class-based solver matching the QSPAlgorithm interface.

    Usage:
        solver = QSPAlgorithmSolver(text_mode='legacy')
        result = solver.run(t=1.0, d=10, x=0.5, backend='torch')
        print(result['Absolute error'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        t: float = 1.0,
        d: int = 10,
        x: float = 0.5,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        result = qsp_evaluate(
            t=t, d=d, x=x, backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Estimated value": result.get("Estimated value"),
            "Ideal value": result.get("Ideal value"),
            "Absolute error": result.get("Absolute error"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "circuit": result.get("circuit"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
        }


# ---------------------------------------------------------------------------
# 6. Analysis utilities
# ---------------------------------------------------------------------------

def qsp_sweep_x(
    t: float,
    d: int,
    n_points: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate QSP approximation quality across the domain [-1, 1].

    Reuses the same optimized phases for all x values.

    Args:
        t: Evolution time.
        d: Polynomial degree.
        n_points: Number of test points in [-1, 1].
        seed: Random seed.
        verbose: Print per-point info.

    Returns:
        Dict with 'x_vals', 'qsp_vals', 'ideal_vals', 'errors', 'max_error'.
    """
    phases = find_qsp_phases(t, d, seed=seed, verbose=False)
    x_vals = np.linspace(-1.0, 1.0, n_points)
    qsp_vals = np.zeros(n_points, dtype=complex)
    ideal_vals = np.cos(t * x_vals)
    errors = np.zeros(n_points)

    for i, xi in enumerate(x_vals):
        qc = build_qsp_circuit(phases, xi)
        result = qc.execute(backend="torch")
        qsp_vals[i] = complex(np.asarray(result.state)[0])
        errors[i] = float(np.abs(qsp_vals[i].real - ideal_vals[i]))

    return {
        "x_vals": x_vals,
        "qsp_vals": qsp_vals.real,
        "ideal_vals": ideal_vals,
        "errors": errors,
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "phases": phases,
    }


# ---------------------------------------------------------------------------
# 7. Test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"t": 0.5, "d": 6, "x": 0.5, "label": "t=0.5, d=6, x=0.5"},
    {"t": 1.0, "d": 8, "x": 0.0, "label": "t=1.0, d=8, x=0 (easy case)"},
    {"t": 1.0, "d": 10, "x": 0.5, "label": "t=1.0, d=10, x=0.5"},
    {"t": 2.0, "d": 12, "x": 0.7, "label": "t=2.0, d=12, x=0.7"},
    {"t": 0.3, "d": 4, "x": 0.9, "label": "t=0.3, d=4, x=0.9"},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    t, d, x, label = case["t"], case["d"], case["x"], case["label"]
    result = qsp_evaluate(t=t, d=d, x=x, verbose=False)
    ok = result["status"] == "ok"
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] {label}")
    print(f"       QSP={result['Estimated value'].real:.6f}, "
          f"ideal={result['Ideal value']:.6f}, "
          f"error={result['Absolute error']:.6e}")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Quantum Signal Processing (QSP) — Manual Implementation")
    print("=" * 60)

    solver = QSPAlgorithmSolver()

    # Demo from the SKILL.md
    print("\n--- Demo: t=1.0, d=10, x=0.5 ---")
    result = solver.run(t=1.0, d=10, x=0.5)
    print(f"  Absolute error: {result['Absolute error']:.6e}")
    print(f"  Status:         {result['status']}")

    # Show phase values
    phases = find_qsp_phases(1.0, 8, seed=42)
    print(f"\n  Optimized phases (d=8, t=1.0):")
    for i, phi in enumerate(phases):
        print(f"    φ[{i}] = {phi:+.6f}")

    # Sweep across x values
    print("\n--- Domain Sweep (t=1.0, d=10) ---")
    sweep = qsp_sweep_x(t=1.0, d=10, n_points=11)
    print(f"  {'x':>6s}  {'QSP(x)':>9s}  {'cos(tx)':>9s}  {'|err|':>9s}")
    for xi, qi, ii, ei in zip(sweep["x_vals"], sweep["qsp_vals"], sweep["ideal_vals"], sweep["errors"]):
        print(f"  {xi:6.3f}  {qi:9.6f}  {ii:9.6f}  {ei:9.2e}")
    print(f"\n  Max error:  {sweep['max_error']:.4e}")
    print(f"  Mean error: {sweep['mean_error']:.4e}")

    # Degree study
    print("\n--- Degree Study (t=1.0, x=0.5) ---")
    for d_val in [4, 6, 8, 10, 12]:
        r = qsp_evaluate(t=1.0, d=d_val, x=0.5, verbose=False)
        print(f"  d={d_val:2d}: error={r['Absolute error']:.4e}")

    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
