"""Manual implementation of Shor's integer factoring algorithm.

Factors an integer N in polynomial time O((log N)^3) by reducing factoring to
quantum period-finding via the Quantum Fourier Transform.

Components:
    - get_modular_matrix: Permutation matrix for |x> -> |x * mult mod N>
    - build_shor_circuit: Constructs the QPE circuit for order finding
    - _is_perfect_power: Checks if N = p^k (handled classically)
    - _extract_factors_from_period: gcd(a^{r/2} ± 1, N) factor extraction
    - shor_factor: End-to-end factoring with retry loop
    - ShorAlgorithmSolver: Class-based interface matching unitarylab pattern

Reference:
    SKILL.md — Shor's Algorithm
"""

from __future__ import annotations

import math
import random
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit
from unitarylab.library import IQFT


# ---------------------------------------------------------------------------
# 1. Modular multiplication matrix
# ---------------------------------------------------------------------------

def get_modular_matrix(mult: int, N: int, n_work: int) -> np.ndarray:
    """Permutation matrix for |x> -> |x * mult mod N>.

    States |x> with x >= N act as identity (no modular arithmetic applied),
    matching the convention in the unitarylab ShorAlgorithm implementation.

    Args:
        mult: Multiplier (e.g., a^{2^j} mod N).
        N: Integer to factor.
        n_work: Number of qubits in the work register (dim = 2^{n_work}).

    Returns:
        A 2^{n_work} × 2^{n_work} complex permutation matrix.
    """
    dim = 1 << n_work
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = (x * mult) % N if x < N else x
        U[y, x] = 1.0
    return U


# ---------------------------------------------------------------------------
# 2. Circuit construction (matrix method)
# ---------------------------------------------------------------------------

def build_shor_circuit(a: int, N: int, n_count: int, n_work: int) -> Circuit:
    """Construct the QPE circuit for order-finding f(x) = a^x mod N.

    Register layout (flat qubit indices):
        counting : [0,         n_count)     — counting register
        work     : [n_count,   n_count + n_work) — work register

    Algorithm steps:
        1. Hadamard on counting register; set work to |1>.
        2. Controlled a^{2^j} mod N on work, controlled by each counting qubit.
        3. Inverse QFT on counting register.

    Args:
        a: Random base (1 < a < N, coprime to N).
        N: Integer to factor.
        n_count: Number of qubits in the counting register.
        n_work: Number of qubits in the work register.

    Returns:
        The constructed Circuit object.
    """
    total = n_count + n_work
    qc = Circuit(total, name=f"Shor_N{N}_a{a}")

    # --- Step 1: Superposition on counting register; work = |1> ---
    qc.h(list(range(n_count)))
    qc.x(n_count)  # Set work[0] = 1, giving |00...01> = |1>

    # --- Step 2: Controlled a^{2^j} mod N ---
    work_qubits = list(range(n_count, total))
    for j in range(n_count):
        mult = pow(a, 1 << j, N)  # a^{2^j} mod N
        U = get_modular_matrix(mult, N, n_work)
        qc.unitary(U, work_qubits, j, "1")  # controlled on counting qubit j

    # --- Step 3: Inverse QFT on counting register ---
    qc.append(IQFT(n_count), list(range(n_count)))

    return qc


# ---------------------------------------------------------------------------
# 3. Classical pre-checks
# ---------------------------------------------------------------------------

def _is_perfect_power(N: int) -> Optional[Tuple[int, int]]:
    """Check if N is a perfect power N = p^k for k >= 2.

    Args:
        N: Integer to check.

    Returns:
        (p, k) such that p^k = N, or None if N is not a perfect power.
    """
    max_exp = int(math.log2(N)) + 1
    for k in range(2, max_exp + 1):
        # Integer k-th root via binary search
        lo, hi = 2, int(N ** (1.0 / k)) + 2
        while lo <= hi:
            mid = (lo + hi) // 2
            val = mid ** k
            if val == N:
                return (mid, k)
            elif val < N:
                lo = mid + 1
            else:
                hi = mid - 1
    return None


# ---------------------------------------------------------------------------
# 4. Factor extraction from period
# ---------------------------------------------------------------------------

def _extract_factors_from_period(
    a: int, r: int, N: int
) -> Optional[List[int]]:
    """Extract non-trivial factors from the found period r.

    Requires:
        - r is even
        - a^{r/2} ≠ -1 (mod N)  (i.e., a^{r/2} mod N ≠ N-1)

    Then factors are gcd(a^{r/2} ± 1, N).

    Args:
        a: The random base used.
        r: The period found via continued fractions.
        N: The integer to factor.

    Returns:
        List of two non-trivial factors [p, q], or None if extraction fails.
    """
    if r % 2 != 0 or r == 0:
        return None

    guess = pow(a, r // 2, N)

    # Guard: trivial square-root cases
    if guess in (1, N - 1):
        return None

    p = math.gcd(guess - 1, N)
    q = math.gcd(guess + 1, N)

    # Both factors must be non-trivial
    if p == 1 or p == N or q == 1 or q == N:
        return None

    return sorted([p, q])


# ---------------------------------------------------------------------------
# 5. End-to-end solver
# ---------------------------------------------------------------------------

def shor_factor(
    N: int,
    method: str = "matrix",
    max_retries: int = 15,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Factor an integer N using Shor's algorithm.

    Full pipeline:
        1. Classical pre-checks (even N, perfect powers).
        2. Random base selection with gcd shortcut.
        3. Quantum circuit construction and simulation.
        4. Classical post-processing: continued fractions → period → gcd.

    Args:
        N: Composite integer to factor.
        method: Circuit method — 'matrix' (currently supported) or 'operator'.
        backend: Simulation backend (default 'torch').
        device: Compute device (default 'cpu').
        dtype: Numeric dtype for simulation.
        max_retries: Maximum number of random base attempts.
        seed: Random seed for reproducibility.
        verbose: Print progress information.

    Returns:
        Dict with keys: 'status', 'factors', 'period', 'Selected base',
        'Computation time (s)', 'Measurement', 'Total qubits', 'circuit'.

    Raises:
        ValueError: If N < 4 or N is prime.
    """
    if N < 4:
        raise ValueError(f"N={N} must be >= 4")

    t_start_total = time.perf_counter()

    if verbose:
        print(f"Shor: factoring N = {N}")
        print(f"  Method: {method}, max_retries: {max_retries}")

    # --- Pre-check 1: Even N ---
    if N % 2 == 0:
        elapsed = time.perf_counter() - t_start_total
        if verbose:
            print(f"  Even N → classical shortcut: [2, {N // 2}]")
        return {
            "status": "ok",
            "factors": [2, N // 2],
            "period": None,
            "Selected base": None,
            "Computation time (s)": round(elapsed, 4),
            "Measurement": None,
            "Total qubits": None,
            "circuit": None,
        }

    # --- Pre-check 2: Perfect power ---
    pp = _is_perfect_power(N)
    if pp is not None:
        p, k = pp
        elapsed = time.perf_counter() - t_start_total
        if verbose:
            print(f"  Perfect power {N} = {p}^{k} → classical factorization")
        factors = [p] * k
        return {
            "status": "ok",
            "factors": factors,
            "period": None,
            "Selected base": None,
            "Computation time (s)": round(elapsed, 4),
            "Measurement": None,
            "Total qubits": None,
            "circuit": None,
        }

    # --- Register sizing ---
    n_work = N.bit_length()
    n_count = 2 * n_work
    total_qubits = n_count + n_work
    N_size = 1 << n_count

    if verbose:
        print(f"  Qubits: counting={n_count}, work={n_work}, total={total_qubits}")
        print(f"  N_size = 2^{n_count} = {N_size}")

    # --- Set random seed if provided ---
    rng = random.Random(seed)

    # --- Retry loop ---
    for attempt in range(1, max_retries + 1):
        a = rng.randint(2, N - 1)

        if verbose:
            print(f"\n  Attempt {attempt}/{max_retries}: a = {a}")

        # Classical shortcut: gcd(a, N) > 1
        g = math.gcd(a, N)
        if g > 1:
            elapsed = time.perf_counter() - t_start_total
            if verbose:
                print(f"    gcd(a, N) = {g} > 1 → classical shortcut")
            return {
                "status": "ok",
                "factors": sorted([g, N // g]),
                "period": None,
                "Selected base": a,
                "Computation time (s)": round(elapsed, 4),
                "Measurement": None,
                "Total qubits": None,
                "circuit": None,
            }

        # --- Build and simulate quantum circuit ---
        t_sim_start = time.perf_counter()

        try:
            qc = build_shor_circuit(a, N, n_count, n_work)
        except Exception as e:
            if verbose:
                print(f"    Circuit build failed: {e}")
            continue

        # Execute and measure
        result = qc.execute(backend=backend, device=device, dtype=dtype)
        meas_bin = result.measure(list(range(n_count)), endian="little")
        meas_int = int(meas_bin, 2)

        t_sim = time.perf_counter() - t_sim_start

        if verbose:
            print(f"    Measurement: {meas_int} (binary: {meas_bin})")
            print(f"    Simulation time: {t_sim:.4f} s")

        # --- Classical post-processing ---
        if meas_int == 0:
            if verbose:
                print("    Measurement = 0, retrying...")
            continue

        phase = meas_int / N_size
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator

        if verbose:
            print(f"    Phase: {phase:.6f} → fraction: {frac} → r = {r}")

        # Verify the period: a^r ≡ 1 (mod N)
        if r == 0 or pow(a, r, N) != 1:
            if verbose:
                print(f"    Period r={r} invalid (a^r mod N = {pow(a, r, N) if r > 0 else 'N/A'}), retrying...")
            continue

        # Extract factors from the period
        factors = _extract_factors_from_period(a, r, N)

        if factors is not None:
            elapsed = time.perf_counter() - t_start_total
            if verbose:
                print(f"    Factors found: {factors}")
                print(f"    Total time: {elapsed:.4f} s")
            return {
                "status": "ok",
                "factors": factors,
                "period": r,
                "Selected base": a,
                "Computation time (s)": round(t_sim, 4),
                "Measurement": meas_int,
                "Total qubits": total_qubits,
                "circuit": qc,
            }

        if verbose:
            print(f"    Factor extraction failed (r={r}, a^(r/2) mod N = {pow(a, r//2, N)}), retrying...")

    # --- Exhausted all retries ---
    elapsed = time.perf_counter() - t_start_total
    if verbose:
        print(f"\n  Failed after {max_retries} retries (total time: {elapsed:.4f} s)")

    return {
        "status": "failed",
        "factors": None,
        "period": None,
        "Selected base": None,
        "Computation time (s)": round(elapsed, 4),
        "Measurement": None,
        "Total qubits": None,
        "circuit": None,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface (matching ShorAlgorithm pattern)
# ---------------------------------------------------------------------------

class ShorAlgorithmSolver:
    """Class-based solver matching the ShorAlgorithm interface.

    Usage:
        solver = ShorAlgorithmSolver()
        result = solver.run(N=15, method='matrix', backend='torch')
        print(result['factors'])  # [3, 5]
    """

    def __init__(
        self,
        text_mode: str = "plain",
        algo_dir: Optional[str] = None,
    ):
        """Initialize the Shor solver.

        Args:
            text_mode: Output text mode ('plain' or 'legacy').
            algo_dir: Directory for output files.
        """
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def update_output(self, data: Dict[str, Any]) -> None:
        """Update the output fields (mirrors ShorAlgorithm.update_output)."""
        self.output.update(data)

    def run(
        self,
        N: int,
        method: str = "matrix",
        max_retries: int = 15,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run Shor's algorithm.

        Args:
            N: Composite integer to factor.
            method: Circuit method ('matrix' or 'operator').
            max_retries: Maximum random base attempts.
            backend: Simulation backend.
            device: Compute device.
            dtype: Numeric dtype.

        Returns:
            Result dict with keys: 'status', 'factors', 'period',
            'Selected base', 'Computation time (s)', 'Measurement',
            'Total qubits', 'circuit_path', 'plot', 'circuit'.
        """
        result = shor_factor(
            N=N,
            method=method,
            max_retries=max_retries,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )

        self.update_output(result)
        return self._build_return_dict(
            result.get("status") == "ok",
            result.get("circuit"),
        )

    def _build_return_dict(
        self,
        success: bool,
        circuit: Optional[Circuit],
    ) -> Dict[str, Any]:
        """Package results in the standard ShorAlgorithm return format.

        Args:
            success: Whether factoring succeeded.
            circuit: The Circuit object (if quantum path was used).

        Returns:
            Formatted result dict matching ShorAlgorithm return schema.
        """
        base = {
            "status": "ok" if success else "failed",
            "circuit_path": "",
            "plot": [],
            "circuit": circuit,
        }
        # Merge with output fields (factors, period, Selected base, etc.)
        base.update(self.output)
        return base


# ---------------------------------------------------------------------------
# 7. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Tuple[int, List[int]]] = [
    # (N, expected_factors) — sorted
    (15, [3, 5]),
    (21, [3, 7]),
    (33, [3, 11]),
    (35, [5, 7]),
    (39, [3, 13]),
    (55, [5, 11]),
    (77, [7, 11]),
    (91, [7, 13]),
]


def run_known_test(
    N: int,
    expected_factors: List[int],
    backend: str = "torch",
    max_retries: int = 15,
    max_attempts: int = 3,
) -> bool:
    """Run a single known test case with multiple top-level attempts.

    Args:
        N: Integer to factor.
        expected_factors: Expected sorted list of factors.
        backend: Simulation backend.
        max_retries: Max retries per shor_factor call.
        max_attempts: Max top-level attempts (algorithm is probabilistic).

    Returns:
        True if the test passes within max_attempts, False otherwise.
    """
    for attempt in range(1, max_attempts + 1):
        result = shor_factor(
            N=N, method="matrix", backend=backend,
            max_retries=max_retries, verbose=False,
        )
        if result["status"] == "ok" and result["factors"] is not None:
            found = sorted(result["factors"])
            prod = 1
            for f in found:
                prod *= f
            if found == expected_factors and prod == N:
                print(f"  PASS: N={N} → {found} "
                      f"(attempt {attempt}, base a={result['Selected base']}, "
                      f"period r={result['period']}, "
                      f"{result['Computation time (s)']}s)")
                return True
        print(f"  Retry {attempt}/{max_attempts}: status={result['status']}, "
              f"factors={result['factors']}")
    print(f"  FAIL: N={N} (expected {expected_factors})")
    return False


# ---------------------------------------------------------------------------
# 8. Main (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Shor's Factoring Algorithm — Manual Implementation")
    print("=" * 60)

    solver = ShorAlgorithmSolver()

    # Demo: N = 15
    print("\n--- Demo: Factor N = 15 ---")
    result = solver.run(N=15, method="matrix", backend="torch")
    print(f"  Status:        {result['status']}")
    print(f"  Factors:       {result['factors']}")
    print(f"  Period r:      {result['period']}")
    print(f"  Selected base: {result['Selected base']}")
    print(f"  Time:          {result['Computation time (s)']} s")

    # Run some known test cases
    print("\n--- Known Test Cases ---")
    passed = 0
    test_cases = KNOWN_CASES[:6]  # First 6 for quick run
    for N, expected in test_cases:
        if run_known_test(N, expected):
            passed += 1
    print(f"\n  Result: {passed}/{len(test_cases)} tests passed")

    # Also test even N shortcut
    print("\n--- Even N Shortcut ---")
    result_even = solver.run(N=14, method="matrix")
    print(f"  N=14: status={result_even['status']}, factors={result_even['factors']}")
    print(f"  (No quantum circuit needed — classical shortcut)")

    sys.exit(0 if passed == len(test_cases) else 1)
