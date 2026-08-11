"""Manual implementation of the Discrete Logarithm quantum algorithm.

Solves g^x ≡ y (mod P) using a two-register QPE extension of Shor's algorithm.

Components:
    - modular_matrix: Builds permutation matrix for |x> -> |x * mult mod P>
    - build_dlp_circuit: Constructs the full DLP quantum circuit
    - solve_dlp: End-to-end solver with classical post-processing
    - DiscreteLogAlgorithmSolver: Class-based interface matching the unitarylab pattern

Reference:
    SKILL.md — Discrete Logarithm Algorithm (DLG)
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab import Circuit, Register
from unitarylab.library import IQFT


# ---------------------------------------------------------------------------
# 1. Modular multiplication matrix
# ---------------------------------------------------------------------------

def modular_matrix(mult: int, P: int, n_work: int) -> np.ndarray:
    """Permutation matrix for the modular multiplication map |x> -> |x * mult mod P>.

    Args:
        mult: Multiplier (e.g., g^{2^i} or (y^{-1})^{2^j}).
        P: Prime modulus.
        n_work: Number of qubits in the work register (dim = 2^{n_work}).

    Returns:
        A 2^{n_work} × 2^{n_work} complex permutation matrix. States |x> with
        x >= P act as identity (no modular arithmetic applied).
    """
    dim = 1 << n_work  # 2^{n_work}
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = (x * mult) % P if x < P else x
        U[y, x] = 1.0
    return U


# ---------------------------------------------------------------------------
# 2. Circuit construction
# ---------------------------------------------------------------------------

def build_dlp_circuit(g: int, y: int, P: int, n_count: int, n_work: int) -> Circuit:
    """Construct the two-register QPE circuit for the DLP g^x ≡ y (mod P).

    Register layout (flat qubit indices):
        reg_a  : [0,           n_count)     — counting register A
        reg_b  : [n_count,     2*n_count)   — counting register B
        reg_w  : [2*n_count,   2*n_count + n_work) — work register

    Algorithm steps:
        1. Hadamard on reg_a and reg_b; set work to |1⟩.
        2. Controlled g^{2^i} mod P on work, controlled by each reg_a bit.
        3. Controlled (y^{-1})^{2^j} mod P on work, controlled by each reg_b bit.
        4. Inverse QFT on both reg_a and reg_b.

    Args:
        g: Base of the discrete logarithm (must be coprime to P).
        y: Target value (must be coprime to P).
        P: Prime modulus.
        n_count: Number of qubits per counting register.
        n_work: Number of qubits in the work register.

    Returns:
        The constructed Circuit object.
    """
    ra = Register("a", n_count)
    rb = Register("b", n_count)
    rw = Register("w", n_work)
    qc = Circuit(ra, rb, rw, name="DLP_circuit")

    # Flat-index offsets for each register
    a_off = 0
    b_off = n_count
    w_off = 2 * n_count

    # --- Step 1: Superposition on counting registers; work = |1> ---
    qc.h(list(range(a_off, a_off + n_count)))
    qc.h(list(range(b_off, b_off + n_count)))
    qc.x(w_off)  # Set work[0] to 1, giving |00...01> = |1>

    # --- Step 2: Controlled g^{2^i} mod P on reg_a ---
    for i in range(n_count):
        mult = pow(g, 1 << i, P)  # g^{2^i} mod P
        qc.unitary(
            modular_matrix(mult, P, n_work),
            list(range(w_off, w_off + n_work)),
            a_off + i,
            "1",
        )

    # --- Step 3: Controlled (y^{-1})^{2^j} mod P on reg_b ---
    y_inv = pow(y, -1, P)  # Modular inverse of y modulo P
    for j in range(n_count):
        mult = pow(y_inv, 1 << j, P)  # (y^{-1})^{2^j} mod P
        qc.unitary(
            modular_matrix(mult, P, n_work),
            list(range(w_off, w_off + n_work)),
            b_off + j,
            "1",
        )

    # --- Step 4: Inverse QFT on both counting registers ---
    qc.append(IQFT(n_count), list(range(a_off, a_off + n_count)))
    qc.append(IQFT(n_count), list(range(b_off, b_off + n_count)))

    return qc


# ---------------------------------------------------------------------------
# 3. Classical post-processing
# ---------------------------------------------------------------------------

def _continued_fraction_period(
    u: int, N_size: int, P: int
) -> Optional[int]:
    """Extract a candidate period r from measurement u via continued fractions.

    Args:
        u: Integer measurement outcome from reg_a.
        N_size: 2^{n_count}, the normalization factor.
        P: Prime modulus (used as denominator bound).

    Returns:
        Candidate period denominator, or None if extraction fails.
    """
    if u == 0:
        return None
    frac = Fraction(u, N_size).limit_denominator(P)
    return frac.denominator


def _find_true_order(g: int, P: int, r_candidate: int, max_multiple: int = 10) -> Optional[int]:
    """Search multiples of r_candidate to find the true group order r.

    The true order r satisfies g^r ≡ 1 (mod P).

    Args:
        g: Base element.
        P: Prime modulus.
        r_candidate: Candidate period from continued fractions.
        max_multiple: Maximum multiplier to search.

    Returns:
        True order r, or None if not found within max_multiple.
    """
    for k in range(1, max_multiple + 1):
        r = r_candidate * k
        if pow(g, r, P) == 1:
            return r
    return None


def _solve_congruence(
    s: int, target: int, r: int
) -> Optional[int]:
    """Solve s·x ≡ target (mod r) for x.

    The equation s·x ≡ target (mod r) has solutions iff gcd(s, r) | target.
    When solvable, returns the smallest non-negative solution.

    Args:
        s: Coefficient (numerator from continued fractions).
        target: Right-hand side value.
        r: Modulus (true group order).

    Returns:
        Smallest non-negative solution x, or None if no solution exists.
    """
    d = math.gcd(s, r)
    if target % d != 0:
        return None
    s_red = s // d
    t_red = target // d
    r_red = r // d
    # Modular inverse of s_red modulo r_red
    try:
        inv = pow(s_red, -1, r_red)
    except ValueError:
        return None
    x = (t_red * inv) % r_red
    return x


def _classical_post_processing(
    probs_dict: Dict[str, Any],
    g: int,
    y: int,
    P: int,
    n_count: int,
    prob_threshold: float = 0.02,
) -> Dict[str, Any]:
    """Classical post-processing of the quantum measurement results.

    Steps:
        1. Sort bitstrings by probability, filter below threshold.
        2. Split each bitstring into reg_a (v) and reg_b (u) halves.
        3. Continued fractions on u/N_size to get candidate (s, r).
        4. Search multiples of r to find true group order.
        5. Compute target = round(v * r / N_size) and solve s·x ≡ -target (mod r).

    Args:
        probs_dict: Dictionary mapping bitstrings to probability data.
        g: Base element.
        y: Target value.
        P: Prime modulus.
        n_count: Number of qubits per counting register.
        prob_threshold: Minimum probability to consider a measurement outcome.

    Returns:
        Dict with keys: 'Found x', 'Detected period r', 'status'.
    """
    N_size = 1 << n_count
    sorted_probs = sorted(
        probs_dict.items(),
        key=lambda item: item[1]["prob"],
        reverse=True,
    )

    for bitstring, data in sorted_probs:
        if data["prob"] < prob_threshold:
            continue

        # reg_a (first n_count bits) → v; reg_b (last n_count bits) → u
        v_bin = bitstring[:n_count]
        u_bin = bitstring[n_count:]

        u = int(u_bin, 2)
        v = int(v_bin, 2)

        # Continued fractions on u/N_size
        r_base = _continued_fraction_period(u, N_size, P)
        if r_base is None:
            continue

        s = Fraction(u, N_size).limit_denominator(P).numerator

        # Find true group order r
        r = _find_true_order(g, P, r_base)
        if r is None:
            continue

        # target = round(v * r / N_size) mod r
        target = round(v * r / N_size) % r
        # Solve: s·x ≡ -target (mod r)  →  s·x ≡ (r - target) (mod r)
        rhs = (-target) % r
        x = _solve_congruence(s, rhs, r)

        if x is not None:
            # Verify the solution
            if pow(g, x, P) == y % P:
                return {
                    "Found x": x,
                    "Detected period r": r,
                    "status": "ok",
                }

    return {
        "Found x": None,
        "Detected period r": None,
        "status": "failed",
    }


# ---------------------------------------------------------------------------
# 4. End-to-end solver
# ---------------------------------------------------------------------------

def solve_dlp(
    g: int,
    y: int,
    P: int,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    prob_threshold: float = 0.02,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the discrete logarithm problem: find x such that g^x ≡ y (mod P).

    Full pipeline:
        1. Validate inputs (coprimality).
        2. Determine register sizes.
        3. Build and simulate the quantum circuit.
        4. Classical post-processing with continued fractions.

    Args:
        g: Base element.
        y: Target value.
        P: Prime modulus.
        backend: Simulation backend (default 'torch').
        device: Compute device (default 'cpu').
        dtype: Numeric dtype for simulation.
        prob_threshold: Minimum probability threshold for measurement outcomes.
        verbose: Print progress information.

    Returns:
        Dict with keys: 'Found x', 'Detected period r', 'status',
        'Computation time (s)', 'circuit'.

    Raises:
        ValueError: If g or y are not coprime to P.
    """
    # --- Validation ---
    if math.gcd(g, P) != 1:
        raise ValueError(f"g={g} is not coprime to P={P} (gcd={math.gcd(g, P)})")
    if math.gcd(y, P) != 1:
        raise ValueError(f"y={y} is not coprime to P={P} (gcd={math.gcd(y, P)})")

    # --- Register sizing ---
    n_work = P.bit_length()
    n_count = 2 * n_work
    N_size = 1 << n_count

    if verbose:
        print(f"DLP: {g}^x ≡ {y} (mod {P})")
        print(f"  Registers: n_count={n_count}, n_work={n_work}")
        print(f"  Total qubits: {2 * n_count + n_work} = {2 * n_count + n_work}")
        print(f"  N_size = 2^{n_count} = {N_size}")

    # --- Circuit construction & simulation ---
    t_start = time.perf_counter()

    qc = build_dlp_circuit(g, y, P, n_count, n_work)

    if verbose:
        print(f"  Circuit built: {qc.name}")

    res_vec = qc.execute(backend=backend, device=device, dtype=dtype)
    probs_dict = res_vec.calculate_state(list(range(2 * n_count)))

    if verbose:
        print(f"  Measurement outcomes: {len(probs_dict)} unique bitstrings")

    # --- Classical post-processing ---
    result = _classical_post_processing(probs_dict, g, y, P, n_count, prob_threshold)

    t_end = time.perf_counter()
    result["Computation time (s)"] = round(t_end - t_start, 4)
    result["circuit"] = qc

    if verbose:
        print(f"  Status: {result['status']}")
        if result["Found x"] is not None:
            print(f"  Found x = {result['Found x']}")
            print(f"  Verification: {g}^{result['Found x']} mod {P} = "
                  f"{pow(g, result['Found x'], P)} (expected {y % P})")
        print(f"  Time: {result['Computation time (s)']} s")

    return result


# ---------------------------------------------------------------------------
# 5. Class-based interface (matching DiscreteLogAlgorithm pattern)
# ---------------------------------------------------------------------------

class DiscreteLogAlgorithmSolver:
    """Class-based solver matching the DiscreteLogAlgorithm interface.

    Usage:
        solver = DiscreteLogAlgorithmSolver()
        result = solver.run(g=3, y=6, P=7, backend='torch')
        print(result['Found x'])  # 3
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        """Initialize the DLP solver.

        Args:
            text_mode: Output text mode ('plain' or 'legacy').
            algo_dir: Directory for output files (unused in manual implementation).
        """
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        g: int,
        y: int,
        P: int,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run the discrete logarithm algorithm.

        Args:
            g: Base element.
            y: Target value.
            P: Prime modulus.
            backend: Simulation backend.
            device: Compute device.
            dtype: Numeric dtype.

        Returns:
            Result dict with keys: 'Found x', 'Detected period r', 'status',
            'Computation time (s)', 'circuit_path', 'plot', 'circuit'.
        """
        result = solve_dlp(
            g=g,
            y=y,
            P=P,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )

        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Package results in the standard return format.

        Args:
            result: Raw result from solve_dlp.

        Returns:
            Formatted result dict matching DiscreteLogAlgorithm return schema.
        """
        return {
            "status": result.get("status", "failed"),
            "Found x": result.get("Found x"),
            "Detected period r": result.get("Detected period r"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 6. Utility: Test known cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Tuple[int, int, int, int]] = [
    # (g, y, P, expected_x)
    (3, 6, 7, 3),    # 3^3 = 27 ≡ 6 mod 7
    (2, 3, 5, 3),    # 2^3 = 8 ≡ 3 mod 5
    (2, 5, 11, 4),   # 2^4 = 16 ≡ 5 mod 11
    (3, 4, 7, 4),    # 3^4 = 81 ≡ 4 mod 7
    (5, 2, 7, 4),    # 5^4 = 625 ≡ 2 mod 7
    (2, 8, 13, 3),   # 2^3 = 8 ≡ 8 mod 13
    (3, 9, 13, 2),   # 3^2 = 9 ≡ 9 mod 13
    (2, 4, 7, 2),    # 2^2 = 4 ≡ 4 mod 7
]


def run_known_test(
    g: int,
    y: int,
    P: int,
    expected: int,
    backend: str = "torch",
    max_retries: int = 5,
) -> bool:
    """Run a single known test case with retries.

    Args:
        g: Base element.
        y: Target value.
        P: Prime modulus.
        expected: Expected discrete logarithm x.
        backend: Simulation backend.
        max_retries: Maximum number of attempts (algorithm is probabilistic).

    Returns:
        True if the test passes within max_retries, False otherwise.
    """
    for attempt in range(1, max_retries + 1):
        result = solve_dlp(g=g, y=y, P=P, backend=backend, verbose=False)
        if result["status"] == "ok" and result["Found x"] == expected:
            print(f"  PASS: {g}^{expected} ≡ {y} mod {P} "
                  f"(attempt {attempt}, {result['Computation time (s)']}s)")
            return True
        print(f"  Retry {attempt}/{max_retries}: status={result['status']}, "
              f"found={result['Found x']}")
    print(f"  FAIL: {g}^x ≡ {y} mod {P} (expected x={expected})")
    return False


# ---------------------------------------------------------------------------
# 7. Main (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Use the class-based interface for a quick demo
    print("=" * 60)
    print("Discrete Logarithm Algorithm — Manual Implementation")
    print("=" * 60)

    solver = DiscreteLogAlgorithmSolver()

    # Demo case from the SKILL.md
    print("\n--- Demo: 3^x ≡ 6 (mod 7) ---")
    result = solver.run(g=3, y=6, P=7, backend="torch")
    print(f"  x = {result['Found x']} (expected 3)")
    print(f"  Status: {result['status']}")
    print(f"  Period r: {result['Detected period r']}")
    print(f"  Time: {result['Computation time (s)']} s")
    print(f"  Verify: 3^{result['Found x']} mod 7 = {pow(3, result['Found x'], 7)}")

    # Run all known test cases
    print("\n--- Known Test Cases ---")
    passed = 0
    for g, y, P, expected in KNOWN_CASES:
        if run_known_test(g, y, P, expected):
            passed += 1
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
