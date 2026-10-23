"""Manual implementation of Simon's algorithm.

Solves Simon's problem: given a black-box function f: {0,1}^n → {0,1}^n with
the promise that f(x) = f(y) iff x ⊕ y = s for some hidden string s, find s
using O(n) quantum queries — an exponential speedup over classical Ω(2^{n/2}).

Components:
    - build_simon_oracle: CNOT-based U_f oracle for hidden string s
    - build_simon_circuit: Full Simon circuit (H → oracle → measure → H)
    - extract_basis: Greedy pivot selection for GF(2) linear independence
    - solve_simon: Back-substitution over GF(2) to recover s
    - simon_find: End-to-end solver
    - SimonAlgorithmSolver: Class-based interface matching unitarylab pattern

Reference:
    SKILL.md — Simon's Algorithm
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit, ClassicalRegister, Register


# ---------------------------------------------------------------------------
# 1. Simon oracle
# ---------------------------------------------------------------------------

def build_simon_oracle(qc: Circuit, s: str, n: int) -> None:
    """Build the U_f oracle for Simon's problem with hidden string s.

    Structure:
        1. Copy x → y via CX gates: for each bit i, cx(i, i+n).
        2. XOR the pivot column into all positions where s[i] = '1'.
           This creates the mapping f(x) = f(x ⊕ s).

    The pivot is the leftmost '1' in s. For each position i where s[i]='1',
    the x-qubit at (n-1-pivot_idx) controls the y-qubit at (n-1-i+n).
    This ensures f(x) = f(x ⊕ s) because XORing the pivot bit flips all
    y-bits that correspond to '1' positions in s.

    Args:
        qc: Circuit to add the oracle gates to.
        s: Hidden binary string (e.g., '1010').
        n: Number of bits (len(s)).
    """
    # Step 1: Copy x → y via CX
    for i in range(n):
        qc.cx(i, i + n)

    # Step 2: Find pivot (leftmost '1' in s)
    pivot_idx = s.find("1")
    if pivot_idx < 0:
        return  # Trivial case: s is all zeros, nothing to do

    # XOR pivot column into each position where s[i] = '1'
    for i in range(n):
        if s[i] == "1":
            # Control: x-qubit at (n-1-pivot_idx)
            # Target:  y-qubit at (n-1-i+n)
            qc.cx(n - 1 - pivot_idx, n - 1 - i + n)


# ---------------------------------------------------------------------------
# 2. Circuit construction
# ---------------------------------------------------------------------------

def build_simon_circuit(s: str) -> Circuit:
    """Build the full Simon circuit for hidden string s.

    Register layout:
        x (input):   qubits [0, n)       — input register
        y (output):  qubits [n, 2n)      — output register (for oracle)
        c (classical): n bits            — classical register for mid-circuit measurement

    Circuit steps:
        1. Hadamard on all x qubits.
        2. Simon oracle U_f (CNOT-based).
        3. Mid-circuit measurement: measure y → c (collapses superposition).
        4. Hadamard on all x qubits again (interference).

    Args:
        s: Hidden binary string.

    Returns:
        The constructed Circuit object.

    Raises:
        ValueError: If s is all zeros.
    """
    if all(bit == "0" for bit in s):
        raise ValueError("Hidden string s must contain at least one '1'")

    n = len(s)

    # Register definitions
    rx = Register("x", n)
    ry = Register("y", n)
    cqr = ClassicalRegister("c", n)
    qc = Circuit(rx, ry, cqr, name=f"Simon_s{s}")

    # Step 1: Hadamard on all x qubits
    qc.h(list(range(n)))

    # Step 2: Simon oracle
    build_simon_oracle(qc, s, n)

    # Step 3: Mid-circuit measurement of y register → classical register
    qc.measure(list(range(n, 2 * n)), list(range(n)))

    # Step 4: Hadamard on all x qubits again
    qc.h(list(range(n)))

    return qc


# ---------------------------------------------------------------------------
# 3. GF(2) linear algebra post-processing
# ---------------------------------------------------------------------------

def extract_basis(state_list: List[str], n: int) -> List[str]:
    """Greedy pivot selection: collect up to n-1 linearly independent vectors.

    Given a list of measured bitstrings (from the x register), selects
    vectors that are linearly independent over GF(2) using the position
    of the leftmost '1' as the pivot.

    Each measured y satisfies y·s ≡ 0 (mod 2). We need n-1 linearly
    independent such equations to uniquely determine s.

    Args:
        state_list: List of binary strings measured from the x register.
        n: Number of bits per string.

    Returns:
        List of up to n-1 linearly independent vectors (bitstrings).
    """
    basis: Dict[int, str] = {}
    for bits in state_list:
        pivot = bits.find("1")
        if pivot >= 0 and pivot not in basis:
            basis[pivot] = bits
        # Stop once we have enough independent vectors
        if len(basis) >= n - 1:
            break
    return list(basis.values())


def solve_simon(basis: List[str], n: int) -> str:
    """Back-substitution over GF(2) to recover the hidden string s.

    Given n-1 linearly independent vectors y such that y·s = 0 (mod 2),
    solve for s. The solution space is 1-dimensional (a single vector up
    to scaling), and we find the unique non-zero vector in the null space.

    Algorithm:
        1. Build a dictionary mapping pivot position → row vector.
        2. Identify the "free variable" (position without a pivot).
        3. Set the free variable to '1'.
        4. Back-substitute from rightmost pivot to leftmost:
           For each pivot position, compute the dot product of known bits
           with the row, and solve for the pivot bit.

    Args:
        basis: List of n-1 linearly independent binary strings.
        n: Number of bits.

    Returns:
        The recovered hidden string s.

    Raises:
        ValueError: If basis has fewer than n-1 vectors.
    """
    if len(basis) < n - 1:
        raise ValueError(
            f"Need at least {n - 1} independent vectors, got {len(basis)}"
        )

    # Map pivot position → row vector
    pivot_positions: Dict[int, str] = {}
    for row in basis:
        pivot = row.find("1")
        if pivot >= 0:
            pivot_positions[pivot] = row

    # Find the free variable (position without a pivot)
    all_positions = set(range(n))
    free_vars = all_positions - set(pivot_positions.keys())

    # Initialize s with '0' everywhere
    s_chars = ["0"] * n

    # Set free variable(s) to '1'
    for pos in free_vars:
        s_chars[pos] = "1"

    # Back-substitute from rightmost pivot to leftmost
    for pivot_pos in sorted(pivot_positions.keys(), reverse=True):
        row = pivot_positions[pivot_pos]
        # Compute: val = Σ_{j ≠ pivot_pos} row[j] * s[j]  (mod 2)
        val = 0
        for j in range(n):
            if j != pivot_pos and row[j] == "1" and s_chars[j] == "1":
                val ^= 1  # XOR (addition mod 2)
        # For y·s = 0: row[pivot_pos]*s[pivot_pos] = val (mod 2)
        # Since row[pivot_pos] = 1: s[pivot_pos] = val
        s_chars[pivot_pos] = str(val)

    return "".join(s_chars)


# ---------------------------------------------------------------------------
# 4. End-to-end solver
# ---------------------------------------------------------------------------

def simon_find(
    s_target: str,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Simon's algorithm to find a hidden binary string s.

    Full pipeline:
        1. Validate input (s must have at least one '1').
        2. Build circuit: H → oracle → mid-circuit measure → H.
        3. Execute simulation (torch backend required for mid-circuit meas).
        4. Extract x-register bitstrings from simulation result.
        5. Greedy pivot selection for n-1 linearly independent vectors.
        6. Back-substitution over GF(2) to recover s.

    Args:
        s_target: The hidden binary string to find (e.g., '1010').
        backend: Simulation backend (must be 'torch' for mid-circuit meas).
        device: Compute device (default 'cpu').
        dtype: Numeric dtype for simulation.
        verbose: Print progress information.

    Returns:
        Dict with keys: 'Computed s', 'status', 'Register size',
        'Equations', 'Valid states', 'computation time (s)', 'circuit'.

    Raises:
        ValueError: If s_target is all zeros.
        ValueError: If backend is not 'torch'.
    """
    # --- Validation ---
    if backend != "torch":
        raise ValueError(
            f"Simon's algorithm requires backend='torch' "
            f"(mid-circuit measurement). Got: '{backend}'"
        )

    if all(bit == "0" for bit in s_target):
        raise ValueError("Hidden string s must contain at least one '1'")

    n = len(s_target)

    if verbose:
        print(f"Simon: finding hidden string of length n = {n}")
        print(f"  Target s: {s_target}")
        print(f"  Qubits: {2 * n} (x={n}, y={n}) + {n} classical bits")

    # --- Circuit construction & simulation ---
    t_start = time.perf_counter()

    qc = build_simon_circuit(s_target)

    if verbose:
        print(f"  Circuit: {qc.name}")

    # Execute (must use torch for mid-circuit measurement)
    result = qc.execute(backend=backend, device=device, dtype=dtype)

    # Extract probability distribution over x register (qubits [0, n))
    state_dict = result.calculate_state(list(range(n)))

    t_quantum = time.perf_counter() - t_start

    if verbose:
        print(f"  Valid states measured: {len(state_dict)}")
        print(f"  Quantum time: {t_quantum:.4f} s")

    # --- Classical post-processing ---
    # Get list of bitstrings (keys of the state dict)
    bitstrings = list(state_dict.keys())

    # Extract n-1 linearly independent vectors
    basis = extract_basis(bitstrings, n)

    if verbose:
        print(f"  Independent equations: {len(basis)}")
        for i, vec in enumerate(basis):
            print(f"    eq[{i}]: {vec}")

    # Solve for s via back-substitution over GF(2)
    found_s = solve_simon(basis, n)

    t_total = time.perf_counter() - t_start

    # --- Success check ---
    is_success = found_s == s_target

    if verbose:
        print(f"  Found s:  {found_s}")
        print(f"  Target s: {s_target}")
        print(f"  Match:    {'✓' if is_success else '✗'}")
        print(f"  Total time: {t_total:.4f} s")

    return {
        "Computed s": found_s,
        "status": "ok" if is_success else "failed",
        "Register size": n,
        "Equations": len(basis),
        "Valid states": len(state_dict),
        "computation time (s)": round(t_quantum, 4),
        "circuit": qc,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface (matching SimonAlgorithm pattern)
# ---------------------------------------------------------------------------

class SimonAlgorithmSolver:
    """Class-based solver matching the SimonAlgorithm interface.

    Usage:
        solver = SimonAlgorithmSolver()
        result = solver.run(s='1010', backend='torch')
        print(result['Computed s'])  # '1010'
    """

    def __init__(
        self,
        text_mode: str = "plain",
        algo_dir: Optional[str] = None,
    ):
        """Initialize the Simon solver.

        Args:
            text_mode: Output text mode ('plain' or 'legacy').
            algo_dir: Directory for output files.
        """
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        s: str,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run Simon's algorithm.

        Args:
            s: Hidden binary string to find.
            backend: Simulation backend (must be 'torch').
            device: Compute device.
            dtype: Numeric dtype.

        Returns:
            Result dict with keys: 'Computed s', 'status', 'circuit_path',
            'plot', 'circuit', 'Valid states', 'computation time (s)',
            'Register size', 'Equations'.
        """
        result = simon_find(
            s_target=s,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )

        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Package results in the standard SimonAlgorithm return format.

        Args:
            result: Raw result from simon_find.

        Returns:
            Formatted result dict matching SimonAlgorithm return schema.
        """
        return {
            "Computed s": result.get("Computed s"),
            "status": result.get("status", "failed"),
            "circuit_path": "",
            "plot": [],
            "circuit": result.get("circuit"),
            "Valid states": result.get("Valid states", 0),
            "computation time (s)": result.get("computation time (s)", 0.0),
            "Register size": result.get("Register size", 0),
            "Equations": result.get("Equations", 0),
        }


# ---------------------------------------------------------------------------
# 6. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[str] = [
    "1010",
    "110",
    "101010",
    "1111",
    "1001",
    "11011",
    "101101",
    "1000001",
]


def run_known_test(s_target: str, backend: str = "torch") -> bool:
    """Run a single known test case.

    Args:
        s_target: Hidden binary string.
        backend: Simulation backend.

    Returns:
        True if s is correctly recovered.
    """
    result = simon_find(s_target=s_target, backend=backend, verbose=False)
    found = result["Computed s"]
    ok = result["status"] == "ok"
    n = len(s_target)
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] s='{s_target}' (n={n}) → found='{found}' "
          f"({result['Valid states']} states, {result['Equations']} eqs, "
          f"{result['computation time (s)']}s)")
    return ok


# ---------------------------------------------------------------------------
# 7. Main (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Simon's Algorithm — Manual Implementation")
    print("=" * 60)

    solver = SimonAlgorithmSolver()

    # Demo case from the SKILL.md
    print("\n--- Demo: Find hidden string s = '1010' ---")
    result = solver.run(s="1010", backend="torch")
    print(f"  Computed s:   {result['Computed s']}")
    print(f"  Status:       {result['status']}")
    print(f"  Register size: {result['Register size']}")
    print(f"  Equations:    {result['Equations']}")
    print(f"  Valid states: {result['Valid states']}")
    print(f"  Time:         {result['computation time (s)']} s")

    # Run all known test cases
    print("\n--- Known Test Cases ---")
    passed = 0
    for s in KNOWN_CASES:
        if run_known_test(s):
            passed += 1
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
