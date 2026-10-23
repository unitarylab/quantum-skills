"""Manual implementation of the Quantum Fourier Transform (QFT).

Maps |x⟩ → (1/√N) Σ_y e^{2πi x y / N} |y⟩ using Hadamard, controlled-phase,
and SWAP gates. The inverse QFT is obtained via circuit dagger.

Circuit construction (for n qubits):
    - H(i) for each qubit i from MSB to LSB.
    - MCP(π/2^{i-j}, j, i) for each lower-index control j.
    - SWAP(i, n-1-i) for bit-reversal.

Verification against NumPy:
    - QFT:  np.fft.ifft(state) * √N
    - IQFT: np.fft.fft(state) / √N

Components:
    - build_qft_circuit: Construct QFT/IQFT circuit
    - qft_apply: Apply QFT to a state and verify
    - QFTAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Quantum Fourier Transform (QFT)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. QFT circuit construction
# ---------------------------------------------------------------------------

def build_qft_circuit(n: int, inverse: bool = False) -> Circuit:
    """Build the QFT (or inverse QFT) circuit for n qubits.

    Construction (forward QFT):
        for i from n-1 down to 0:
            H(i)
            for j from i-1 down to 0:
                MCP(π / 2^{i-j}, control=j, target=i)
        for i from 0 to n//2 - 1:
            SWAP(i, n-1-i)

    Inverse QFT is obtained via qft.dagger().

    Args:
        n: Number of qubits.
        inverse: If True, return inverse QFT circuit.

    Returns:
        QFT or IQFT Circuit object.
    """
    qft = Circuit(n, name="QFT")

    # Phase estimation layers: H + controlled rotations
    for i in range(n - 1, -1, -1):
        qft.h(i)
        for j in range(i - 1, -1, -1):
            angle = math.pi / (1 << (i - j))  # π / 2^{i-j}
            qft.mcp(angle, j, i)

    # Bit-reversal swaps
    for i in range(n // 2):
        qft.swap(i, n - 1 - i)

    if inverse:
        qft = qft.dagger()
        qft.update_name("IQFT")
        qft.gate_sequence.update_name("IQFT")

    return qft


# ---------------------------------------------------------------------------
# 2. Matrix-level QFT (educational reference)
# ---------------------------------------------------------------------------

def qft_matrix(n: int) -> np.ndarray:
    """Compute the exact QFT matrix for n qubits.

    QFT[x, y] = (1/√N) · ω^{x·y}  where ω = e^{2πi/N}, N = 2^n.

    This is the mathematical target; the circuit implements this
    via the H + MCP + SWAP decomposition.

    Args:
        n: Number of qubits.

    Returns:
        N×N complex matrix where N = 2^n.
    """
    N = 1 << n
    omega = np.exp(2.0j * math.pi / N)
    Q = np.zeros((N, N), dtype=complex)
    for x in range(N):
        for y in range(N):
            Q[x, y] = omega ** (x * y)
    return Q / math.sqrt(N)


# ---------------------------------------------------------------------------
# 3. End-to-end QFT application
# ---------------------------------------------------------------------------

def qft_apply(
    state: Optional[np.ndarray] = None,
    n: int = 3,
    inverse: bool = False,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Apply QFT (or IQFT) to a quantum state.

    Pipeline:
        1. Validate state or start from |0...0⟩.
        2. Build QFT/IQFT circuit.
        3. Initialize state, append QFT, execute simulation.
        4. Verify against NumPy FFT/iFFT.

    Args:
        state: Optional initial state vector (length 2^n).
               If None, starts from |0...0⟩.
        n: Number of qubits (required if state is None).
        inverse: If True, apply inverse QFT.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress.

    Returns:
        Dict with: 'status', 'Final state', 'Expected state',
        'Verification error', 'Computation time (s)', 'circuit',
        'circuit_path', 'plot'.
    """
    N = 1 << n

    # --- Validate state ---
    if state is None:
        state = np.zeros(N, dtype=complex)
        state[0] = 1.0  # |0...0⟩
    else:
        state = np.asarray(state, dtype=complex)
        if state.shape != (N,):
            raise ValueError(f"State must have length 2^n = {N}, got {state.shape}")

    # Normalize
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        raise ValueError("State must be non-zero")
    state = state / norm

    if verbose:
        print(f"QFT (n={n}, inverse={inverse})")
        print(f"  Input state: {np.array2string(state, precision=3, suppress_small=True)}")

    t_start = time.perf_counter()

    # --- Build QFT circuit ---
    qft = build_qft_circuit(n, inverse=inverse)

    # --- Assemble and execute ---
    qc = Circuit(n, name="QFT Example")
    qc.initialize(state, list(range(n)))
    qc.append(qft, list(range(n)))

    result = qc.execute(backend=backend, device=device, dtype=dtype)
    final_state = np.asarray(result.state, dtype=complex)

    # --- Classical verification ---
    # QFT convention: verify against ifft(state) * √N
    # IQFT convention: verify against fft(state) / √N
    if inverse:
        expected = np.fft.fft(state) / math.sqrt(N)
    else:
        expected = np.fft.ifft(state) * math.sqrt(N)

    error = float(np.linalg.norm(final_state - expected))

    t_elapsed = time.perf_counter() - t_start

    is_success = error < 1e-7

    if verbose:
        print(f"\n  Results:")
        print(f"  Final state:    {np.array2string(final_state, precision=4, suppress_small=True)}")
        print(f"  Expected state: {np.array2string(expected, precision=4, suppress_small=True)}")
        print(f"  Verification error: {error:.6e}")
        print(f"  Status:         {'ok' if is_success else 'failed'}")
        print(f"  Time:           {t_elapsed:.4f}s")

    return {
        "status": "ok" if is_success else "failed",
        "Final state": final_state,
        "Expected state": expected,
        "Verification error": error,
        "Computation time (s)": round(t_elapsed, 4),
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
    }


# ---------------------------------------------------------------------------
# 4. Class-based interface
# ---------------------------------------------------------------------------

class QFTAlgorithmSolver:
    """Class-based solver matching the QFTAlgorithm interface.

    Usage:
        solver = QFTAlgorithmSolver(text_mode='plain')
        result = solver.run(n=3, state=state, inverse=False, backend='torch')
        print(result['Verification error'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        n: int = 3,
        state: Optional[np.ndarray] = None,
        inverse: bool = False,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        result = qft_apply(
            state=state, n=n, inverse=inverse,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Final state": result.get("Final state"),
            "Expected state": result.get("Expected state"),
            "Verification error": result.get("Verification error"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 5. Test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"n": 3, "state": None, "inverse": False, "label": "n=3, |000⟩, QFT"},
    {"n": 3, "state": None, "inverse": True, "label": "n=3, |000⟩, IQFT"},
    {"n": 4, "state": None, "inverse": False, "label": "n=4, |0000⟩, QFT"},
    {"n": 4, "state": None, "inverse": True, "label": "n=4, |0000⟩, IQFT"},
]


def make_basis_state(n: int, k: int) -> np.ndarray:
    """Create the |k⟩ basis state vector for n qubits."""
    N = 1 << n
    state = np.zeros(N, dtype=complex)
    state[k] = 1.0
    return state


def run_known_test(case: Dict[str, Any]) -> bool:
    n, inv = case["n"], case["inverse"]
    state = case["state"]
    if state is None:
        state = make_basis_state(n, 0)
    result = qft_apply(state=state, n=n, inverse=inv, verbose=False)
    ok = result["status"] == "ok"
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] {case['label']}")
    print(f"       Error: {result['Verification error']:.6e}, time={result['Computation time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Quantum Fourier Transform — Manual Implementation")
    print("=" * 60)

    solver = QFTAlgorithmSolver()

    # Demo from the SKILL.md
    print("\n--- Demo: n=3, |001⟩, QFT ---")
    state_demo = np.zeros(8, dtype=complex)
    state_demo[1] = 1.0  # |001⟩
    result = solver.run(n=3, state=state_demo, inverse=False)
    print(f"  Verification error: {result['Verification error']:.2e}")
    print(f"  Status:             {result['status']}")

    # QFT ↔ IQFT roundtrip
    print("\n--- Roundtrip: QFT → IQFT = Identity ---")
    state_orig = make_basis_state(3, 3)  # |011⟩
    # Apply QFT then IQFT
    r_qft = qft_apply(state=state_orig, n=3, inverse=False, verbose=False)
    state_after_qft = r_qft["Final state"]
    r_iqft = qft_apply(state=state_after_qft, n=3, inverse=True, verbose=False)
    roundtrip_err = float(np.linalg.norm(r_iqft["Final state"] - state_orig))
    print(f"  Original:   {np.array2string(state_orig, precision=3, suppress_small=True)}")
    print(f"  After QFT→IQFT: {np.array2string(r_iqft['Final state'], precision=4, suppress_small=True)}")
    print(f"  Roundtrip error: {roundtrip_err:.2e}")

    # QFT matrix comparison
    print("\n--- QFT Matrix (n=2) — Circuit vs Exact ---")
    Q_exact = qft_matrix(2)
    print(f"  Exact QFT matrix:")
    for row in Q_exact:
        print(f"    {np.array2string(row, precision=4, suppress_small=True)}")

    # Circuit-based matrix verification (each basis state)
    max_err = 0.0
    for k in range(4):
        bs = make_basis_state(2, k)
        r = qft_apply(state=bs, n=2, inverse=False, verbose=False)
        expected_col = Q_exact[:, k]  # QFT|k⟩ is k-th column
        err = float(np.linalg.norm(r["Final state"] - expected_col))
        max_err = max(max_err, err)
    print(f"  Max column error (n=2): {max_err:.2e}")

    # Test cases
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))

    # Additional: superposition input
    print(f"  Additional tests:")
    sup_state = np.ones(8, dtype=complex) / math.sqrt(8)  # uniform superposition
    r = qft_apply(state=sup_state, n=3, inverse=False, verbose=False)
    ok = r["status"] == "ok"
    print(f"  [{'✓' if ok else '✗'}] n=3, uniform superposition, QFT")
    print(f"       Error: {r['Verification error']:.6e}")
    if ok:
        passed += 1

    total = len(KNOWN_CASES) + 1
    print(f"\n  Result: {passed}/{total} tests passed")
    sys.exit(0 if passed == total else 1)
