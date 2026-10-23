"""Manual implementation of Quantum Phase Estimation (QPE).

Given a unitary operator U and one of its eigenstates |ψ⟩ satisfying
U|ψ⟩ = e^{2πiφ}|ψ⟩, QPE uses d auxiliary qubits and the inverse QFT
to extract a binary approximation of the phase φ with precision 1/2^d.

Algorithm:
    1. Initialize d-qubit phase register in |0⟩^d and target in |ψ⟩.
    2. Apply H to all phase qubits (uniform superposition).
    3. For k = 0..d-1: apply controlled-U^(2^k) from phase qubit k.
    4. Apply inverse QFT to phase register.
    5. Measure phase register → d-bit binary representation of φ.

The estimated phase is φ_est = int(best_bits, 2) / 2^d ∈ [0, 1).

Components:
    - build_qpe_circuit: Reusable QPE circuit builder (public API for HHL etc.)
    - qpe_run: End-to-end QPE pipeline
    - QPESolver: Class-based interface
    - phase_histogram_from_statevector: Manual marginalization fallback

Reference:
    SKILL.md — Quantum Phase Estimation (QPE)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit

# Try to import IQFT from the library; fall back to manual iQFT if unavailable
try:
    from unitarylab.library import IQFT as _IQFT

    _HAS_LIBRARY_IQFT = True
except ImportError:
    _HAS_LIBRARY_IQFT = False


# ---------------------------------------------------------------------------
# 1. Manual iQFT (fallback when unitarylab.library.IQFT is unavailable)
# ---------------------------------------------------------------------------

def _build_iqft(n: int) -> Circuit:
    """Build inverse QFT circuit manually.

    Used as fallback when unitarylab.library.IQFT is not available.

    Args:
        n: Number of qubits.

    Returns:
        iQFT Circuit.
    """
    qc = Circuit(n, name=f"iQFT_{n}")
    # Swap qubits
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    # Controlled-phase rotations + Hadamard
    for j in range(n):
        for k in range(j):
            angle = -np.pi / (2 ** (j - k))
            qc.mcp(angle, k, j)
        qc.h(j)
    return qc


def _get_iqft(d: int) -> Circuit:
    """Get iQFT circuit, preferring library version if available."""
    if _HAS_LIBRARY_IQFT:
        return _IQFT(d)
    return _build_iqft(d)


# ---------------------------------------------------------------------------
# 2. Phase histogram extraction
# ---------------------------------------------------------------------------

def phase_histogram_from_statevector(
    statevector: np.ndarray,
    d: int,
    threshold: float = 1e-12,
) -> Dict[str, float]:
    """Marginalize target register from full statevector.

    Groups probability by phase-register bit-string via `idx % 2^d`.
    Returns a dict sorted by descending probability.

    Args:
        statevector: Full flattened statevector.
        d: Number of phase register qubits.
        threshold: Minimum probability to include.

    Returns:
        Dict mapping phase bit-string → probability, sorted descending.
    """
    probs = np.abs(statevector) ** 2
    counts: Dict[str, float] = {}
    modulus = 1 << d
    for idx, p in enumerate(probs):
        if p < threshold:
            continue
        k = idx % modulus
        bits = format(k, f"0{d}b")
        counts[bits] = counts.get(bits, 0.0) + float(p)
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# 3. QPE circuit builder (reusable public API)
# ---------------------------------------------------------------------------

def build_qpe_circuit(
    U: Circuit,
    d: int,
    prepare_target: Optional[Circuit] = None,
) -> Circuit:
    """Build a standalone QPE circuit, suitable for embedding in other algorithms.

    Layout:
        - Phase register: qubits 0..d-1
        - Target register: qubits d..d+n_target-1

    Steps:
        1. Optionally append prepare_target to target register.
        2. Apply H to all phase qubits.
        3. Apply controlled-U^(2^k) from phase qubit k for k=0..d-1.
        4. Append iQFT to phase register.

    This method is designed to be called directly by parent algorithms
    (e.g., HHL, QAE) without going through the full run() pipeline.

    Args:
        U: Unitary operator whose phase is to be estimated.
        d: Number of phase register qubits. Precision = 1/2^d.
        prepare_target: Optional circuit preparing eigenstate |ψ⟩.
            If None, target starts in |0⟩^n (only correct if |0⟩
            is an eigenstate of U).

    Returns:
        Full QPE Circuit of size d + n_target.

    Raises:
        ValueError: If prepare_target qubit count doesn't match U.
    """
    n_target = U.get_num_qubits()
    qc = Circuit(d + n_target, name=f"QPE_d{d}")
    phase_qubits = list(range(d))
    target_qubits = list(range(d, d + n_target))

    # Eigenstate preparation
    if prepare_target is not None:
        if prepare_target.get_num_qubits() != n_target:
            raise ValueError(
                f"prepare_target has {prepare_target.get_num_qubits()} qubits, "
                f"but U expects {n_target}"
            )
        qc.append(prepare_target, target_qubits)

    # Uniform superposition on phase register
    for q in phase_qubits:
        qc.h(q)

    # Controlled powers of U: U^(2^k) for each phase qubit k
    for k in range(d):
        power = 1 << k  # 2^k
        qc.append(
            U.repeat(power),
            target=target_qubits,
            control=phase_qubits[k],
            control_state="1",
        )

    # Inverse QFT
    iqft = _get_iqft(d)
    qc.append(iqft, phase_qubits)

    return qc


# ---------------------------------------------------------------------------
# 4. End-to-end QPE
# ---------------------------------------------------------------------------

def qpe_run(
    U: Circuit,
    d: int,
    prepare_target: Optional[Circuit] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Quantum Phase Estimation.

    Pipeline:
        1. Validate parameters and compute qubit layout.
        2. Build QPE circuit via build_qpe_circuit().
        3. Execute statevector simulation.
        4. Extract phase histogram; pick best bit-string.
        5. Compute φ_est = int(best_bits, 2) / 2^d.

    Args:
        U: Unitary operator whose eigenphase is to be estimated.
        d: Number of phase register qubits. Precision = 1/2^d.
        prepare_target: Optional circuit preparing eigenstate |ψ⟩.
        backend: Simulation backend ('torch').
        device: Compute device ('cpu' or 'cuda').
        dtype: Numerical dtype.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - Estimated phase: Float φ ∈ [0, 1)
            - Best phase bit string: Binary string of best phase
            - Best phase probability: Probability of the best bits
            - Phase probabilities: List of top-3 (bits, prob) pairs
            - Computation time (s): Wall-clock simulation time
            - circuit: The full QPE Circuit
            - n_phase_qubits: d
            - n_target_qubits: U.get_num_qubits()

    Raises:
        ValueError: If d < 1 or prepare_target mismatch.
    """
    # --- Stage 1: Validation ---
    if d < 1:
        raise ValueError(f"Phase register size d must be >= 1, got {d}")

    n_target = U.get_num_qubits()
    total_qubits = d + n_target
    resolution = 1.0 / (1 << d)

    if verbose:
        print(f"Quantum Phase Estimation (QPE)")
        print(f"  Phase register:  {d} qubits")
        print(f"  Target register: {n_target} qubits")
        print(f"  Total qubits:    {total_qubits}")
        print(f"  Resolution:       1/2^{d} = {resolution:.6f}")

    # --- Stage 2: Circuit construction ---
    if verbose:
        print(f"  Building QPE circuit...")

    qc = build_qpe_circuit(U, d, prepare_target)

    if verbose:
        print(f"  Controlled-U^(2^k) sequence built")
        print(f"  iQFT appended (source: "
              f"{'library' if _HAS_LIBRARY_IQFT else 'manual'})")

    # --- Stage 3: Simulation ---
    if verbose:
        print(f"  Executing simulation...")

    t_start = time.perf_counter()
    final_state = qc.execute(backend=backend, device=device, dtype=dtype)
    statevector = np.asarray(final_state.state, dtype=complex).reshape(-1)
    comp_time = time.perf_counter() - t_start

    # --- Stage 4: Phase extraction ---
    # Try to use the built-in phase_probabilities; fall back to manual
    phase_qubits = list(range(d))
    try:
        phase_probs = final_state._phase_probabilities_from_state(
            phase_qubits, endian="little", threshold=1e-8,
        )
        use_builtin = True
    except (AttributeError, TypeError):
        phase_probs = phase_histogram_from_statevector(statevector, d, threshold=1e-8)
        use_builtin = False

    # Sort by probability descending
    sorted_phases = sorted(phase_probs.items(), key=lambda kv: kv[1], reverse=True)
    best_bits = sorted_phases[0][0]
    best_prob = sorted_phases[0][1]

    # Phase estimate: binary → decimal
    phi_est = int(best_bits, 2) / (1 << d)

    if verbose:
        print(f"  Simulation time:  {comp_time:.4f}s")
        print(f"  Extraction method: {'built-in' if use_builtin else 'manual'}")
        print(f"  Best phase bits:  |{best_bits}⟩ (prob={best_prob:.6f})")
        print(f"  Estimated φ:      {phi_est:.6f}")
        if len(sorted_phases) > 1:
            print(f"  Runner-up:        |{sorted_phases[1][0]}⟩ "
                  f"(prob={sorted_phases[1][1]:.6f})")
        print(f"  Status:           ok")

    return {
        "status": "ok",
        "Estimated phase": phi_est,
        "Best phase bit string": best_bits,
        "Best phase probability": best_prob,
        "Phase probabilities": sorted_phases[:3],
        "Computation time (s)": round(comp_time, 4),
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
        "n_phase_qubits": d,
        "n_target_qubits": n_target,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class QPESolver:
    """Class-based solver for Quantum Phase Estimation.

    Usage:
        solver = QPESolver(text_mode='plain')
        result = solver.run(U=circuit, d=6, prepare_target=prep)
        print(result['Estimated phase'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        U: Circuit,
        d: int,
        prepare_target: Optional[Circuit] = None,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run QPE. See qpe_run() for docs."""
        result = qpe_run(
            U=U, d=d, prepare_target=prepare_target,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "Estimated phase": result.get("Estimated phase"),
            "Best phase bit string": result.get("Best phase bit string"),
            "Best phase probability": result.get("Best phase probability"),
            "Phase probabilities": result.get("Phase probabilities"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "n_phase_qubits": result.get("n_phase_qubits"),
            "n_target_qubits": result.get("n_target_qubits"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 6. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    # (gate_type, eigenstate_type, d, expected_phi, label)
    {"gate": "S", "eigenstate": "|1⟩", "d": 4, "expected": 0.25,
     "label": "S gate, |1⟩ eigenstate, d=4, φ=0.25"},
    {"gate": "T", "eigenstate": "|1⟩", "d": 6, "expected": 0.125,
     "label": "T gate, |1⟩ eigenstate, d=6, φ=0.125"},
    {"gate": "Z", "eigenstate": "|1⟩", "d": 4, "expected": 0.5,
     "label": "Z gate, |1⟩ eigenstate, d=4, φ=0.5"},
    {"gate": "S", "eigenstate": "|1⟩", "d": 8, "expected": 0.25,
     "label": "S gate, |1⟩, d=8 (high precision)"},
    {"gate": "T", "eigenstate": "|1⟩", "d": 3, "expected": 0.125,
     "label": "T gate, |1⟩, d=3 (low res: 1/8 steps)"},
]


def _build_gate_unitary(gate_type: str) -> Circuit:
    """Build a single-qubit phase gate.

    S gate:  diag(1, i)   → phase 0.25 on |1⟩
    T gate:  diag(1, e^{iπ/4}) → phase 0.125 on |1⟩
    Z gate:  diag(1, -1)  → phase 0.5 on |1⟩
    """
    U = Circuit(1, name=f"{gate_type}_gate")
    if gate_type == "S":
        U.s(0)
    elif gate_type == "T":
        U.t(0)
    elif gate_type == "Z":
        U.z(0)
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
    return U


def _build_eigenstate_prep(eigenstate_type: str) -> Circuit:
    """Prepare eigenstate. |1⟩ = X|0⟩; |0⟩ = I|0⟩."""
    qc = Circuit(1, name=eigenstate_type)
    if eigenstate_type == "|1⟩":
        qc.x(0)
    elif eigenstate_type == "|0⟩":
        pass  # Default |0⟩ state
    else:
        raise ValueError(f"Unknown eigenstate: {eigenstate_type}")
    return qc


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case and print result."""
    gate_type = case["gate"]
    eigenstate_type = case["eigenstate"]
    d = case["d"]
    expected = case["expected"]
    label = case["label"]

    U = _build_gate_unitary(gate_type)
    prep = _build_eigenstate_prep(eigenstate_type)

    result = qpe_run(U=U, d=d, prepare_target=prep, backend="torch", verbose=False)

    est = result["Estimated phase"]
    resolution = 1.0 / (1 << d)
    error = abs(est - expected)
    # Exact if expected is representable in d bits; otherwise within 1 step
    ok = error <= resolution + 1e-10
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {label}")
    print(f"       expected={expected}, est={est:.6f}, error={error:.2e}, "
          f"best_bits={result['Best phase bit string']}, "
          f"prob={result['Best phase probability']:.4f}")
    return ok


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Quantum Phase Estimation (QPE) — Manual Implementation")
    print(f"  iQFT source: {'unitarylab.library.IQFT' if _HAS_LIBRARY_IQFT else 'manual build'}")
    print("=" * 60)

    solver = QPESolver()

    # --- Demo 1: S gate (φ = 0.25) ---
    print("\n--- Demo 1: S gate on |1⟩, d=4 (φ = 0.25 = 1/4) ---")
    U1 = Circuit(1, name="S_gate")
    U1.s(0)
    prep1 = Circuit(1, name="prep_|1⟩")
    prep1.x(0)

    result1 = solver.run(U=U1, d=4, prepare_target=prep1)
    print(f"  Expected φ:       0.25")
    print(f"  Estimated φ:      {result1['Estimated phase']:.6f}")
    print(f"  Best bits:        |{result1['Best phase bit string']}⟩")
    print(f"  Best probability: {result1['Best phase probability']:.4f}")
    print(f"  Error:            {abs(result1['Estimated phase'] - 0.25):.2e}")

    # --- Demo 2: T gate (φ = 0.125) ---
    print("\n--- Demo 2: T gate on |1⟩, d=6 (φ = 0.125 = 1/8) ---")
    U2 = Circuit(1, name="T_gate")
    U2.t(0)
    prep2 = Circuit(1, name="prep_|1⟩")
    prep2.x(0)

    result2 = solver.run(U=U2, d=6, prepare_target=prep2)
    print(f"  Expected φ:       0.125")
    print(f"  Estimated φ:      {result2['Estimated phase']:.6f}")
    print(f"  Best bits:        |{result2['Best phase bit string']}⟩")
    print(f"  Best probability: {result2['Best phase probability']:.4f}")

    # --- Demo 3: Z gate (φ = 0.5) ---
    print("\n--- Demo 3: Z gate on |1⟩, d=4 (φ = 0.5) ---")
    U3 = Circuit(1, name="Z_gate")
    U3.z(0)

    result3 = solver.run(U=U3, d=4, prepare_target=prep1)
    print(f"  Expected φ:       0.5")
    print(f"  Estimated φ:      {result3['Estimated phase']:.6f}")
    print(f"  Best bits:        |{result3['Best phase bit string']}⟩")

    # --- Demo 4: Precision scaling ---
    print("\n--- Demo 4: Precision vs d (S gate, φ=0.25) ---")
    for d_val in [2, 3, 4, 5, 6, 7, 8]:
        result4 = qpe_run(
            U=U1, d=d_val, prepare_target=prep1,
            backend="torch", verbose=False,
        )
        err4 = abs(result4["Estimated phase"] - 0.25)
        res4 = 1.0 / (1 << d_val)
        perfect = err4 < 1e-12
        marker = " ✓ exact" if perfect else ""
        print(f"  d={d_val}: φ={result4['Estimated phase']:.6f}, error={err4:.2e}, "
              f"resolution=1/2^{d_val}={res4:.6f}, bits={result4['Best phase bit string']}{marker}")

    # --- Demo 5: Phase probability distribution ---
    print("\n--- Demo 5: Top phase probabilities (T gate, d=5) ---")
    result5 = qpe_run(
        U=U2, d=5, prepare_target=prep2,
        backend="torch", verbose=False,
    )
    for rank, (bits, prob) in enumerate(result5["Phase probabilities"]):
        phi_val = int(bits, 2) / (1 << 5)
        print(f"  #{rank+1}: |{bits}⟩ → φ={phi_val:.4f}, prob={prob:.6f}")

    # --- Demo 6: build_qpe_circuit as subroutine ---
    print("\n--- Demo 6: build_qpe_circuit() as reusable subroutine ---")
    qpe_circ = build_qpe_circuit(U1, d=3, prepare_target=prep1)
    print(f"  Circuit name:      {qpe_circ.name}")
    print(f"  Total qubits:      {qpe_circ.get_num_qubits()}")
    state = qpe_circ.execute(backend="torch")
    sv = np.asarray(state.state, dtype=complex).reshape(-1)
    hist = phase_histogram_from_statevector(sv, 3)
    best_bits6 = next(iter(hist))
    phi6 = int(best_bits6, 2) / 8.0
    print(f"  Embedded QPE φ:    {phi6:.3f} (bits=|{best_bits6}⟩, prob={hist[best_bits6]:.4f})")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")

    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
