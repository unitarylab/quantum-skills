"""Manual implementation of Quantum Amplitude Estimation (QAE).

Amplitude Estimation combines Grover/Amplitude Amplification with Quantum
Phase Estimation (QPE) to estimate the success probability `a` of a quantum
circuit U (i.e., U|0⟩ = √a|good⟩ + √(1-a)|bad⟩) with O(1/ε) circuit
evaluations — a quadratic speedup over the classical O(1/ε²) Monte Carlo.

Algorithm:
    1. Build the Grover operator Q = S_ψ ∘ S_f, which acts as a rotation
       by angle 2θ in the 2D good/bad subspace, with sin²θ = a.
    2. Construct a QPE circuit with d phase-register qubits, applying
       controlled-Q^(2^k) for k = 0..d-1, followed by the inverse QFT.
    3. Simulate the circuit; marginalize to get the phase histogram.
    4. Extract the most likely phase φ ≈ 2θ/(2π), fold to [0, 0.5],
       and recover â = sin²(π·φ).

Precision scales as δa ≈ π/2^d. Total qubits = d + n_data + 1.

Components:
    - build_iqft: Inverse Quantum Fourier Transform circuit
    - build_phase_oracle: Phase-kickback oracle for good states
    - build_grover_operator: Single Grover iteration G (with global phase fix)
    - build_qpe_circuit: Full QPE circuit around G
    - phase_histogram: Marginalize statevector to phase-register probabilities
    - amplitude_estimation: End-to-end QAE pipeline
    - AmplitudeEstimationSolver: Class-based interface

Reference:
    SKILL.md — Amplitude Estimation (QAE)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Inverse QFT circuit
# ---------------------------------------------------------------------------

def build_iqft(n: int, do_swaps: bool = True) -> Circuit:
    """Build the inverse Quantum Fourier Transform circuit.

    Structure:
        - (optional) Swap qubits: i ↔ n-1-i for i = 0..⌊n/2⌋-1
        - For each qubit j (0..n-1):
            - Controlled-phase rotations mcp(-π/2^(j-k)) from k to j
            - Hadamard on qubit j

    Args:
        n: Number of qubits.
        do_swaps: Whether to reverse qubit order via swaps at the start.

    Returns:
        iQFT Circuit of size n.
    """
    qc = Circuit(n, name=f"iQFT_{n}")
    if do_swaps:
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
    for j in range(n):
        for k in range(j):
            angle = -np.pi / (2 ** (j - k))
            qc.mcp(angle, k, j)  # controlled-phase rotation
        qc.h(j)
    return qc


# ---------------------------------------------------------------------------
# 2. Ancilla state management (phase kickback)
# ---------------------------------------------------------------------------

def _prepare_kickback_ancilla_minus(qc: Circuit, ancilla: int) -> None:
    """Prepare ancilla qubit in |−⟩ = H·X|0⟩ for phase kickback."""
    qc.x(ancilla)
    qc.h(ancilla)


def _unprepare_kickback_ancilla_minus(qc: Circuit, ancilla: int) -> None:
    """Restore ancilla qubit from |−⟩ back to |0⟩."""
    qc.h(ancilla)
    qc.x(ancilla)


# ---------------------------------------------------------------------------
# 3. Phase oracle (good-state marker)
# ---------------------------------------------------------------------------

def build_phase_oracle(
    n_data: int,
    good_zero_qubits: List[int],
) -> Circuit:
    """Build the phase-kickback oracle.

    Flips the phase of computational basis states where all `good_zero_qubits`
    are |0⟩. The oracle uses a kickback ancilla in the |−⟩ state:

        - Prepare ancilla in |−⟩
        - X-flip controlled qubits (|0⟩-control → |1⟩-control)
        - Multi-controlled-X targeting ancilla
        - Unflip controlled qubits
        - Unprepare ancilla

    Net effect on data register: |x⟩ → (-1)^{f(x)}|x⟩
    where f(x) = 1 iff all good_zero_qubits are |0⟩.

    Args:
        n_data: Number of data qubits.
        good_zero_qubits: Qubit indices that must be |0⟩ for the good state.

    Returns:
        Oracle Circuit of size n_data + 1.
    """
    qc = Circuit(n_data + 1, name="PhaseOracle")
    ancilla = n_data

    _prepare_kickback_ancilla_minus(qc, ancilla)

    # |0⟩-control → |1⟩-control
    for q in good_zero_qubits:
        qc.x(q)

    controls = list(good_zero_qubits)
    if len(controls) == 0:
        qc.x(ancilla)
    elif len(controls) == 1:
        qc.cx(controls[0], ancilla)
    else:
        qc.mcx(controls, ancilla)

    # Restore
    for q in good_zero_qubits:
        qc.x(q)

    _unprepare_kickback_ancilla_minus(qc, ancilla)
    return qc


# ---------------------------------------------------------------------------
# 4. Grover operator (oracle + diffuser + global phase correction)
# ---------------------------------------------------------------------------

def build_grover_operator(
    U: Circuit,
    good_zero_qubits: List[int],
) -> Circuit:
    """Build a single Grover iteration G = S_ψ ∘ S_f.

    In the 2D invariant subspace spanned by |good⟩ and |bad⟩, G acts as
    a rotation by angle 2θ with eigenvalues e^{±i2θ}.

    Steps:
        1. Oracle S_f: phase flip on good states.
        2. Diffuser S_ψ = U(2|0ⁿ⟩⟨0ⁿ| − I)U†: reflection about |ψ⟩.
        3. Global phase correction (−1): X-kick on ancilla so that
           controlled-G works correctly in QPE.

    Args:
        U: State preparation circuit (data qubits only).
        good_zero_qubits: Qubit indices defining the good state.

    Returns:
        Grover iteration Circuit of size n_data + 1.
    """
    n_data = U.get_num_qubits()
    ancilla = n_data
    data = list(range(n_data))

    qc = Circuit(n_data + 1, name="GroverIter")

    # --- Oracle S_f ---
    oracle = build_phase_oracle(n_data, good_zero_qubits)
    qc.append(oracle, list(range(n_data + 1)))

    # --- Diffuser S_ψ = U(2|0ⁿ⟩⟨0ⁿ|−I)U† ---
    qc.append(U.dagger(), data)
    all_zeros_oracle = build_phase_oracle(n_data, data)
    qc.append(all_zeros_oracle, list(range(n_data + 1)))
    qc.append(U, data)

    # --- Global phase correction (−1) ---
    # The diffuser sign convention introduces a (−1) factor.
    # This correction cancels it so controlled-G works in QPE.
    _prepare_kickback_ancilla_minus(qc, ancilla)
    qc.x(ancilla)
    _unprepare_kickback_ancilla_minus(qc, ancilla)

    return qc


# ---------------------------------------------------------------------------
# 5. QPE circuit around the Grover operator
# ---------------------------------------------------------------------------

def build_qpe_circuit(
    G: Circuit,
    d: int,
    prepare_target: Optional[Circuit] = None,
) -> Circuit:
    """Build the full QPE circuit around the Grover operator G.

    Layout:
        - Phase register: qubits 0..d-1
        - Target register: qubits d..d+n_target-1

    Steps:
        1. Append state preparation to target register.
        2. Apply H to all phase qubits (uniform superposition).
        3. Apply controlled-G^(2^k) for each phase qubit k.
        4. Append iQFT on the phase register.

    Args:
        G: Grover iteration circuit (n_target qubits).
        d: Number of phase register qubits (precision).
        prepare_target: Optional state preparation circuit for target.

    Returns:
        Full QPE Circuit of size d + n_target.
    """
    n_target = G.get_num_qubits()
    qc = Circuit(d + n_target, name=f"QPE_d{d}")

    phase = list(range(d))
    target = list(range(d, d + n_target))

    # State preparation on target
    if prepare_target is not None:
        qc.append(prepare_target, target)

    # Hadamard on all phase qubits
    for q in phase:
        qc.h(q)

    # Controlled powers of G: G^(2^k) for each phase qubit k
    for k in range(d):
        power = 2 ** k
        qc.append(
            G.repeat(power),
            target=target,
            control=phase[k],
            control_state='1',
        )

    # Inverse QFT
    iqft = build_iqft(d, do_swaps=True)
    qc.append(iqft, phase)

    return qc


# ---------------------------------------------------------------------------
# 6. Phase histogram extraction
# ---------------------------------------------------------------------------

def phase_histogram(statevector: np.ndarray, d: int) -> Dict[str, float]:
    """Marginalize all non-phase qubits from the full statevector.

    Groups probability by phase-register bit-string via `idx % 2^d`,
    producing a histogram sorted by descending probability.

    Args:
        statevector: Full complex statevector (flattened).
        d: Number of phase register qubits.

    Returns:
        Dict mapping phase bit-string → probability, sorted descending.
    """
    probs = np.abs(statevector) ** 2
    counts: Dict[str, float] = {}
    modulus = 2 ** d
    for idx, p in enumerate(probs):
        if p < 1e-12:
            continue
        k = idx % modulus
        bits = format(k, f'0{d}b')
        counts[bits] = counts.get(bits, 0.0) + float(p)
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# 7. End-to-end amplitude estimation
# ---------------------------------------------------------------------------

def amplitude_estimation(
    U: Circuit,
    good_zero_qubits: List[int],
    d: int = 6,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Quantum Amplitude Estimation.

    Pipeline:
        1. Build Grover operator G from U and good_zero_qubits.
        2. Build QPE circuit with d phase qubits around G.
        3. Simulate the circuit to get the statevector.
        4. Extract phase histogram; find the most likely phase.
        5. Fold phase to [0, 0.5] and compute â = sin²(π·φ).

    Args:
        U: State preparation circuit (data qubits only, no ancilla).
        good_zero_qubits: Qubit indices that must all be |0⟩ for good state.
        d: Number of phase register qubits. Precision scales as π/2^d.
        backend: Simulation backend ('torch' only).
        device: Compute device ('cpu' or 'cuda').
        dtype: Numerical dtype for simulation.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - Target amplitude: Estimated success probability â
            - Phase: Estimated phase φ ∈ [0, 0.5]
            - Most likely phase (bits): Bit-string of the top peak
            - Computation time (s): Wall-clock simulation time
            - Total qubits: d + n_data + 1
            - circuit: The constructed QPE Circuit
            - phase_histogram: Full phase histogram dict

    Raises:
        ValueError: If d < 1.
    """
    if d < 1:
        raise ValueError(f"Phase register size d must be >= 1, got {d}")

    # --- Stage 1: Parameter setup ---
    n_data = U.get_num_qubits()
    total_qubits = d + n_data + 1  # phase + data + Grover ancilla
    n_target = n_data + 1  # data + ancilla (G operates on this)

    if verbose:
        print(f"Amplitude Estimation (QAE)")
        print(f"  Phase register: {d} qubits")
        print(f"  Data register:  {n_data} qubits (+1 Grover ancilla)")
        print(f"  Total qubits:   {total_qubits}")
        print(f"  Good state:     qubits {good_zero_qubits} all |0⟩")
        print(f"  Resolution:     δa ≈ π/2^{d} = {math.pi / 2**d:.6f}")

    # --- Stage 2: Circuit construction ---
    if verbose:
        print(f"  Building Grover operator...")

    G = build_grover_operator(U, good_zero_qubits)

    # State preparation for target register (U on data qubits, ancilla in |0⟩)
    prepare = Circuit(n_target, name="prepare")
    prepare.append(U, list(range(n_data)))

    if verbose:
        print(f"  Building QPE circuit (d={d})...")

    qpe_circ = build_qpe_circuit(G, d=d, prepare_target=prepare)

    # --- Stage 3: Simulation ---
    if verbose:
        print(f"  Executing simulation...")

    t_start = time.perf_counter()
    state = qpe_circ.execute(backend=backend, device=device, dtype=dtype).state
    statevector = np.asarray(state, dtype=complex).reshape(-1)
    comp_time = time.perf_counter() - t_start

    # --- Stage 4: Classical post-processing ---
    histogram = phase_histogram(statevector, d=d)

    best_bits = next(iter(histogram))
    best_prob = histogram[best_bits]
    phi_raw = int(best_bits, 2) / (2 ** d)

    # Phase folding: QPE produces symmetric peaks at φ and 1-φ
    phi = min(phi_raw, 1.0 - phi_raw)

    # Amplitude recovery: a = sin²(π·φ)
    est_amp = float(np.sin(np.pi * phi) ** 2)

    if verbose:
        print(f"  Simulation time:        {comp_time:.4f}s")
        print(f"  Phase histogram entries: {len(histogram)}")
        print(f"  Best phase bits:        {best_bits} (prob={best_prob:.6f})")
        print(f"  Raw phase φ_raw:        {phi_raw:.6f}")
        print(f"  Folded phase φ:         {phi:.6f}")
        print(f"  Estimated amplitude â:  {est_amp:.6f}")
        print(f"  Status:                 ok")

    return {
        "status": "ok",
        "Target amplitude": est_amp,
        "Phase": phi,
        "Most likely phase (bits)": best_bits,
        "Phase histogram": histogram,
        "Computation time (s)": round(comp_time, 4),
        "Total qubits": total_qubits,
        "circuit": qpe_circ,
        "circuit_path": "",
        "plot": [],
    }


# ---------------------------------------------------------------------------
# 8. Class-based interface
# ---------------------------------------------------------------------------

class AmplitudeEstimationSolver:
    """Class-based solver for amplitude estimation.

    Usage:
        solver = AmplitudeEstimationSolver(text_mode='plain')
        result = solver.run(
            U=circuit, good_zero_qubits=[0], d=6,
        )
        print(result['Target amplitude'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        U: Circuit,
        good_zero_qubits: List[int],
        d: int = 6,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run amplitude estimation. See amplitude_estimation() for docs."""
        result = amplitude_estimation(
            U=U, good_zero_qubits=good_zero_qubits, d=d,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "Target amplitude": result.get("Target amplitude"),
            "Phase": result.get("Phase"),
            "Most likely phase (bits)": result.get("Most likely phase (bits)"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "Total qubits": result.get("Total qubits"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 9. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    # (theta, n_qubits, good_qubits, d, label)
    {"theta": np.pi / 8, "n_qubits": 1, "good_qubits": [0], "d": 5, "label": "n=1, d=5, p≈0.854"},
    {"theta": 0.5, "n_qubits": 1, "good_qubits": [0], "d": 5, "label": "n=1, d=5, p=cos²(0.5)"},
    {"theta": 0.8, "n_qubits": 1, "good_qubits": [0], "d": 6, "label": "n=1, d=6, p=cos²(0.8)"},
    {"theta": np.pi / 6, "n_qubits": 2, "good_qubits": [0, 1], "d": 5, "label": "n=2, d=5, p=cos⁴(π/6)"},
    {"theta": 1.0, "n_qubits": 2, "good_qubits": [0], "d": 5, "label": "n=2, d=5, single good qubit"},
]


def _build_test_state_prep(theta: float, n_qubits: int, good_qubits: List[int]) -> Circuit:
    """Build test state preparation: ry(2θ) on each good qubit.

    After this circuit, each good qubit has probability cos²(θ) of being |0⟩.
    Since qubits are independent, P(all good = |0⟩) = cos(θ)^(2·|good_qubits|).
    """
    U = Circuit(n_qubits, name="TestPrep")
    for q in good_qubits:
        U.ry(2 * theta, q)
    return U


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case and print result."""
    theta = case["theta"]
    n_qubits = case["n_qubits"]
    good_qubits = case["good_qubits"]
    d = case["d"]
    label = case["label"]

    true_p = float(np.cos(theta) ** (2 * len(good_qubits)))  # cos²(θ) per qubit → cos(θ)^(2n)

    U = _build_test_state_prep(theta, n_qubits, good_qubits)

    result = amplitude_estimation(
        U=U, good_zero_qubits=good_qubits, d=d,
        backend="torch", verbose=False,
    )

    est_amp = result["Target amplitude"]
    error = abs(est_amp - true_p)
    # Expected error bound: ~π/2^d
    max_error = math.pi / (2 ** d)

    ok = error < max_error * 1.5  # allow some margin
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {label}")
    print(f"       true_p={true_p:.6f}, est_p={est_amp:.6f}, error={error:.6f} "
          f"(bound≈{max_error:.6f}), φ={result['Phase']:.6f}")
    return ok


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Amplitude Estimation (QAE) — Manual Implementation")
    print("=" * 60)

    solver = AmplitudeEstimationSolver()

    # --- Demo 1: 1-qubit, known amplitude ---
    print("\n--- Demo 1: 1-qubit, cos²(π/8) ≈ 0.854 ---")
    theta1 = np.pi / 8
    true_p1 = float(np.cos(theta1) ** 2)
    U1 = Circuit(1, name="Prep1")
    U1.ry(2 * theta1, 0)

    result1 = solver.run(U=U1, good_zero_qubits=[0], d=6)
    print(f"  True amplitude:     {true_p1:.6f}")
    print(f"  Estimated amplitude: {result1['Target amplitude']:.6f}")
    print(f"  Error:              {abs(result1['Target amplitude'] - true_p1):.6e}")
    print(f"  Phase φ:            {result1['Phase']:.6f}")
    print(f"  Best bits:          {result1['Most likely phase (bits)']}")
    print(f"  Total qubits:       {result1['Total qubits']}")

    # --- Demo 2: Effect of d on precision ---
    print("\n--- Demo 2: Precision vs phase qubit count d ---")
    theta2 = 0.8
    true_p2 = float(np.cos(theta2) ** 2)
    U2 = Circuit(1, name="Prep2")
    U2.ry(2 * theta2, 0)

    for d_val in [3, 4, 5, 6, 7, 8]:
        result2 = amplitude_estimation(
            U=U2, good_zero_qubits=[0], d=d_val,
            backend="torch", verbose=False,
        )
        error2 = abs(result2["Target amplitude"] - true_p2)
        bound2 = math.pi / (2 ** d_val)
        print(f"  d={d_val}: est={result2['Target amplitude']:.6f}, "
              f"error={error2:.6e} (bound=π/2^{d_val}={bound2:.6f}), "
              f"bits={result2['Most likely phase (bits)']}")

    # --- Demo 3: 2-qubit example ---
    print("\n--- Demo 3: 2-qubit, |00⟩ target, cos⁴(1.0) ≈ 0.093 ---")
    theta3 = 1.0
    true_p3 = float(np.cos(theta3) ** 4)  # cos²(θ) per qubit, 2 qubits
    U3 = Circuit(2, name="Prep3")
    U3.ry(2 * theta3, 0)
    U3.ry(2 * theta3, 1)

    result3 = solver.run(U=U3, good_zero_qubits=[0, 1], d=6)
    print(f"  True amplitude:     {true_p3:.6f}")
    print(f"  Estimated amplitude: {result3['Target amplitude']:.6f}")
    print(f"  Error:              {abs(result3['Target amplitude'] - true_p3):.6e}")
    print(f"  Phase φ:            {result3['Phase']:.6f}")
    print(f"  Total qubits:       {result3['Total qubits']}")

    # --- Demo 4: Phase histogram details ---
    print("\n--- Demo 4: Phase Histogram (top entries) ---")
    theta4 = np.pi / 8
    U4 = Circuit(1, name="Prep4")
    U4.ry(2 * theta4, 0)

    result4 = amplitude_estimation(
        U=U4, good_zero_qubits=[0], d=5,
        backend="torch", verbose=False,
    )
    histogram4 = result4["Phase histogram"]
    for i, (bits, prob) in enumerate(histogram4.items()):
        if i >= 6:
            print(f"  ... ({len(histogram4) - 6} more entries)")
            break
        phi_raw = int(bits, 2) / (2 ** 5)
        phi_folded = min(phi_raw, 1.0 - phi_raw)
        print(f"  [{bits}] prob={prob:.6f}  φ_raw={phi_raw:.6f}  φ_folded={phi_folded:.6f}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")

    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
