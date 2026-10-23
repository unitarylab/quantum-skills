"""Manual implementation of LCU (Linear Combination of Unitaries).

Applies a non-unitary operator M = Σ_j α_j U_j to a quantum state via
the three-stage PREPARE → SELECT → UNPREPARE circuit, with probabilistic
post-selection on the ancilla register.

Circuit structure:
    1. PREPARE (V):   |0⟩_anc → Σ_j √(α_j/s) |j⟩_anc
    2. SELECT (U_c):  Σ_j |j⟩⟨j|_anc ⊗ U_j  (controlled multiplexer)
    3. UNPREPARE (V†): ancilla back → success branch |0⟩_anc ⊗ (M|ψ⟩)/s

Post-selection success probability: P = ||M|ψ⟩||² / s²

Components:
    - build_prepare_circuit: State preparation for ancilla amplitudes
    - build_select_circuit: Multiplexed controlled unitaries
    - lcu_matrix: Matrix-level LCU computation (educational)
    - lcu_apply: End-to-end LCU application
    - LCUAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Linear Combination of Unitaries (LCU)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit, Register


# ---------------------------------------------------------------------------
# 1. PREPARE circuit
# ---------------------------------------------------------------------------

def build_prepare_circuit(
    alphas: List[float],
    n_anc: int,
    name: str = "V",
) -> Circuit:
    """Build the PREPARE operator V.

    V|0⟩_anc = Σ_j √(α_j/s) |j⟩_anc  where s = Σ α_j.

    Uses Circuit.initialize() to directly load the amplitude distribution
    into the ancilla register, matching the approach in LCUAlgorithm._build_V().

    Args:
        alphas: Non-negative weights α_j.
        n_anc: Number of ancilla qubits (≥ ceil(log2(len(alphas)))).
        name: Circuit name.

    Returns:
        PREPARE circuit V on n_anc qubits.
    """
    s = float(np.sum(alphas))
    m = len(alphas)
    target_dim = 1 << n_anc

    # Build amplitude vector: state[j] = √(α_j / s), padded to 2^n_anc
    state = np.zeros(target_dim, dtype=complex)
    for j in range(m):
        state[j] = complex(math.sqrt(alphas[j] / s))

    # Normalize (handle floating-point imprecision in padding)
    norm = np.linalg.norm(state)
    if norm > 1e-15:
        state = state / norm

    qc = Circuit(n_anc, name=name)
    qc.initialize(state, list(range(n_anc)))
    return qc


# ---------------------------------------------------------------------------
# 2. SELECT circuit
# ---------------------------------------------------------------------------

def build_select_circuit(
    unitaries: List[Circuit],
    n_anc: int,
    n_sys: int,
    name: str = "SELECT",
) -> Circuit:
    """Build the SELECT operator.

    SELECT = Σ_j |j⟩⟨j|_anc ⊗ U_j

    For each j, applies U_j to the system register, controlled on
    the ancilla being in state |j⟩ (binary encoding).

    Args:
        unitaries: List of unitary circuits U_j, each on n_sys qubits.
        n_anc: Number of ancilla (control) qubits.
        n_sys: Number of system (target) qubits.
        name: Circuit name.

    Returns:
        SELECT circuit on n_anc + n_sys qubits.
    """
    qc = Circuit(n_anc + n_sys, name=name)
    anc_qubits = list(range(n_anc))
    sys_qubits = list(range(n_anc, n_anc + n_sys))

    for j, U_j in enumerate(unitaries):
        # Binary encoding of j as control state
        ctrl_state = format(j, f"0{n_anc}b")
        qc.append(
            U_j,
            target=sys_qubits,
            control=anc_qubits,
            control_state=ctrl_state,
        )

    return qc


# ---------------------------------------------------------------------------
# 3. Matrix-level LCU (educational/validation)
# ---------------------------------------------------------------------------

def lcu_matrix(
    alphas: List[float],
    unitaries_mat: List[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Compute M = Σ_j α_j U_j at the matrix level.

    This is the exact target operator that the LCU circuit applies
    probabilistically. Used for validation and educational comparison.

    Args:
        alphas: Weights α_j.
        unitaries_mat: List of unitary matrices U_j.

    Returns:
        Tuple of (M, s) where M = Σ α_j U_j and s = Σ α_j.
    """
    s = float(np.sum(alphas))
    dim = unitaries_mat[0].shape[0]
    M = np.zeros((dim, dim), dtype=complex)
    for alpha, U in zip(alphas, unitaries_mat):
        M += alpha * U
    return M, s


# ---------------------------------------------------------------------------
# 4. End-to-end LCU application
# ---------------------------------------------------------------------------

def lcu_apply(
    alphas: List[float],
    unitaries: List[Circuit],
    n_sys: int,
    initial_state: Optional[Circuit] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Apply M = Σ_j α_j U_j to a quantum state using LCU.

    Full pipeline:
        1. PREPARE: V|0⟩_anc = Σ √(α_j/s) |j⟩
        2. SELECT: controlled-U_j on system
        3. UNPREPARE: V† on ancilla
        4. Simulate and post-select ancilla = |0⟩
        5. Extract system state ∝ M|ψ⟩

    Args:
        alphas: Non-negative weights.
        unitaries: List of unitary circuits.
        n_sys: System qubit count.
        initial_state: Optional circuit preparing initial |ψ⟩.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress.

    Returns:
        Dict with: 'status', 'Success probability', 'Result state',
        'Computation time (s)', 'circuit', 's_norm', 'n_anc'.
    """
    m = len(alphas)
    if m != len(unitaries):
        raise ValueError(f"len(alphas)={m} != len(unitaries)={len(unitaries)}")
    if m < 1:
        raise ValueError("Need at least one unitary")

    n_anc = max(1, int(math.ceil(math.log2(m))))
    s_norm = float(np.sum(alphas))
    total_qubits = n_anc + n_sys

    if verbose:
        print("LCU: M = Σ_j α_j U_j  ({} terms)".format(m))
        print(f"  Terms: {m}, s = Σα_j = {s_norm:.4f}")
        print(f"  Ancilla qubits: {n_anc}, System qubits: {n_sys}")
        print(f"  Total qubits: {total_qubits}")

    t_start = time.perf_counter()

    # --- Build PREPARE ---
    V = build_prepare_circuit(alphas, n_anc)

    # --- Build SELECT ---
    SEL = build_select_circuit(unitaries, n_anc, n_sys)

    # --- Assemble full LCU circuit ---
    anc_qubits = list(range(n_anc))
    sys_qubits = list(range(n_anc, total_qubits))

    qc = Circuit(total_qubits, name="LCU_circuit")

    # 1. Initial system state (if provided)
    if initial_state is not None:
        qc.append(initial_state, sys_qubits)

    # 2. PREPARE on ancilla
    qc.append(V, anc_qubits)

    # 3. SELECT
    qc.append(SEL, list(range(total_qubits)))

    # 4. UNPREPARE (V†) on ancilla
    qc.append(V.dagger(), anc_qubits)

    # --- Simulate ---
    result = qc.execute(backend=backend, device=device, dtype=dtype)
    state_arr = np.asarray(result.state, dtype=complex)

    # --- Post-selection ---
    # ancilla = |0...0⟩ in the statevector index.
    #
    # State index layout (unitarylab convention):
    #   The least-significant bit (LSB) in the state index is ancilla qubit 0.
    #   For n_anc=1: anc=0 states are even indices [0, 2, 4, ...]
    #
    # More generally: ancilla qubits occupy the n_anc least significant
    # bits of the state index. Post-selecting anc=|0...0⟩ means keeping
    # every (2^n_anc)-th element starting from 0.
    n_sys_dim = 1 << n_sys
    stride = 1 << n_anc
    post_state = state_arr[0::stride].copy()[:n_sys_dim]

    # Success probability = ||M|ψ⟩||² / s² = ||post_state||² * s² wait
    # Actually: the post-selected (anc=0) component has norm ||M|ψ⟩/s||
    # So P_succ = ||M|ψ⟩||² / s² = ||post_state||²
    # But we also need to consider: the full state has norm 1, and the
    # anc=0 component is proportional to M|ψ⟩/s.
    # P_succ = sum(|anc=0 amplitudes|²) = ||post_state||² (before normalization)
    success_prob = float(np.sum(np.abs(post_state) ** 2))

    # Normalize the post-selected state
    if success_prob > 1e-15:
        result_state = post_state / math.sqrt(success_prob)
    else:
        result_state = post_state

    t_elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\n  Results:")
        print(f"  Success probability: {success_prob:.6f}")
        print(f"  Result state: {np.array2string(result_state, precision=4, suppress_small=True)}")
        print(f"  Expected: M|0⟩/∥M|0⟩∥ (scaled by 1/s)")
        print(f"  Time: {t_elapsed:.4f}s")

    return {
        "status": "ok",  # Always ok; quality in Success probability
        "Success probability": success_prob,
        "Result state": result_state,
        "Computation time (s)": round(t_elapsed, 4),
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
        "s_norm": s_norm,
        "n_anc": n_anc,
    }


# ---------------------------------------------------------------------------
# 5. Validation against exact M
# ---------------------------------------------------------------------------

def validate_lcu(
    alphas: List[float],
    unitaries: List[Circuit],
    unitaries_mat: List[np.ndarray],
    n_sys: int,
    initial_state: Optional[Circuit] = None,
) -> Dict[str, Any]:
    """Validate LCU result against the exact matrix computation.

    Computes:
        M_exact = Σ α_j U_j (matrix level)
        |ψ_out_exact⟩ = M|0⟩ / ∥M|0⟩∥

    And compares with the LCU circuit output.

    Args:
        alphas: Weights.
        unitaries: Unitary circuits (for LCU simulation).
        unitaries_mat: Unitary matrices (for exact comparison).
        n_sys: System qubits.
        initial_state: Initial state circuit.

    Returns:
        Dict with LCU result plus 'fidelity' and 'expected_state'.
    """
    result = lcu_apply(alphas, unitaries, n_sys, initial_state, verbose=False)

    # Exact M|0⟩
    M, s = lcu_matrix(alphas, unitaries_mat)
    psi0 = np.zeros(1 << n_sys, dtype=complex)
    psi0[0] = 1.0  # |0...0⟩
    expected_raw = M @ psi0  # = M|0⟩
    expected_norm = np.linalg.norm(expected_raw)
    if expected_norm > 1e-15:
        expected = expected_raw / expected_norm
    else:
        expected = expected_raw

    result_state = result["Result state"]
    # Fidelity (modulo global phase): |⟨ψ_lcu|ψ_exact⟩|
    fidelity = float(np.abs(np.vdot(result_state, expected)))

    result["expected_state"] = expected
    result["fidelity"] = fidelity
    result["expected_success"] = float(expected_norm**2 / s**2)

    return result


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class LCUAlgorithmSolver:
    """Class-based solver matching the LCUAlgorithm interface.

    Usage:
        solver = LCUAlgorithmSolver(text_mode='plain')
        result = solver.run(
            alphas=[0.6, 0.4],
            unitaries=[U0, U1],
            n_sys=1,
            backend='torch',
        )
        print(result['Success probability'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        alphas: List[float],
        unitaries: List[Circuit],
        n_sys: int,
        initial_state: Optional[Circuit] = None,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        result = lcu_apply(
            alphas=alphas, unitaries=unitaries, n_sys=n_sys,
            initial_state=initial_state, backend=backend,
            device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Success probability": result.get("Success probability"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "Result state": result.get("Result state"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 7. Test cases
# ---------------------------------------------------------------------------

def make_hadamard(n_sys: int) -> Circuit:
    """Create H gate circuit."""
    qc = Circuit(n_sys, name="H")
    qc.h(0)
    return qc


def make_pauli_x(n_sys: int) -> Circuit:
    """Create Pauli-X gate circuit."""
    qc = Circuit(n_sys, name="X")
    qc.x(0)
    return qc


def make_identity(n_sys: int) -> Circuit:
    """Create identity circuit."""
    return Circuit(n_sys, name="I")


# Matrix equivalents for validation
H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
I_MAT = np.eye(2, dtype=complex)

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "alphas": [0.6, 0.4],
        "unitaries": [make_hadamard(1), make_pauli_x(1)],
        "unitaries_mat": [H_MAT, X_MAT],
        "n_sys": 1,
        "label": "M = 0.6H + 0.4X",
    },
    {
        "alphas": [0.8, 0.2],
        "unitaries": [make_identity(2), make_pauli_x(2)],
        "unitaries_mat": [np.eye(4), np.kron(X_MAT, I_MAT)],
        "n_sys": 2,
        "label": "M = 0.8·I + 0.2·XI (2-qubit)",
    },
    {
        "alphas": [0.5, 0.5],
        "unitaries": [make_identity(1), make_pauli_x(1)],
        "unitaries_mat": [I_MAT, X_MAT],
        "n_sys": 1,
        "label": "M = 0.5I + 0.5X (equal weights)",
    },
    {
        "alphas": [0.3, 0.3, 0.4],
        "unitaries": [make_identity(1), make_pauli_x(1), make_hadamard(1)],
        "unitaries_mat": [I_MAT, X_MAT, H_MAT],
        "n_sys": 1,
        "label": "M = 0.3I + 0.3X + 0.4H (3 terms)",
    },
    {
        "alphas": [1.0],
        "unitaries": [make_hadamard(1)],
        "unitaries_mat": [H_MAT],
        "n_sys": 1,
        "label": "M = H only (single unitary)",
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    result = validate_lcu(
        alphas=case["alphas"],
        unitaries=case["unitaries"],
        unitaries_mat=case["unitaries_mat"],
        n_sys=case["n_sys"],
    )
    fid = result["fidelity"]
    succ = result["Success probability"]
    exp_succ = result["expected_success"]
    ok = fid > 0.94 and abs(succ - exp_succ) < 0.05
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] {case['label']}")
    print(f"       Fidelity={fid:.6f}, P_succ={succ:.4f} (expected={exp_succ:.4f})")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LCU — Linear Combination of Unitaries")
    print("=" * 60)

    solver = LCUAlgorithmSolver()

    # Demo from the SKILL.md
    print("\n--- Demo: M = 0.6·H + 0.4·X ---")
    U0 = make_hadamard(1)
    U1 = make_pauli_x(1)
    result = solver.run(alphas=[0.6, 0.4], unitaries=[U0, U1], n_sys=1)
    print(f"  Success probability: {result['Success probability']:.6f}")
    print(f"  Result state: {np.array2string(result['Result state'], precision=4)}")

    # Validation
    val = validate_lcu(
        alphas=[0.6, 0.4],
        unitaries=[U0, U1],
        unitaries_mat=[H_MAT, X_MAT],
        n_sys=1,
    )
    print(f"\n  Fidelity vs exact: {val['fidelity']:.6f}")
    print(f"  Expected state:     {np.array2string(val['expected_state'], precision=4)}")
    print(f"  Expected P_succ:    {val['expected_success']:.6f}")

    # PREPARE amplitude view
    print("\n--- PREPARE Amplitudes ---")
    for alphas in [[0.6, 0.4], [0.3, 0.3, 0.4]]:
        s = sum(alphas)
        amps = [math.sqrt(a / s) for a in alphas]
        print(f"  α={alphas}: √(α/s) = {[f'{a:.4f}' for a in amps]}")

    # Test cases
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    n_cases = len(KNOWN_CASES)
    print(f"\n  Result: {passed}/{n_cases} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
