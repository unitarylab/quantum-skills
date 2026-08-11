"""Manual implementation of Amplitude Amplification.

Amplitude Amplification generalizes Grover's search algorithm. Given a unitary
operator U that prepares a state with a small initial success probability p,
this algorithm iteratively applies a Grover-style operator to amplify the
probability of the "good" (target) states, achieving probability close to 1
after O(1/√p) iterations.

Algorithm:
    1. Prepare initial state |ψ⟩ = U|0⟩ using a user-supplied Circuit U.
    2. Define "good" state condition: qubits listed in `good_zero_qubits`
       must all be in state |0⟩.
    3. Repeat Grover iterations: each iteration applies an **oracle**
       (phase-flipper for good states) followed by a **diffuser**
       (reflection about |ψ⟩).
    4. Measure the amplified probability in the data register.

The number of iterations is chosen automatically from the initial probability p:
    k ≈ π/(4θ) - 1/2,  where sinθ = √p
or set manually via `reps`.

Components:
    - _get_optimal_iterations: Compute optimal Grover iteration count from p
    - _prepare_kickback_ancilla_minus / _unprepare_kickback_ancilla_minus:
      Ancilla state management for phase kickback
    - _build_oracle: Phase-kickback oracle marking good states
    - _build_diffuser: Grover diffuser reflecting about |ψ⟩
    - amplitude_amplification: End-to-end amplitude amplification
    - AmplitudeAmplificationSolver: Class-based interface

Reference:
    SKILL.md — Amplitude Amplification
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Iteration count computation
# ---------------------------------------------------------------------------

def _get_optimal_iterations(p: float) -> int:
    """Calculate optimal Grover iteration count based on initial probability p.

    k = round(π / (4·arcsin(√p)) - 0.5)

    Uses round() instead of floor() to avoid floating-point precision errors
    near integer boundaries.

    Args:
        p: Initial success probability. Must satisfy 0 < p < 1.

    Returns:
        Optimal number of Grover iterations (≥ 0).

    Raises:
        ValueError: If p is not in (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"Initial probability p must satisfy 0 < p < 1, got {p}")
    theta = math.asin(math.sqrt(p))
    r = int(round((math.pi / (4.0 * theta)) - 0.5))
    return max(0, r)


# ---------------------------------------------------------------------------
# 2. Ancilla state management (phase kickback)
# ---------------------------------------------------------------------------

def _prepare_kickback_ancilla_minus(qc: Circuit, ancilla: int) -> None:
    """Prepare ancilla qubit in |−⟩ = H·X|0⟩ state.

    The |−⟩ state is used for phase kickback in the oracle:
    applying X⊗...⊗X-controlled-X on the ancilla flips the phase
    of the data-register state when all control conditions are met.

    Args:
        qc: Circuit to add gates to.
        ancilla: Index of the ancilla qubit.
    """
    qc.x(ancilla)
    qc.h(ancilla)


def _unprepare_kickback_ancilla_minus(qc: Circuit, ancilla: int) -> None:
    """Restore ancilla qubit from |−⟩ back to |0⟩ state.

    Applies the inverse of _prepare_kickback_ancilla_minus.

    Args:
        qc: Circuit to add gates to.
        ancilla: Index of the ancilla qubit.
    """
    qc.h(ancilla)
    qc.x(ancilla)


# ---------------------------------------------------------------------------
# 3. Oracle construction
# ---------------------------------------------------------------------------

def _build_oracle(
    qc: Circuit,
    zero_qubits: List[int],
    ancilla: int,
) -> None:
    """Build the phase-kickback oracle.

    Applies a phase flip (-1) to computational basis states where all
    `zero_qubits` are |0⟩. The oracle uses a kickback ancilla in the
    |−⟩ state:

        - Prepare ancilla in |−⟩
        - Flip target qubits with X (converting |0⟩-control to |1⟩-control)
        - Apply multi-controlled-X targeting the ancilla
        - Unflip target qubits
        - Unprepare ancilla to |0⟩

    Net effect on data register: |x⟩ → (-1)^{f(x)}|x⟩
    where f(x) = 1 iff all qubits in `zero_qubits` are |0⟩.

    Args:
        qc: Circuit to add gates to.
        zero_qubits: Qubit indices that must be |0⟩ for the "good" state.
        ancilla: Index of the ancilla qubit.
    """
    _prepare_kickback_ancilla_minus(qc, ancilla)

    # Convert |0⟩-control to |1⟩-control by flipping all zero_qubits
    for q in zero_qubits:
        qc.x(q)

    # Multi-controlled-X gate targeting the ancilla
    controls = list(zero_qubits)
    if len(controls) == 0:
        qc.z(ancilla)
    elif len(controls) == 1:
        qc.cx(controls[0], ancilla)
    else:
        qc.mcx(controls, ancilla)

    # Unflip the zero_qubits
    for q in zero_qubits:
        qc.x(q)

    _unprepare_kickback_ancilla_minus(qc, ancilla)


# ---------------------------------------------------------------------------
# 4. Diffuser construction
# ---------------------------------------------------------------------------

def _build_diffuser(
    qc: Circuit,
    U: Circuit,
    data_qubits: List[int],
    ancilla: int,
) -> None:
    """Build the Grover diffusion operator.

    Implements D = U(2|0ⁿ⟩⟨0ⁿ| − I)U†, reflecting the state about |ψ⟩ = U|0⟩.

    Steps:
        1. Apply U† (inverse of state preparation)
        2. Apply the all-zeros phase oracle via _build_oracle on all data qubits
        3. Apply U (re-prepare)

    The all-zeros oracle component (2|0ⁿ⟩⟨0ⁿ| − I) is implemented by
    calling _build_oracle with all data qubits as zero_qubits.

    Args:
        qc: Circuit to add gates to.
        U: State preparation circuit.
        data_qubits: Indices of all data qubits.
        ancilla: Index of the ancilla qubit.
    """
    qc.append(U.dagger(), data_qubits)
    _build_oracle(qc, zero_qubits=list(data_qubits), ancilla=ancilla)
    qc.append(U, data_qubits)


# ---------------------------------------------------------------------------
# 5. End-to-end amplitude amplification
# ---------------------------------------------------------------------------

def amplitude_amplification(
    U: Circuit,
    good_zero_qubits: List[int],
    p: float,
    reps: Optional[int] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run amplitude amplification on a given state preparation circuit.

    Pipeline:
        1. Resolve parameters: Determine Grover iteration count from p.
        2. Build circuit: Append U, then `reps` × (oracle → diffuser).
        3. Simulate: Execute statevector simulation.
        4. Post-process: Extract probability of good states.
        5. Export: Save circuit diagram and result text.

    Args:
        U: State preparation circuit (acts on data qubits only, no ancilla).
        good_zero_qubits: Qubit indices that must all be |0⟩ for the
            "good" state.
        p: Estimated initial success probability. Must satisfy 0 < p < 1.
        reps: Manual override for number of Grover iterations. If None,
            computed automatically from p.
        backend: Simulation backend ('torch' only).
        device: Compute device ('cpu' or 'cuda').
        dtype: Numerical dtype for simulation.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok' if amplified probability exceeds initial p
            - Amplified Target Probability: Float, measured probability
            - Initial Success Probability: Float, the p value passed in
            - Repetitions: Int, number of Grover iterations used
            - Computation Time (s): Float, wall-clock simulation time
            - Data register size: Int, number of data qubits
            - circuit: The constructed Circuit object

    Raises:
        ValueError: If p is not in (0, 1) and reps is None.
    """
    if verbose:
        print(f"Amplitude Amplification")
        print(f"  good_zero_qubits={good_zero_qubits}, p={p}")

    # --- Stage 1: Parameter resolution ---
    n_data = U.get_num_qubits()
    ancilla = n_data  # Ancilla is the last qubit

    if reps is None:
        reps = _get_optimal_iterations(p)

    theta = math.asin(math.sqrt(p))

    if verbose:
        print(f"  Data qubits: {n_data} (total: {n_data + 1} with ancilla)")
        print(f"  Initial angle θ = arcsin(√p): {theta:.4f} rad")
        print(f"  Grover iterations: {reps}")
        print(f"  Expected amplified angle (2k+1)θ: {(2 * reps + 1) * theta:.4f} rad")

    # --- Stage 2: Circuit construction ---
    qc = Circuit(n_data + 1, name="Amplitude_Amplification")
    data_qubits = list(range(n_data))

    # Append initial state preparation
    qc.append(U, data_qubits)

    # Grover iteration loop
    for _ in range(reps):
        _build_oracle(qc, zero_qubits=good_zero_qubits, ancilla=ancilla)
        _build_diffuser(qc, U=U, data_qubits=data_qubits, ancilla=ancilla)

    # --- Stage 3: Simulation ---
    t_start = time.perf_counter()
    result = qc.execute(backend=backend, device=device, dtype=dtype)
    state_basis_dict = result.calculate_state(data_qubits)
    comp_time = time.perf_counter() - t_start

    if verbose:
        print(f"  Simulation time: {comp_time:.4f}s")
        print(f"  Measured basis states: {len(state_basis_dict)}")

    # --- Stage 4: Post-processing ---
    target_prob = 0.0
    for basis_str, state_info in state_basis_dict.items():
        # Check if all good_zero_qubits are '0' in this basis state
        is_target = all(basis_str[q] == '0' for q in good_zero_qubits)
        if is_target:
            if isinstance(state_info, dict):
                if 'prob' in state_info:
                    target_prob += float(state_info['prob'])
                elif 'probability' in state_info:
                    target_prob += float(state_info['probability'])
                elif 'amp' in state_info:
                    target_prob += abs(state_info['amp']) ** 2
                else:
                    target_prob += float(list(state_info.values())[0])
            else:
                target_prob += float(state_info)

    is_success = target_prob > p

    if verbose:
        print(f"  Initial success probability: {p:.6f}")
        print(f"  Amplified target probability: {target_prob:.6f}")
        print(f"  Amplification factor: {target_prob / p:.2f}×")
        print(f"  Status: {'ok' if is_success else 'failed'}")

    return {
        "status": "ok" if is_success else "failed",
        "Amplified Target Probability": target_prob,
        "Initial Success Probability": p,
        "Repetitions": reps,
        "Computation Time (s)": round(comp_time, 4),
        "Data register size": n_data,
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class AmplitudeAmplificationSolver:
    """Class-based solver for amplitude amplification.

    Usage:
        solver = AmplitudeAmplificationSolver(text_mode='plain')
        result = solver.run(
            U=circuit, good_zero_qubits=[0, 1], p=0.05,
        )
        print(result['Amplified Target Probability'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        U: Circuit,
        good_zero_qubits: List[int],
        p: float,
        reps: Optional[int] = None,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run amplitude amplification. See amplitude_amplification() for docs."""
        result = amplitude_amplification(
            U=U, good_zero_qubits=good_zero_qubits,
            p=p, reps=reps,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "Amplified Target Probability": result.get("Amplified Target Probability"),
            "Initial Success Probability": result.get("Initial Success Probability"),
            "Repetitions": result.get("Repetitions"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "Data register size": result.get("Data register size"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 7. Test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    # (theta, n_qubits, good_zero_qubits, label)
    {"theta": 1.35, "n_qubits": 1, "good_qubits": [0], "label": "1-qubit, small initial prob"},
    {"theta": 1.2, "n_qubits": 2, "good_qubits": [0, 1], "label": "2-qubit, |00⟩ target"},
    {"theta": 1.0, "n_qubits": 2, "good_qubits": [0], "label": "2-qubit, q0=|0⟩ target only"},
    {"theta": 1.4, "n_qubits": 3, "good_qubits": [0, 1], "label": "3-qubit, q0,q1=|0⟩ target"},
    {"theta": 0.8, "n_qubits": 2, "good_qubits": [0, 1], "label": "2-qubit, moderate initial prob"},
]


def _build_test_state_prep(theta: float, n_qubits: int, good_qubits: List[int]) -> Circuit:
    """Build a test state preparation circuit.

    Creates a state where the target qubits have probability cos²(θ) of
    being |0⟩. Each target qubit gets ry(2θ) applied; non-target qubits
    get no rotation (stay in |0⟩).

    Args:
        theta: Angle controlling initial success probability.
        n_qubits: Total number of data qubits.
        good_qubits: Which qubits are part of the target condition.

    Returns:
        State preparation Circuit.
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
    label = case["label"]

    p_initial = float(np.cos(theta) ** (2 * len(good_qubits)))  # cos²(θ) per qubit → cos(θ)^(2n)

    U = _build_test_state_prep(theta, n_qubits, good_qubits)

    result = amplitude_amplification(
        U=U, good_zero_qubits=good_qubits, p=p_initial,
        backend="torch", verbose=False,
    )

    ok = result["status"] == "ok"
    amp_prob = result["Amplified Target Probability"]
    factor = amp_prob / p_initial if p_initial > 0 else float("inf")

    icon = "ok" if ok else "FAIL"
    print(f"  [{icon}] {label}")
    print(f"       p_initial={p_initial:.6f}, p_amplified={amp_prob:.6f}, "
          f"factor={factor:.2f}×, reps={result['Repetitions']}")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Amplitude Amplification — Manual Implementation")
    print("=" * 60)

    solver = AmplitudeAmplificationSolver()

    # --- Demo 1: 1-qubit, small initial probability ---
    print("\n--- Demo 1: 1-qubit, small initial probability ---")
    theta1 = 1.35
    U1 = Circuit(1, name="Demo1")
    U1.ry(2 * theta1, 0)
    p1 = float(np.cos(theta1) ** 2)

    result1 = solver.run(U=U1, good_zero_qubits=[0], p=p1)
    print(f"  Initial p:            {p1:.6f}")
    print(f"  Amplified probability: {result1['Amplified Target Probability']:.6f}")
    print(f"  Amplification factor:  {result1['Amplified Target Probability'] / p1:.2f}×")
    print(f"  Iterations:            {result1['Repetitions']}")
    print(f"  Status:                {result1['status']}")

    # --- Demo 2: 2-qubit, both qubits |0⟩ ---
    print("\n--- Demo 2: 2-qubit, |00⟩ target ---")
    theta2 = 1.2
    U2 = Circuit(2, name="Demo2")
    U2.ry(2 * theta2, 0)
    U2.ry(2 * theta2, 1)
    p2 = float(np.cos(theta2) ** 4)  # Both qubits independent

    result2 = solver.run(U=U2, good_zero_qubits=[0, 1], p=p2)
    print(f"  Initial p:            {p2:.6f}")
    print(f"  Amplified probability: {result2['Amplified Target Probability']:.6f}")
    print(f"  Amplification factor:  {result2['Amplified Target Probability'] / p2:.2f}×")
    print(f"  Iterations:            {result2['Repetitions']}")
    print(f"  Status:                {result2['status']}")

    # --- Demo 3: Comparison with different p estimates ---
    print("\n--- Demo 3: Effect of p estimate on amplification ---")
    theta3 = 1.0
    U3 = Circuit(2, name="Demo3")
    U3.ry(2 * theta3, 0)
    U3.ry(2 * theta3, 1)
    true_p = float(np.cos(theta3) ** 4)

    for est_p in [0.01, 0.05, true_p, 0.5]:
        result3 = amplitude_amplification(
            U=U3, good_zero_qubits=[0, 1], p=est_p,
            backend="torch", verbose=False,
        )
        print(f"  p_est={est_p:.4f} → reps={result3['Repetitions']}, "
              f"amplified_prob={result3['Amplified Target Probability']:.6f}, "
              f"status={result3['status']}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")

    # --- Optimal iteration count table ---
    print("\n--- Optimal Iterations k(p) ---")
    for p_val in [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]:
        k = _get_optimal_iterations(p_val)
        theta_val = math.asin(math.sqrt(p_val))
        amp = math.sin((2 * k + 1) * theta_val) ** 2
        print(f"  p={p_val:.3f} → k={k}, theoretical amplified prob={amp:.4f}")

    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
