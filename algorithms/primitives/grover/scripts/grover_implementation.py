"""Manual implementation of Grover's Search algorithm.

Finds a marked computational-basis state in an unstructured search space of size
N = 2^n with quadratic quantum speedup: O(√N) queries vs O(N) classical.

Algorithm steps:
    1. Prepare uniform superposition: H^{⊗n}|0^n⟩
    2. Oracle: mark target state via phase flip (ancilla kickback)
    3. Diffuser: reflect about the uniform superposition
    4. Repeat oracle + diffuser for optimal iterations: ⌊π/(4·arcsin(√p)) - ½⌉
    5. Measure: most likely state is the target

This implements the uniform-superposition, single-target special case of
amplitude amplification.

Components:
    - build_oracle: Mark target bitstring via MCX + kickback ancilla
    - build_diffuser: Reflect about uniform superposition
    - get_optimal_iterations: Compute near-optimal iteration count
    - build_grover_circuit: Full Grover circuit
    - grover_search: End-to-end Grover search pipeline
    - GroverAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — Grover Search
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Optimal iteration count
# ---------------------------------------------------------------------------

def get_optimal_iterations(p: float) -> int:
    """Compute the near-optimal number of Grover iterations.

    For initial success probability p with one marked state:
        k_opt = ⌊π / (4·arcsin(√p)) - ½⌉

    Uses round() to avoid floating-point precision errors near integer boundaries.

    Args:
        p: Initial success probability (p = 1/2^n for one marked state).

    Returns:
        Optimal iteration count (minimum 1).
    """
    theta = math.asin(math.sqrt(p))
    k = round(math.pi / (4.0 * theta) - 0.5)
    return max(1, k)


# ---------------------------------------------------------------------------
# 2. Oracle construction
# ---------------------------------------------------------------------------

def build_oracle(
    qc: Circuit,
    data_qubits: List[int],
    target: str,
    ancilla: int,
) -> None:
    """Add the phase oracle to the circuit.

    Marks the target computational-basis state by flipping its phase via a
    kickback ancilla prepared in |−⟩. Target bits of '0' are handled by
    surrounding controls with X gates (flip → control → unflip).

    Net effect: |x⟩ → (-1)^{f(x)}|x⟩ where f(x) = 1 iff x = target.

    Args:
        qc: Circuit to modify in-place.
        data_qubits: Indices of data register qubits.
        target: Binary string (e.g. '101') of length n.
        ancilla: Ancilla qubit index.
    """
    n = len(data_qubits)
    if len(target) != n:
        raise ValueError(f"Target '{target}' has length {len(target)}, expected {n}")

    # Prepare ancilla in |−⟩
    qc.x(ancilla)
    qc.h(ancilla)

    # Flip qubits where target bit is '0' (control on |1⟩ for those positions)
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(data_qubits[i])

    # Multi-controlled-X: flips ancilla iff all data bits match target
    if n == 1:
        qc.cx(data_qubits[0], ancilla)
    else:
        qc.mcx(data_qubits, ancilla)

    # Undo the bit flips
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(data_qubits[i])

    # Return ancilla to |0⟩
    qc.h(ancilla)
    qc.x(ancilla)


# ---------------------------------------------------------------------------
# 3. Diffuser construction
# ---------------------------------------------------------------------------

def build_diffuser(
    qc: Circuit,
    data_qubits: List[int],
    ancilla: int,
) -> None:
    """Add the Grover diffuser to the circuit.

    D = 2|s⟩⟨s| - I = H^{⊗n} · (2|0^n⟩⟨0^n| - I) · H^{⊗n}

    Implemented as: H^{⊗n} → all-zeros oracle → H^{⊗n}
    where the all-zeros oracle flips phase of |0^n⟩.

    Args:
        qc: Circuit to modify in-place.
        data_qubits: Indices of data register qubits.
        ancilla: Ancilla qubit index.
    """
    n = len(data_qubits)

    # H^{⊗n} on data qubits
    for q in data_qubits:
        qc.h(q)

    # All-zeros phase oracle: mark |0...0⟩
    qc.x(ancilla)
    qc.h(ancilla)
    for q in data_qubits:
        qc.x(q)  # Flip so |0⟩ → |1⟩; now target |1...1⟩
    if n == 1:
        qc.cx(data_qubits[0], ancilla)
    else:
        qc.mcx(data_qubits, ancilla)
    for q in data_qubits:
        qc.x(q)  # Unflip
    qc.h(ancilla)
    qc.x(ancilla)

    # H^{⊗n} on data qubits
    for q in data_qubits:
        qc.h(q)


# ---------------------------------------------------------------------------
# 4. Full Grover circuit
# ---------------------------------------------------------------------------

def build_grover_circuit(
    n: int,
    target: str,
) -> Circuit:
    """Build the full Grover search circuit.

    Qubit layout: data qubits 0..n-1, ancilla at index n.

    Args:
        n: Number of data qubits (search space size 2^n).
        target: Target bit string of length n (e.g. '101').

    Returns:
        Grover Circuit with n+1 qubits.
    """
    # Validate target
    if len(target) != n:
        raise ValueError(f"Target '{target}' length {len(target)} != n={n}")
    if not all(c in '01' for c in target):
        raise ValueError(f"Target must be a binary string, got '{target}'")

    p = 1.0 / (1 << n)
    reps = get_optimal_iterations(p)

    ancilla = n
    data_qubits = list(range(n))

    qc = Circuit(n + 1, name=f"Grover_n{n}_target_{target}")

    # Step 1: Uniform superposition on data qubits
    for q in data_qubits:
        qc.h(q)

    # Step 2: Grover iterations (oracle + diffuser)
    for _ in range(reps):
        build_oracle(qc, data_qubits, target, ancilla)
        build_diffuser(qc, data_qubits, ancilla)

    return qc


# ---------------------------------------------------------------------------
# 5. End-to-end Grover search
# ---------------------------------------------------------------------------

def grover_search(
    n: int,
    target: str,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run Grover's search algorithm.

    Pipeline:
        1. Validate target and compute optimal iterations.
        2. Build Grover circuit (H^n → oracle+diffuser × reps).
        3. Execute statevector simulation.
        4. Extract most likely state (argmax of probabilities).

    Args:
        n: Number of data qubits (search space N = 2^n).
        target: Target bit string of length n.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - status: 'success' if Result == target, 'partial_success' otherwise
            - Result: Most likely state (binary string of length n)
            - Amplified target-state probability: Probability of most likely state
            - Optimal Iterations: Number of Grover iterations used
            - Computation Time (s): Wall-clock time
            - circuit: The Grover Circuit object
    """
    # --- Validation ---
    if not all(c in '01' for c in target):
        raise ValueError(f"Target must be a binary string, got '{target}'")
    if len(target) != n:
        raise ValueError(f"Target length {len(target)} != n={n}")

    t_start = time.perf_counter()

    p = 1.0 / (1 << n)
    reps = get_optimal_iterations(p)

    if verbose:
        print(f"Grover Search")
        print(f"  Qubits (n):    {n}")
        print(f"  Search space:  N = 2^{n} = {1 << n}")
        print(f"  Target:        |{target}⟩")
        print(f"  Iterations:    {reps} (optimal ⌊π/(4θ) - ½⌉)")

    # --- Build circuit ---
    qc = build_grover_circuit(n, target)

    # --- Execute ---
    result_sim = qc.execute(backend=backend, device=device, dtype=dtype)

    # Extract probabilities of data qubits
    probs = result_sim.calculate_state(list(range(n)))

    # Find most likely state
    # calculate_state returns {bitstring: prob} or {bitstring: {prob: ...}}
    best_state = None
    best_prob = -1.0
    for bitstring, data in probs.items():
        if isinstance(data, dict):
            prob = data.get("prob", data.get("probability", 0.0))
        else:
            prob = float(data)

        if prob > best_prob:
            best_prob = prob
            best_state = bitstring

    # Pad to length n
    if best_state is not None:
        best_state = best_state.zfill(n)

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    status = "success" if best_state == target else "partial_success"

    if verbose:
        print(f"  Most likely:   |{best_state}⟩ (prob = {best_prob:.6f})")
        print(f"  Status:        {status}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": status,
        "Result": best_state,
        "Amplified target-state probability": best_prob,
        "Optimal Iterations": reps,
        "Computation Time (s)": comp_time,
        "circuit": qc,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class GroverAlgorithmSolver:
    """Class-based Grover search solver.

    Usage:
        solver = GroverAlgorithmSolver()
        result = solver.run(n=3, target="101")
        print(result['Result'])  # '101'
        print(result['status'])  # 'success'
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        n: int,
        target: str,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run Grover's search algorithm.

        Args:
            n: Number of data qubits.
            target: Target bit string of length n (e.g. '101').
            backend: Simulation backend.
            device: Compute device.
            dtype: Numerical dtype.

        Returns:
            Result dict with keys: status, Result, Amplified target-state
            probability, Optimal Iterations, Computation Time (s), circuit.
        """
        result = grover_search(
            n=n,
            target=target,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Result": result.get("Result"),
            "Amplified target-state probability": result.get(
                "Amplified target-state probability", 0.0
            ),
            "Optimal Iterations": result.get("Optimal Iterations", 0),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 7. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"n": 3, "target": "101", "description": "3-qubit, target |101⟩"},
    {"n": 3, "target": "000", "description": "3-qubit, target |000⟩"},
    {"n": 3, "target": "111", "description": "3-qubit, target |111⟩"},
    {"n": 4, "target": "1101", "description": "4-qubit, target |1101⟩, N=16"},
    {"n": 5, "target": "11010", "description": "5-qubit, target |11010⟩, N=32"},
    {"n": 2, "target": "10", "description": "2-qubit, target |10⟩, N=4"},
    {"n": 1, "target": "1", "description": "1-qubit, target |1⟩, N=2"},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case with retries (Grover is probabilistic on real HW,
    but deterministic on statevector simulators)."""
    n = case["n"]
    target = case["target"]
    desc = case["description"]

    result = grover_search(n=n, target=target, backend="torch", verbose=False)
    match = result["Result"] == target
    ok = result["status"] == "success" and match
    icon = "ok" if ok else "FAIL"

    print(f"  [{icon}] {desc}")
    print(f"       n={n}, target=|{target}⟩, result=|{result['Result']}⟩, "
          f"prob={result['Amplified target-state probability']:.4f}, "
          f"iters={result['Optimal Iterations']}, "
          f"time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Grover's Search — Manual Implementation")
    print("=" * 60)

    solver = GroverAlgorithmSolver()

    # --- Demo 1: 3-qubit search ---
    print("\n--- Demo 1: 3-qubit Grover, target |101⟩ ---")
    result1 = solver.run(n=3, target="101")
    print(f"  Result:    |{result1['Result']}⟩")
    print(f"  Target:    |101⟩")
    print(f"  Match:     {result1['Result'] == '101'}")
    print(f"  Status:    {result1['status']}")
    print(f"  Prob:      {result1['Amplified target-state probability']:.4f}")
    print(f"  Iterations:{result1['Optimal Iterations']}")

    # --- Demo 2: Multiple 3-qubit targets ---
    print("\n--- Demo 2: All 3-qubit targets ---")
    for target in ["000", "101", "111"]:
        result2 = solver.run(n=3, target=target)
        icon = "✓" if result2["Result"] == target else "✗"
        print(f"  {icon} Target=|{target}⟩ → Result=|{result2['Result']}⟩, "
              f"prob={result2['Amplified target-state probability']:.4f}")

    # --- Demo 3: 4-qubit (N=16) ---
    print("\n--- Demo 3: 4-qubit Grover, target |1101⟩ (N=16) ---")
    result3 = solver.run(n=4, target="1101")
    print(f"  N=16, target=|1101⟩, result=|{result3['Result']}⟩, "
          f"match={result3['Result'] == '1101'}, "
          f"iters={result3['Optimal Iterations']}")

    # --- Demo 4: 5-qubit (N=32) — quadratic speedup demo ---
    print("\n--- Demo 4: 5-qubit Grover, target |11010⟩ (N=32) ---")
    print(f"  Classical: O(32) queries average")
    result4 = solver.run(n=5, target="11010")
    print(f"  Quantum:   {result4['Optimal Iterations']} Grover iterations")
    print(f"  Result:    |{result4['Result']}⟩, match={result4['Result'] == '11010'}")

    # --- Demo 5: Optimal iteration count visualization ---
    print("\n--- Demo 5: Optimal Grover iterations for n = 1..10 ---")
    for n_val in range(1, 11):
        p_val = 1.0 / (1 << n_val)
        k_val = get_optimal_iterations(p_val)
        theoretical = int(math.pi / 4 * math.sqrt(1 << n_val))
        print(f"  n={n_val:>2}, N=2^{n_val}={1<<n_val:>4}, "
              f"iterations={k_val:>3} (≈ π/4·√N ≈ {theoretical})")

    # --- Demo 6: 1-qubit and 2-qubit minimal cases ---
    print("\n--- Demo 6: Minimal cases (n=1, n=2) ---")
    for n_val, tgt in [(1, "1"), (2, "10")]:
        result6 = solver.run(n=n_val, target=tgt)
        print(f"  n={n_val}, target=|{tgt}⟩, result=|{result6['Result']}⟩, "
              f"match={result6['Result'] == tgt}, iters={result6['Optimal Iterations']}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
