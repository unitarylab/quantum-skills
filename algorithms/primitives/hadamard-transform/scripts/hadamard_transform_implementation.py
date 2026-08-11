"""Manual implementation of the Hadamard Transform (H^{⊗n}).

The Hadamard Transform applies a Hadamard gate to every qubit simultaneously,
mapping the computational basis to a uniform superposition and vice versa.
It is a foundational building block in virtually all quantum algorithms.

Two supported modes:
    1. 'superposition': Apply H^{⊗n} to |0⟩ⁿ, producing the uniform
       superposition (1/√(2ⁿ)) Σ |x⟩. Verifies all 2ⁿ basis states
       have equal probability 1/2ⁿ.
    2. 'reflexive_test': Initialize a random quantum state |ψ⟩, apply
       H^{⊗n} twice, and verify H² = I (the state is recovered).

Properties:
    - Single-qubit: H = 1/√2 [[1, 1], [1, -1]]
    - H|0⟩ = |+⟩, H|1⟩ = |−⟩
    - Self-inverse: H² = I
    - H^{⊗n}|x⟩ = (1/√(2ⁿ)) Σ_y (-1)^{x·y} |y⟩
    - From |0⟩ⁿ: produces uniform superposition over all 2ⁿ states

Components:
    - hadamard_transform_circuit: Build n-qubit H^{⊗n} circuit
    - hadamard_transform_run: End-to-end two-mode pipeline
    - HadamardTransformSolver: Class-based interface
    - probabilities_from_statevec: Extract bitstring→prob dict from statevector

Reference:
    SKILL.md — Hadamard Transform
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Core Hadamard transform circuit
# ---------------------------------------------------------------------------

def hadamard_transform_circuit(n: int) -> Circuit:
    """Build the n-qubit Hadamard transform circuit H^{⊗n}.

    Simply applies an H gate to every qubit 0..n-1.

    Args:
        n: Number of qubits. Must be >= 1.

    Returns:
        Circuit of size n with H on all qubits.
    """
    qc = Circuit(n, name="H^{%d}" % n if n <= 5 else f"H^{n}")
    for q in range(n):
        qc.h(q)
    return qc


# ---------------------------------------------------------------------------
# 2. Statevector utilities
# ---------------------------------------------------------------------------

def probabilities_from_statevec(
    statevec: np.ndarray,
    threshold: float = 1e-12,
) -> Dict[str, float]:
    """Convert a statevector to a sorted bitstring → probability dict.

    Args:
        statevec: Complex statevector (flattened, length 2ⁿ).
        threshold: Probabilities below this value are filtered out.

    Returns:
        Dict mapping binary bitstring → probability, sorted by bitstring.

    Raises:
        ValueError: If statevector length is not a power of 2.
    """
    dim = len(statevec)
    n = int(np.log2(dim))
    if 2 ** n != dim:
        raise ValueError(f"Statevector length {dim} is not a power of 2")
    probs = np.abs(statevec) ** 2
    out: Dict[str, float] = {}
    for idx, p in enumerate(probs):
        if p < threshold:
            continue
        bits = format(idx, f"0{n}b")
        out[bits] = float(p)
    return dict(sorted(out.items()))


def _generate_random_state(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a normalized random complex statevector of dimension 2ⁿ.

    Uses independent Gaussian real/imag parts, then normalizes.
    This produces a Haar-uniform random state.

    Args:
        n: Number of qubits (statevector dimension = 2ⁿ).
        seed: Optional seed for reproducibility.

    Returns:
        Normalized complex statevector of shape (2ⁿ,).
    """
    rng = np.random.default_rng(seed)
    dim = 1 << n
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi = psi.astype(np.complex128)
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# 3. End-to-end Hadamard Transform
# ---------------------------------------------------------------------------

def hadamard_transform_run(
    n: int = 3,
    mode: str = "superposition",
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    random_seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the n-qubit Hadamard Transform.

    Modes:
        'superposition':  H^{⊗n}|0⟩ⁿ → uniform superposition.
        'reflexive_test': |ψ⟩ → H^{⊗n} → H^{⊗n} → verify |ψ⟩ recovered.

    Args:
        n: Number of qubits (>= 1).
        mode: 'superposition' or 'reflexive_test'.
        backend: Simulation backend ('torch').
        device: Compute device ('cpu' or 'cuda').
        dtype: Numerical dtype.
        random_seed: Optional seed for random state in reflexive_test.
        verbose: Print progress information.

    Returns:
        Dict with keys:
            - status: 'ok' on success, 'failed' otherwise
            - State vector: Final statevector as np.ndarray
            - Probability distribution: Dict bitstring→prob (superposition only)
            - Computation time (s): Wall-clock simulation time
            - circuit: The constructed Circuit
            - max_deviation: Max |amp| deviation (reflexive_test only)
            - n_qubits: Number of qubits used
            - mode: The mode that was run

    Raises:
        ValueError: If n < 1 or mode is invalid.
    """
    # --- Stage 1: Validation ---
    if n < 1:
        raise ValueError(f"Number of qubits n must be >= 1, got {n}")
    valid_modes = {"superposition", "reflexive_test"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

    if verbose:
        print(f"Hadamard Transform")
        print(f"  Qubits: {n}")
        print(f"  Mode:   {mode}")

    # --- Stage 2: Circuit construction ---
    qc = Circuit(n, name=f"Hadamard_{mode}")
    target_qubits = list(range(n))
    original_state: Optional[np.ndarray] = None

    if mode == "superposition":
        for q in target_qubits:
            qc.h(q)
        if verbose:
            print(f"  Applied H to all {n} qubits (|0⟩ⁿ → uniform superposition)")

    elif mode == "reflexive_test":
        original_state = _generate_random_state(n, seed=random_seed)
        qc.initialize(original_state, target=target_qubits)
        # Apply H⊗n twice (H² = I)
        for _ in range(2):
            for q in target_qubits:
                qc.h(q)
        if verbose:
            print(f"  Initialized random state (dim={1<<n})")
            print("  Applied H^{⊗n} twice (reflexivity test)")

    # --- Stage 3: Simulation ---
    if verbose:
        print(f"  Executing simulation...")

    t_start = time.perf_counter()
    result = qc.execute(backend=backend, device=device, dtype=dtype)
    statevec = np.asarray(result.state, dtype=complex).reshape(-1)
    comp_time = time.perf_counter() - t_start

    if verbose:
        print(f"  Simulation time: {comp_time:.4f}s")
        print(f"  Statevector dim: {len(statevec)}")

    # --- Stage 4: Post-processing ---
    prob_dict: Dict[str, float] = {}
    is_success = False
    max_deviation = 0.0

    if mode == "superposition":
        expected_prob = 1.0 / (1 << n)  # 1/2ⁿ
        prob_dict = probabilities_from_statevec(statevec)
        # Check: all 2ⁿ states present, each probability ≈ 1/2ⁿ
        all_uniform = all(
            np.isclose(p, expected_prob, atol=1e-5)
            for p in prob_dict.values()
        )
        is_success = all_uniform and len(prob_dict) == (1 << n)
        max_deviation = max(
            abs(p - expected_prob) for p in prob_dict.values()
        ) if prob_dict else float("inf")

        if verbose:
            print(f"  Expected probability per state: {expected_prob:.6f}")
            print(f"  Measured states: {len(prob_dict)} / {1 << n}")
            print(f"  Max deviation from uniform: {max_deviation:.2e}")
            print(f"  Uniformity check: {'passed' if is_success else 'FAILED'}")

    elif mode == "reflexive_test":
        is_success = bool(np.allclose(statevec, original_state, atol=1e-10))
        max_deviation = float(np.max(np.abs(statevec - original_state)))
        prob_dict = {}

        if verbose:
            print(f"  Max amplitude deviation: {max_deviation:.2e}")
            print(f"  H²=I check: {'passed' if is_success else 'FAILED'}")

    if verbose:
        print(f"  Status: {'ok' if is_success else 'failed'}")

    return {
        "status": "ok" if is_success else "failed",
        "State vector": statevec,
        "Probability distribution": prob_dict,
        "Computation time (s)": round(comp_time, 4),
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
        "n_qubits": n,
        "mode": mode,
        "max_deviation": max_deviation,
    }


# ---------------------------------------------------------------------------
# 4. Class-based interface
# ---------------------------------------------------------------------------

class HadamardTransformSolver:
    """Class-based solver for the Hadamard Transform.

    Usage:
        solver = HadamardTransformSolver(text_mode='plain')
        result = solver.run(n=4, mode='superposition')
        print(result['Probability distribution'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        n: int = 3,
        mode: str = "superposition",
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run Hadamard Transform. See hadamard_transform_run() for docs."""
        result = hadamard_transform_run(
            n=n, mode=mode, backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build standardized return dict."""
        return {
            "status": result.get("status", "failed"),
            "State vector": result.get("State vector"),
            "Probability distribution": result.get("Probability distribution"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
            "max_deviation": result.get("max_deviation"),
            "n_qubits": result.get("n_qubits"),
            "mode": result.get("mode"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ---------------------------------------------------------------------------
# 5. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"n": 1, "mode": "superposition", "label": "n=1, |+⟩ superposition"},
    {"n": 2, "mode": "superposition", "label": "n=2, 4-state uniform"},
    {"n": 3, "mode": "superposition", "label": "n=3, 8-state uniform"},
    {"n": 4, "mode": "superposition", "label": "n=4, 16-state uniform"},
    {"n": 1, "mode": "reflexive_test", "label": "n=1, H²=I reflexivity"},
    {"n": 3, "mode": "reflexive_test", "label": "n=3, H²=I reflexivity"},
    {"n": 5, "mode": "reflexive_test", "label": "n=5, H²=I reflexivity"},
    {"n": 8, "mode": "superposition", "label": "n=8, 256-state uniform"},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case and print result."""
    n = case["n"]
    mode = case["mode"]
    label = case["label"]

    result = hadamard_transform_run(
        n=n, mode=mode, backend="torch", verbose=False,
    )

    ok = result["status"] == "ok"
    icon = "ok" if ok else "FAIL"

    if mode == "superposition":
        n_states = len(result["Probability distribution"])
        max_dev = result["max_deviation"]
        print(f"  [{icon}] {label}")
        print(f"       states={n_states}/{1<<n}, max_dev={max_dev:.2e}, "
              f"time={result['Computation time (s)']:.4f}s")
    else:
        max_dev = result["max_deviation"]
        print(f"  [{icon}] {label}")
        print(f"       max_dev={max_dev:.2e} (expect <1e-10), "
              f"time={result['Computation time (s)']:.4f}s")

    return ok


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Hadamard Transform (H^{⊗n}) — Manual Implementation")
    print("=" * 60)

    solver = HadamardTransformSolver()

    # --- Demo 1: Superposition mode (n=3) ---
    print("\n--- Demo 1: Superposition mode, n=3 ---")
    result1 = solver.run(n=3, mode="superposition")
    prob_dict1 = result1["Probability distribution"]
    print(f"  States produced: {len(prob_dict1)} / {1 << 3}")
    print(f"  First 4 probabilities: "
          f"{[f'{p:.6f}' for p in list(prob_dict1.values())[:4]]}")
    print(f"  Expected each: 1/8 = {1/8:.6f}")

    # --- Demo 2: Reflexive test (n=3) ---
    print("\n--- Demo 2: Reflexive test (H²=I), n=3 ---")
    result2 = solver.run(n=3, mode="reflexive_test")
    print(f"  Max deviation: {result2['max_deviation']:.2e}")
    print(f"  State restored: {result2['status']}")

    # --- Demo 3: Scaling behavior ---
    print("\n--- Demo 3: Scaling with n (superposition mode) ---")
    for n_val in [1, 2, 4, 6, 8, 10]:
        result3 = hadamard_transform_run(
            n=n_val, mode="superposition", backend="torch", verbose=False,
        )
        print(f"  n={n_val:>2}: {len(result3['Probability distribution']):>5} states, "
              f"time={result3['Computation time (s)']:.4f}s, "
              f"status={result3['status']}")

    # --- Demo 4: Uniformity precision ---
    print("\n--- Demo 4: Uniformity precision (n=5, all probs ≈ 1/32) ---")
    result4 = hadamard_transform_run(
        n=5, mode="superposition", backend="torch", verbose=False,
    )
    probs4 = list(result4["Probability distribution"].values())
    expected4 = 1.0 / 32
    deviations = [abs(p - expected4) for p in probs4]
    print(f"  Expected: {expected4:.10f}")
    print(f"  Min prob: {min(probs4):.10f}")
    print(f"  Max prob: {max(probs4):.10f}")
    print(f"  Max deviation: {max(deviations):.2e}")
    print(f"  Mean deviation: {np.mean(deviations):.2e}")

    # --- Demo 5: Reflexive test precision across n ---
    print("\n--- Demo 5: Reflexivity precision across n ---")
    for n_val in [1, 2, 3, 4, 5, 6]:
        result5 = hadamard_transform_run(
            n=n_val, mode="reflexive_test", backend="torch", verbose=False,
        )
        dev = result5["max_deviation"]
        print(f"  n={n_val}: max |H²ψ - ψ| = {dev:.2e}  "
              f"({'✓' if dev < 1e-10 else '✗'} H²=I)")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")

    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
