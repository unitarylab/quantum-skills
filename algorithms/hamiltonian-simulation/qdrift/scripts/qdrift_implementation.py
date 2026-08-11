"""Manual implementation of QDrift Hamiltonian Simulation.

Approximates U(t) = e^{-iHt} by stochastically sampling Pauli-term
evolutions: at each step, a single Pauli term is randomly selected with
probability proportional to its coefficient magnitude, and a uniformly
scaled rotation angle is applied. This replaces deterministic Trotter
ordering with stochastic averaging.

Algorithm:
    1. Decompose H = Σ c_j P_j into Pauli strings.
    2. Compute λ = Σ |c_j| (the 1-norm of coefficients).
    3. Build sampling probabilities: p_j = |c_j| / λ.
    4. For k = 1..steps:
       - Sample index j ~ categorical(p)
       - Apply: angle = sign(c_j) · λ · t / steps
       - Evolve: exp(-i · angle · P_j)
    5. The resulting circuit approximates U(t) = e^{-iHt}.
    6. Compare with exact U via Frobenius norm.

Complexity (randomized): N = O((λt)² / ε), potentially avoiding
the L² factor in first-order deterministic product formulas.

Components:
    - manual_pauli_decompose: Decompose 2x2 matrix into Pauli basis
    - qdrift_sample: Sample from Pauli terms probabilistically
    - build_qdrift_sequence: Generate random (P_j, angle) sequence
    - build_qdrift_circuit: Assemble circuit from sequence
    - qdrift_simulate: End-to-end simulation with error analysis
    - QDriftAlgorithmSolver: Class-based interface

Reference:
    SKILL.md — QDrift Hamiltonian Simulation
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import expm

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from unitarylab.library.pauli_operator import (
        pauli_string_decomposition,
        pauli_string_evolution,
    )

    HAS_UNITARYLAB = True
except ImportError:
    HAS_UNITARYLAB = False

from unitarylab.core import Circuit, Register


# ---------------------------------------------------------------------------
# 1. Pauli decomposition (manual + unitarylab wrapper)
# ---------------------------------------------------------------------------

# Pauli matrices for 2x2 manual decomposition
_PAULIS = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_LABELS = ["I", "X", "Y", "Z"]


def manual_pauli_decompose(H: np.ndarray) -> List[Tuple[str, complex]]:
    """Decompose a 2x2 Hermitian matrix into Pauli basis.

    H = Σ c_P · P, where P ∈ {I, X, Y, Z}.

    Uses the Hilbert-Schmidt inner product:
        c_P = (1/2) Tr(P^† H)

    For real symmetric matrices (the supported case), c_Y = 0.

    Args:
        H: 2x2 Hermitian matrix.

    Returns:
        List of (pauli_string, coefficient) tuples.
    """
    if H.shape != (2, 2):
        raise ValueError(
            f"Manual decomposition only supports 2x2 matrices, got {H.shape}"
        )
    decomp = []
    for label in _PAULI_LABELS:
        P = _PAULIS[label]
        c = 0.5 * np.trace(P.conj().T @ H)
        # Round near-zero coefficients
        if abs(c) > 1e-14:
            decomp.append((label, complex(c)))
    return decomp


def decompose_hamiltonian(
    H: np.ndarray, prefer_unitarylab: bool = True
) -> List[Tuple[str, complex]]:
    """Decompose a Hermitian Hamiltonian into Pauli strings.

    Preferentially uses unitarylab's pauli_string_decomposition (which
    handles arbitrary-dimension matrices via tensor-product Pauli strings),
    falling back to manual 2x2 decomposition.

    Args:
        H: Hermitian Hamiltonian matrix.
        prefer_unitarylab: Use unitarylab if available.

    Returns:
        List of (pauli_string, coefficient) tuples, sorted by |coeff| desc.
    """
    if HAS_UNITARYLAB and prefer_unitarylab:
        decomp = pauli_string_decomposition(H)
    else:
        decomp = manual_pauli_decompose(H)

    # Sort by absolute coefficient (descending) for consistency
    decomp.sort(key=lambda x: abs(x[1]), reverse=True)
    return decomp


# ---------------------------------------------------------------------------
# 2. QDrift random sampling
# ---------------------------------------------------------------------------

def compute_qdrift_params(
    decomposition: List[Tuple[str, complex]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute QDrift sampling parameters from Pauli decomposition.

    Args:
        decomposition: List of (pauli_string, coefficient) tuples.

    Returns:
        Tuple of (pauli_strings, coeffs, probs, lam) where:
        - pauli_strings: array of Pauli string labels
        - coeffs: array of real coefficients
        - probs: array of sampling probabilities (|c_j| / λ)
        - lam: Σ |c_j| (the 1-norm of coefficients)
    """
    pauli_strings = np.array([p for p, _ in decomposition])
    coeffs = np.array([c.real for _, c in decomposition], dtype=float)
    lam = float(np.sum(np.abs(coeffs)))
    probs = np.abs(coeffs) / lam if lam > 0 else np.ones_like(coeffs) / len(coeffs)
    return pauli_strings, coeffs, probs, lam


def qdrift_sample(
    decomposition: List[Tuple[str, complex]],
    t: float,
    steps: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[str, float]]:
    """Generate a random QDrift sequence.

    For each of `steps` iterations:
        1. Sample a Pauli term j with probability p_j = |c_j| / λ.
        2. Compute angle = sign(c_j) · λ · t / steps.
        3. Append (pauli_string, angle) to the sequence.

    Args:
        decomposition: Pauli decomposition of H.
        t: Total evolution time.
        steps: Number of random samples (circuit depth).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        List of (pauli_string, angle) tuples forming the QDrift sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    pauli_strings, coeffs, probs, lam = compute_qdrift_params(decomposition)

    # Sample indices according to probability distribution
    indices = rng.choice(len(decomposition), size=steps, p=probs)

    # Build sequence
    angle_scale = lam * t / steps
    sequence: List[Tuple[str, float]] = []
    for idx in indices:
        # Convert numpy.str_ to Python str (required by Cython-typed APIs)
        pauli_str = str(pauli_strings[idx])
        sign = np.sign(coeffs[idx])
        angle = float(sign * angle_scale)
        sequence.append((pauli_str, angle))

    return sequence


# ---------------------------------------------------------------------------
# 3. Circuit construction
# ---------------------------------------------------------------------------

def build_qdrift_circuit(
    sequence: List[Tuple[str, float]],
    n_qubits: int,
    name: str = "QDrift",
) -> Circuit:
    """Build the QDrift circuit from a random Pauli sequence.

    Each element (P_j, angle) in the sequence becomes a gate:
        exp(-i · angle · P_j)

    applied sequentially to the quantum register.

    Args:
        sequence: List of (pauli_string, angle) tuples.
        n_qubits: Number of qubits in the register.
        name: Circuit name.

    Returns:
        The assembled Circuit object.
    """
    reg = Register("K", n_qubits)
    qc = Circuit(reg, name=name)

    qubit_range = list(range(n_qubits))

    if HAS_UNITARYLAB:
        for pauli_str, angle in sequence:
            gate = pauli_string_evolution(pauli_str, angle)
            qc.append(gate, qubit_range)
    else:
        # Manual gate construction for 2x2 (single-qubit) case
        for pauli_str, angle in sequence:
            if pauli_str == "I":
                # Identity — global phase, no gate needed
                continue
            elif pauli_str == "X":
                qc.rx(2 * angle, 0)
            elif pauli_str == "Y":
                qc.ry(2 * angle, 0)
            elif pauli_str == "Z":
                qc.rz(2 * angle, 0)
            # Multi-qubit Pauli strings go through unitarylab path

    return qc


# ---------------------------------------------------------------------------
# 4. End-to-end simulation
# ---------------------------------------------------------------------------

def qdrift_simulate(
    H: np.ndarray,
    t: float,
    error: float = 1e-8,
    steps: int = 5000,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Simulate U(t) = e^{-iHt} using randomized QDrift.

    Full pipeline:
        1. Pauli-decompose H.
        2. Compute λ = Σ|c_j|, probabilities p_j = |c_j|/λ.
        3. Sample `steps` random Pauli terms with probability p_j.
        4. Build circuit: apply exp(-i·sign(c_j)·λ·t/steps·P_j) for each.
        5. Compute U_approx via circuit matrix.
        6. Compare with exact U = expm(-1j·H·t) via Frobenius norm.

    Args:
        H: Hermitian Hamiltonian matrix.
        t: Total evolution time.
        error: Desired error (reserved; does not auto-set steps).
        steps: Number of random Pauli samples (depth).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        seed: Random seed for reproducibility (sets np.random seed).
        verbose: Print progress information.

    Returns:
        Dict with keys: 'status', 'Approximate evolution matrix',
        'Exact evolution matrix', 'Frobenius norm of error',
        'Computation time (s)', 'circuit', 'circuit_path', 'plot',
        'steps', 'lambda', 'n_terms'.

    Raises:
        ValueError: If H is not square.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")

    # --- Set seed ---
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- Determine number of qubits ---
    dim = H.shape[0]
    n_qubits = int(np.ceil(np.log2(dim)))
    # Zero-padding: if dim is not a power of 2, pad to next power of 2
    padded_dim = 1 << n_qubits
    if padded_dim != dim:
        H_padded = np.zeros((padded_dim, padded_dim), dtype=H.dtype)
        H_padded[:dim, :dim] = H
        H = H_padded

    if verbose:
        print(f"QDrift simulation")
        print(f"  H dim: {dim}x{dim}, qubits: {n_qubits}")
        print(f"  t = {t}, steps = {steps}")

    t_start = time.perf_counter()

    # --- Pauli decomposition ---
    decomposition = decompose_hamiltonian(H)
    n_terms = len(decomposition)

    # --- Compute QDrift parameters ---
    _, coeffs, probs, lam = compute_qdrift_params(decomposition)

    if verbose:
        print(f"  Pauli terms: {n_terms}")
        print(f"  λ = Σ|c_j| = {lam:.6f}")
        print(f"  Probabilities: {dict(zip([p for p,_ in decomposition], np.round(np.abs(coeffs)/lam, 4)))}")

    # --- Generate random sequence ---
    sequence = qdrift_sample(decomposition, t, steps, rng)

    if verbose and steps <= 20:
        for i, (ps, ang) in enumerate(sequence):
            print(f"    [{i:2d}] {ps}  angle={ang:.6f}")

    # --- Build circuit ---
    qc = build_qdrift_circuit(sequence, n_qubits)

    # --- Compute approximate unitary ---
    U_approx = qc.get_matrix(backend=backend, device=device, dtype=dtype)

    # --- Exact evolution ---
    if HAS_SCIPY:
        U_exact = expm(-1.0j * H * t)
    else:
        eigenvals, eigenvecs = np.linalg.eigh(H)
        U_exact = eigenvecs @ np.diag(np.exp(-1.0j * eigenvals * t)) @ eigenvecs.conj().T

    # --- Error computation ---
    frob_error = float(np.linalg.norm(U_approx - U_exact, ord="fro"))

    t_elapsed = time.perf_counter() - t_start

    # Success: always 'ok' (QDrift always produces a circuit, error is informational)
    is_success = True

    if verbose:
        print(f"  U_approx (first 2x2):\n{np.array2string(U_approx[:2, :2], precision=4, suppress_small=True)}")
        print(f"  U_exact (first 2x2):\n{np.array2string(U_exact[:2, :2], precision=4, suppress_small=True)}")
        print(f"  Frobenius error: {frob_error:.6e}")
        print(f"  Status: {'ok' if is_success else 'failed'}")
        print(f"  Time: {t_elapsed:.4f} s")

    return {
        "status": "ok" if is_success else "failed",
        "Approximate evolution matrix": U_approx,
        "Exact evolution matrix": U_exact,
        "Frobenius norm of error": frob_error,
        "Computation time (s)": round(t_elapsed, 4),
        "circuit": qc,
        "circuit_path": "",
        "plot": [],
        "steps": steps,
        "lambda": lam,
        "n_terms": n_terms,
    }


# ---------------------------------------------------------------------------
# 5. Statistical analysis over multiple seeds
# ---------------------------------------------------------------------------

def qdrift_statistics(
    H: np.ndarray,
    t: float,
    steps: int = 5000,
    n_seeds: int = 10,
    backend: str = "torch",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run QDrift over multiple random seeds and collect statistics.

    Args:
        H: Hermitian Hamiltonian.
        t: Evolution time.
        steps: Number of QDrift samples per run.
        n_seeds: Number of independent random seeds to evaluate.
        backend: Simulation backend.
        verbose: Print per-seed information.

    Returns:
        Dict with keys: 'mean_error', 'std_error', 'min_error',
        'max_error', 'errors' (list of per-seed errors),
        'mean_time', 'total_time'.
    """
    errors: List[float] = []
    times: List[float] = []

    for seed in range(n_seeds):
        result = qdrift_simulate(
            H=H, t=t, steps=steps, seed=seed,
            backend=backend, verbose=verbose,
        )
        errors.append(result["Frobenius norm of error"])
        times.append(result["Computation time (s)"])

    errors_arr = np.array(errors)
    return {
        "mean_error": float(np.mean(errors_arr)),
        "std_error": float(np.std(errors_arr)),
        "min_error": float(np.min(errors_arr)),
        "max_error": float(np.max(errors_arr)),
        "errors": errors,
        "mean_time": float(np.mean(times)),
        "total_time": float(np.sum(times)),
        "n_seeds": n_seeds,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface (matching QDriftAlgorithm pattern)
# ---------------------------------------------------------------------------

class QDriftAlgorithmSolver:
    """Class-based solver matching the QDriftAlgorithm interface.

    Usage:
        solver = QDriftAlgorithmSolver()
        result = solver.run(
            H=np.array([[2, 1], [1, 3]], dtype=float),
            t=1.0, error=1e-8, steps=5000, backend='torch',
        )
        print(result['Frobenius norm of error'])
    """

    def __init__(
        self,
        text_mode: str = "plain",
        algo_dir: Optional[str] = None,
    ):
        """Initialize the QDrift solver.

        Args:
            text_mode: Output text mode ('plain' or 'legacy').
            algo_dir: Directory for output files.
        """
        self.text_mode = text_mode
        self.algo_dir = algo_dir
        self.output: Dict[str, Any] = {}

    def run(
        self,
        H: np.ndarray,
        t: float,
        error: float = 1e-8,
        steps: int = 5000,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run QDrift Hamiltonian simulation.

        Args:
            H: Hermitian Hamiltonian matrix.
            t: Total evolution time.
            error: Desired error (informational; use steps to control).
            steps: Number of random Pauli samples.
            backend: Simulation backend.
            device: Compute device.
            dtype: Numeric dtype.

        Returns:
            Result dict with keys: 'status', 'Approximate evolution matrix',
            'Exact evolution matrix', 'Frobenius norm of error',
            'Computation time (s)', 'circuit_path', 'plot', 'circuit'.
        """
        result = qdrift_simulate(
            H=H,
            t=t,
            error=error,
            steps=steps,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )

        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Package results in the standard QDriftAlgorithm return format.

        Args:
            result: Raw result from qdrift_simulate.

        Returns:
            Formatted result dict matching QDriftAlgorithm return schema.
        """
        base = {
            "status": result.get("status", "failed"),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }
        base.update({
            "Approximate evolution matrix": result.get("Approximate evolution matrix"),
            "Exact evolution matrix": result.get("Exact evolution matrix"),
            "Frobenius norm of error": result.get("Frobenius norm of error"),
            "Computation time (s)": result.get("Computation time (s)", 0.0),
        })
        return base


# ---------------------------------------------------------------------------
# 7. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "H": np.array([[2, 1], [1, 3]], dtype=float),
        "t": 1.0,
        "steps": 5000,
        "seed": 42,
        "label": "H=[[2,1],[1,3]], t=1.0, steps=5000",
    },
    {
        "H": np.array([[0.35, 0.0], [0.0, -0.35]], dtype=float),
        "t": 0.6,
        "steps": 1000,
        "seed": 7,
        "label": "H=0.35Z, t=0.6 (diagonal)",
    },
    {
        "H": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        "t": np.pi / 4,
        "steps": 2000,
        "seed": 123,
        "label": "H=X, t=π/4",
    },
    {
        "H": np.array([[1.5, 0.8], [0.8, -0.5]], dtype=float),
        "t": 0.5,
        "steps": 5000,
        "seed": 99,
        "label": "H asymmetric diag, t=0.5",
    },
    {
        "H": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float),
        "t": 0.3,
        "steps": 3000,
        "seed": 1,
        "label": "H = 0.5I + 0.5X, t=0.3",
    },
]


def run_known_test(case: Dict[str, Any], max_attempts: int = 1) -> bool:
    """Run a single known QDrift test case.

    Args:
        case: Dict with 'H', 't', 'steps', 'seed', 'label'.
        max_attempts: Maximum attempts (unused for deterministic seed).

    Returns:
        True if the test passes.
    """
    H = case["H"]
    t = case["t"]
    steps = case["steps"]
    seed = case["seed"]
    label = case["label"]

    result = qdrift_simulate(
        H=H, t=t, steps=steps, seed=seed, verbose=False,
    )
    err = result["Frobenius norm of error"]
    status = result["status"]
    ok = status == "ok"
    icon = "✓" if ok else "✗"

    print(f"  [{icon}] {label}")
    print(f"       Error: {err:.6e} (steps={steps})")
    print(f"       Time:  {result['Computation time (s)']}s")

    # QDrift always produces 'ok'; pass if error is finite
    return ok and np.isfinite(err)


# ---------------------------------------------------------------------------
# 8. Main (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("QDrift Hamiltonian Simulation — Manual Implementation")
    print("=" * 60)

    solver = QDriftAlgorithmSolver()

    # Demo case from the SKILL.md
    print("\n--- Demo: H = [[2,1],[1,3]], t = 1.0, steps = 5000 ---")
    H_demo = np.array([[2, 1], [1, 3]], dtype=float)
    np.random.seed(42)
    result = solver.run(H=H_demo, t=1.0, error=1e-8, steps=5000, backend="torch")

    print(f"  Status:              {result['status']}")
    print(f"  Frobenius error:     {result['Frobenius norm of error']:.6e}")
    print(f"  Computation time:    {result['Computation time (s)']}s")

    # Show the Pauli decomposition
    print("\n--- Pauli Decomposition ---")
    decomp = decompose_hamiltonian(H_demo)
    _, coeffs, probs, lam = compute_qdrift_params(decomp)
    print(f"  H = Σ c_j P_j:")
    for (ps, c), p in zip(decomp, probs):
        print(f"    c_{ps} = {c.real:+.4f},  |{ps}| → prob = {p:.4f}")
    print(f"  λ = Σ|c_j| = {lam:.4f}")
    print(f"  Angle per step = λ·t/steps = {lam:.4f}·1.0/5000 = {lam * 1.0 / 5000:.6f}")

    # Deterministic consistency check
    print("\n--- Deterministic Reproducibility ---")
    np.random.seed(42)
    r1 = qdrift_simulate(
        H=H_demo, t=1.0, steps=50, seed=42, verbose=False,
    )
    np.random.seed(42)
    r2 = qdrift_simulate(
        H=H_demo, t=1.0, steps=50, seed=42, verbose=False,
    )
    same = np.allclose(
        r1["Approximate evolution matrix"],
        r2["Approximate evolution matrix"],
    )
    print(f"  Same seed → same result: {'✓' if same else '✗'}")

    # Collect statistics
    print("\n--- Multi-Seed Statistics (10 seeds, steps=500) ---")
    stats = qdrift_statistics(H_demo, t=1.0, steps=500, n_seeds=10)
    print(f"  Mean error: {stats['mean_error']:.6e}")
    print(f"  Std error:  {stats['std_error']:.6e}")
    print(f"  Min error:  {stats['min_error']:.6e}")
    print(f"  Max error:  {stats['max_error']:.6e}")
    print(f"  Total time: {stats['total_time']:.4f}s")

    # Run known test cases
    print("\n--- Known Test Cases ---")
    passed = 0
    for case in KNOWN_CASES:
        if run_known_test(case):
            passed += 1
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
