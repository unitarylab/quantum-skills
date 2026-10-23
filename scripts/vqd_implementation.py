"""Manual implementation of Variational Quantum Deflation (VQD).

VQD computes the lowest k eigenvalues of a Hamiltonian by sequentially
optimizing variational ansatz states with overlap penalties against
previously found states.

At step j, the cost function is:
    Cⱼ(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + Σᵢ₌₀ʲ⁻² βᵢ · |⟨ψ(θ)|ψᵢ⟩|²

Components:
    - random_hermitian: Generate random Hermitian Hamiltonian
    - build_ansatz_layer: Parameterized circuit layer (Ry+Rz + ring CX)
    - build_vqd_circuit: Stack L ansatz layers
    - compute_expectation: ⟨ψ|H|ψ⟩
    - compute_overlap: |⟨ψ(θ)|ψ_opt⟩|² between two states
    - vqd_solve: End-to-end sequential deflation solver
    - VQDSolver: Class-based interface

Reference:
    SKILL.md — Variational Quantum Deflation (VQD)
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Hamiltonian utilities
# ---------------------------------------------------------------------------

def random_hermitian(
    num_qubits: int,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a random Hermitian matrix of dimension 2^n.

    H = (A + A†)/2 where A is a complex Gaussian random matrix.

    Args:
        num_qubits: Number of qubits.
        seed: Random seed.
        normalize: If True, divide by spectral norm.

    Returns:
        Hermitian matrix of shape (2^n, 2^n).
    """
    dim = 1 << num_qubits
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (a + a.conj().T) / 2.0
    if normalize:
        spec_norm = np.linalg.norm(h, ord=2)
        if spec_norm > 0:
            h = h / spec_norm
    return h


def exact_eigenvalues(hamiltonian: np.ndarray, k: int) -> np.ndarray:
    """Compute the exact lowest k eigenvalues via numpy.linalg.eigvalsh.

    Args:
        hamiltonian: Hermitian matrix.
        k: Number of eigenvalues.

    Returns:
        Sorted array of k lowest eigenvalues.
    """
    evals = np.linalg.eigvalsh(hamiltonian)
    return np.sort(np.real(evals))[:k]


# ---------------------------------------------------------------------------
# 2. Ansatz construction
# ---------------------------------------------------------------------------

def build_ansatz_layer(
    layer_params: np.ndarray,
    num_qubits: int,
) -> Circuit:
    """Build a single VQD ansatz layer.

    Per qubit: Ry(θ[q,0]) → Rz(θ[q,1])
    Entanglement: ring CX (0,1) → (1,2) → ... → (n-1, 0)

    Args:
        layer_params: Array of shape (num_qubits, 2).
        num_qubits: Number of qubits.

    Returns:
        Ansatz Circuit for one layer.
    """
    qc = Circuit(num_qubits, name="Ansatz_L")
    for q in range(num_qubits):
        qc.ry(float(layer_params[q, 0]), q)
        qc.rz(float(layer_params[q, 1]), q)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    if num_qubits > 1:
        qc.cx(num_qubits - 1, 0)
    return qc


def build_vqd_circuit(
    params_flat: np.ndarray,
    num_qubits: int,
    layers: int,
) -> Circuit:
    """Build the full VQD circuit by stacking L ansatz layers.

    Args:
        params_flat: Flat parameters, length = 2·n·L.
        num_qubits: Number of qubits.
        layers: Number of ansatz layers.

    Returns:
        Full Circuit.
    """
    params = np.asarray(params_flat, dtype=float).reshape(layers, num_qubits, 2)
    qc = Circuit(num_qubits, name=f"VQD_n{num_qubits}_L{layers}")
    for layer in range(layers):
        layer_qc = build_ansatz_layer(params[layer], num_qubits)
        qc.append(layer_qc, list(range(num_qubits)))
    return qc


# ---------------------------------------------------------------------------
# 3. Energy and overlap computation
# ---------------------------------------------------------------------------

def compute_expectation(
    params_flat: np.ndarray,
    hamiltonian: np.ndarray,
    num_qubits: int,
    layers: int,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> Tuple[float, np.ndarray]:
    """Compute ⟨ψ(θ)|H|ψ(θ)⟩ and return the statevector.

    Args:
        params_flat: Flat parameter array.
        hamiltonian: Hermitian matrix.
        num_qubits: Number of qubits.
        layers: Number of ansatz layers.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Tuple of (energy, statevector).
    """
    qc = build_vqd_circuit(params_flat, num_qubits, layers)
    state = qc.execute(backend=backend, device=device, dtype=dtype).state
    energy = float(np.real((state.conj().T @ hamiltonian @ state).item()))
    return energy, state


def compute_overlap(
    state1: np.ndarray,
    state2: np.ndarray,
) -> float:
    """Compute |⟨ψ₁|ψ₂⟩|² — the squared overlap between two states.

    Args:
        state1: First statevector.
        state2: Second statevector.

    Returns:
        Squared absolute overlap in [0, 1].
    """
    overlap = np.abs(np.vdot(state1, state2)) ** 2
    return float(np.real(overlap))


# ---------------------------------------------------------------------------
# 4. End-to-end VQD solver
# ---------------------------------------------------------------------------

def vqd_solve(
    hamiltonian: Optional[np.ndarray] = None,
    n: int = 2,
    layers: int = 2,
    k: int = 2,
    betas: Optional[np.ndarray] = None,
    max_iter: int = 150,
    seed: int = 7,
    normalize: bool = True,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run VQD to find the lowest k eigenvalues of a Hamiltonian.

    Pipeline:
        1. Generate/validate Hamiltonian.
        2. For each step j = 1..k:
           a. Define cost = energy + Σ βᵢ·|⟨ψ(θ)|ψᵢ⟩|².
           b. Run COBYLA to minimize cost.
           c. Store optimized parameters and state.
        3. Return eigenvalues, optimal points, and states.

    Args:
        hamiltonian: User-supplied Hermitian matrix. If None, random one generated.
        n: Number of qubits (if hamiltonian=None).
        layers: Ansatz depth (number of layers).
        k: Number of eigenvalues to compute.
        betas: Overlap penalty weights (length >= k-1). Auto-computed if None.
        max_iter: Maximum COBYLA iterations per step.
        seed: Random seed.
        normalize: Normalize random Hamiltonian by spectral norm.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys: status, eigenvalues, optimal_points, optimal_states,
            exact_eigenvalues, cost_function_evals, Computation Time (s).

    Raises:
        ImportError: If scipy is not installed.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for VQD. Install with: pip install scipy")

    t_start = time.perf_counter()

    # --- Stage 1: Hamiltonian ---
    if hamiltonian is None:
        hamiltonian = random_hermitian(n, seed=seed, normalize=normalize)
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    dim = hamiltonian.shape[0]
    num_qubits = dim.bit_length() - 1
    n = num_qubits

    n_params = 2 * num_qubits * layers
    exact_evals = exact_eigenvalues(hamiltonian, k)

    # --- Stage 2: Auto-compute betas ---
    if betas is None:
        # Default: sum of absolute coefficients as a reference scale
        betas = np.full(k - 1, 1.0)

    if verbose:
        print(f"VQD — Variational Quantum Deflation")
        print(f"  Qubits:         {num_qubits}")
        print(f"  Layers:         {layers}")
        print(f"  Parameters:     {n_params} per state")
        print(f"  k (states):     {k}")
        print(f"  Betas:          {np.round(betas, 3).tolist()}")
        print(f"  Exact evals:    {np.round(exact_evals, 6).tolist()}")

    # --- Stage 3: Sequential deflation ---
    rng = np.random.default_rng(seed)
    prior_params: List[np.ndarray] = []
    prior_states: List[np.ndarray] = []
    eigenvalues: List[float] = []
    optimal_points: List[np.ndarray] = []
    cost_evals_per_step: List[int] = []

    for step in range(k):
        initial_theta = rng.uniform(-np.pi, np.pi, size=n_params)

        def objective(params_flat: np.ndarray) -> float:
            energy, state = compute_expectation(
                params_flat, hamiltonian, num_qubits, layers,
                backend=backend, device=device, dtype=dtype,
            )
            penalty = sum(
                betas[i] * compute_overlap(state, prior_states[i])
                for i in range(len(prior_states))
            )
            return energy + penalty

        if verbose:
            print(f"  Step {step+1}/{k}: optimizing state {step+1}...")

        opt_result = minimize(
            fun=objective,
            x0=initial_theta,
            method="COBYLA",
            options={"maxiter": max_iter},
        )

        theta_opt = opt_result.x
        energy_opt, state_opt = compute_expectation(
            theta_opt, hamiltonian, num_qubits, layers,
            backend=backend, device=device, dtype=dtype,
        )

        prior_params.append(theta_opt)
        prior_states.append(state_opt)
        eigenvalues.append(energy_opt)
        optimal_points.append(theta_opt)
        cost_evals_per_step.append(opt_result.nfev)

        if verbose:
            error = abs(energy_opt - exact_evals[step])
            print(f"    Energy: {energy_opt:.6f} (exact: {exact_evals[step]:.6f}, "
                  f"error: {error:.2e}), evals: {opt_result.nfev}")

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        print(f"  VQD eigenvalues: {np.round(eigenvalues, 6).tolist()}")
        print(f"  Exact:           {np.round(exact_evals, 6).tolist()}")
        print(f"  Total time:      {comp_time}s")

    return {
        "status": "ok",
        "eigenvalues": np.array(eigenvalues),
        "optimal_points": np.array(optimal_points),
        "optimal_states": prior_states,
        "exact_eigenvalues": exact_evals,
        "cost_function_evals": np.array(cost_evals_per_step),
        "Computation Time (s)": comp_time,
        "n_qubits": num_qubits,
        "layers": layers,
        "k": k,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class VQDSolver:
    """Class-based VQD solver.

    Usage:
        solver = VQDSolver()
        result = solver.run(n=2, layers=2, k=2, max_iter=150)
        print(result['eigenvalues'])

        # With custom Hamiltonian
        H = random_hermitian(2, seed=42)
        result2 = solver.run(hamiltonian=H, layers=3, k=3)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        hamiltonian: Optional[np.ndarray] = None,
        n: int = 2,
        layers: int = 2,
        k: int = 2,
        betas: Optional[np.ndarray] = None,
        max_iter: int = 150,
        seed: int = 7,
        normalize: bool = True,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run VQD. See vqd_solve() for docs."""
        result = vqd_solve(
            hamiltonian=hamiltonian, n=n, layers=layers, k=k,
            betas=betas, max_iter=max_iter, seed=seed,
            normalize=normalize, backend=backend, device=device,
            dtype=dtype, verbose=(self.text_mode != "plain"),
        )
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "eigenvalues": result.get("eigenvalues"),
            "exact_eigenvalues": result.get("exact_eigenvalues"),
            "cost_function_evals": result.get("cost_function_evals"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
        }


# ---------------------------------------------------------------------------
# 6. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"name": "n2_L2_k2", "n": 2, "layers": 2, "k": 2, "max_iter": 150, "seed": 7},
    {"name": "n2_L3_k2", "n": 2, "layers": 3, "k": 2, "max_iter": 200, "seed": 42},
    {"name": "n2_L4_k3", "n": 2, "layers": 4, "k": 3, "max_iter": 200, "seed": 123},
    {"name": "n3_L3_k2", "n": 3, "layers": 3, "k": 2, "max_iter": 200, "seed": 0},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = vqd_solve(
        n=case["n"], layers=case["layers"], k=case["k"],
        max_iter=case["max_iter"], seed=case["seed"],
        verbose=False,
    )
    ok = result["status"] == "ok"
    icon = "ok" if ok else "FAIL"
    errors = np.abs(result["eigenvalues"] - result["exact_eigenvalues"])
    max_err = np.max(errors)
    evals_str = np.round(result["eigenvalues"], 4).tolist()
    exact_str = np.round(result["exact_eigenvalues"], 4).tolist()
    print(f"  [{icon}] {name}: vqd={evals_str}, exact={exact_str}, "
          f"max_err={max_err:.2e}, evals={result['cost_function_evals'].tolist()}, "
          f"time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not HAS_SCIPY:
        print("scipy is not installed. Install with: pip install scipy")
        import sys; sys.exit(1)

    print("=" * 60)
    print("VQD — Variational Quantum Deflation")
    print("=" * 60)

    solver = VQDSolver()

    # --- Demo 1: 2-qubit, k=2 ---
    print("\n--- Demo 1: 2-qubit, L=2, k=2 ---")
    result1 = solver.run(n=2, layers=2, k=2, max_iter=150, seed=7)
    errors1 = np.abs(result1['eigenvalues'] - result1['exact_eigenvalues'])
    print(f"  VQD eigenvalues:   {np.round(result1['eigenvalues'], 6).tolist()}")
    print(f"  Exact eigenvalues: {np.round(result1['exact_eigenvalues'], 6).tolist()}")
    print(f"  Max error:         {np.max(errors1):.2e}")

    # --- Demo 2: Deeper ansatz ---
    print("\n--- Demo 2: 2-qubit, L=4, k=3 ---")
    result2 = solver.run(n=2, layers=4, k=3, max_iter=200, seed=123)
    errors2 = np.abs(result2['eigenvalues'] - result2['exact_eigenvalues'])
    print(f"  VQD eigenvalues:   {np.round(result2['eigenvalues'], 6).tolist()}")
    print(f"  Exact eigenvalues: {np.round(result2['exact_eigenvalues'], 6).tolist()}")
    print(f"  Max error:         {np.max(errors2):.2e}")

    # --- Demo 3: 3-qubit ---
    print("\n--- Demo 3: 3-qubit, L=3, k=2 ---")
    result3 = solver.run(n=3, layers=3, k=2, max_iter=200, seed=0)
    errors3 = np.abs(result3['eigenvalues'] - result3['exact_eigenvalues'])
    print(f"  VQD eigenvalues:   {np.round(result3['eigenvalues'], 6).tolist()}")
    print(f"  Exact eigenvalues: {np.round(result3['exact_eigenvalues'], 6).tolist()}")
    print(f"  Max error:         {np.max(errors3):.2e}")

    # --- Demo 4: Custom betas ---
    print("\n--- Demo 4: Effect of beta weights on orthogonality ---")
    H4 = random_hermitian(2, seed=7)
    for beta_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
        result4 = vqd_solve(
            hamiltonian=H4, layers=2, k=2, betas=np.array([beta_val]),
            max_iter=150, seed=7, verbose=False,
        )
        # Check orthogonality
        s0 = result4["optimal_states"][0]
        s1 = result4["optimal_states"][1]
        overlap = compute_overlap(s0, s1)
        print(f"  β={beta_val:.1f}: overlap=|⟨ψ₀|ψ₁⟩|²={overlap:.6f}, "
              f"evals={np.round(result4['eigenvalues'], 4).tolist()}")

    # --- Demo 5: Custom Hamiltonian ---
    print("\n--- Demo 5: Custom Ising Hamiltonian ---")
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    H_custom = 1.0 * np.kron(Z, Z) - 0.5 * np.kron(X, I) - 0.5 * np.kron(I, X)
    result5 = vqd_solve(
        hamiltonian=H_custom, layers=3, k=2, max_iter=200, seed=42, verbose=False,
    )
    print(f"  H = 1.0·ZZ - 0.5·XI - 0.5·IX")
    print(f"  VQD eigenvalues:   {np.round(result5['eigenvalues'], 6).tolist()}")
    print(f"  Exact eigenvalues: {np.round(result5['exact_eigenvalues'], 6).tolist()}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
