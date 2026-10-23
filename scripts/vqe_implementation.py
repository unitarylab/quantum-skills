"""Manual implementation of Variational Quantum Eigensolver (VQE).

VQE is a hybrid quantum-classical algorithm that finds the ground-state
energy of a Hamiltonian by variationally optimizing a parameterized
quantum circuit (ansatz). It accepts any Hermitian matrix or generates
a random one for benchmarking.

Algorithm:
    1. Validate or generate a Hermitian Hamiltonian H (2ⁿ×2ⁿ).
    2. Initialize parameters θ ∈ R^{2·n·L} uniformly in [−π, π].
    3. COBYLA optimization: per iteration, build ansatz circuit,
       compute ⟨ψ(θ)|H|ψ(θ)⟩, append to energy history.
    4. Return VQE energy, exact energy, and absolute error.

Ansatz per layer: Ry(θ[q,0]) → Rz(θ[q,1]) on each qubit, then ring CX.

Total parameters: 2 × n × layers.

Components:
    - random_hermitian: Generate random Hermitian matrix
    - validate_hamiltonian: Validate and extract qubit count
    - build_ansatz: Single ansatz layer (Ry+Rz + ring CX)
    - build_vqe_circuit: Stack L ansatz layers
    - expectation: Compute ⟨ψ(θ)|H|ψ(θ)⟩
    - vqe_solve: Full COBYLA hybrid loop
    - VQEAlgorithm: Class-based interface

Reference:
    SKILL.md — VQE
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from unitarylab.core import Circuit


# ===================================================================
# 1. Hamiltonian utilities
# ===================================================================


def validate_hamiltonian(
    hamiltonian: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Validate a Hamiltonian matrix and extract the qubit count.

    Checks: square, power-of-2 dimension, Hermitian.

    Args:
        hamiltonian: Input matrix.

    Returns:
        Tuple of (validated_array, num_qubits).

    Raises:
        ValueError: If validation fails.
    """
    h = np.asarray(hamiltonian, dtype=np.complex128)

    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")

    dim = h.shape[0]
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hamiltonian dimension {dim} must be a power of 2")

    if not np.allclose(h, h.conj().T):
        raise ValueError("Hamiltonian must be Hermitian (H = H†)")

    num_qubits = dim.bit_length() - 1
    return h, num_qubits


def random_hermitian(
    num_qubits: int,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate a random Hermitian matrix of dimension 2^n.

    Builds H = (A + A†)/2 from a random complex Gaussian matrix A.
    Optionally normalizes by spectral norm (‖H‖₂).

    Args:
        num_qubits: Number of qubits (dimension = 2^n).
        seed: Random seed for reproducibility.
        normalize: If True, divide by spectral norm.

    Returns:
        Hermitian matrix of shape (2^n, 2^n), dtype complex128.
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


def exact_ground_energy(hamiltonian: np.ndarray) -> float:
    """Compute the exact ground-state energy of a Hermitian matrix.

    Uses numpy.linalg.eigvalsh (specialized for Hermitian matrices).

    Args:
        hamiltonian: Hermitian matrix.

    Returns:
        Minimum eigenvalue (float).
    """
    evals = np.linalg.eigvalsh(hamiltonian)
    return float(np.min(np.real(evals)))


# ===================================================================
# 2. VQE ansatz builder
# ===================================================================


def build_ansatz(
    layer_parameters: np.ndarray,
    num_qubits: int,
) -> Circuit:
    """Build a single VQE ansatz layer.

    Per qubit: Ry(θ[q,0]) → Rz(θ[q,1])
    Then ring CX: CX(0,1) → CX(1,2) → ... → CX(n-1, 0)

    Args:
        layer_parameters: Array of shape (num_qubits, 2).
        num_qubits: Number of qubits.

    Returns:
        Ansatz Circuit for one layer.
    """
    ansatz = Circuit(num_qubits, name=f"Ansatz_L")

    # Rotation gates per qubit
    for q in range(num_qubits):
        ansatz.ry(float(layer_parameters[q, 0]), q)
        ansatz.rz(float(layer_parameters[q, 1]), q)

    # Ring CX entanglement
    for q in range(num_qubits - 1):
        ansatz.cx(q, q + 1)
    if num_qubits > 1:
        ansatz.cx(num_qubits - 1, 0)

    return ansatz


def build_vqe_circuit(
    parameters_flat: np.ndarray,
    num_qubits: int,
    layers: int,
) -> Circuit:
    """Build the full VQE circuit by stacking L ansatz layers.

    Args:
        parameters_flat: Flat parameter array of length 2·n·L.
        num_qubits: Number of qubits.
        layers: Number of variational layers.

    Returns:
        Full VQE Circuit.
    """
    params = np.asarray(parameters_flat, dtype=float).reshape(layers, num_qubits, 2)
    qc = Circuit(num_qubits, name=f"VQE_n{num_qubits}_L{layers}")

    for layer in range(layers):
        layer_qc = build_ansatz(params[layer], num_qubits)
        qc.append(layer_qc, list(range(num_qubits)))

    return qc


# ===================================================================
# 3. Energy expectation value
# ===================================================================


def compute_expectation(
    parameters_flat: np.ndarray,
    hamiltonian: np.ndarray,
    num_qubits: int,
    layers: int,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> float:
    """Compute ⟨ψ(θ)|H|ψ(θ)⟩ via statevector simulation.

    Args:
        parameters_flat: Flat parameter array.
        hamiltonian: Hermitian matrix (2ⁿ×2ⁿ).
        num_qubits: Number of qubits.
        layers: Number of ansatz layers.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Energy expectation value (real float).
    """
    qc = build_vqe_circuit(parameters_flat, num_qubits, layers)
    state = qc.execute(backend=backend, device=device, dtype=dtype).state
    energy = float(np.real((state.conj().T @ hamiltonian @ state).item()))
    return energy


# ===================================================================
# 4. End-to-end VQE solver
# ===================================================================


def vqe_solve(
    hamiltonian: Optional[np.ndarray] = None,
    n: int = 2,
    layers: int = 2,
    max_iter: int = 150,
    seed: int = 7,
    normalize: bool = True,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run VQE to find the ground-state energy of a Hamiltonian.

    Pipeline:
        1. Validate or generate Hermitian Hamiltonian.
        2. Initialize parameters θ ∈ [−π, π].
        3. COBYLA minimizes ⟨ψ(θ)|H|ψ(θ)⟩.
        4. Compare VQE energy with exact ground energy.

    Args:
        hamiltonian: User-supplied Hermitian matrix (2ⁿ×2ⁿ). If None,
            a random Hamiltonian is generated.
        n: Number of qubits (only used if hamiltonian=None).
        layers: Number of variational layers. More = more expressive.
        max_iter: Maximum COBYLA iterations.
        seed: Random seed for parameter init and random Hamiltonian.
        normalize: If True, normalize random Hamiltonian by spectral norm.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - status: 'ok' on success
            - VQE Energy: Optimized energy from COBYLA
            - Exact Energy: True ground-state energy
            - Absolute Error: |VQE − Exact|
            - Optimizer Message: COBYLA termination reason
            - energy_history: Per-iteration energies
            - Quantum Comp Time (s): Optimization wall-clock time
            - n_qubits, layers, n_params: Config values

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If hamiltonian validation fails.
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for VQE. Install with: pip install scipy"
        )

    # --- Stage 1: Hamiltonian + Parameters ---
    if hamiltonian is None:
        hamiltonian = random_hermitian(n, seed=seed, normalize=normalize)
        source = "random"
    else:
        source = "user-supplied"

    hamiltonian, num_qubits = validate_hamiltonian(hamiltonian)
    if source == "user-supplied":
        n = num_qubits

    exact_energy = exact_ground_energy(hamiltonian)
    n_params = 2 * num_qubits * layers

    rng = np.random.default_rng(seed)
    initial_theta = rng.uniform(-np.pi, np.pi, size=n_params)

    if verbose:
        print(f"VQE — Ground State Energy Estimation")
        print(f"  Hamiltonian:    {source}, dim={hamiltonian.shape[0]} (2^{num_qubits})")
        print(f"  Qubits:         {num_qubits}")
        print(f"  Layers:         {layers}")
        print(f"  Parameters:     {n_params} (= 2 × {num_qubits} × {layers})")
        print(f"  Exact E₀:       {exact_energy:.6f}")
        print(f"  COBYLA maxiter: {max_iter}")

    # --- Stage 2: Preview circuit ---
    qc_preview = build_vqe_circuit(initial_theta, num_qubits, layers)

    # --- Stage 3: COBYLA optimization ---
    if verbose:
        print(f"  Running COBYLA...")

    energy_history: List[float] = []

    def objective(params_flat: np.ndarray) -> float:
        energy = compute_expectation(
            params_flat, hamiltonian, num_qubits, layers,
            backend=backend, device=device, dtype=dtype,
        )
        energy_history.append(energy)
        return energy

    opt_start = time.perf_counter()
    opt_result = minimize(
        fun=objective,
        x0=initial_theta,
        method="COBYLA",
        options={"maxiter": max_iter},
    )
    opt_time = time.perf_counter() - opt_start

    vqe_energy = float(opt_result.fun)
    abs_error = abs(vqe_energy - exact_energy)
    n_evals = len(energy_history)

    if verbose:
        print(f"  Optimization done: {opt_time:.2f}s")
        print(f"  Function evals:    {n_evals}")
        print(f"  VQE Energy:        {vqe_energy:.6f}")
        print(f"  Exact Energy:      {exact_energy:.6f}")
        print(f"  Absolute Error:    {abs_error:.2e}")
        print(f"  COBYLA message:    {opt_result.message}")
        print(f"  Status:            ok")

    return {
        "status": "ok",
        "VQE Energy": vqe_energy,
        "Exact Energy": exact_energy,
        "Absolute Error": abs_error,
        "Optimizer Message": str(opt_result.message),
        "energy_history": energy_history,
        "Quantum Comp Time (s)": round(opt_time, 2),
        "n_evals": n_evals,
        "n_qubits": num_qubits,
        "layers": layers,
        "n_params": n_params,
        "circuit": qc_preview,
        "circuit_path": "",
        "plot": [],
    }


# ===================================================================
# 5. Visualization
# ===================================================================


def plot_vqe_convergence(
    energy_history: List[float],
    exact_energy: Optional[float] = None,
    output_path: str = "VQE_Convergence.svg",
) -> str:
    """Plot energy convergence over COBYLA iterations.

    Args:
        energy_history: Per-evaluation energy values.
        exact_energy: Optional exact ground energy for reference line.
        output_path: Output file path.

    Returns:
        Absolute path to saved plot.
    """
    if not HAS_MATPLOTLIB:
        return ""

    import os

    plt.figure(figsize=(6, 4))
    plt.plot(energy_history, color="#9b59b6", lw=2, label="VQE Energy")
    if exact_energy is not None:
        plt.axhline(
            y=exact_energy, color="#e74c3c", ls="--", lw=1.5,
            label=f"Exact E₀ = {exact_energy:.4f}",
        )
    plt.xlabel("Function Evaluation")
    plt.ylabel("Energy ⟨H⟩")
    plt.title("VQE Energy Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.abspath(output_path)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# ===================================================================
# 6. Class-based interface
# ===================================================================


class VQEAlgorithm:
    """Class-based VQE solver.

    Usage:
        solver = VQEAlgorithm()
        result = solver.run(n=2, layers=3, max_iter=200)
        print(result['VQE Energy'])

        # With custom Hamiltonian
        H = np.array([[...], [...]])  # 2ⁿ×2ⁿ Hermitian
        result2 = solver.run(hamiltonian=H, layers=3)
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        n: int = 2,
        layers: int = 2,
        max_iter: int = 150,
        seed: int = 7,
        hamiltonian: Optional[np.ndarray] = None,
        normalize: bool = True,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run VQE. See vqe_solve() for docs."""
        result = vqe_solve(
            hamiltonian=hamiltonian, n=n, layers=layers,
            max_iter=max_iter, seed=seed, normalize=normalize,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        # Generate convergence plot
        if self.algo_dir and HAS_MATPLOTLIB:
            import os

            os.makedirs(self.algo_dir, exist_ok=True)
            plot_path = plot_vqe_convergence(
                result["energy_history"], result["Exact Energy"],
                os.path.join(self.algo_dir, "VQE_Convergence.svg"),
            )
            result["plot"] = [{"format": "svg", "filename": plot_path}]
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "VQE Energy": result.get("VQE Energy"),
            "Exact Energy": result.get("Exact Energy"),
            "Absolute Error": result.get("Absolute Error"),
            "Optimizer Message": result.get("Optimizer Message"),
            "Quantum Comp Time (s)": result.get("Quantum Comp Time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


# ===================================================================
# 7. Known test cases
# ===================================================================

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "n2_L2_m150",
        "n": 2, "layers": 2, "max_iter": 150, "seed": 7,
        "max_error": 0.01,
    },
    {
        "name": "n2_L4_m200",
        "n": 2, "layers": 4, "max_iter": 200, "seed": 42,
        "max_error": 0.001,
    },
    {
        "name": "n3_L3_m200",
        "n": 3, "layers": 3, "max_iter": 200, "seed": 0,
        "max_error": 0.05,
    },
    {
        "name": "n2_L6_m300",
        "n": 2, "layers": 6, "max_iter": 300, "seed": 123,
        "max_error": 0.001,
    },
    {
        "name": "n1_L2_m100",
        "n": 1, "layers": 2, "max_iter": 100, "seed": 42,
        "max_error": 0.01,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = vqe_solve(
        n=case["n"], layers=case["layers"],
        max_iter=case["max_iter"], seed=case["seed"],
        verbose=False,
    )
    error = result["Absolute Error"]
    ok = error < case["max_error"]
    icon = "ok" if ok else "WARN"
    print(f"  [{icon}] {name}: VQE={result['VQE Energy']:.6f}, "
          f"Exact={result['Exact Energy']:.6f}, "
          f"error={error:.2e} (max={case['max_error']:.0e}), "
          f"evals={result['n_evals']}")
    return ok


# ===================================================================
# 8. Main
# ===================================================================

def main() -> None:
    """Run the complete VQE demonstration pipeline."""
    if not HAS_SCIPY:
        print("scipy is not installed. Install with: pip install scipy")
        return

    print("=" * 60)
    print("VQE — Variational Quantum Eigensolver")
    print("  Ground-State Energy Estimation")
    print("=" * 60)

    solver = VQEAlgorithm()

    # --- Demo 1: Basic 2-qubit ---
    print("\n--- Demo 1: 2-qubit random Hamiltonian (L=2, max_iter=150) ---")
    result1 = solver.run(n=2, layers=2, max_iter=150, seed=7)
    print(f"  VQE Energy:    {result1['VQE Energy']:.6f}")
    print(f"  Exact Energy:  {result1['Exact Energy']:.6f}")
    print(f"  Error:         {result1['Absolute Error']:.2e}")

    # --- Demo 2: Deeper ansatz ---
    print("\n--- Demo 2: 2-qubit, deeper ansatz (L=4, max_iter=200) ---")
    result2 = solver.run(n=2, layers=4, max_iter=200, seed=42)
    print(f"  VQE Energy:    {result2['VQE Energy']:.6f}")
    print(f"  Exact Energy:  {result2['Exact Energy']:.6f}")
    print(f"  Error:         {result2['Absolute Error']:.2e}")

    # --- Demo 3: 3-qubit ---
    print("\n--- Demo 3: 3-qubit (L=3, max_iter=200) ---")
    result3 = solver.run(n=3, layers=3, max_iter=200, seed=0)
    print(f"  VQE Energy:    {result3['VQE Energy']:.6f}")
    print(f"  Exact Energy:  {result3['Exact Energy']:.6f}")
    print(f"  Error:         {result3['Absolute Error']:.2e}")

    # --- Demo 4: Effect of layers ---
    print("\n--- Demo 4: Effect of ansatz depth (n=2, same Hamiltonian) ---")
    H4 = random_hermitian(2, seed=123)
    exact4 = exact_ground_energy(H4)
    print(f"  Exact E₀: {exact4:.6f}")
    for l_val in [1, 2, 3, 4, 6, 8]:
        result4 = vqe_solve(
            hamiltonian=H4, layers=l_val, max_iter=200,
            seed=42, verbose=False,
        )
        print(f"  L={l_val}: VQE={result4['VQE Energy']:.6f}, "
              f"error={result4['Absolute Error']:.2e}, "
              f"evals={result4['n_evals']}")

    # --- Demo 5: Energy convergence ---
    print("\n--- Demo 5: Energy Convergence (n=2, L=3) ---")
    result5 = vqe_solve(n=2, layers=3, max_iter=150, seed=42, verbose=False)
    hist = result5["energy_history"]
    if len(hist) > 1:
        print(f"  Start energy: {hist[0]:.6f}")
        print(f"  End energy:   {hist[-1]:.6f}")
        print(f"  Improvement:  {hist[0] - hist[-1]:.6f}")
        print(f"  Evaluations:  {len(hist)}")

    # --- Demo 6: Reproducibility ---
    print("\n--- Demo 6: Reproducibility (same seed → same result) ---")
    for run in range(3):
        result6 = vqe_solve(
            n=2, layers=2, max_iter=100, seed=42, verbose=False,
        )
        print(f"  Run {run+1}: VQE={result6['VQE Energy']:.6f}, "
              f"error={result6['Absolute Error']:.2e}")

    # --- Demo 7: Custom Hamiltonian ---
    print("\n--- Demo 7: Custom Hamiltonian (Ising ZZ + X terms) ---")
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    H_custom = (
        1.0 * np.kron(Z, Z)
        - 0.5 * np.kron(X, I)
        - 0.5 * np.kron(I, X)
    )
    result7 = vqe_solve(
        hamiltonian=H_custom, layers=3, max_iter=200, seed=42, verbose=False,
    )
    print(f"  H = 1.0·ZZ − 0.5·XI − 0.5·IX")
    print(f"  VQE Energy:    {result7['VQE Energy']:.6f}")
    print(f"  Exact Energy:  {result7['Exact Energy']:.6f}")
    print(f"  Error:         {result7['Absolute Error']:.2e}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")


if __name__ == "__main__":
    main()
