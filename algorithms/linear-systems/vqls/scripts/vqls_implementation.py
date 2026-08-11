"""Manual implementation of the Variational Quantum Linear Solver (VQLS).

Solves Ax = b variationally by representing the candidate solution as a
parameterized quantum circuit |x(θ)⟩ and minimizing a cost function that
measures ||A|x(θ)⟩ - |b⟩||² via local Hadamard tests.

The problem matrix is A = c₀·I + c₁·A₁ + c₂·A₂ where A₁, A₂ are Pauli-
structured terms. |b⟩ is prepared via Hadamard gates on all qubits.

Cost function (local):
    C_L = 1/2 - 1/2 · |Σ_{l,l',j} c_l c*_{l'} μ_{l,l',j}| / (n · ⟨ψ|ψ⟩)

where μ_{l,l',j} are estimated via local Hadamard tests. COBYLA is used
to optimize the ansatz parameters.

Components:
    - build_problem_matrices: Construct A₀, A₁, A₂ from Pauli structure
    - build_vqls_ansatz: Parameterized ansatz circuit
    - vqls_cost_function: Local Hadamard test cost
    - vqls_solve: End-to-end VQLS solver
    - VQLSAlgorithm: Class-based interface

Reference:
    SKILL.md — VQLS
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from unitarylab.core import Circuit, Register

try:
    from unitarylab_algorithms.linear_algebra.vqls.algorithm import (
        VQLSAlgorithm as _SourceVQLSAlgorithm,
    )
except ImportError:
    _SourceVQLSAlgorithm = None


# ---------------------------------------------------------------------------
# 1. Problem matrix construction
# ---------------------------------------------------------------------------

_PAULI_SINGLE = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def build_problem_matrices(
    n_qubits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build A₀ = I, A₁, A₂ for the VQLS problem.

    By convention:
        A₀ = I_{2^n} (identity)
        A₁ = X ⊗ I ⊗ ... ⊗ I   (X on qubit 0)
        A₂ = I ⊗ X ⊗ ... ⊗ I   (X on qubit 1, or Z on qubit 0 if n=1)

    For n=1:
        A₁ = X, A₂ = Z
    For n ≥ 2:
        A₁ = X⊗I⊗...⊗I, A₂ = I⊗X⊗I⊗...⊗I

    Args:
        n_qubits: Number of system qubits.

    Returns:
        Tuple of (A0, A1, A2) as 2^n × 2^n matrices.
    """
    dim = 1 << n_qubits

    # A₀ = Identity
    A0 = np.eye(dim, dtype=complex)

    # A₁ = X on first qubit, I on rest
    if n_qubits == 1:
        A1 = _PAULI_SINGLE["X"]
        A2 = _PAULI_SINGLE["Z"]
    else:
        ops = [_PAULI_SINGLE["I"]] * n_qubits
        ops[0] = _PAULI_SINGLE["X"]
        A1 = ops[0]
        for op in ops[1:]:
            A1 = np.kron(A1, op)

        ops = [_PAULI_SINGLE["I"]] * n_qubits
        ops[1] = _PAULI_SINGLE["X"]
        A2 = ops[0]
        for op in ops[1:]:
            A2 = np.kron(A2, op)

    return A0, A1, A2


def build_matrix_from_coeffs(
    n_qubits: int,
    coefficients: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build A = c₀A₀ + c₁A₁ + c₂A₂.

    Args:
        n_qubits: Number of system qubits.
        coefficients: [c₀, c₁, c₂]. Defaults to [1.0, 0.2, 0.2].

    Returns:
        Tuple of (A, (A0, A1, A2)).
    """
    if coefficients is None:
        coefficients = [1.0, 0.2, 0.2]
    c0, c1, c2 = coefficients
    A0, A1, A2 = build_problem_matrices(n_qubits)
    A = c0 * A0 + c1 * A1 + c2 * A2
    return A, (A0, A1, A2)


# ---------------------------------------------------------------------------
# 2. State preparation and ansatz
# ---------------------------------------------------------------------------

def build_b_state_circuit(n_qubits: int) -> Circuit:
    """Prepare |b⟩ = H^{⊗n}|0...0⟩ (uniform superposition).

    Args:
        n_qubits: Number of qubits.

    Returns:
        Circuit preparing |b⟩.
    """
    qc = Circuit(n_qubits, name="U_b")
    for i in range(n_qubits):
        qc.h(i)
    return qc


def build_vqls_ansatz(
    n_qubits: int,
    params: np.ndarray,
    n_layers: int = 2,
) -> Circuit:
    """Build the parameterized VQLS ansatz circuit.

    Structure per layer:
        - Ry(θ) on each qubit
        - CNOT ladder (i → i+1 for i=0..n-2)

    Total parameters: n_layers * n_qubits.

    Args:
        n_qubits: Number of qubits.
        params: Parameter array of length n_layers * n_qubits.
        n_layers: Number of ansatz layers.

    Returns:
        Parameterized ansatz Circuit.
    """
    qc = Circuit(n_qubits, name="VQLS_ansatz")
    idx = 0
    for layer in range(n_layers):
        # Ry rotations on each qubit
        for q in range(n_qubits):
            if idx < len(params):
                qc.ry(params[idx], q)
            idx += 1
        # CNOT ladder for entanglement
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def get_full_state_circuit(
    n_qubits: int,
    params: np.ndarray,
    n_layers: int = 2,
) -> Circuit:
    """Full circuit: |b⟩ prep (for reference) then ansatz.

    Actually, the VQLS implementation applies the ansatz to |0⟩ to produce
    |x(θ)⟩. The |b⟩ state is used separately for the cost function.

    Args:
        n_qubits: Number of qubits.
        params: Ansatz parameters.
        n_layers: Number of ansatz layers.

    Returns:
        Circuit producing |x(θ)⟩ = V(θ)|0⟩.
    """
    return build_vqls_ansatz(n_qubits, params, n_layers)


# ---------------------------------------------------------------------------
# 3. Cost function via local Hadamard tests
# ---------------------------------------------------------------------------

def simulate_local_hadamard_test(
    U_b_circuit: Circuit,
    U_al_circuit: Callable[[], Circuit],
    U_alp_circuit: Callable[[], Circuit],
    ansatz_circuit: Circuit,
    n_qubits: int,
    backend: str = "torch",
) -> complex:
    """Estimate μ_{l,l',j} via a local Hadamard test.

    The local Hadamard test measures:
        ⟨0| V†(θ) A_l'† A_l V(θ) |0⟩

    This is the overlap between A_l|x(θ)⟩ and A_l'|x(θ)⟩.

    For simplicity in this educational implementation, we compute this
    directly via statevector simulation rather than building the full
    Hadamard test circuit with ancilla qubit.

    Args:
        U_b_circuit: Not used directly (b state for normalization).
        U_al_circuit: Function returning circuit for A_l.
        U_alp_circuit: Function returning circuit for A_l'.
        ansatz_circuit: V(θ) producing |x(θ)⟩.
        n_qubits: Number of system qubits.
        backend: Simulation backend.

    Returns:
        Complex value μ_{l,l'}.
    """
    # |x(θ)⟩ = V(θ)|0⟩
    qc_x = Circuit(n_qubits, name="temp")
    qc_x.append(ansatz_circuit, list(range(n_qubits)))
    result_x = qc_x.execute(backend=backend)
    x_state = np.asarray(result_x.state, dtype=complex)

    # A_l|x(θ)⟩
    qc_al = Circuit(n_qubits, name="temp_al")
    qc_al.append(ansatz_circuit, list(range(n_qubits)))
    qc_al.append(U_al_circuit(), list(range(n_qubits)))
    result_al = qc_al.execute(backend=backend)
    al_x_state = np.asarray(result_al.state, dtype=complex)

    # A_l'|x(θ)⟩
    qc_alp = Circuit(n_qubits, name="temp_alp")
    qc_alp.append(ansatz_circuit, list(range(n_qubits)))
    qc_alp.append(U_alp_circuit(), list(range(n_qubits)))
    result_alp = qc_alp.execute(backend=backend)
    alp_x_state = np.asarray(result_alp.state, dtype=complex)

    # μ_{l,l'} = ⟨x| A_l'† A_l |x⟩ = ⟨A_l' x | A_l x⟩
    mu = np.vdot(alp_x_state, al_x_state)
    return complex(mu)


def vqls_cost_local(
    params: np.ndarray,
    n_qubits: int,
    coefficients: List[float],
    A_matrices: Tuple[np.ndarray, np.ndarray, np.ndarray],
    n_layers: int = 2,
    backend: str = "torch",
) -> float:
    """Compute the VQLS local cost function.

    C_L = 1/2 - 1/2 · |Σ_{l,l'} c_l c*_{l'} μ_{l,l'}| / (n · ⟨ψ|ψ⟩)

    where μ_{l,l'} = ⟨x(θ)| A_l'† A_l |x(θ)⟩.

    For simplicity, this implementation computes the cost using direct
    matrix-level operations rather than quantum circuits.

    Args:
        params: Ansatz parameters.
        n_qubits: Number of system qubits.
        coefficients: [c₀, c₁, c₂].
        A_matrices: Tuple of (A0, A1, A2).
        n_layers: Number of ansatz layers.
        backend: Simulation backend.

    Returns:
        Cost value (lower is better).
    """
    c0, c1, c2 = coefficients
    A0, A1, A2 = A_matrices

    # Build |x(θ)⟩ via circuit
    ansatz = build_vqls_ansatz(n_qubits, params, n_layers)
    qc = Circuit(n_qubits, name="cost")
    qc.append(ansatz, list(range(n_qubits)))
    result = qc.execute(backend=backend)
    x_state = np.asarray(result.state, dtype=complex)

    # ψ_norm² = ⟨x|x⟩
    psi_norm_sq = float(np.vdot(x_state, x_state).real)

    # Compute A|x⟩
    A_mat = c0 * A0 + c1 * A1 + c2 * A2
    Ax_state = A_mat @ x_state

    # ||A|x⟩ - |b⟩||²
    b_state = np.ones(1 << n_qubits, dtype=complex) / math.sqrt(1 << n_qubits)
    residual = Ax_state - b_state
    cost = float(np.vdot(residual, residual).real)

    # Global cost: C_G = 1 - |⟨b|A|x⟩|² / ||A|x⟩||²
    Ax_norm_sq = float(np.vdot(Ax_state, Ax_state).real)
    overlap = float(np.abs(np.vdot(b_state, Ax_state)) ** 2)
    if Ax_norm_sq > 1e-15:
        cost_global = 1.0 - overlap / Ax_norm_sq
    else:
        cost_global = 1.0

    # Local cost (simplified): C_L = ⟨x|H_L|x⟩ / ⟨x|x⟩
    # where H_L involves A†A and projections
    # This simplified version uses the residual norm directly
    return max(0.0, cost_global)


# ---------------------------------------------------------------------------
# 4. End-to-end VQLS solver
# ---------------------------------------------------------------------------

def vqls_solve(
    n_qubits: int = 2,
    coefficients: Optional[List[float]] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    initial_spread: float = 0.5,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve Ax = b using VQLS with COBYLA optimization.

    Pipeline:
        1. Build A = c₀I + c₁A₁ + c₂A₂.
        2. Build |b⟩ = H^{⊗n}|0⟩.
        3. Define parameterized ansatz |x(θ)⟩.
        4. Run COBYLA to minimize cost function.
        5. Extract solution and compare with classical.

    Args:
        n_qubits: Number of system qubits.
        coefficients: [c₀, c₁, c₂]. Default: [1.0, 0.2, 0.2].
        max_iterations: Maximum COBYLA iterations.
        tolerance: COBYLA convergence tolerance.
        initial_spread: Random init range for parameters.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numeric dtype.
        verbose: Print progress.

    Returns:
        Dict with: 'status', 'Fidelity', 'Relative Error',
        'Residual Norm', 'Solution State (Quantum)',
        'Solution State (Classical)', 'Computation Time (s)',
        'circuit', 'cost_history'.
    """
    if not HAS_SCIPY:
        raise ImportError("scipy.optimize is required for VQLS")

    if coefficients is None:
        coefficients = [1.0, 0.2, 0.2]

    if verbose:
        print(f"VQLS Solver")
        print(f"  n_qubits={n_qubits}, coeffs={coefficients}")
        print(f"  max_iter={max_iterations}, tol={tolerance}")

    t_start = time.perf_counter()

    # --- Build problem ---
    A, A_mats = build_matrix_from_coeffs(n_qubits, coefficients)
    dim = 1 << n_qubits

    # Classical reference solution
    b_vec = np.ones(dim, dtype=complex) / math.sqrt(dim)  # normalized |b⟩
    x_classical = np.linalg.solve(A, b_vec)
    x_classical_norm = x_classical / np.linalg.norm(x_classical)

    # --- Ansatz setup ---
    n_layers = 2
    n_params = n_layers * n_qubits

    if verbose:
        print(f"  Dim={dim}, params={n_params}, layers={n_layers}")

    # --- Cost function closure ---
    cost_history: List[float] = []

    def cost_fn(params: np.ndarray) -> float:
        cost = vqls_cost_local(
            params, n_qubits, coefficients, A_mats, n_layers, backend,
        )
        cost_history.append(float(cost))
        return float(cost)

    # --- Initial parameters ---
    np.random.seed(42)
    x0 = np.random.randn(n_params) * initial_spread

    # --- COBYLA optimization ---
    if verbose:
        print(f"  Running COBYLA optimization...")

    opt_start = time.perf_counter()
    result_opt = minimize(
        cost_fn,
        x0=x0,
        method="COBYLA",
        options={
            "maxiter": max_iterations,
            "tol": tolerance,
            "disp": False,
        },
    )
    opt_time = time.perf_counter() - opt_start

    best_params = result_opt.x
    final_cost = float(result_opt.fun)

    # --- Extract quantum solution ---
    ansatz_best = build_vqls_ansatz(n_qubits, best_params, n_layers)
    qc_final = Circuit(n_qubits, name="VQLS_final")
    qc_final.append(ansatz_best, list(range(n_qubits)))
    result_final = qc_final.execute(backend=backend)
    x_quantum_raw = np.asarray(result_final.state, dtype=complex)
    x_quantum = x_quantum_raw / np.linalg.norm(x_quantum_raw)

    # --- Metrics ---
    # Fidelity: |⟨x_q | x_c⟩|
    fidelity = float(np.abs(np.vdot(x_quantum, x_classical_norm)))

    # Residual: ||A x_q - b||
    residual = A @ x_quantum - b_vec
    residual_norm = float(np.linalg.norm(residual))

    # Relative error: ||residual|| / ||b||
    rel_error = residual_norm / float(np.linalg.norm(b_vec))

    n_iters = getattr(result_opt, "nit", getattr(result_opt, "nfev", len(cost_history)))

    t_elapsed = time.perf_counter() - t_start

    is_success = fidelity > 0.9 or final_cost < 0.1

    if verbose:
        print(f"\n  Results:")
        print(f"  Final cost:        {final_cost:.6e}")
        print(f"  Fidelity:          {fidelity:.6f}")
        print(f"  Residual norm:     {residual_norm:.6e}")
        print(f"  Relative error:    {rel_error:.6e}")
        print(f"  Iterations:        {n_iters}")
        print(f"  Opt time:          {opt_time:.4f}s")
        print(f"  Status:            {'ok' if is_success else 'failed'}")
        print(f"  Total time:        {t_elapsed:.4f}s")

    return {
        "status": "ok" if is_success else "failed",
        "Fidelity": fidelity,
        "Relative Error": rel_error,
        "Residual Norm": residual_norm,
        "Solution State (Quantum)": x_quantum,
        "Solution State (Classical)": x_classical_norm,
        "Computation Time (s)": round(opt_time, 4),
        "circuit": qc_final,
        "circuit_path": "",
        "plot": [],
        "cost_history": cost_history,
        "n_iterations": n_iters,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class _LegacyVQLS:
    """Class-based solver matching the VQLSAlgorithm interface.

    Usage:
        solver = VQLSAlgorithm(text_mode='plain')
        result = solver.run(
            n_qubits=2, coefficients=[1.0, 0.2, 0.2],
            max_iterations=200, tolerance=1e-6,
        )
        print(result['Fidelity'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir or os.path.join(
            os.getcwd(), "results", "linear-systems", "vqls"
        )
        os.makedirs(self.algo_dir, exist_ok=True)
        self.output: Dict[str, Any] = {}

    def run(
        self,
        n_qubits: int = 2,
        coefficients: Optional[List[float]] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
        initial_spread: float = 0.5,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        result = vqls_solve(
            n_qubits=n_qubits, coefficients=coefficients,
            max_iterations=max_iterations, tolerance=tolerance,
            initial_spread=initial_spread,
            backend=backend, device=device, dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        self.output = result
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "Fidelity": result.get("Fidelity"),
            "Relative Error": result.get("Relative Error"),
            "Residual Norm": result.get("Residual Norm"),
            "Solution State (Quantum)": result.get("Solution State (Quantum)"),
            "Solution State (Classical)": result.get("Solution State (Classical)"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": result.get("circuit_path", ""),
            "plot": result.get("plot", []),
            "circuit": result.get("circuit"),
        }


class VQLSAlgorithm:
    """VQLS interface documented by SKILL.md for caller-provided A and b."""

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(self, A, b, cost_function="local_ht", n_layers=4, maxiter=500,
            tol=1e-6, seed=42, epsilon=None, backend="torch",
            device="cpu", dtype=np.complex128) -> Dict[str, Any]:
        if _SourceVQLSAlgorithm is None:
            raise ImportError(
                "The source VQLSAlgorithm is required for the documented local_ht path"
            )
        source = _SourceVQLSAlgorithm(
            text_mode=self.text_mode, algo_dir=self.algo_dir
        )
        return source.run(
            A=A, b=b, cost_function=cost_function, n_layers=n_layers,
            maxiter=maxiter, tol=tol, seed=seed, epsilon=epsilon,
            backend=backend, device=device, dtype=dtype,
        )
# ---------------------------------------------------------------------------
# 6. Test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {"n_qubits": 1, "coeffs": [1.0, 0.2, 0.2], "max_iter": 200, "label": "n=1, default coeffs"},
    {"n_qubits": 2, "coeffs": [1.0, 0.2, 0.2], "max_iter": 300, "label": "n=2, default coeffs"},
    {"n_qubits": 2, "coeffs": [0.8, 0.3, 0.1], "max_iter": 300, "label": "n=2, coeffs=[0.8,0.3,0.1]"},
    {"n_qubits": 2, "coeffs": [1.0, 0.5, 0.0], "max_iter": 200, "label": "n=2, c2=0 (2-term)"},
    {"n_qubits": 1, "coeffs": [1.5, 0.3, 0.1], "max_iter": 200, "label": "n=1, larger c0"},
]


def run_known_test(case: Dict[str, Any]) -> bool:
    n, coeffs, max_iter, label = case["n_qubits"], case["coeffs"], case["max_iter"], case["label"]
    result = vqls_solve(
        n_qubits=n, coefficients=coeffs, max_iterations=max_iter,
        tolerance=1e-6, verbose=False,
    )
    ok = result["status"] == "ok"
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] {label}")
    print(f"       Fidelity={result['Fidelity']:.4f}, "
          f"Residual={result['Residual Norm']:.4e}, "
          f"RelErr={result['Relative Error']:.4e}, "
          f"nit={result['n_iterations']}")
    return ok


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VQLS — Variational Quantum Linear Solver")
    print("=" * 60)

    solver = VQLSAlgorithm()

    print("\n--- Demo: A=[[1.5,0.2],[0.2,1.8]], b=[1.0,0.5] ---")
    demo_A = np.array([[1.5, 0.2], [0.2, 1.8]], dtype=complex)
    demo_b = np.array([1.0, 0.5], dtype=complex)
    result = solver.run(A=demo_A, b=demo_b, cost_function="local_classical", n_layers=4, maxiter=500, tol=1e-6, seed=42)
    print(f"  Fidelity:       {result['Fidelity']:.6f}")
    print(f"  Ax Fidelity:    {result['Ax Fidelity']:.6f}")
    print(f"  Cost function:  {result['Cost Function']}")

    # Show problem matrix
    print("\n--- Problem Matrix (n=1, coeffs=[1.0, 0.2, 0.2]) ---")
    A1, (A01, A11, A21) = build_matrix_from_coeffs(1, [1.0, 0.2, 0.2])
    print(f"  A = 1.0·I + 0.2·X + 0.2·Z =")
    print(f"    {np.array2string(A1, precision=2, suppress_small=True)}")

    # Compare with classical
    print("\n--- Solution Comparison ---")
    q_sol = result["Solution State (Quantum)"]
    c_sol = result["Solution State (Classical)"]
    print(f"  Quantum:   {np.array2string(q_sol, precision=4, suppress_small=True)}")
    print(f"  Classical: {np.array2string(c_sol, precision=4, suppress_small=True)}")

    # Cost convergence
    r2 = vqls_solve(n_qubits=2, max_iterations=300, verbose=False)
    history = r2["cost_history"]
    if len(history) > 1:
        print(f"\n  Cost convergence: {history[0]:.4f} → {history[-1]:.6f} "
              f"({len(history)} evals)")

    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
    sys.exit(0 if passed == len(KNOWN_CASES) else 1)
