"""Manual implementation of SPSA (Simultaneous Perturbation Stochastic Approximation) Gradient.

SPSA estimates gradients using random perturbation vectors, requiring only
2·batch_size circuit evaluations per gradient call — independent of the number
of parameters. This makes it efficient for circuits with many parameters.

For each random perturbation δ ∈ {+1, -1}ᵈ:
    ĝ(θ) = mean_k[(f(θ + ε·δ^{(k)}) - f(θ - ε·δ^{(k)})) / (2ε) ⊙ δ^{(k)}]

Components:
    - build_parameterized_circuit: Build circuit from gate sequence
    - compute_expectation: Evaluate ⟨ψ|O|ψ⟩
    - spsa_gradient: Core SPSA gradient estimation
    - spsa_estimator_solve: End-to-end gradient pipeline
    - SPSAEstimatorGradientSolver: Class-based interface

Reference:
    SKILL.md — SPSA Gradient Estimation
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Pauli / observable utilities
# ---------------------------------------------------------------------------

_PAULI_MAP: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Convert Pauli string to dense matrix."""
    result = np.array([[1.0]], dtype=np.complex128)
    for ch in pauli_str:
        result = np.kron(result, _PAULI_MAP.get(ch, _PAULI_MAP["I"]))
    return result


def observable_to_matrix(pauli_list: List[Tuple[str, float]]) -> np.ndarray:
    """Build observable matrix from Pauli terms."""
    if not pauli_list:
        raise ValueError("pauli_list must not be empty")
    num_qubits = len(pauli_list[0][0])
    matrix = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=np.complex128)
    for pauli_str, coeff in pauli_list:
        matrix += coeff * pauli_string_to_matrix(pauli_str)
    return matrix


# ---------------------------------------------------------------------------
# 2. Parameterized circuit
# ---------------------------------------------------------------------------

def build_parameterized_circuit(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Circuit:
    """Build a circuit from gate sequence.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: List of (gate, qubits).
        params: Parameter values.

    Returns:
        Circuit.
    """
    qc = Circuit(num_qubits, name="SPSA")
    param_idx = 0
    for gate, qubits in gate_sequence:
        q0 = qubits[0]
        if gate in ("rx", "ry", "rz"):
            val = float(params[param_idx])
            param_idx += 1
            if gate == "rx":
                qc.rx(val, q0)
            elif gate == "ry":
                qc.ry(val, q0)
            elif gate == "rz":
                qc.rz(val, q0)
        elif gate == "cx":
            qc.cx(q0, qubits[1])
        elif gate == "h":
            qc.h(q0)
        elif gate == "x":
            qc.x(q0)
        elif gate == "y":
            qc.y(q0)
        elif gate == "z":
            qc.z(q0)
        else:
            raise ValueError(f"Unsupported gate: {gate}")
    return qc


def compute_expectation(
    qc: Circuit,
    observable_matrix: np.ndarray,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> float:
    """Compute ⟨ψ|O|ψ⟩ via statevector simulation."""
    state = qc.execute(backend=backend, device=device, dtype=dtype).state
    return float(np.real((state.conj().T @ observable_matrix @ state).item()))


# ---------------------------------------------------------------------------
# 3. SPSA gradient
# ---------------------------------------------------------------------------

def spsa_gradient(
    eval_fn,
    theta: np.ndarray,
    epsilon: float = 0.01,
    batch_size: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Compute SPSA gradient estimate.

    For each of `batch_size` random perturbations δ^{(k)} ∈ {+1, -1}ᵈ:
        ĝᵢ = mean_k[(f(θ + εδ^{(k)}) - f(θ - εδ^{(k)})) / (2ε) · 1/δᵢ^{(k)}]

    Since δᵢ ∈ {+1, -1}, dividing by δᵢ is equivalent to multiplying by δᵢ.

    Args:
        eval_fn: Function f(θ) -> float.
        theta: Parameter vector of length n.
        epsilon: Perturbation size (> 0).
        batch_size: Number of random perturbations to average.
        seed: RNG seed for reproducibility.

    Returns:
        Gradient estimate of length n.

    Raises:
        ValueError: If epsilon <= 0.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    n = len(theta)
    rng = np.random.default_rng(seed)

    grads = np.zeros((batch_size, n), dtype=float)

    for k in range(batch_size):
        # Random sign vector: δ ∈ {+1, -1}ⁿ
        delta = (-1) ** rng.integers(0, 2, size=n).astype(float)

        f_plus = eval_fn(theta + epsilon * delta)
        f_minus = eval_fn(theta - epsilon * delta)

        # ĝᵢ = (f_plus - f_minus) / (2ε) · (1/δᵢ)
        # Since δᵢ = ±1, 1/δᵢ = δᵢ
        grads[k] = ((f_plus - f_minus) / (2.0 * epsilon)) * delta

    return np.mean(grads, axis=0)


# ---------------------------------------------------------------------------
# 4. End-to-end solver
# ---------------------------------------------------------------------------

def spsa_estimator_solve(
    pauli_list: List[Tuple[str, float]],
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    parameter_values: np.ndarray,
    epsilon: float = 0.01,
    batch_size: int = 4,
    seed: Optional[int] = None,
    parameters: Optional[List[int]] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Estimate gradients via SPSA.

    Pipeline:
        1. Build observable matrix.
        2. Define evaluation function.
        3. Generate batch_size random sign vectors.
        4. Evaluate f at θ ± εδ for each perturbation.
        5. Average gradient estimates over batch.

    Args:
        pauli_list: Observable as Pauli terms.
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        parameter_values: Current parameter values.
        epsilon: Perturbation size.
        batch_size: Number of SPSA samples.
        seed: RNG seed.
        parameters: Indices of params to differentiate (None = all).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys: status, gradients, batch_size, epsilon, num_evals,
            Computation Time (s).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    t_start = time.perf_counter()

    # --- Build observable ---
    observable_matrix = observable_to_matrix(pauli_list)

    # --- Identify parameters ---
    param_indices = [i for i, (g, _) in enumerate(gate_sequence) if g in ("rx", "ry", "rz")]
    if parameters is not None:
        effective_indices = [param_indices[idx] for idx in parameters]
    else:
        effective_indices = param_indices

    theta_full = np.asarray(parameter_values, dtype=float)

    # --- Evaluation function ---
    def eval_fn(params: np.ndarray) -> float:
        qc = build_parameterized_circuit(num_qubits, gate_sequence, params)
        return compute_expectation(qc, observable_matrix, backend=backend, device=device, dtype=dtype)

    # --- SPSA gradient for the full parameter set ---
    full_grad = spsa_gradient(
        eval_fn, theta_full, epsilon=epsilon, batch_size=batch_size, seed=seed,
    )

    # --- Select requested parameters ---
    gradient = full_grad[effective_indices]

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)
    num_evals = 2 * batch_size  # θ+εδ and θ-εδ for each perturbation

    if verbose:
        print(f"SPSA Gradient")
        print(f"  n_params:      {len(effective_indices)}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Epsilon:       {epsilon}")
        print(f"  Evaluations:   {num_evals} (= 2 × batch_size)")
        print(f"  Gradient:      {np.round(gradient, 6).tolist()}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "gradients": gradient,
        "full_gradients": full_grad,
        "batch_size": batch_size,
        "epsilon": epsilon,
        "num_evals": num_evals,
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class SPSAEstimatorGradientSolver:
    """Class-based solver for SPSA Gradient.

    Usage:
        solver = SPSAEstimatorGradientSolver()
        result = solver.run(
            pauli_list=[("ZZ", 1.0)],
            num_qubits=2,
            gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
            parameter_values=[0.5, 1.0],
            epsilon=0.01, batch_size=4, seed=123,
        )
        print(result['gradients'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        pauli_list: List[Tuple[str, float]],
        num_qubits: int,
        gate_sequence: List[Tuple[str, List[int]]],
        parameter_values: List[float],
        epsilon: float = 0.01,
        batch_size: int = 4,
        seed: Optional[int] = None,
        parameters: Optional[List[int]] = None,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run SPSA gradient estimation."""
        result = spsa_estimator_solve(
            pauli_list=pauli_list,
            num_qubits=num_qubits,
            gate_sequence=gate_sequence,
            parameter_values=np.array(parameter_values),
            epsilon=epsilon,
            batch_size=batch_size,
            seed=seed,
            parameters=parameters,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "gradients": result.get("gradients"),
            "batch_size": result.get("batch_size"),
            "epsilon": result.get("epsilon"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
        }


# ---------------------------------------------------------------------------
# 6. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "2q_RY_B4",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "epsilon": 0.01,
        "batch_size": 4,
        "seed": 123,
    },
    {
        "name": "2q_RY_B10",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "epsilon": 0.01,
        "batch_size": 10,
        "seed": 42,
    },
    {
        "name": "1q_RY_B4",
        "pauli_list": [("Z", 1.0)],
        "num_qubits": 1,
        "gate_sequence": [("ry", [0])],
        "parameter_values": [0.5],
        "epsilon": 0.01,
        "batch_size": 4,
        "seed": 7,
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = spsa_estimator_solve(
        pauli_list=case["pauli_list"],
        num_qubits=case["num_qubits"],
        gate_sequence=case["gate_sequence"],
        parameter_values=np.array(case["parameter_values"]),
        epsilon=case["epsilon"],
        batch_size=case["batch_size"],
        seed=case["seed"],
        verbose=False,
    )
    ok = result["status"] == "ok"
    icon = "ok" if ok else "FAIL"
    grad_str = np.round(result["gradients"], 4).tolist()
    print(f"  [{icon}] {name}: grad={grad_str}, B={result['batch_size']}, "
          f"evals={result['num_evals']}, time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SPSA Gradient — Manual Implementation")
    print("=" * 60)

    solver = SPSAEstimatorGradientSolver()

    # --- Demo 1: Basic SPSA ---
    print("\n--- Demo 1: ⟨ZZ⟩ SPSA gradient, 2-qubit RY + CX, B=4 ---")
    pauli_zz = [("ZZ", 1.0)]
    gate_seq = [("ry", [0]), ("ry", [1]), ("cx", [0, 1])]
    params_test = [0.5, 1.0]

    result1 = solver.run(
        pauli_list=pauli_zz,
        num_qubits=2,
        gate_sequence=gate_seq,
        parameter_values=params_test,
        epsilon=0.01, batch_size=4, seed=123,
    )
    print(f"  SPSA gradient: {np.round(result1['gradients'], 6).tolist()}")

    # --- Demo 2: Compare with exact (parameter-shift) gradient ---
    print("\n--- Demo 2: SPSA vs analytic parameter-shift comparison ---")
    obs_mat = observable_to_matrix(pauli_zz)

    # Analytic parameter-shift gradient
    def exact_grad_fn(theta):
        n = len(theta)
        g = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            qcp = build_parameterized_circuit(2, gate_seq, theta + (np.pi/2) * ei)
            qcm = build_parameterized_circuit(2, gate_seq, theta - (np.pi/2) * ei)
            fp = compute_expectation(qcp, obs_mat)
            fm = compute_expectation(qcm, obs_mat)
            g[i] = (fp - fm) / 2.0
        return g

    exact_grad = exact_grad_fn(np.array(params_test))
    spsa_grad = result1["gradients"]
    print(f"  SPSA (B=4):       {np.round(spsa_grad, 6).tolist()}")
    print(f"  Exact (PS rule):  {np.round(exact_grad, 6).tolist()}")
    print(f"  Difference:       {np.max(np.abs(spsa_grad - exact_grad)):.2e}")
    print(f"  Note: SPSA is stochastic; larger batch_size reduces variance.")

    # --- Demo 3: Effect of batch size ---
    print("\n--- Demo 3: Effect of batch size on SPSA accuracy ---")
    exact_g = exact_grad_fn(np.array(params_test))
    for B in [1, 2, 4, 8, 16, 32]:
        result3 = spsa_estimator_solve(
            pauli_list=pauli_zz,
            num_qubits=2,
            gate_sequence=gate_seq,
            parameter_values=np.array(params_test),
            epsilon=0.01, batch_size=B, seed=42,
            verbose=False,
        )
        err = np.max(np.abs(result3["gradients"] - exact_g))
        # Expected: error ∝ 1/√B
        expected_sigma = 1.0 / np.sqrt(B)
        print(f"  B={B:>3}: grad={np.round(result3['gradients'], 4).tolist()}, "
              f"max_err={err:.4f}, expected σ∝{expected_sigma:.4f}")

    # --- Demo 4: Effect of epsilon ---
    print("\n--- Demo 4: Effect of epsilon on SPSA accuracy ---")
    for eps in [0.1, 0.05, 0.01, 0.005, 0.001]:
        result4 = spsa_estimator_solve(
            pauli_list=pauli_zz,
            num_qubits=2,
            gate_sequence=gate_seq,
            parameter_values=np.array(params_test),
            epsilon=eps, batch_size=8, seed=42,
            verbose=False,
        )
        err = np.max(np.abs(result4["gradients"] - exact_g))
        print(f"  ε={eps:.3f}: max_err={err:.4e}")

    # --- Demo 5: Subset of parameters ---
    print("\n--- Demo 5: Differentiate only θ₀ ---")
    result5 = solver.run(
        pauli_list=pauli_zz,
        num_qubits=2,
        gate_sequence=gate_seq,
        parameter_values=params_test,
        epsilon=0.01, batch_size=4, seed=123,
        parameters=[0],
    )
    print(f"  ∂E/∂θ₀ = {np.round(result5['gradients'], 6).tolist()} (length=1)")

    # --- Demo 6: Reproducibility with seed ---
    print("\n--- Demo 6: Reproducibility (same seed → same result) ---")
    for run in range(3):
        result6 = spsa_estimator_solve(
            pauli_list=pauli_zz,
            num_qubits=2,
            gate_sequence=gate_seq,
            parameter_values=np.array(params_test),
            epsilon=0.01, batch_size=4, seed=42,
            verbose=False,
        )
        print(f"  Run {run+1}: {np.round(result6['gradients'], 6).tolist()}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
