"""Manual implementation of Finite Difference Gradient.

Numerically approximates parameter gradients of parameterized quantum circuits
using finite difference schemes: central, forward, and backward. Universal —
works with any differentiable gate without gate-set restrictions.

The gradient of f(θ) w.r.t. θᵢ is approximated as:
    Central:   (f(θ + ε·êᵢ) - f(θ - ε·êᵢ)) / (2ε)     [O(ε²)]
    Forward:   (f(θ + ε·êᵢ) - f(θ)          ) / ε       [O(ε)]
    Backward:  (f(θ)           - f(θ - ε·êᵢ)) / ε       [O(ε)]

Components:
    - build_parameterized_circuit: Build a circuit with RX/RY/RZ gates
    - compute_expectation: Evaluate ⟨ψ(θ)|O|ψ(θ)⟩ via statevector simulation
    - finite_difference_gradient: Core gradient estimation
    - finite_diff_estimator_solve: End-to-end gradient pipeline
    - FiniteDiffEstimatorGradientSolver: Class-based interface

Reference:
    SKILL.md — Finite Difference Gradient
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Pauli / expectation utilities
# ---------------------------------------------------------------------------

_PAULI_MAP: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """Convert a Pauli string to its dense matrix representation."""
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
# 2. Parameterized circuit construction
# ---------------------------------------------------------------------------

def build_parameterized_circuit(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Circuit:
    """Build a parameterized circuit with named rotation gates.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: List of (gate_type, qubit_indices) where gate_type
            is one of 'rx', 'ry', 'rz'.
        params: Flat parameter array, one value per parameterized gate.

    Returns:
        Circuit with gates applied.
    """
    qc = Circuit(num_qubits, name="ParamCircuit")
    p_idx = 0  # separate counter for parameterized gates only
    for gate, qubits in gate_sequence:
        if gate in ("rx", "ry", "rz"):
            val = float(params[p_idx])
            p_idx += 1
            qb = qubits[0]
            if gate == "rx":
                qc.rx(val, qb)
            elif gate == "ry":
                qc.ry(val, qb)
            else:
                qc.rz(val, qb)
        elif gate == "cx":
            qc.cx(qubits[0], qubits[1])
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
    """Compute ⟨ψ|O|ψ⟩ for state produced by circuit qc.

    Args:
        qc: Circuit producing state |ψ⟩.
        observable_matrix: Hermitian observable matrix.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Real expectation value (float).
    """
    state = qc.execute(backend=backend, device=device, dtype=dtype).state
    return float(np.real((state.conj().T @ observable_matrix @ state).item()))


# ---------------------------------------------------------------------------
# 3. Finite difference gradient computation
# ---------------------------------------------------------------------------

def finite_difference_gradient(
    eval_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 1e-3,
    method: str = "central",
) -> np.ndarray:
    """Compute gradient via finite difference.

    Args:
        eval_fn: Function f(θ) -> float to differentiate.
        theta: Current parameter vector of length n.
        epsilon: Perturbation step size (> 0).
        method: 'central', 'forward', or 'backward'.

    Returns:
        Gradient vector of length n.

    Raises:
        ValueError: If epsilon <= 0 or method is invalid.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    valid_methods = {"central", "forward", "backward"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    n = len(theta)
    grad = np.zeros(n, dtype=float)

    f0 = eval_fn(theta) if method in ("forward", "backward") else None

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = 1.0

        if method == "central":
            fp = eval_fn(theta + epsilon * ei)
            fm = eval_fn(theta - epsilon * ei)
            grad[i] = (fp - fm) / (2.0 * epsilon)
        elif method == "forward":
            fp = eval_fn(theta + epsilon * ei)
            grad[i] = (fp - f0) / epsilon
        elif method == "backward":
            fm = eval_fn(theta - epsilon * ei)
            grad[i] = (f0 - fm) / epsilon

    return grad


# ---------------------------------------------------------------------------
# 4. Analytic gradient (for comparison)
# ---------------------------------------------------------------------------

def analytic_gradient_rz(
    theta: np.ndarray,
    num_qubits: int,
    observable_matrix: np.ndarray,
    backend: str = "torch",
) -> np.ndarray:
    """Compute analytic gradient for circuits with only RZ gates.

    For RZ(θ) gates: ∂⟨O⟩/∂θᵢ = (⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)) / 2

    This is the parameter-shift rule, used here for comparison.
    """
    n = len(theta)
    grad = np.zeros(n)

    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = 1.0

        qc_plus = build_parameterized_circuit(
            num_qubits,
            [("rz", [q]) for q in range(n)],
            theta + (np.pi / 2) * ei,
        )
        fp = compute_expectation(qc_plus, observable_matrix, backend=backend)

        qc_minus = build_parameterized_circuit(
            num_qubits,
            [("rz", [q]) for q in range(n)],
            theta - (np.pi / 2) * ei,
        )
        fm = compute_expectation(qc_minus, observable_matrix, backend=backend)

        grad[i] = (fp - fm) / 2.0

    return grad


# ---------------------------------------------------------------------------
# 5. End-to-end gradient pipeline
# ---------------------------------------------------------------------------

def finite_diff_estimator_solve(
    pauli_list: List[Tuple[str, float]],
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    parameter_values: np.ndarray,
    epsilon: float = 1e-2,
    method: str = "central",
    parameters: Optional[List[int]] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute gradients of expectation values via finite difference.

    Pipeline:
        1. Build observable matrix from Pauli terms.
        2. Define the evaluation function f(θ) = ⟨ψ(θ)|O|ψ(θ)⟩.
        3. Compute gradient via finite difference scheme.
        4. Optionally compute analytic gradient for comparison.

    Args:
        pauli_list: Observable defined as Pauli terms.
        num_qubits: Number of qubits.
        gate_sequence: Gate layout for the parameterized circuit.
        parameter_values: Current parameter values (length = n_params).
        epsilon: Perturbation step size.
        method: 'central', 'forward', or 'backward'.
        parameters: Indices of parameters to differentiate (None = all).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys:
            - status: 'ok'
            - gradients: Estimated gradient array
            - analytic_gradient: Parameter-shift gradient (if applicable)
            - method, epsilon: Config values
            - num_evals: Number of function evaluations
            - Computation Time (s): Wall-clock time
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Invalid method: '{method}'")

    t_start = time.perf_counter()

    # --- Build observable ---
    observable_matrix = observable_to_matrix(pauli_list)

    # --- Restrict to requested parameters ---
    theta = np.asarray(parameter_values, dtype=float)
    if parameters is not None:
        theta = theta[parameters]

    # --- Define evaluation function ---
    def eval_fn(params: np.ndarray) -> float:
        full_params = np.asarray(parameter_values, dtype=float).copy()
        if parameters is not None:
            for j, idx in enumerate(parameters):
                full_params[idx] = params[j]
        else:
            full_params = params
        qc = build_parameterized_circuit(num_qubits, gate_sequence, full_params)
        return compute_expectation(qc, observable_matrix, backend=backend, device=device, dtype=dtype)

    # --- Compute gradient ---
    n_evals_map = {"central": 2 * len(theta), "forward": len(theta) + 1, "backward": len(theta) + 1}
    n_evals = n_evals_map[method]

    gradient = finite_difference_gradient(eval_fn, theta, epsilon=epsilon, method=method)

    # --- Analytic comparison (if circuit is RZ-only) ---
    analytic_grad: Optional[np.ndarray] = None
    if all(g[0] == "rz" for g in gate_sequence):
        analytic_grad = analytic_gradient_rz(theta, num_qubits, observable_matrix, backend=backend)

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        print(f"Finite Difference Gradient")
        print(f"  Method:        {method}")
        print(f"  Epsilon:       {epsilon}")
        print(f"  n_params:      {len(theta)}")
        print(f"  Evaluations:   {n_evals}")
        print(f"  Gradient:      {np.round(gradient, 6).tolist()}")
        if analytic_grad is not None:
            diff = np.max(np.abs(gradient - analytic_grad))
            print(f"  Analytic diff: {diff:.2e}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "gradients": gradient,
        "analytic_gradient": analytic_grad,
        "method": method,
        "epsilon": epsilon,
        "num_evals": n_evals,
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class FiniteDiffEstimatorGradientSolver:
    """Class-based solver for Finite Difference Gradient.

    Usage:
        solver = FiniteDiffEstimatorGradientSolver()
        result = solver.run(
            pauli_list=[("ZZ", 1.0)],
            num_qubits=2,
            gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
            parameter_values=[0.5, 1.0],
            epsilon=1e-2,
            method="central",
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
        epsilon: float = 1e-2,
        method: str = "central",
        parameters: Optional[List[int]] = None,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run finite difference gradient estimation."""
        result = finite_diff_estimator_solve(
            pauli_list=pauli_list,
            num_qubits=num_qubits,
            gate_sequence=gate_sequence,
            parameter_values=np.array(parameter_values),
            epsilon=epsilon,
            method=method,
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
            "method": result.get("method"),
            "epsilon": result.get("epsilon"),
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
        }


# ---------------------------------------------------------------------------
# 7. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "2q_RY_central",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "epsilon": 1e-2,
        "method": "central",
    },
    {
        "name": "2q_RY_forward",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "epsilon": 1e-2,
        "method": "forward",
    },
    {
        "name": "2q_RZ_central",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("rz", [0]), ("rz", [1])],
        "parameter_values": [0.5, 1.0],
        "epsilon": 1e-3,
        "method": "central",
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = finite_diff_estimator_solve(
        pauli_list=case["pauli_list"],
        num_qubits=case["num_qubits"],
        gate_sequence=case["gate_sequence"],
        parameter_values=np.array(case["parameter_values"]),
        epsilon=case["epsilon"],
        method=case["method"],
        verbose=False,
    )
    ok = result["status"] == "ok"
    icon = "ok" if ok else "FAIL"
    grad_str = np.round(result["gradients"], 4).tolist()
    diff_str = ""
    if result["analytic_gradient"] is not None:
        diff = np.max(np.abs(result["gradients"] - result["analytic_gradient"]))
        diff_str = f", max_diff={diff:.2e}"
    print(f"  [{icon}] {name}: grad={grad_str}, evals={result['num_evals']}, "
          f"time={result['Computation Time (s)']}s{diff_str}")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Finite Difference Gradient — Manual Implementation")
    print("=" * 60)

    solver = FiniteDiffEstimatorGradientSolver()

    # --- Demo 1: Central difference, RY gates ---
    print("\n--- Demo 1: Central difference, 2-qubit RY circuit ---")
    result1 = solver.run(
        pauli_list=[("ZZ", 1.0)],
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
        epsilon=1e-2,
        method="central",
    )
    print(f"  Gradient: {np.round(result1['gradients'], 6).tolist()}")

    # --- Demo 2: Compare central/forward/backward ---
    print("\n--- Demo 2: Method comparison (same circuit) ---")
    for method in ["central", "forward", "backward"]:
        result2 = solver.run(
            pauli_list=[("ZZ", 1.0)],
            num_qubits=2,
            gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
            parameter_values=[0.5, 1.0],
            epsilon=1e-3,
            method=method,
        )
        print(f"  {method:>10}: {np.round(result2['gradients'], 6).tolist()}")

    # --- Demo 3: Central vs analytic (parameter-shift) for RZ circuit ---
    print("\n--- Demo 3: Central FD vs analytic parameter-shift (RZ circuit) ---")
    pauli_zz = [("ZZ", 1.0)]
    params_rz = np.array([0.5, 1.0])
    gate_seq_rz = [("rz", [0]), ("rz", [1])]
    obs_matrix = observable_to_matrix(pauli_zz)

    result3 = finite_diff_estimator_solve(
        pauli_list=pauli_zz,
        num_qubits=2,
        gate_sequence=gate_seq_rz,
        parameter_values=params_rz,
        epsilon=1e-3,
        method="central",
        verbose=False,
    )
    analytic_grad = result3["analytic_gradient"]
    fd_grad = result3["gradients"]

    print(f"  Finite diff gradient: {np.round(fd_grad, 6).tolist()}")
    print(f"  Analytic gradient:    {np.round(analytic_grad, 6).tolist()}")
    print(f"  Max difference:       {np.max(np.abs(fd_grad - analytic_grad)):.2e}")

    # --- Demo 4: Effect of epsilon ---
    print("\n--- Demo 4: Effect of epsilon on accuracy (RZ circuit) ---")
    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        result4 = finite_diff_estimator_solve(
            pauli_list=pauli_zz,
            num_qubits=2,
            gate_sequence=gate_seq_rz,
            parameter_values=params_rz,
            epsilon=eps,
            method="central",
            verbose=False,
        )
        diff = np.max(np.abs(result4["gradients"] - result4["analytic_gradient"]))
        print(f"  ε={eps:.0e}: max diff = {diff:.2e}")

    # --- Demo 5: Parameter subset differentiation ---
    print("\n--- Demo 5: Differentiate only first parameter (θ₀) ---")
    result5 = solver.run(
        pauli_list=[("ZZ", 1.0)],
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
        epsilon=1e-2,
        method="central",
        parameters=[0],
    )
    print(f"  ∂E/∂θ₀ = {np.round(result5['gradients'], 6).tolist()} (length {len(result5['gradients'])})")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
