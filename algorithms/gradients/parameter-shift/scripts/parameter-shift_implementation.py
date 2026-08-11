"""Manual implementation of Parameter Shift Gradient.

Computes exact analytic gradients of parameterized quantum circuits via the
parameter shift rule. For any gate generator with eigenvalues ±½:

    ∂f/∂θᵢ = (f(θ + π/2·êᵢ) - f(θ - π/2·êᵢ)) / 2

Unlike finite difference, this introduces NO approximation error — it is exact.
Requires gates from the supported set: rx, ry, rz, rzx, rzz, ryy, rxx, cx, cy,
cz, h, x, y, z, p.

Components:
    - build_parameterized_circuit: Build circuit from gate sequence
    - compute_expectation: Evaluate ⟨ψ|O|ψ⟩ via statevector simulation
    - parameter_shift_gradient: Core gradient via ±π/2 shifts
    - param_shift_estimator_solve: End-to-end gradient pipeline
    - ParamShiftEstimatorGradientSolver: Class-based interface

Reference:
    SKILL.md — Parameter Shift Gradient
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# Supported gates for parameter shift
SUPPORTED_PARAM_GATES = {"rx", "ry", "rz", "rzx", "rzz", "ryy", "rxx", "p"}


def build_parameterized_circuit(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Circuit:
    """Build a circuit from a sequence of parameterized and fixed gates.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: List of (gate_type, qubit_list). Gate types: rx, ry, rz,
            rzx, rzz, ryy, rxx, cx, cy, cz, h, x, y, z, p.
        params: Flat parameter array for parameterized gates.

    Returns:
        Circuit with all gates applied.
    """
    qc = Circuit(num_qubits, name="ParamShift")
    param_idx = 0
    for gate, qubits in gate_sequence:
        if gate in SUPPORTED_PARAM_GATES:
            val = float(params[param_idx])
            param_idx += 1
            _apply_gate(qc, gate, qubits, val)
        else:
            _apply_gate(qc, gate, qubits)
    return qc


def _apply_gate(qc: Circuit, gate: str, qubits: List[int], val: Optional[float] = None) -> None:
    """Apply a gate to the circuit."""
    q0 = qubits[0]
    q1 = qubits[1] if len(qubits) > 1 else -1
    if gate == "rx":
        qc.rx(val, q0)
    elif gate == "ry":
        qc.ry(val, q0)
    elif gate == "rz":
        qc.rz(val, q0)
    elif gate == "p":
        qc.p(val, q0)
    elif gate == "rzx":
        qc.rzx(val, q0, q1)
    elif gate == "rzz":
        qc.rzz(val, q0, q1)
    elif gate == "ryy":
        qc.ryy(val, q0, q1)
    elif gate == "rxx":
        qc.rxx(val, q0, q1)
    elif gate == "cx":
        qc.cx(q0, q1)
    elif gate == "cy":
        qc.cy(q0, q1)
    elif gate == "cz":
        qc.cz(q0, q1)
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


# ---------------------------------------------------------------------------
# 3. Expectation value evaluation
# ---------------------------------------------------------------------------

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
# 4. Parameter shift gradient
# ---------------------------------------------------------------------------

def parameter_shift_gradient(
    eval_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    shift: float = np.pi / 2,
) -> np.ndarray:
    """Compute exact analytic gradient via parameter shift rule.

    ∂f/∂θᵢ = (f(θ + s·êᵢ) - f(θ - s·êᵢ)) / 2

    where s = π/2 for standard single-parameter rotation gates.

    Args:
        eval_fn: Function f(θ) -> float.
        theta: Parameter vector of length n.
        shift: Shift magnitude (default π/2).

    Returns:
        Exact gradient vector of length n.
    """
    n = len(theta)
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = 1.0
        fp = eval_fn(theta + shift * ei)
        fm = eval_fn(theta - shift * ei)
        grad[i] = (fp - fm) / 2.0
    return grad


# ---------------------------------------------------------------------------
# 5. End-to-end solver
# ---------------------------------------------------------------------------

def param_shift_estimator_solve(
    pauli_list: List[Tuple[str, float]],
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    parameter_values: np.ndarray,
    parameters: Optional[List[int]] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute exact gradients of expectation values via parameter shift.

    Pipeline:
        1. Build observable matrix from Pauli terms.
        2. Define evaluation function wrapping circuit execution.
        3. For each target parameter, evaluate at θ ± π/2.
        4. Assemble gradient vector.

    Args:
        pauli_list: Observable as Pauli terms.
        num_qubits: Number of qubits.
        gate_sequence: Gate layout, e.g. [("ry", [0]), ("ry", [1]), ("cx", [0, 1])].
        parameter_values: Current parameter values.
        parameters: Indices of params to differentiate (None = all).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys: status, gradients, n_params, num_evals, Computation Time (s).
    """
    t_start = time.perf_counter()

    # --- Build observable ---
    observable_matrix = observable_to_matrix(pauli_list)

    # --- Identify which gates are parameterized ---
    n_total_params = sum(g in SUPPORTED_PARAM_GATES for g, _ in gate_sequence)
    effective_indices = list(range(n_total_params)) if parameters is None else list(parameters)
    if any(idx < 0 or idx >= n_total_params for idx in effective_indices):
        raise IndexError(
            f"Parameter indices {effective_indices} are invalid for {n_total_params} parameters"
        )
    n_target = len(effective_indices)

    # --- Evaluation function ---
    def eval_fn(full_params: np.ndarray) -> float:
        qc = build_parameterized_circuit(num_qubits, gate_sequence, full_params)
        return compute_expectation(qc, observable_matrix, backend=backend, device=device, dtype=dtype)

    # --- Compute gradient for each target parameter ---
    theta_full = np.asarray(parameter_values, dtype=float)
    gradient = np.zeros(n_target, dtype=float)

    for i, param_idx in enumerate(effective_indices):
        ei = np.zeros(n_total_params, dtype=float)
        ei[param_idx] = 1.0

        fp = eval_fn(theta_full + (np.pi / 2) * ei)
        fm = eval_fn(theta_full - (np.pi / 2) * ei)
        gradient[i] = (fp - fm) / 2.0

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)
    num_evals = 2 * n_target

    if verbose:
        print(f"Parameter Shift Gradient")
        print(f"  n_params:      {n_total_params} (differentiating {n_target})")
        print(f"  Evaluations:   {num_evals} (2 per parameter)")
        print(f"  Gradient:      {np.round(gradient, 6).tolist()}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "gradients": gradient,
        "n_params": n_total_params,
        "num_evals": num_evals,
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class ParamShiftEstimatorGradientSolver:
    """Class-based solver for Parameter Shift Gradient.

    Usage:
        solver = ParamShiftEstimatorGradientSolver()
        result = solver.run(
            pauli_list=[("ZZ", 1.0)],
            num_qubits=2,
            gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
            parameter_values=[0.5, 1.0],
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
        parameters: Optional[List[int]] = None,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run parameter shift gradient estimation."""
        result = param_shift_estimator_solve(
            pauli_list=pauli_list,
            num_qubits=num_qubits,
            gate_sequence=gate_sequence,
            parameter_values=np.array(parameter_values),
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
            "Computation Time (s)": result.get("Computation Time (s)", 0.0),
            "circuit_path": "",
            "plot": [],
        }


# ---------------------------------------------------------------------------
# 7. Known test cases
# ---------------------------------------------------------------------------

KNOWN_CASES: List[Dict[str, Any]] = [
    {
        "name": "2q_RY_2params",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "description": "⟨ZZ⟩ gradient w.r.t. two RY parameters",
    },
    {
        "name": "2q_RZ_2params",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("rz", [0]), ("rz", [1])],
        "parameter_values": [0.5, 1.0],
        "description": "No entanglement — RZ on each qubit",
    },
    {
        "name": "3q_RY_3params",
        "pauli_list": [("ZZZ", 1.0)],
        "num_qubits": 3,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("ry", [2]), ("cx", [0, 1]), ("cx", [1, 2])],
        "parameter_values": [0.3, -0.2, 0.7],
        "description": "3-qubit with ring CX entangler",
    },
    {
        "name": "2q_RY_subset",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "parameters": [0],
        "description": "Differentiate only θ₀",
    },
    {
        "name": "1q_H_then_RY",
        "pauli_list": [("Z", 1.0)],
        "num_qubits": 1,
        "gate_sequence": [("h", [0]), ("ry", [0])],
        "parameter_values": [0.5],
        "description": "Fixed gate before the parameterized gate",
    },
]

KNOWN_CASES_SUBSET = {i for i, c in enumerate(KNOWN_CASES) if "parameters" in c}


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single known test case."""
    name = case["name"]
    result = param_shift_estimator_solve(
        pauli_list=case["pauli_list"],
        num_qubits=case["num_qubits"],
        gate_sequence=case["gate_sequence"],
        parameter_values=np.array(case["parameter_values"]),
        parameters=case.get("parameters"),
        verbose=False,
    )
    params = np.asarray(case["parameter_values"], dtype=float)
    observable = observable_to_matrix(case["pauli_list"])

    def eval_fn(theta: np.ndarray) -> float:
        qc = build_parameterized_circuit(case["num_qubits"], case["gate_sequence"], theta)
        return compute_expectation(qc, observable)

    finite_diff = np.zeros_like(params)
    epsilon = 1e-3
    for idx in range(len(params)):
        offset = np.zeros_like(params)
        offset[idx] = epsilon
        finite_diff[idx] = (eval_fn(params + offset) - eval_fn(params - offset)) / (2 * epsilon)
    selected = case.get("parameters", list(range(len(params))))
    expected = finite_diff[selected]
    max_error = float(np.max(np.abs(result["gradients"] - expected)))
    ok = result["status"] == "ok" and max_error < 1e-4
    icon = "ok" if ok else "FAIL"
    grad_str = np.round(result["gradients"], 4).tolist()
    print(f"  [{icon}] {name}: grad={grad_str}, fd={np.round(expected, 4).tolist()}, "
          f"max_error={max_error:.2e}, evals={result['num_evals']}, "
          f"time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Parameter Shift Gradient — Manual Implementation")
    print("=" * 60)

    solver = ParamShiftEstimatorGradientSolver()

    # --- Demo 1: Basic 2-qubit RY circuit ---
    print("\n--- Demo 1: ⟨ZZ⟩ gradient, 2-qubit RY + CX ---")
    result1 = solver.run(
        pauli_list=[("ZZ", 1.0)],
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
    )
    print(f"  Gradient: {np.round(result1['gradients'], 6).tolist()}")

    # --- Demo 2: 3-qubit circuit ---
    print("\n--- Demo 2: 3-qubit RY + ring CX, ⟨ZZZ⟩ ---")
    result2 = solver.run(
        pauli_list=[("ZZZ", 1.0)],
        num_qubits=3,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("ry", [2]), ("cx", [0, 1]), ("cx", [1, 2])],
        parameter_values=[0.3, -0.2, 0.7],
    )
    print(f"  Gradient: {np.round(result2['gradients'], 6).tolist()}")

    # --- Demo 3: Subset of parameters ---
    print("\n--- Demo 3: Differentiate only θ₀ (skip θ₁) ---")
    result3 = solver.run(
        pauli_list=[("ZZ", 1.0)],
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
        parameters=[0],
    )
    print(f"  ∂E/∂θ₀ = {np.round(result3['gradients'], 6).tolist()}")

    # --- Demo 4: Compare with finite difference ---
    print("\n--- Demo 4: Parameter shift vs finite difference comparison ---")
    from unitarylab.core import Circuit as _Circuit

    params_test = np.array([0.5, 1.0])
    pauli_zz = [("ZZ", 1.0)]
    gate_seq = [("ry", [0]), ("ry", [1]), ("cx", [0, 1])]
    obs_mat = observable_to_matrix(pauli_zz)

    # Parameter shift
    def eval_fn(full_params):
        qc = build_parameterized_circuit(2, gate_seq, full_params)
        return compute_expectation(qc, obs_mat, backend="torch")

    ps_grad = parameter_shift_gradient(eval_fn, params_test)

    # Finite difference (central)
    def fd_grad_fn(theta, eps):
        n = len(theta)
        g = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            fp = eval_fn(theta + eps * ei)
            fm = eval_fn(theta - eps * ei)
            g[i] = (fp - fm) / (2.0 * eps)
        return g

    fd_grad = fd_grad_fn(params_test, 1e-3)

    print(f"  Parameter shift: {np.round(ps_grad, 8).tolist()}")
    print(f"  Finite diff:     {np.round(fd_grad, 8).tolist()}")
    print(f"  Difference:      {np.max(np.abs(ps_grad - fd_grad)):.2e}")

    # --- Demo 5: Compare with RZ gate (analytic = parameter shift) ---
    print("\n--- Demo 5: RZ circuit (shift value = π/2 is exact) ---")
    gate_seq_rz = [("rz", [0]), ("rz", [1])]

    def eval_fn_rz(params):
        qc = build_parameterized_circuit(2, gate_seq_rz, params)
        return compute_expectation(qc, obs_mat, backend="torch")

    ps_grad_rz = parameter_shift_gradient(eval_fn_rz, params_test)

    # RZ gates only add phases to |00⟩. Therefore ⟨ZZ⟩ remains exactly 1
    # for all θ₁ and θ₂, and both derivatives are zero.
    expected_rz = np.zeros(2)
    print(f"  Parameter shift: {np.round(ps_grad_rz, 8).tolist()}")
    print(f"  Expected (exact): {np.round(expected_rz, 8).tolist()}")
    print(f"  Difference:       {np.max(np.abs(ps_grad_rz - expected_rz)):.2e}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
