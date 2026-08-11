"""Manual implementation of Linear Combination of Unitaries (LCU) Gradient.

Computes analytic parameter gradients from the LCU derivative decomposition.
This manual reference constructs the exact derivative branches and evaluates
the contraction measured by an ancilla-augmented LCU circuit,
2 Re[⟨ψ|O|∂ᵢψ⟩], without numerical approximation.

The LCU method requires gates from the supported set: rx, ry, rz, rzx, rzz, ryy,
rxx, cx, cy, cz, ccx, swap, iswap, h, t, s, sdg, x, y, z.

Components:
    - build_lcu_derivative_states: Build exact state and derivative branches
    - compute_expectation: Evaluate ⟨ψ|O|ψ⟩
    - lin_comb_estimator_gradient: Core analytic gradient computation
    - lin_comb_estimator_solve: End-to-end gradient pipeline
    - LinCombEstimatorGradientSolver: Class-based interface

Reference:
    SKILL.md — Linear Combination of Unitaries (LCU) Gradient
"""

from __future__ import annotations

import math
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
# 2. Circuit construction
# ---------------------------------------------------------------------------

# Gate generators: the derivative of exp(-i θ G/2) w.r.t θ at θ = 0
# For Pauli rotation exp(-i θ P/2): d/dθ = -i/2 · P
# In LCU, we represent this via controlled operations

SUPPORTED_PARAM_GATES = {"rx", "ry", "rz", "rzx", "rzz", "ryy", "rxx"}


def build_base_circuit(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Circuit:
    """Build a parameterized circuit from gate sequence.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: List of (gate_type, qubits).
        params: Flat parameter array.

    Returns:
        Circuit with all gates applied.
    """
    qc = Circuit(num_qubits, name="Base")
    param_idx = 0
    for gate, qubits in gate_sequence:
        q0 = qubits[0]
        if gate in SUPPORTED_PARAM_GATES:
            val = float(params[param_idx])
            param_idx += 1
            _apply_param_gate(qc, gate, qubits, val)
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


def _apply_param_gate(qc: Circuit, gate: str, qubits: List[int], val: float) -> None:
    """Apply a parameterized gate."""
    q0 = qubits[0]
    q1 = qubits[1] if len(qubits) > 1 else -1
    if gate == "rx":
        qc.rx(val, q0)
    elif gate == "ry":
        qc.ry(val, q0)
    elif gate == "rz":
        qc.rz(val, q0)
    elif gate == "rzx":
        qc.rzx(val, q0, q1)
    elif gate == "rzz":
        qc.rzz(val, q0, q1)
    elif gate == "ryy":
        qc.ryy(val, q0, q1)
    elif gate == "rxx":
        qc.rxx(val, q0, q1)


def compute_expectation(
    qc: Circuit,
    observable_matrix: np.ndarray,
    ancilla: Optional[int] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> float:
    """Compute ⟨ψ|O|ψ⟩ via statevector simulation.

    If ancilla is provided, measures ⟨X_anc ⊗ O⟩ using the LCU ancilla
    measurement protocol.

    Args:
        qc: Circuit producing state |ψ⟩.
        observable_matrix: Observable matrix.
        ancilla: Ancilla qubit index (for LCU X measurement) or None.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Expectation value (float).
    """
    state = qc.execute(backend=backend, device=device, dtype=dtype).state

    if ancilla is not None:
        # Ancilla is qubit 0 (least-significant bit), so X_anc ⊗ O_data is
        # represented as O_data ⊗ X in the dense big-endian matrix ordering.
        x_anc = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        x_measure = np.kron(observable_matrix, x_anc)
        return float(np.real((state.conj().T @ x_measure @ state).item()))
    else:
        return float(np.real((state.conj().T @ observable_matrix @ state).item()))


# ---------------------------------------------------------------------------
# 3. Exact derivative-operator reconstruction
# ---------------------------------------------------------------------------

def _expand_single_qubit_operator(
    num_qubits: int, target: int, operator: np.ndarray,
) -> np.ndarray:
    """Expand a single-qubit operator with qubit 0 as the least-significant bit."""
    identity = np.eye(2, dtype=np.complex128)
    factors = [operator if q == target else identity for q in reversed(range(num_qubits))]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _expand_pauli_product(
    num_qubits: int, assignments: Dict[int, str],
) -> np.ndarray:
    """Expand a Pauli product such as X(q0)X(q1) to the full Hilbert space."""
    factors = [_PAULI_MAP[assignments.get(q, "I")] for q in reversed(range(num_qubits))]
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def _cx_matrix(num_qubits: int, control: int, target: int) -> np.ndarray:
    """Build a dense CX matrix in the same little-endian qubit convention."""
    dim = 1 << num_qubits
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        output = basis ^ (1 << target) if ((basis >> control) & 1) else basis
        matrix[output, basis] = 1.0
    return matrix


def _gate_and_derivative_matrices(
    num_qubits: int,
    gate: str,
    qubits: List[int],
    value: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return U and analytic dU/dθ for one supported gate.

    For a Pauli rotation U(θ)=exp(-iθP/2), use
    dU/dθ=(-i/2)P U. This is the derivative operator decomposed by the LCU
    method before its real-part contraction with the observable branch.
    """
    if gate in SUPPORTED_PARAM_GATES:
        if value is None:
            raise ValueError(f"Missing parameter value for {gate}")
        if gate == "rx":
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "X"})
        elif gate == "ry":
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "Y"})
        elif gate == "rz":
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "Z"})
        elif gate == "rxx":
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "X", qubits[1]: "X"})
        elif gate == "ryy":
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "Y", qubits[1]: "Y"})
        elif gate == "rzz":
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "Z", qubits[1]: "Z"})
        else:  # rzx: Z on the first qubit, X on the second
            generator = _expand_pauli_product(num_qubits, {qubits[0]: "Z", qubits[1]: "X"})
        identity = np.eye(1 << num_qubits, dtype=np.complex128)
        unitary = math.cos(value / 2) * identity - 1j * math.sin(value / 2) * generator
        derivative = (-0.5j) * generator @ unitary
        return unitary, derivative

    if gate == "cx":
        return _cx_matrix(num_qubits, qubits[0], qubits[1]), None
    if gate in ("h", "x", "y", "z"):
        local = {
            "h": np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2),
            "x": _PAULI_MAP["X"],
            "y": _PAULI_MAP["Y"],
            "z": _PAULI_MAP["Z"],
        }[gate]
        return _expand_single_qubit_operator(num_qubits, qubits[0], local), None
    raise ValueError(f"Unsupported gate: {gate}")


def build_lcu_derivative_states(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Build |ψ⟩ and every exact |∂ᵢψ⟩ branch from gate derivatives."""
    dim = 1 << num_qubits
    initial = np.zeros(dim, dtype=np.complex128)
    initial[0] = 1.0

    matrices: List[np.ndarray] = []
    derivative_matrices: List[Tuple[int, np.ndarray]] = []
    param_idx = 0
    for gate_idx, (gate, qubits) in enumerate(gate_sequence):
        value = float(params[param_idx]) if gate in SUPPORTED_PARAM_GATES else None
        unitary, derivative = _gate_and_derivative_matrices(
            num_qubits, gate, qubits, value,
        )
        matrices.append(unitary)
        if derivative is not None:
            derivative_matrices.append((gate_idx, derivative))
            param_idx += 1

    if param_idx != len(params):
        raise ValueError(f"Expected {param_idx} parameter values, received {len(params)}")

    forward_states = [initial]
    for unitary in matrices:
        forward_states.append(unitary @ forward_states[-1])
    final_state = forward_states[-1]

    derivative_states: List[np.ndarray] = []
    for gate_idx, derivative in derivative_matrices:
        branch = derivative @ forward_states[gate_idx]
        for unitary in matrices[gate_idx + 1:]:
            branch = unitary @ branch
        derivative_states.append(branch)
    return final_state, derivative_states


# ---------------------------------------------------------------------------
# 4. LCU gradient computation
# ---------------------------------------------------------------------------

def lin_comb_estimator_gradient(
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
    """Compute the exact LCU estimator gradient from derivative branches.

    Reconstruct the same contraction measured by the ancilla-augmented LCU
    circuit: 2 Re[⟨ψ|O|∂ᵢψ⟩]. No finite differences or parameter shifts are
    used.

    Args:
        pauli_list: Observable as Pauli terms.
        num_qubits: Number of data qubits.
        gate_sequence: Gate layout for the parameterized circuit.
        parameter_values: Current parameter values.
        parameters: Indices of params to differentiate (None = all).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys: status, gradients, n_params, num_evals,
            Computation Time (s).
    """
    t_start = time.perf_counter()

    n_params = sum(g in SUPPORTED_PARAM_GATES for g, _ in gate_sequence)
    effective_indices = list(range(n_params)) if parameters is None else list(parameters)
    if any(idx < 0 or idx >= n_params for idx in effective_indices):
        raise IndexError(f"Parameter indices {effective_indices} are invalid for {n_params} parameters")
    n_target = len(effective_indices)
    params = np.asarray(parameter_values, dtype=float)

    # --- Build observable ---
    observable_matrix = observable_to_matrix(pauli_list)

    final_state, derivative_states = build_lcu_derivative_states(
        num_qubits, gate_sequence, params,
    )
    gradient = np.array([
        2.0 * np.real(np.vdot(final_state, observable_matrix @ derivative_states[idx]))
        for idx in effective_indices
    ], dtype=float)

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        print(f"LCU Estimator Gradient")
        print(f"  n_params:      {n_params} (differentiating {n_target})")
        print(f"  Evaluations:   {n_target} (1 per param)")
        print(f"  Gradient:      {np.round(gradient, 6).tolist()}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "gradients": gradient,
        "n_params": n_params,
        "num_evals": n_target,
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 5. Class-based interface
# ---------------------------------------------------------------------------

class LinCombEstimatorGradientSolver:
    """Class-based solver for LCU Gradient.

    Usage:
        solver = LinCombEstimatorGradientSolver()
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
        """Run LCU gradient estimation."""
        result = lin_comb_estimator_gradient(
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
# 6. Finite difference comparison
# ---------------------------------------------------------------------------

def finite_diff_gradient(
    eval_fn, theta: np.ndarray, epsilon: float = 1e-3,
) -> np.ndarray:
    """Central finite difference gradient for comparison."""
    n = len(theta)
    grad = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        fp = eval_fn(theta + epsilon * ei)
        fm = eval_fn(theta - epsilon * ei)
        grad[i] = (fp - fm) / (2.0 * epsilon)
    return grad


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
        "description": "⟨ZZ⟩ LCU gradient, 2 RY gates",
    },
    {
        "name": "2q_RZ_2params",
        "pauli_list": [("ZZ", 1.0)],
        "num_qubits": 2,
        "gate_sequence": [("rz", [0]), ("rz", [1])],
        "parameter_values": [0.5, 1.0],
        "description": "⟨ZZ⟩ LCU, 2 RZ gates",
    },
    {
        "name": "1q_RY_1param",
        "pauli_list": [("Z", 1.0)],
        "num_qubits": 1,
        "gate_sequence": [("ry", [0])],
        "parameter_values": [0.5],
        "description": "⟨Z⟩ gradient for RY on 1 qubit",
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


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = lin_comb_estimator_gradient(
        pauli_list=case["pauli_list"],
        num_qubits=case["num_qubits"],
        gate_sequence=case["gate_sequence"],
        parameter_values=np.array(case["parameter_values"]),
        verbose=False,
    )
    params = np.asarray(case["parameter_values"], dtype=float)
    observable = observable_to_matrix(case["pauli_list"])

    def eval_fn(theta: np.ndarray) -> float:
        qc = build_base_circuit(case["num_qubits"], case["gate_sequence"], theta)
        state = qc.execute(backend="torch").state
        return float(np.real(np.vdot(state, observable @ state)))

    # A moderate step avoids amplifying simulator roundoff while remaining
    # tight enough to catch an incorrect analytic derivative construction.
    finite_diff = finite_diff_gradient(eval_fn, params, epsilon=1e-3)
    max_error = float(np.max(np.abs(result["gradients"] - finite_diff)))
    ok = result["status"] == "ok" and max_error < 1e-4
    icon = "ok" if ok else "FAIL"
    grad_str = np.round(result["gradients"], 4).tolist()
    print(f"  [{icon}] {name}: grad={grad_str}, fd={np.round(finite_diff, 4).tolist()}, "
          f"max_error={max_error:.2e}, evals={result['num_evals']}, "
          f"time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LCU Gradient — Linear Combination of Unitaries")
    print("=" * 60)

    solver = LinCombEstimatorGradientSolver()

    # --- Demo 1: 2-qubit RY circuit ---
    print("\n--- Demo 1: ⟨ZZ⟩ gradient, 2-qubit RY + CX ---")
    pauli_zz = [("ZZ", 1.0)]
    gate_seq = [("ry", [0]), ("ry", [1]), ("cx", [0, 1])]
    params_test = [0.5, 1.0]

    result1 = solver.run(
        pauli_list=pauli_zz,
        num_qubits=2,
        gate_sequence=gate_seq,
        parameter_values=params_test,
    )
    print(f"  LCU Gradient: {np.round(result1['gradients'], 6).tolist()}")

    # --- Demo 2: Compare with finite difference ---
    print("\n--- Demo 2: LCU vs Finite Difference comparison ---")
    obs_mat = observable_to_matrix(pauli_zz)

    def eval_fn(theta):
        qc = build_base_circuit(2, gate_seq, theta)
        state = qc.execute(backend="torch").state
        return float(np.real((state.conj().T @ obs_mat @ state).item()))

    fd_grad = finite_diff_gradient(eval_fn, np.array(params_test), epsilon=1e-3)
    lcu_grad = result1["gradients"]

    print(f"  LCU gradient:     {np.round(lcu_grad, 6).tolist()}")
    print(f"  Finite diff grad: {np.round(fd_grad, 6).tolist()}")
    print(f"  Difference:       {np.max(np.abs(lcu_grad - fd_grad)):.2e}")

    # --- Demo 3: 1-qubit RY ---
    print("\n--- Demo 3: ⟨Z⟩ gradient, 1-qubit RY(0.5) ---")
    # |ψ⟩ = RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    # ⟨Z⟩ = cos²(θ/2) - sin²(θ/2) = cos(θ)
    # ∂⟨Z⟩/∂θ = -sin(θ)
    result3 = solver.run(
        pauli_list=[("Z", 1.0)],
        num_qubits=1,
        gate_sequence=[("ry", [0])],
        parameter_values=[0.5],
    )
    expected3 = -math.sin(0.5)  # exact analytic gradient
    print(f"  LCU Gradient:    {result3['gradients'][0]:.6f}")
    print(f"  Exact (-sin(θ)):  {expected3:.6f}")
    print(f"  Difference:       {abs(result3['gradients'][0] - expected3):.2e}")

    # --- Demo 4: 2-qubit RZ (no entanglement) ---
    print("\n--- Demo 4: ⟨ZZ⟩ gradient, 2-qubit RZ (analytic comparison) ---")
    # RZ(θ₁)⊗RZ(θ₂)|00⟩, ⟨ZZ⟩ = 1, ∂/∂θ₁ = ∂/∂θ₂ = 0
    result4 = solver.run(
        pauli_list=[("ZZ", 1.0)],
        num_qubits=2,
        gate_sequence=[("rz", [0]), ("rz", [1])],
        parameter_values=[0.5, 1.0],
    )
    print(f"  LCU Gradient:     {np.round(result4['gradients'], 8).tolist()}")
    print(f"  Expected:         [0.0, 0.0] (ZZ is diagonal in computational basis)")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
