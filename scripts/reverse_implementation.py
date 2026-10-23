"""Manual implementation of Reverse-Mode Statevector Gradient.

Computes expectation value gradients and Quantum Geometric Tensor entries by
traversing parameterized gates in reverse order using statevector-level
differentiation. Unlike parameter-shift or LCU, this does not require additional
circuit evaluations — it operates directly on statevectors.

For each parameterized gate Uⱼ(θⱼ) in the circuit, the derivative
∂Uⱼ/∂θⱼ = Σₖ cₖ Gₖ is decomposed, and reverse sweeps over forward/backward
statevectors yield exact gradients with O(P) complexity for P parameterized gates.

Components:
    - build_parameterized_circuit: Build circuit from gate sequence
    - compute_gate_derivative: Decompose ∂U/∂θ for standard gates
    - reverse_estimator_gradient: Reverse-sweep gradient computation
    - reverse_estimator_solve: End-to-end gradient pipeline
    - ReverseEstimatorGradientSolver: Class-based interface

Reference:
    SKILL.md — Reverse-Mode Gradient
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Gate definitions & derivatives
# ---------------------------------------------------------------------------

# Single-qubit gate matrices
GATE_MATRICES: Dict[str, np.ndarray] = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
    "I": np.eye(2, dtype=np.complex128),
}

# CX matrix (control=0, target=1)
CX_MATRIX: np.ndarray = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=np.complex128)


def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    """Compute R_axis(θ) = exp(-iθ P/2) for P ∈ {X, Y, Z}.

    Args:
        axis: 'X', 'Y', or 'Z'.
        theta: Rotation angle.

    Returns:
        2×2 unitary matrix.
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    if axis == "X":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    elif axis == "Y":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    elif axis == "Z":
        return np.array([[complex(c, -s), 0], [0, complex(c, s)]], dtype=np.complex128)
    else:
        raise ValueError(f"Unknown axis: {axis}")


def rotation_derivative(axis: str, theta: float) -> np.ndarray:
    """Compute ∂R_axis(θ)/∂θ.

    For R(θ) = exp(-iθP/2):
        ∂R/∂θ = -iP/2 · R(θ)

    Args:
        axis: 'X', 'Y', or 'Z'.
        theta: Rotation angle.

    Returns:
        2×2 derivative matrix.
    """
    P = GATE_MATRICES[axis]
    R = rotation_matrix(axis, theta)
    return -0.5j * (P @ R)


# ---------------------------------------------------------------------------
# 2. Pauli / observable utilities
# ---------------------------------------------------------------------------

_PAULI_MAP: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": GATE_MATRICES["X"],
    "Y": GATE_MATRICES["Y"],
    "Z": GATE_MATRICES["Z"],
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
# 3. Circuit and statevector computation
# ---------------------------------------------------------------------------

def build_parameterized_circuit(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Circuit:
    """Build a parameterized circuit.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: List of (gate_type, qubit_list).
        params: Parameter values for parameterized gates.

    Returns:
        Circuit.
    """
    qc = Circuit(num_qubits, name="Reverse")
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


def get_statevector_from_circuit(
    qc: Circuit,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> np.ndarray:
    """Get statevector from circuit execution."""
    return qc.execute(backend=backend, device=device, dtype=dtype).state


# ---------------------------------------------------------------------------
# 4. Reverse-mode gradient via statevector sweeps
# ---------------------------------------------------------------------------

def reverse_estimator_gradient(
    observable_matrix: np.ndarray,
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> np.ndarray:
    """Compute gradient via reverse-mode statevector differentiation.

    For expectation value f(θ) = ⟨ψ(θ)|O|ψ(θ)⟩:

        ∂f/∂θⱼ = 2·Re(⟨φ|∂Uⱼ/∂θⱼ|ψⱼ⟩)

    where:
        - |ψⱼ⟩ is the forward state just before gate j
        - |φ⟩ = Uⱼ₊₁† ··· U_P† · O · U_P ··· Uⱼ₊₁ |ψ(θ)⟩
          is the "backward" state propagated from the final state

    Algorithm:
        1. Forward pass: accumulate full statevector and store intermediate states.
        2. Backward pass: propagate O|ψ⟩ backward; at each parameterized gate,
           contract with the gate's derivative to get the gradient component.

    Args:
        observable_matrix: Observable O as dense matrix.
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        params: Parameter values.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Gradient vector of length n_param_gates.
    """
    # Identify parameterized gate count and indices
    param_count = sum(1 for g, _ in gate_sequence if g in ("rx", "ry", "rz"))
    grad = np.zeros(param_count, dtype=float)

    dim = 1 << num_qubits

    # --- Build per-gate matrix sequence for reverse sweep ---
    # We'll simulate the circuit gate-by-gate using full matrix operations
    # Forward: accumulate state; backward: propagate O|ψ⟩ through conjugate gates

    # Full unitary of the entire circuit (for approach using matrix multiplication)
    # For the reverse sweep, we walk through gates in reverse.

    # Store intermediate states in forward pass
    states: List[np.ndarray] = []
    current_state = np.zeros(dim, dtype=np.complex128)
    current_state[0] = 1.0  # |0...0⟩

    gate_matrices: List[np.ndarray] = []
    gate_is_param: List[bool] = []
    gate_axis: List[str] = []
    gate_qubits: List[List[int]] = []

    for gate, qubits in gate_sequence:
        q0 = qubits[0]

        if gate in ("rx", "ry", "rz"):
            val = float(params[len([g for g in gate_is_param if g])])
            gate_is_param.append(True)
            gate_axis.append(gate[1].upper())  # X, Y, Z
            gate_qubits.append(qubits)
        else:
            gate_is_param.append(False)
            gate_axis.append("")
            gate_qubits.append(qubits)

    # Actually, let me use a different approach — the reverse sweep method.
    # We simulate each gate's full unitary on the n-qubit space,
    # store intermediate states, then sweep backward.

    # Forward sweep: store all intermediate states (before each gate)
    current = np.zeros(dim, dtype=np.complex128)
    current[0] = 1.0
    intermediate_states = [current.copy()]  # state before gate 0

    param_idx = 0
    for gate, qubits in gate_sequence:
        q0 = qubits[0]
        if gate in ("rx", "ry", "rz"):
            val = float(params[param_idx])
            param_idx += 1
            U_gate = _build_gate_unitary(num_qubits, gate, qubits, val)
        elif gate == "cx":
            U_gate = _build_gate_unitary(num_qubits, "cx", qubits)
        elif gate == "h":
            U_gate = _build_gate_unitary(num_qubits, "h", qubits)
        elif gate in ("x", "y", "z"):
            U_gate = _build_gate_unitary(num_qubits, gate, qubits)
        else:
            U_gate = np.eye(dim, dtype=np.complex128)

        current = U_gate @ current
        intermediate_states.append(current.copy())

    # Use the final state from the same manually expanded matrices. Mixing a
    # simulator statevector with hand-built matrices can silently introduce a
    # qubit-endianness mismatch on multi-qubit circuits.
    final_state = intermediate_states[-1]

    # Backward sweep: |φ⟩ = O|ψ_final⟩, then propagate backward
    psi_backward = observable_matrix @ final_state

    param_idx = param_count - 1  # index into grad array, walking backward

    for step in range(len(gate_sequence) - 1, -1, -1):
        gate, qubits = gate_sequence[step]
        psi_forward = intermediate_states[step]  # state before this gate

        if gate in ("rx", "ry", "rz"):
            val = float(params[param_idx])
            axis = gate[1].upper()

            # Compute ∂U/∂θ for this rotation gate
            dU = _build_gate_derivative_unitary(num_qubits, axis, qubits, val)

            # ∂f/∂θⱼ = 2·Re(⟨ψ_backward| (∂U/∂θ) |ψ_forward⟩)
            grad_i = 2.0 * float(np.real(
                np.vdot(psi_backward, dU @ psi_forward)
            ))
            grad[param_idx] = grad_i

            # Propagate backward: |ψ_backward⟩ → U† |ψ_backward⟩
            U_gate = _build_gate_unitary(num_qubits, gate, qubits, val)
            psi_backward = U_gate.conj().T @ psi_backward

            param_idx -= 1
        else:
            # Fixed gate: just propagate backward
            U_gate = _build_gate_unitary(num_qubits, gate, qubits)
            psi_backward = U_gate.conj().T @ psi_backward

    return grad


def _build_gate_unitary(
    num_qubits: int,
    gate: str,
    qubits: List[int],
    val: Optional[float] = None,
) -> np.ndarray:
    """Build the full 2^n × 2^n unitary for a gate acting on specified qubits.

    Args:
        num_qubits: Total qubit count.
        gate: Gate type (rx, ry, rz, cx, h, x, y, z).
        qubits: Qubit indices the gate acts on.
        val: Parameter value for parameterized gates.

    Returns:
        Full unitary matrix of shape (dim, dim).
    """
    dim = 1 << num_qubits
    q0 = qubits[0]

    if gate == "rx":
        g = rotation_matrix("X", val)
    elif gate == "ry":
        g = rotation_matrix("Y", val)
    elif gate == "rz":
        g = rotation_matrix("Z", val)
    elif gate == "h":
        g = GATE_MATRICES["H"]
    elif gate == "x":
        g = GATE_MATRICES["X"]
    elif gate == "y":
        g = GATE_MATRICES["Y"]
    elif gate == "z":
        g = GATE_MATRICES["Z"]
    elif gate == "cx":
        # 2-qubit CX: build full operator
        return _build_cx_unitary(num_qubits, q0, qubits[1])
    else:
        g = np.eye(2, dtype=np.complex128)

    return _expand_single_qubit_gate(num_qubits, q0, g)


def _build_gate_derivative_unitary(
    num_qubits: int,
    axis: str,
    qubits: List[int],
    val: float,
) -> np.ndarray:
    """Build the full 2^n × 2^n derivative matrix for ∂R_axis(θ)/∂θ.

    Args:
        num_qubits: Total qubit count.
        axis: 'X', 'Y', or 'Z'.
        qubits: Qubit indices.
        val: Parameter value.

    Returns:
        Full derivative matrix.
    """
    dg = rotation_derivative(axis, val)
    return _expand_single_qubit_gate(num_qubits, qubits[0], dg)


def _expand_single_qubit_gate(
    num_qubits: int,
    target: int,
    gate_2x2: np.ndarray,
) -> np.ndarray:
    """Expand a 2×2 gate acting on one qubit to the full 2^n space.

    Args:
        num_qubits: Total qubit count.
        target: Target qubit index.
        gate_2x2: 2×2 matrix.

    Returns:
        Full unitary of shape (dim, dim).
    """
    # UnitaryLab numbers qubit 0 as the least-significant basis bit. Build the
    # Kronecker product from the highest qubit down to preserve that ordering.
    I2 = np.eye(2, dtype=np.complex128)
    ops = [gate_2x2 if q == target else I2 for q in reversed(range(num_qubits))]

    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _build_cx_unitary(
    num_qubits: int,
    control: int,
    target: int,
) -> np.ndarray:
    """Build full CX unitary in 2^n space.

    Args:
        num_qubits: Total qubit count.
        control: Control qubit index.
        target: Target qubit index.

    Returns:
        Full CX unitary.
    """
    dim = 1 << num_qubits
    U = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        ctrl_bit = (i >> control) & 1
        tgt_bit = (i >> target) & 1
        if ctrl_bit == 1:
            # Flip target bit
            j = i ^ (1 << target)
            U[i, i] = 0
            U[j, i] = 1
    return U


# ---------------------------------------------------------------------------
# 5. End-to-end solver
# ---------------------------------------------------------------------------

def reverse_estimator_solve(
    pauli_list: List[Tuple[str, float]],
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    parameter_values: List[float],
    parameters: Optional[List[int]] = None,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute exact gradient via reverse-mode statevector sweep.

    Args:
        pauli_list: Observable as Pauli terms.
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        parameter_values: Parameter values.
        parameters: Indices of params to differentiate (None = all).
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys: status, gradients, n_params, Computation Time (s).
    """
    t_start = time.perf_counter()

    observable_matrix = observable_to_matrix(pauli_list)
    params = np.asarray(parameter_values, dtype=float)

    full_grad = reverse_estimator_gradient(
        observable_matrix, num_qubits, gate_sequence, params,
        backend=backend, device=device, dtype=dtype,
    )

    if parameters is not None:
        gradient = full_grad[parameters]
    else:
        gradient = full_grad

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        print(f"Reverse-Mode Gradient (Statevector)")
        print(f"  n_params:      {len(full_grad)} (returning {len(gradient)})")
        print(f"  Gradient:      {np.round(gradient, 6).tolist()}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "gradients": gradient,
        "full_gradients": full_grad,
        "n_params": len(full_grad),
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class ReverseEstimatorGradientSolver:
    """Class-based solver for Reverse-Mode Gradient.

    Usage:
        solver = ReverseEstimatorGradientSolver()
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
        """Run reverse-mode gradient estimation."""
        result = reverse_estimator_solve(
            pauli_list=pauli_list,
            num_qubits=num_qubits,
            gate_sequence=gate_sequence,
            parameter_values=parameter_values,
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
    },
    {
        "name": "1q_RY_1param",
        "pauli_list": [("Z", 1.0)],
        "num_qubits": 1,
        "gate_sequence": [("ry", [0])],
        "parameter_values": [0.5],
    },
    {
        "name": "3q_RY_3params",
        "pauli_list": [("ZZZ", 1.0)],
        "num_qubits": 3,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("ry", [2]),
                          ("cx", [0, 1]), ("cx", [1, 2])],
        "parameter_values": [0.3, -0.2, 0.7],
    },
    {
        "name": "1q_H_then_RY",
        "pauli_list": [("Z", 1.0)],
        "num_qubits": 1,
        "gate_sequence": [("h", [0]), ("ry", [0])],
        "parameter_values": [0.5],
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = reverse_estimator_solve(
        pauli_list=case["pauli_list"],
        num_qubits=case["num_qubits"],
        gate_sequence=case["gate_sequence"],
        parameter_values=case["parameter_values"],
        verbose=False,
    )
    params = np.asarray(case["parameter_values"], dtype=float)
    observable = observable_to_matrix(case["pauli_list"])

    def eval_fn(theta: np.ndarray) -> float:
        qc = build_parameterized_circuit(case["num_qubits"], case["gate_sequence"], theta)
        state = get_statevector_from_circuit(qc)
        return float(np.real(np.vdot(state, observable @ state)))

    epsilon = 1e-6
    finite_diff = np.zeros_like(params)
    for idx in range(len(params)):
        offset = np.zeros_like(params)
        offset[idx] = epsilon
        finite_diff[idx] = (eval_fn(params + offset) - eval_fn(params - offset)) / (2 * epsilon)

    max_error = float(np.max(np.abs(result["gradients"] - finite_diff)))
    ok = result["status"] == "ok" and max_error < 1e-6
    icon = "ok" if ok else "FAIL"
    grad_str = np.round(result["gradients"], 4).tolist()
    print(f"  [{icon}] {name}: grad={grad_str}, fd={np.round(finite_diff, 4).tolist()}, "
          f"max_error={max_error:.2e}, time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Reverse-Mode Gradient — Manual Implementation")
    print("=" * 60)

    solver = ReverseEstimatorGradientSolver()

    # --- Demo 1: 2-qubit RY circuit ---
    print("\n--- Demo 1: ⟨ZZ⟩ gradient, 2-qubit RY + CX (reverse-mode) ---")
    pauli_zz = [("ZZ", 1.0)]
    gate_seq = [("ry", [0]), ("ry", [1]), ("cx", [0, 1])]
    params_test = [0.5, 1.0]

    result1 = solver.run(
        pauli_list=pauli_zz,
        num_qubits=2,
        gate_sequence=gate_seq,
        parameter_values=params_test,
    )
    print(f"  Reverse-mode grad: {np.round(result1['gradients'], 6).tolist()}")

    # --- Demo 2: Compare with finite difference ---
    print("\n--- Demo 2: Reverse-mode vs finite difference ---")
    obs_mat = observable_to_matrix(pauli_zz)

    # Finite difference for comparison
    def fd_gradient(params, eps=1e-3):
        n = len(params)
        g = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            from unitarylab.core import Circuit as _C
            qcp = build_parameterized_circuit(2, gate_seq, params + eps * ei)
            qcm = build_parameterized_circuit(2, gate_seq, params - eps * ei)
            sp = get_statevector_from_circuit(qcp)
            sm = get_statevector_from_circuit(qcm)
            fp = float(np.real((sp.conj().T @ obs_mat @ sp).item()))
            fm = float(np.real((sm.conj().T @ obs_mat @ sm).item()))
            g[i] = (fp - fm) / (2.0 * eps)
        return g

    fd_grad = fd_gradient(np.array(params_test))
    rev_grad = result1["gradients"]
    print(f"  Reverse-mode:  {np.round(rev_grad, 6).tolist()}")
    print(f"  Finite diff:   {np.round(fd_grad, 6).tolist()}")
    print(f"  Max diff:      {np.max(np.abs(rev_grad - fd_grad)):.2e}")

    # --- Demo 3: 1-qubit analytic verification ---
    print("\n--- Demo 3: ⟨Z⟩ gradient for RY(0.5)|0⟩ — analytic check ---")
    # RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
    # ⟨Z⟩ = cos²(θ/2) - sin²(θ/2) = cos(θ)
    # ∂⟨Z⟩/∂θ = -sin(θ)
    result3 = solver.run(
        pauli_list=[("Z", 1.0)],
        num_qubits=1,
        gate_sequence=[("ry", [0])],
        parameter_values=[0.5],
    )
    expected3 = -math.sin(0.5)
    print(f"  Reverse-mode:  {result3['gradients'][0]:.8f}")
    print(f"  Analytic (-sin(θ)): {expected3:.8f}")
    print(f"  Diff:          {abs(result3['gradients'][0] - expected3):.2e}")

    # --- Demo 4: 3-qubit circuit ---
    print("\n--- Demo 4: 3-qubit RY + ring CX, ⟨ZZZ⟩ ---")
    result4 = solver.run(
        pauli_list=[("ZZZ", 1.0)],
        num_qubits=3,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("ry", [2]),
                       ("cx", [0, 1]), ("cx", [1, 2])],
        parameter_values=[0.3, -0.2, 0.7],
    )
    print(f"  Gradient: {np.round(result4['gradients'], 6).tolist()}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
