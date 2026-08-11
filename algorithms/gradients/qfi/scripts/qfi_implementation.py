"""Manual implementation of Quantum Fisher Information (QFI).

Computes the Quantum Fisher Information matrix for pure parameterized quantum
states. QFI is derived from the Quantum Geometric Tensor (QGT):

    QFI = 4 · Re(QGT)

where QGTᵢⱼ = ⟨∂ᵢψ|∂ⱼψ⟩ - ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩.

The QFI matrix is real, symmetric, positive semi-definite, and serves as the
metric tensor of the quantum state manifold — used in natural gradient descent
and variational optimization.

Components:
    - compute_qgt_entry: Estimate one QGT entry via finite difference
    - compute_qfi_from_qgt: Build full QFI matrix from QGT
    - qfi_solve: End-to-end QFI computation pipeline
    - QFISolver: Class-based interface

Reference:
    SKILL.md — Quantum Fisher Information (QFI)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unitarylab.core import Circuit


# ---------------------------------------------------------------------------
# 1. Pauli utilities
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


# ---------------------------------------------------------------------------
# 2. Parameterized circuit
# ---------------------------------------------------------------------------

def build_parameterized_circuit(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
) -> Circuit:
    """Build a circuit from parameterized and fixed gates.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: List of (gate, qubits).
        params: Parameter values.

    Returns:
        Circuit.
    """
    qc = Circuit(num_qubits, name="QFI_Circuit")
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
        else:
            raise ValueError(f"Unsupported gate: {gate}")
    return qc


# ---------------------------------------------------------------------------
# 3. State and derivative computation
# ---------------------------------------------------------------------------

def get_statevector(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> np.ndarray:
    """Get the statevector from a parameterized circuit.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        params: Parameter values.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Statevector as numpy array.
    """
    qc = build_parameterized_circuit(num_qubits, gate_sequence, params)
    return qc.execute(backend=backend, device=device, dtype=dtype).state


def compute_partial_derivative(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
    param_idx: int,
    epsilon: float = 1e-4,
    backend: str = "torch",
    dtype: type = np.complex128,
) -> np.ndarray:
    """Compute |∂ψ/∂θᵢ⟩ via central finite difference.

    |∂ψ/∂θᵢ⟩ ≈ (|ψ(θ + ε·êᵢ)⟩ - |ψ(θ - ε·êᵢ)⟩) / (2ε)

    Args:
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        params: Parameter values.
        param_idx: Index of parameter to differentiate.
        epsilon: Finite difference step.
        backend: Simulation backend.
        dtype: Numerical dtype.

    Returns:
        Partial derivative statevector.
    """
    ei = np.zeros(len(params), dtype=float)
    ei[param_idx] = 1.0

    psi_plus = get_statevector(num_qubits, gate_sequence, params + epsilon * ei, backend=backend, dtype=dtype)
    psi_minus = get_statevector(num_qubits, gate_sequence, params - epsilon * ei, backend=backend, dtype=dtype)

    return (psi_plus - psi_minus) / (2.0 * epsilon)


# ---------------------------------------------------------------------------
# 4. QGT and QFI computation
# ---------------------------------------------------------------------------

def compute_qgt_entry(
    psi: np.ndarray,
    dpsi_i: np.ndarray,
    dpsi_j: np.ndarray,
    phase_fix: bool = True,
) -> complex:
    """Compute one entry of the Quantum Geometric Tensor.

    QGTᵢⱼ = ⟨∂ᵢψ|∂ⱼψ⟩ - ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩  (phase_fix=True)
    QGTᵢⱼ = ⟨∂ᵢψ|∂ⱼψ⟩                      (phase_fix=False)

    Args:
        psi: Statevector |ψ⟩.
        dpsi_i: Partial derivative |∂ᵢψ⟩.
        dpsi_j: Partial derivative |∂ⱼψ⟩.
        phase_fix: Whether to subtract the phase-fix term.

    Returns:
        Complex QGT entry.
    """
    inner = complex(np.vdot(dpsi_i, dpsi_j))
    if phase_fix:
        inner -= complex(np.vdot(dpsi_i, psi)) * complex(np.vdot(psi, dpsi_j))
    return inner


def compute_qfi_from_qgt(
    qgt: np.ndarray,
) -> np.ndarray:
    """Compute QFI from QGT: QFI = 4 · Re(QGT).

    Args:
        qgt: Complex QGT matrix of shape (n, n).

    Returns:
        Real symmetric QFI matrix of shape (n, n).
    """
    return 4.0 * np.real(qgt)


def compute_qfi_matrix(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    params: np.ndarray,
    phase_fix: bool = True,
    epsilon: float = 1e-4,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the full QFI matrix for a parameterized circuit.

    Pipeline:
        1. Compute statevector |ψ(θ)⟩
        2. Compute all partial derivatives |∂ᵢψ⟩ via finite difference
        3. Assemble QGT: QGTᵢⱼ = ⟨∂ᵢψ|∂ⱼψ⟩ - ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩
        4. Extract QFI: QFI = 4 · Re(QGT)

    Args:
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        params: Parameter values.
        phase_fix: Subtract the phase-fix term from QGT.
        epsilon: Finite difference step for derivatives.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.

    Returns:
        Tuple of (qfi_matrix, qgt_matrix).
            qfi_matrix: (n, n) real symmetric array.
            qgt_matrix: (n, n) complex array.
    """
    n_params = len(params)

    # --- Stage 1: Statevector ---
    psi = get_statevector(num_qubits, gate_sequence, params, backend=backend, device=device, dtype=dtype)

    # --- Stage 2: All partial derivatives ---
    dpsis = []
    for i in range(n_params):
        dpsi_i = compute_partial_derivative(
            num_qubits, gate_sequence, params, i, epsilon=epsilon, backend=backend, dtype=dtype,
        )
        dpsis.append(dpsi_i)

    # --- Stage 3: Assemble QGT ---
    qgt = np.zeros((n_params, n_params), dtype=np.complex128)
    for i in range(n_params):
        for j in range(n_params):
            qgt[i, j] = compute_qgt_entry(psi, dpsis[i], dpsis[j], phase_fix=phase_fix)

    # --- Stage 4: Extract QFI ---
    qfi = compute_qfi_from_qgt(qgt)

    return qfi, qgt


# ---------------------------------------------------------------------------
# 5. End-to-end solver
# ---------------------------------------------------------------------------

def qfi_solve(
    num_qubits: int,
    gate_sequence: List[Tuple[str, List[int]]],
    parameter_values: List[float],
    parameters: Optional[List[int]] = None,
    phase_fix: bool = True,
    epsilon: float = 1e-4,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute the QFI matrix for a parameterized quantum circuit.

    Args:
        num_qubits: Number of qubits.
        gate_sequence: Gate layout.
        parameter_values: Parameter values.
        parameters: Indices of parameters to include (None = all).
        phase_fix: Subtract phase-fix term.
        epsilon: FD step for derivative estimation.
        backend: Simulation backend.
        device: Compute device.
        dtype: Numerical dtype.
        verbose: Print progress.

    Returns:
        Dict with keys: status, qfis, qgt, n_params, is_symmetric,
            is_psd, Computation Time (s).
    """
    t_start = time.perf_counter()

    params = np.asarray(parameter_values, dtype=float)
    if parameters is not None:
        params = params[parameters]

    n = len(params)

    if verbose:
        print(f"Quantum Fisher Information")
        print(f"  Qubits:        {num_qubits}")
        print(f"  n_params:      {n}")
        print(f"  Phase fix:     {phase_fix}")
        print(f"  Epsilon:       {epsilon}")

    qfi_matrix, qgt_matrix = compute_qfi_matrix(
        num_qubits, gate_sequence, params,
        phase_fix=phase_fix, epsilon=epsilon,
        backend=backend, device=device, dtype=dtype,
    )

    # Verify: QFI should be real, symmetric, positive semi-definite
    is_symmetric = bool(np.allclose(qfi_matrix, qfi_matrix.T))
    evals = np.linalg.eigvalsh(qfi_matrix)
    is_psd = bool(np.all(evals >= -1e-10))

    t_end = time.perf_counter()
    comp_time = round(t_end - t_start, 4)

    if verbose:
        print(f"  QFI matrix:\n{np.round(qfi_matrix, 4)}")
        print(f"  Symmetric:     {is_symmetric}")
        print(f"  PSD:           {is_psd}")
        print(f"  QFI det:       {np.linalg.det(qfi_matrix):.6f}")
        print(f"  Time:          {comp_time}s")

    return {
        "status": "ok",
        "qfis": qfi_matrix,
        "qgt": qgt_matrix,
        "n_params": n,
        "is_symmetric": is_symmetric,
        "is_psd": is_psd,
        "Computation Time (s)": comp_time,
    }


# ---------------------------------------------------------------------------
# 6. Class-based interface
# ---------------------------------------------------------------------------

class QFISolver:
    """Class-based QFI solver.

    Usage:
        solver = QFISolver()
        result = solver.run(
            num_qubits=2,
            gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
            parameter_values=[0.5, 1.0],
        )
        print(result['qfis'])
    """

    def __init__(self, text_mode: str = "plain", algo_dir: Optional[str] = None):
        self.text_mode = text_mode
        self.algo_dir = algo_dir

    def run(
        self,
        num_qubits: int,
        gate_sequence: List[Tuple[str, List[int]]],
        parameter_values: List[float],
        parameters: Optional[List[int]] = None,
        phase_fix: bool = True,
        epsilon: float = 1e-4,
        backend: str = "torch",
        device: str = "cpu",
        dtype: type = np.complex128,
    ) -> Dict[str, Any]:
        """Run QFI computation. See qfi_solve() for docs."""
        result = qfi_solve(
            num_qubits=num_qubits,
            gate_sequence=gate_sequence,
            parameter_values=parameter_values,
            parameters=parameters,
            phase_fix=phase_fix,
            epsilon=epsilon,
            backend=backend,
            device=device,
            dtype=dtype,
            verbose=(self.text_mode != "plain"),
        )
        return self._build_return_dict(result)

    def _build_return_dict(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": result.get("status", "failed"),
            "qfis": result.get("qfis"),
            "qgt": result.get("qgt"),
            "is_symmetric": result.get("is_symmetric", False),
            "is_psd": result.get("is_psd", False),
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
        "num_qubits": 2,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        "parameter_values": [0.5, 1.0],
        "description": "QFI for 2 RY params with CX entanglement",
    },
    {
        "name": "1q_RY_1param",
        "num_qubits": 1,
        "gate_sequence": [("ry", [0])],
        "parameter_values": [0.5],
        "description": "Single-qubit RY: QFI = [[1]]",
    },
    {
        "name": "3q_RY_3params",
        "num_qubits": 3,
        "gate_sequence": [("ry", [0]), ("ry", [1]), ("ry", [2]),
                          ("cx", [0, 1]), ("cx", [1, 2])],
        "parameter_values": [0.3, -0.2, 0.7],
        "description": "3-qubit ring CX, QFI shape (3,3)",
    },
]


def run_known_test(case: Dict[str, Any]) -> bool:
    """Run a single test case."""
    name = case["name"]
    result = qfi_solve(
        num_qubits=case["num_qubits"],
        gate_sequence=case["gate_sequence"],
        parameter_values=case["parameter_values"],
        verbose=False,
    )
    qfi = result["qfis"]
    ok = (result["status"] == "ok" and result["is_symmetric"]
          and qfi.shape == (result["n_params"], result["n_params"]))
    icon = "ok" if ok else "FAIL"
    print(f"  [{icon}] {name}: shape={qfi.shape}, symmetric={result['is_symmetric']}, "
          f"PSD={result['is_psd']}, det={np.linalg.det(qfi):.6f}, "
          f"time={result['Computation Time (s)']}s")
    return ok


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Fisher Information — Manual Implementation")
    print("=" * 60)

    solver = QFISolver()

    # --- Demo 1: 2-qubit RY circuit ---
    print("\n--- Demo 1: QFI for 2-qubit RY + CX ---")
    result1 = solver.run(
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
    )
    print(f"  QFI matrix:\n{np.round(result1['qfis'], 4)}")
    print(f"  QFI trace: {np.trace(result1['qfis']):.6f}")
    print(f"  QFI determinant: {np.linalg.det(result1['qfis']):.6f}")

    # --- Demo 2: Verify QFI = 4·Re(QGT) ---
    print("\n--- Demo 2: Verify QFI = 4·Re(QGT) relationship ---")
    result2 = qfi_solve(
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
        verbose=False,
    )
    qfi2 = result2["qfis"]
    qgt2 = result2["qgt"]
    expected_qfi = 4.0 * np.real(qgt2)
    match = np.allclose(qfi2, expected_qfi)
    print(f"  QFI == 4·Re(QGT): {match}")
    print(f"  Max deviation: {np.max(np.abs(qfi2 - expected_qfi)):.2e}")

    # --- Demo 3: Phase-fix effect ---
    print("\n--- Demo 3: Effect of phase_fix on QFI ---")
    result3a = qfi_solve(
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
        phase_fix=True, verbose=False,
    )
    result3b = qfi_solve(
        num_qubits=2,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("cx", [0, 1])],
        parameter_values=[0.5, 1.0],
        phase_fix=False, verbose=False,
    )
    diff = np.max(np.abs(result3a["qfis"] - result3b["qfis"]))
    print(f"  QFI (phase_fix=True):  trace={np.trace(result3a['qfis']):.4f}")
    print(f"  QFI (phase_fix=False): trace={np.trace(result3b['qfis']):.4f}")
    print(f"  Max difference: {diff:.2e}")

    # --- Demo 4: 3-qubit circuit ---
    print("\n--- Demo 4: QFI for 3-qubit circuit ---")
    result4 = solver.run(
        num_qubits=3,
        gate_sequence=[("ry", [0]), ("ry", [1]), ("ry", [2]),
                       ("cx", [0, 1]), ("cx", [1, 2])],
        parameter_values=[0.3, -0.2, 0.7],
    )
    print(f"  QFI shape: {result4['qfis'].shape}")
    print(f"  QFI trace: {np.trace(result4['qfis']):.6f}")

    # --- Known test cases ---
    print("\n--- Known Test Cases ---")
    passed = sum(1 for c in KNOWN_CASES if run_known_test(c))
    print(f"\n  Result: {passed}/{len(KNOWN_CASES)} tests passed")
