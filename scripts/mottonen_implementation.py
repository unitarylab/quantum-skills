# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from unitarylab.core import Circuit

try:
    from ...algo_base import BaseAlgorithm
except ImportError:
    # 单独运行时，将上级目录加入 sys.path，使 base 模块可被找到
    import sys
    _algorithms_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _algorithms_dir not in sys.path:
        sys.path.insert(0, _algorithms_dir)
    from unitarylab_algorithms.algo_base import BaseAlgorithm



class MottonenAlgorithm(BaseAlgorithm):
    """Standalone Mottönen state-preparation algorithm module."""

    def __init__(self, text_mode: str = "plain", algo_dir: str = 'circuits'):
        if algo_dir is None:
            _this = os.path.abspath(__file__)
            _directory = os.path.dirname(_this)
            algo_dir = os.path.join(
                os.getcwd(),
                "results",
                os.path.basename(os.path.dirname(_directory)),
                os.path.basename(_directory),
            )
        os.makedirs(algo_dir, exist_ok=True)
        super().__init__(
            name="Mottonen State Preparation Algorithm",
            prefix="MOT",
            text_mode=text_mode,
            algo_dir=algo_dir,
        )

    def run(
        self,
        Psi,
        target_qubits: int,
        target_error: float = 1e-6,
        backend='torch',
        device='cpu',
        dtype=np.complex128,
    ) -> Dict[str, Any]:
        """Run Mottönen state preparation for a target state vector."""
        psi = np.asarray(Psi, dtype=np.complex128)
        self.update_input({
            "Method": "mottonen",
            "Target qubits": target_qubits,
            "Target error": target_error,
            "State vector length": int(psi.size),
        })

        self.log("Stage 1: Validating and normalizing target state")
        start_time = time.time()
        result = MottonenAlgorithm.Mottonen(psi, int(target_qubits), float(target_error))

        self.log("Stage 2: Building state-preparation circuit")
        circuit = result.circuit

        self.log("Stage 3: Computing emitted unitary and preparation error")
        prepared_state = np.asarray(result.evolution_result, dtype=np.complex128)[:, 0]
        total_error = float(result.total_error)
        comp_time = time.time() - start_time
        is_success = total_error <= max(float(target_error), 1e-10)

        self.log(f"  Total error: {total_error:.6e}")
        self.log(f"  Computation time: {comp_time:.4f} s")

        self.log("Stage 4: Exporting circuit diagram")
        self.update_output({
            "Prepared state": prepared_state,
            "Total error": total_error,
            "Computation time (s)": round(comp_time, 4),
        })
        self.status = "success" if is_success else "failed"
        self.summary = f"Mottonen state preparation completed with error {total_error:.6e}."

        circuit_path = self.save_circuit(circuit)
        filename = self.save_txt()
        return self._build_return_dict(is_success, circuit_path, filename, circuit)


    def _phase_invariant_error(reference: np.ndarray, candidate: np.ndarray) -> float:
        """Return the same global-phase-invariant state error used in test_pauli.py."""
        reference = np.asarray(reference, dtype=np.complex128)
        candidate = np.asarray(candidate, dtype=np.complex128)
        overlap = np.vdot(reference, candidate)
        if abs(overlap) > 1e-12:
            candidate = candidate * np.conj(overlap / abs(overlap))
        return float(np.linalg.norm(reference - candidate))


    def _normalize_state_vector(state: np.ndarray, tol: float) -> np.ndarray:
        """Return a normalized complex state vector, rejecting only the zero vector."""
        psi = np.asarray(state, dtype=np.complex128)
        if psi.ndim != 1:
            raise ValueError("Psi must be a one-dimensional state vector.")
        if psi.size == 0:
            raise ValueError("Psi must not be empty.")
        if not np.all(np.isfinite(psi)):
            raise ValueError("Psi entries must be finite.")

        norm = float(np.linalg.norm(psi))
        if norm <= tol:
            raise ValueError("Psi must not be the zero vector.")

        return np.ascontiguousarray(psi / norm)


    @dataclass(slots=True)
    class StatePreparationResult:
        """Result container used by the Mottonen state-preparation implementation."""

        method: str
        Psi: np.ndarray
        target_qubits: int
        target_error: float
        tol: float = 1e-12
        dim: int = field(init=False)
        padded_dim: int = field(init=False)
        _circuit: Optional[Circuit] = field(init=False, repr=False, default=None)
        _total_error: Optional[float] = field(init=False, repr=False, default=None)
        _evolution_result: Optional[np.ndarray] = field(init=False, repr=False, default=None)

        def __post_init__(self) -> None:
            if self.target_error <= 0:
                raise ValueError("target_error must be positive.")
            if self.tol <= 0:
                raise ValueError("tol must be positive.")
            if isinstance(self.target_qubits, bool) or not isinstance(self.target_qubits, (int, np.integer)):
                raise TypeError("target_qubits must be an integer.")
            if self.target_qubits < 0:
                raise ValueError("target_qubits must be non-negative.")

            psi = MottonenAlgorithm._normalize_state_vector(self.Psi, self.tol)
            self.dim = int(psi.size)
            padded_dim = 1 << int(self.target_qubits)
            if self.dim > padded_dim:
                raise ValueError(f"State vector length {self.dim} exceeds target Hilbert dimension {padded_dim}.")

            if self.dim < padded_dim:
                padded_psi = np.zeros(padded_dim, dtype=np.complex128)
                padded_psi[:self.dim] = psi
                warnings.warn(
                    f"State vector length {self.dim} is smaller than 2**target_qubits; padded to {padded_dim}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                padded_psi = psi

            self.padded_dim = padded_dim
            self.Psi = np.ascontiguousarray(padded_psi)

        def __repr__(self) -> str:
            return (
                f"StatePreparationResult(method={self.method}, "
                f"target_error={self.target_error}, qubits={self.target_qubits})"
            )

        @staticmethod
        def _circuit_has_blocks(circuit: Optional[Circuit]) -> bool:
            if circuit is None:
                return False
            return any(getattr(gate, 'block_gate_sequence', None) is not None for gate in circuit.data().data())

        def _flatten_circuit(self, circuit: Optional[Circuit], max_depth: int = 64) -> Optional[Circuit]:
            if circuit is None:
                return None

            original_name = getattr(circuit, 'name', None)
            flattened = circuit
            depth = 0
            while self._circuit_has_blocks(flattened):
                if depth >= max_depth:
                    raise ValueError(f"Circuit decomposition exceeded max_depth={max_depth}")
                flattened = flattened.decompose(1)
                depth += 1

            if original_name is not None and hasattr(flattened, 'update_name'):
                flattened.update_name(original_name)
            return flattened

        @property
        def circuit(self) -> Circuit:
            if self._circuit is None:
                self._run()
            self._circuit = self._flatten_circuit(self._circuit)
            return self._circuit

        @property
        def total_error(self) -> float:
            if self._total_error is None:
                if self._evolution_result is None:
                    self._run()
                prepared_state = np.asarray(self._evolution_result, dtype=np.complex128)[:, 0]
                self._total_error = MottonenAlgorithm._phase_invariant_error(self.Psi, prepared_state)
            return self._total_error

        @property
        def evolution_result(self) -> np.ndarray:
            if self._evolution_result is None:
                self._run()
            return np.asarray(self._evolution_result, dtype=np.complex128)

    def _bit_reversed_state_vector(state_vector: np.ndarray, num_qubits: int) -> np.ndarray:
        """Reorder amplitudes so the recursive schedule matches repository wire ordering."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        if num_qubits <= 1:
            return state_vector.copy()

        reordered = np.empty_like(state_vector)
        for index, amplitude in enumerate(state_vector):
            reversed_index = int(format(index, f'0{num_qubits}b')[::-1], 2)
            reordered[reversed_index] = amplitude
        return reordered


    def _gray_code(num_bits: int) -> np.ndarray:
        """Return the reflected Gray-code sequence used in the uniform-rotation decomposition."""
        if num_bits < 0:
            raise ValueError('num_bits must be non-negative.')
        if num_bits == 0:
            return np.array([0], dtype=np.int64)

        return np.array([index ^ (index >> 1) for index in range(1 << num_bits)], dtype=np.int64)


    def _compute_theta(alphas: np.ndarray) -> np.ndarray:
        """Convert uniformly controlled rotation angles into the Gray-code ladder angles."""
        alphas = np.asarray(alphas, dtype=np.float64)
        if alphas.ndim != 1 or alphas.size == 0:
            raise ValueError('alphas must be a non-empty one-dimensional vector.')

        size = int(alphas.size)
        if size & (size - 1):
            raise ValueError('alphas length must be a power of 2.')

        num_controls = size.bit_length() - 1
        gray = MottonenAlgorithm._gray_code(num_controls)
        theta = np.zeros(size, dtype=np.float64)
        scale = 1.0 / float(size)

        for i, gray_code in enumerate(gray):
            accumulator = 0.0
            for j, alpha in enumerate(alphas):
                parity = ((j & int(gray_code)).bit_count()) & 1
                accumulator += -float(alpha) if parity else float(alpha)
            theta[i] = scale * accumulator

        return theta


    def _compute_alpha_z(phases: np.ndarray, num_qubits: int, k: int) -> np.ndarray:
        """Compute the uniformly controlled RZ angles from Eq. (5) of the paper."""
        phases = np.asarray(phases, dtype=np.float64)
        num_pairs = 1 << (num_qubits - k)
        block = 1 << (k - 1)
        alpha_z = np.zeros(num_pairs, dtype=np.float64)

        for j in range(num_pairs):
            start0 = (2 * j) * block
            start1 = (2 * j + 1) * block
            phase_diff = phases[start1:start1 + block] - phases[start0:start0 + block]
            alpha_z[j] = float(np.sum(phase_diff) / block)

        return alpha_z


    def _compute_alpha_y(amplitudes: np.ndarray, num_qubits: int, k: int) -> np.ndarray:
        """Compute the uniformly controlled RY angles for the amplitude-preparation stage."""
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        num_pairs = 1 << (num_qubits - k)
        half_block = 1 << (k - 1)
        full_block = 1 << k
        alpha_y = np.zeros(num_pairs, dtype=np.float64)

        for j in range(num_pairs):
            block_start = j * full_block
            numerator = float(np.sum(amplitudes[block_start + half_block:block_start + full_block] ** 2))
            denominator = float(np.sum(amplitudes[block_start:block_start + full_block] ** 2))
            if denominator <= 1e-15:
                alpha_y[j] = 0.0
                continue

            ratio = float(np.clip(numerator / denominator, 0.0, 1.0))
            alpha_y[j] = float(2.0 * np.arcsin(np.sqrt(ratio)))

        return alpha_y


    def _ry_matrix(angle: float) -> np.ndarray:
        """Return the standard single-qubit RY matrix."""
        half_angle = 0.5 * float(angle)
        return np.array(
            [
                [np.cos(half_angle), -np.sin(half_angle)],
                [np.sin(half_angle), np.cos(half_angle)],
            ],
            dtype=np.complex128,
        )


    def _rz_matrix(angle: float) -> np.ndarray:
        """Return the standard single-qubit RZ matrix."""
        half_angle = 0.5 * float(angle)
        return np.array(
            [
                [np.exp(-1j * half_angle), 0.0],
                [0.0, np.exp(1j * half_angle)],
            ],
            dtype=np.complex128,
        )


    def _uniform_rotation_operations(
        axis: str,
        alphas: np.ndarray,
        control_wires: list[int],
        target_wire: int,
    ) -> list[dict[str, float | int]]:
        """Return the Gray-code gate ladder for one uniformly controlled rotation."""
        alphas = np.asarray(alphas, dtype=np.float64)
        if alphas.ndim != 1 or alphas.size == 0:
            raise ValueError('alphas must be a non-empty one-dimensional vector.')

        if not control_wires:
            angle = float(alphas[0])
            if abs(angle) <= 1e-15:
                return []
            return [{'name': axis, 'angle': angle, 'target': int(target_wire)}]

        theta = MottonenAlgorithm._compute_theta(alphas)
        gray = MottonenAlgorithm._gray_code(len(control_wires))
        operations: list[dict[str, float | int]] = []

        for index, angle in enumerate(theta):
            angle = float(angle)
            if abs(angle) > 1e-15:
                operations.append({'name': axis, 'angle': angle, 'target': int(target_wire)})

            changed = int(gray[index] ^ gray[(index + 1) % len(gray)])
            changed_bit = changed.bit_length() - 1
            operations.append(
                {
                    'name': 'cx',
                    'control': int(control_wires[changed_bit]),
                    'target': int(target_wire),
                }
            )

        return operations


    def _mottonen_operations(state_vector: np.ndarray, num_qubits: int) -> list[dict[str, float | int]]:
        """Return the explicit RY/RZ/CNOT schedule from the Mottönen construction."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        expected_dim = 1 << num_qubits
        if state_vector.ndim != 1 or state_vector.size != expected_dim:
            raise ValueError(f'state_vector must have shape ({expected_dim},).')

        if num_qubits == 0:
            return []

        ordered_state = MottonenAlgorithm._bit_reversed_state_vector(state_vector, num_qubits)
        amplitudes = np.abs(ordered_state)
        phases = np.angle(ordered_state)
        wires_reversed = list(range(num_qubits))[::-1]
        operations: list[dict[str, float | int]] = []

        for k in range(num_qubits, 0, -1):
            alpha_y = MottonenAlgorithm._compute_alpha_y(amplitudes, num_qubits, k)
            control_wires = list(wires_reversed[k:])
            target_wire = int(wires_reversed[k - 1])
            operations.extend(
                MottonenAlgorithm._uniform_rotation_operations('ry', alpha_y, control_wires, target_wire)
            )

        if not np.allclose(phases, 0.0, atol=1e-15):
            for k in range(num_qubits, 0, -1):
                alpha_z = MottonenAlgorithm._compute_alpha_z(phases, num_qubits, k)
                control_wires = list(wires_reversed[k:])
                target_wire = int(wires_reversed[k - 1])
                operations.extend(
                    MottonenAlgorithm._uniform_rotation_operations('rz', alpha_z, control_wires, target_wire)
                )

        return operations


    def _apply_single_qubit_gate(
        evolution: np.ndarray,
        matrix_2x2: np.ndarray,
        target_wire: int,
    ) -> np.ndarray:
        """Left-multiply a full evolution matrix by a single-qubit gate on one wire."""
        dim = evolution.shape[0]
        updated = np.array(evolution, dtype=np.complex128, copy=True)
        step = 1 << target_wire
        block = step << 1

        for base in range(0, dim, block):
            for offset in range(step):
                index0 = base + offset
                index1 = index0 + step
                row0 = evolution[index0].copy()
                row1 = evolution[index1].copy()
                updated[index0] = matrix_2x2[0, 0] * row0 + matrix_2x2[0, 1] * row1
                updated[index1] = matrix_2x2[1, 0] * row0 + matrix_2x2[1, 1] * row1

        return updated


    def _apply_cnot(evolution: np.ndarray, control_wire: int, target_wire: int) -> np.ndarray:
        """Left-multiply a full evolution matrix by a CNOT permutation."""
        dim = evolution.shape[0]
        permutation = np.arange(dim, dtype=np.int64)

        for basis_index in range(dim):
            if (basis_index >> control_wire) & 1:
                permutation[basis_index] = basis_index ^ (1 << target_wire)

        return np.asarray(evolution[permutation, :], dtype=np.complex128)


    def _build_evolution_matrix(operations: list[dict[str, float | int]], num_qubits: int) -> np.ndarray:
        """Accumulate the exact unitary emitted by the explicit gate schedule."""
        dim = 1 << num_qubits
        evolution = np.eye(dim, dtype=np.complex128)

        for operation in operations:
            name = str(operation['name'])
            if name == 'ry':
                evolution = MottonenAlgorithm._apply_single_qubit_gate(
                    evolution,
                    MottonenAlgorithm._ry_matrix(float(operation['angle'])),
                    int(operation['target']),
                )
            elif name == 'rz':
                evolution = MottonenAlgorithm._apply_single_qubit_gate(
                    evolution,
                    MottonenAlgorithm._rz_matrix(float(operation['angle'])),
                    int(operation['target']),
                )
            elif name == 'cx':
                evolution = MottonenAlgorithm._apply_cnot(
                    evolution,
                    int(operation['control']),
                    int(operation['target']),
                )
            else:
                raise ValueError(f'Unsupported gate operation {name!r}.')

        return evolution


    def _build_mottonen_circuit(operations: list[dict[str, float | int]], num_qubits: int) -> Circuit:
        """Build the emitted UnitaryLab circuit from the explicit paper decomposition."""
        qc = Circuit(num_qubits, name='Quantum circuit for Mottonen State Preparation')
        for operation in operations:
            name = str(operation['name'])
            if name == 'ry':
                qc.ry(float(operation['angle']), int(operation['target']))
            elif name == 'rz':
                qc.rz(float(operation['angle']), int(operation['target']))
            elif name == 'cx':
                qc.cx(int(operation['control']), int(operation['target']))
            else:
                raise ValueError(f'Unsupported gate operation {name!r}.')
        return qc


    def mottonen_state_preparation_circuit(state_vector: np.ndarray, num_qubits: int) -> Circuit:
        """Construct the Mottonen state-preparation circuit from explicit elementary gates."""
        operations = MottonenAlgorithm._mottonen_operations(state_vector, num_qubits)
        return MottonenAlgorithm._build_mottonen_circuit(operations, num_qubits)


    def mottonen_state_preparation_matrix(state_vector: np.ndarray, num_qubits: int) -> np.ndarray:
        """Return the exact unitary emitted by the explicit Mottonen gate schedule."""
        operations = MottonenAlgorithm._mottonen_operations(state_vector, num_qubits)
        return MottonenAlgorithm._build_evolution_matrix(operations, num_qubits)


    class Mottonen(StatePreparationResult):
        """State-preparation interface for the Mottönen method."""

        def __init__(self, Psi: np.ndarray, target_qubits: int, target_error: float) -> None:
            super().__init__('mottonen', Psi, target_qubits, target_error)
            self._run()

        def _run(self) -> None:
            """Build the Mottonen circuit and cache the exact emitted unitary and error."""
            operations = MottonenAlgorithm._mottonen_operations(self.Psi, self.target_qubits)
            self._circuit = self._flatten_circuit(
                MottonenAlgorithm._build_mottonen_circuit(operations, self.target_qubits)
            )
            self._evolution_result = MottonenAlgorithm._build_evolution_matrix(operations, self.target_qubits)
            prepared_state = self._evolution_result[:, 0]
            self._total_error = MottonenAlgorithm._phase_invariant_error(self.Psi, prepared_state)



def test(Psi=None, target_qubits: int = 2, target_error: float = 1e-6):
    """Test Mottönen state preparation with a Bell-like default state."""
    if Psi is None:
        Psi = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    algo = MottonenAlgorithm(text_mode="legacy")
    return algo.run(Psi=Psi, target_qubits=int(target_qubits), target_error=float(target_error))


if __name__ == "__main__":
    Psi = [1, 0, 0, 1]  # [PARAM]
    target_qubits = 2  # [PARAM]
    target_error = 1e-6  # [PARAM]
    test(Psi=Psi, target_qubits=target_qubits, target_error=target_error)
