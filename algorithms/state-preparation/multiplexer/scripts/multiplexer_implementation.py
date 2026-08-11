# -*- coding: utf-8 -*-

"""Multiplexer state preparation via recursive amplitude splitting."""

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


class MultiplexerAlgorithm(BaseAlgorithm):
    """Standalone multiplexer state-preparation algorithm module."""

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
        super().__init__("Multiplexer State Preparation Algorithm", "MUX", text_mode, algo_dir)

    def run(
        self,
        Psi,
        target_qubits: int,
        target_error: float = 1e-6,
        backend='torch',
        device='cpu',
        dtype=np.complex128,
    ) -> Dict[str, Any]:
        psi = np.asarray(Psi, dtype=np.complex128)
        self.update_input({
            "Method": "multiplexer",
            "Target qubits": target_qubits,
            "Target error": target_error,
            "State vector length": int(psi.size),
        })
        start_time = time.time()
        self.log("Stage 1: Validating and normalizing target state")
        result = MultiplexerAlgorithm.Multiplexer(psi, int(target_qubits), float(target_error))
        self.log("Stage 2: Building multiplexer circuit")
        circuit = result.circuit
        self.log("Stage 3: Computing emitted unitary and preparation error")
        prepared_state = np.asarray(result.evolution_result, dtype=np.complex128)[:, 0]
        total_error = float(result.total_error)
        comp_time = time.time() - start_time
        is_success = total_error <= max(float(target_error), 1e-10)
        self.update_output({
            "Prepared state": prepared_state,
            "Total error": total_error,
            "Computation time (s)": round(comp_time, 4),
        })
        self.status = "success" if is_success else "failed"
        self.summary = f"Multiplexer state preparation completed with error {total_error:.6e}."
        self.log("Stage 4: Exporting circuit diagram")
        circuit_path = self.save_circuit(circuit)
        filename = self.save_txt()
        return self._build_return_dict(is_success, circuit_path, filename, circuit)



    def _bit_reversed_state_vector(state_vector: np.ndarray, num_wires: int) -> np.ndarray:
        """Reorder amplitudes so the recursive schedule matches repository wire ordering."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        if num_wires <= 1:
            return state_vector.copy()

        reordered = np.empty_like(state_vector)
        for index, amplitude in enumerate(state_vector):
            reversed_index = int(format(index, f'0{num_wires}b')[::-1], 2)
            reordered[reversed_index] = amplitude
        return reordered


    def _state_vector_error(reference: np.ndarray, candidate: np.ndarray, tol: float) -> float:
        """Return a global-phase-invariant state-vector error."""
        reference = np.asarray(reference, dtype=np.complex128)
        candidate = np.asarray(candidate, dtype=np.complex128)
        overlap = np.vdot(reference, candidate)
        if abs(overlap) > tol:
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
        """Result container used by the multiplexer state-preparation implementation."""

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

            psi = MultiplexerAlgorithm._normalize_state_vector(self.Psi, self.tol)
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
                f"MultiplexerAlgorithm.StatePreparationResult(method={self.method}, "
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
                self._total_error = MultiplexerAlgorithm._state_vector_error(self.Psi, prepared_state, self.tol)
            return self._total_error

        @property
        def evolution_result(self) -> np.ndarray:
            if self._evolution_result is None:
                self._run()
            return np.asarray(self._evolution_result, dtype=np.complex128)


    def _multiplexer_gate_spec(amplitudes: np.ndarray) -> list[dict[str, object]]:
        """Return the recursive RY/mcry schedule from the multiplexer state-prep construction."""
        amplitudes = np.asarray(amplitudes, dtype=np.float64)
        if amplitudes.ndim != 1 or amplitudes.size == 0:
            raise ValueError('amplitudes must be a non-empty one-dimensional vector.')

        dim = amplitudes.size
        if dim & (dim - 1):
            raise ValueError('amplitudes length must be a power of 2.')

        norm = float(np.linalg.norm(amplitudes))
        if norm <= 1e-12:
            raise ValueError('amplitudes must not be the zero vector.')

        amplitudes = amplitudes / norm
        # Prefix sums let each recursive split query left/right probability mass in O(1).
        probs = amplitudes ** 2
        prefix = np.concatenate(([0.0], np.cumsum(probs)))
        num_wires = dim.bit_length() - 1
        gates: list[dict[str, object]] = []

        def build(level: int, start: int, length: int, controls: tuple[int, ...]) -> None:
            if level == num_wires:
                return

            half = length // 2
            # Each node splits its probability mass between the left and right subtree.
            left_norm = float(np.sqrt(prefix[start + half] - prefix[start]))
            right_norm = float(np.sqrt(prefix[start + length] - prefix[start + half]))

            if left_norm + right_norm <= 1e-15:
                theta = 0.0
            else:
                theta = float(2.0 * np.arctan2(right_norm, left_norm))

            if controls:
                gates.append(
                    {
                        'name': 'MCRY',
                        # `controls` stores the branch bits for wires 0..level-1.
                        'target': level,
                        'controls': list(range(len(controls))),
                        'control_values': list(controls),
                        'angle': theta,
                    }
                )
            else:
                gates.append(
                    {
                        'name': 'RY',
                        'target': level,
                        'controls': None,
                        'control_values': None,
                        'angle': theta,
                    }
                )

            build(level + 1, start, half, controls + (0,))
            build(level + 1, start + half, half, controls + (1,))

        build(0, 0, dim, ())
        return gates


    def _apply_controlled_ry(
        qc: Circuit,
        angle: float,
        target: int,
        controls: list[int] | None,
        control_values: list[int] | None,
    ) -> None:
        """Emit one amplitude-splitting gate using explicit rotation primitives."""
        if abs(angle) <= 1e-15:
            return

        if not controls:
            qc.ry(angle, target)
        elif len(controls) == 1:
            qc.cry(angle, controls[0], target, control_values)
        else:
            qc.mcry(angle, controls, target, control_values)


    def _apply_controlled_phase(
        qc: Circuit,
        angle: float,
        target: int,
        controls: list[int] | None,
        control_values: list[int] | None,
    ) -> None:
        """Emit one phase gate using explicit phase primitives."""
        if abs(angle) <= 1e-15:
            return

        if not controls:
            qc.p(angle, target)
        elif len(controls) == 1:
            qc.cp(angle, controls[0], target, control_values)
        else:
            qc.mcp(angle, controls, target, control_values)


    def _apply_basis_state_phase(
        qc: Circuit,
        basis_index: int,
        phase_angle: float,
        num_wires: int,
    ) -> None:
        """Apply a phase to exactly one computational-basis state."""
        if abs(phase_angle) <= 1e-15:
            return

        bits = [int(bit) for bit in format(basis_index, f'0{num_wires}b')]
        target = num_wires - 1
        controls = list(range(target))
        control_values = bits[:-1]
        # Convert the selected basis state into an active-high controlled phase on the last wire.
        needs_flip = bits[target] == 0

        if needs_flip:
            qc.x(target)

        MultiplexerAlgorithm._apply_controlled_phase(
            qc,
            phase_angle,
            target,
            controls,
            control_values,
        )

        if needs_flip:
            qc.x(target)


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


    def _phase_matrix(angle: float) -> np.ndarray:
        """Return the single-qubit phase matrix used by Circuit.p."""
        return np.array(
            [
                [1.0, 0.0],
                [0.0, np.exp(1j * float(angle))],
            ],
            dtype=np.complex128,
        )


    def _x_matrix() -> np.ndarray:
        """Return the single-qubit X matrix."""
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


    def _controls_match(
        basis_index: int,
        controls: list[int] | None,
        control_values: list[int] | None,
    ) -> bool:
        """Return whether one computational-basis index matches a control pattern."""
        if not controls:
            return True
        values = control_values if control_values is not None else [1] * len(controls)
        return all(((basis_index >> control) & 1) == int(value) for control, value in zip(controls, values))


    def _apply_controlled_single_qubit_matrix(
        evolution: np.ndarray,
        matrix_2x2: np.ndarray,
        target: int,
        controls: list[int] | None = None,
        control_values: list[int] | None = None,
    ) -> np.ndarray:
        """Left-multiply a dense evolution matrix by a controlled one-qubit gate."""
        dim = evolution.shape[0]
        updated = np.array(evolution, dtype=np.complex128, copy=True)
        target_mask = 1 << target

        for index0 in range(dim):
            if index0 & target_mask:
                continue
            if not MultiplexerAlgorithm._controls_match(index0, controls, control_values):
                continue

            index1 = index0 | target_mask
            row0 = evolution[index0].copy()
            row1 = evolution[index1].copy()
            updated[index0] = matrix_2x2[0, 0] * row0 + matrix_2x2[0, 1] * row1
            updated[index1] = matrix_2x2[1, 0] * row0 + matrix_2x2[1, 1] * row1

        return updated


    def _apply_basis_state_phase_dense(
        evolution: np.ndarray,
        basis_index: int,
        phase_angle: float,
        num_wires: int,
    ) -> np.ndarray:
        """Apply a basis-state-selective phase to a dense evolution matrix."""
        if abs(phase_angle) <= 1e-15:
            return evolution

        bits = [int(bit) for bit in format(basis_index, f'0{num_wires}b')]
        target = num_wires - 1
        controls = list(range(target))
        control_values = bits[:-1]
        needs_flip = bits[target] == 0

        if needs_flip:
            evolution = MultiplexerAlgorithm._apply_controlled_single_qubit_matrix(
                evolution,
                MultiplexerAlgorithm._x_matrix(),
                target,
            )

        evolution = MultiplexerAlgorithm._apply_controlled_single_qubit_matrix(
            evolution,
            MultiplexerAlgorithm._phase_matrix(phase_angle),
            target,
            controls,
            control_values,
        )

        if needs_flip:
            evolution = MultiplexerAlgorithm._apply_controlled_single_qubit_matrix(
                evolution,
                MultiplexerAlgorithm._x_matrix(),
                target,
            )

        return evolution


    def _build_multiplexer_dense_matrix(
        state_vector: np.ndarray,
        num_wires: int,
    ) -> np.ndarray:
        """Construct the multiplexer unitary directly from the explicit gate schedule."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        expected_dim = 1 << num_wires
        if state_vector.ndim != 1 or state_vector.size != expected_dim:
            raise ValueError(f'state_vector must have shape ({expected_dim},).')
        ordered_state = MultiplexerAlgorithm._bit_reversed_state_vector(state_vector, num_wires)

        evolution = np.eye(expected_dim, dtype=np.complex128)
        if num_wires == 0:
            return evolution

        for gate in MultiplexerAlgorithm._multiplexer_gate_spec(np.abs(ordered_state)):
            evolution = MultiplexerAlgorithm._apply_controlled_single_qubit_matrix(
                evolution,
                MultiplexerAlgorithm._ry_matrix(float(gate['angle'])),
                int(gate['target']),
                gate['controls'],
                gate['control_values'],
            )

        for basis_index, phase_angle in enumerate(np.angle(ordered_state)):
            evolution = MultiplexerAlgorithm._apply_basis_state_phase_dense(
                evolution,
                basis_index,
                float(phase_angle),
                num_wires,
            )

        return evolution


    def _build_multiplexer_gate_circuit(
        state_vector: np.ndarray,
        num_wires: int,
    ) -> Circuit:
        """Construct the multiplexer circuit directly from explicit UnitaryLab gates."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        expected_dim = 1 << num_wires
        if state_vector.ndim != 1 or state_vector.size != expected_dim:
            raise ValueError(f'state_vector must have shape ({expected_dim},).')
        ordered_state = MultiplexerAlgorithm._bit_reversed_state_vector(state_vector, num_wires)

        qc = Circuit(num_wires, name='Quantum circuit for Multiplexer State Preparation')
        if num_wires == 0:
            return qc

        for gate in MultiplexerAlgorithm._multiplexer_gate_spec(np.abs(ordered_state)):
            MultiplexerAlgorithm._apply_controlled_ry(
                qc,
                float(gate['angle']),
                int(gate['target']),
                gate['controls'],
                gate['control_values'],
            )

        for basis_index, phase_angle in enumerate(np.angle(ordered_state)):
            MultiplexerAlgorithm._apply_basis_state_phase(qc, basis_index, float(phase_angle), num_wires)

        return qc


    def multiplexer_state_preparation_circuit(state_vector: np.ndarray, num_wires: int) -> Circuit:
        """Construct the multiplexer state-preparation circuit from explicit gates."""
        return MultiplexerAlgorithm._build_multiplexer_gate_circuit(state_vector, num_wires)


    def multiplexer_state_preparation_matrix(state_vector: np.ndarray, num_wires: int) -> np.ndarray:
        """Construct the emitted multiplexer unitary directly from the gate schedule."""
        return MultiplexerAlgorithm._build_multiplexer_dense_matrix(state_vector, num_wires)


    class Multiplexer(StatePreparationResult):
        """
        State-preparation interface for the multiplexer method.

        The emitted circuit is built directly from explicit UnitaryLab rotation and
        phase primitives, and `evolution_result` is computed from that circuit via
        `Circuit.get_matrix()`.
        """

        def __init__(self, Psi: np.ndarray, target_qubits: int, target_error: float) -> None:
            """Initialize the multiplexer state-preparation solver."""
            super().__init__('multiplexer', Psi, target_qubits, target_error)
            self._run()

        def _run(self) -> None:
            """Build the multiplexer circuit and cache the emitted unitary and error."""
            self._circuit = self._flatten_circuit(
                MultiplexerAlgorithm.multiplexer_state_preparation_circuit(self.Psi, self.target_qubits)
            )
            self._evolution_result = MultiplexerAlgorithm.multiplexer_state_preparation_matrix(
                self.Psi,
                self.target_qubits,
            )
            prepared_state = self._evolution_result[:, 0]
            self._total_error = MultiplexerAlgorithm._state_vector_error(self.Psi, prepared_state, self.tol)


def test(Psi=None, target_qubits: int = 2, target_error: float = 1e-6):
    """Test multiplexer state preparation with a default complex state."""
    if Psi is None:
        Psi = np.array([1, 1j, 0, 1], dtype=np.complex128)
        Psi = Psi / np.linalg.norm(Psi)
    algo = MultiplexerAlgorithm(text_mode="legacy")
    return algo.run(Psi=Psi, target_qubits=int(target_qubits), target_error=float(target_error))


if __name__ == "__main__":
    Psi = [1, 1j, 0, 1]  # [PARAM]
    target_qubits = 2  # [PARAM]
    target_error = 1e-6  # [PARAM]
    test(Psi=Psi, target_qubits=target_qubits, target_error=target_error)
