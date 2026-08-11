# -*- coding: utf-8 -*-
from __future__ import annotations
import math
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



class SuperpositionAlgorithm(BaseAlgorithm):
    """Standalone sparse-superposition state-preparation algorithm module."""

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
        super().__init__("Superposition State Preparation Algorithm", "SUP", text_mode, algo_dir)

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
            "Method": "superposition",
            "Target qubits": target_qubits,
            "Target error": target_error,
            "State vector length": int(psi.size),
        })
        start_time = time.time()
        self.log("Stage 1: Extracting sparse support")
        result = SuperpositionAlgorithm.Superposition(psi, int(target_qubits), float(target_error))
        self.log("Stage 2: Building sparse-superposition circuit")
        circuit = result.circuit
        self.log("Stage 3: Computing emitted unitary and preparation error")
        prepared_state = np.asarray(result.evolution_result, dtype=np.complex128)[:, 0]
        total_error = float(result.total_error)
        comp_time = time.time() - start_time
        is_success = total_error <= max(float(target_error), 1e-10)
        self.update_output({
            "Prepared state": prepared_state,
            "Total error": total_error,
            "Support size": len(result.basis_states),
            "Index register qubits": result.index_register_qubits,
            "Computation time (s)": round(comp_time, 4),
        })
        self.status = "success" if is_success else "failed"
        self.summary = f"Superposition state preparation completed with error {total_error:.6e}."
        self.log("Stage 4: Exporting circuit diagram")
        circuit_path = self.save_circuit(circuit)
        filename = self.save_txt()
        return self._build_return_dict(is_success, circuit_path, filename, circuit)



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
        """Result container used by the sparse-superposition implementation."""

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

            psi = SuperpositionAlgorithm._normalize_state_vector(self.Psi, self.tol)
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
                f"SuperpositionAlgorithm.StatePreparationResult(method={self.method}, "
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
                self._total_error = SuperpositionAlgorithm._state_vector_error(self.Psi, prepared_state, self.tol)
            return self._total_error

        @property
        def evolution_result(self) -> np.ndarray:
            if self._evolution_result is None:
                self._run()
            return np.asarray(self._evolution_result, dtype=np.complex128)

    def order_states(basis_states: list[list[int]]) -> dict[tuple[int, ...], tuple[int, ...]]:
        """
        Map a support basis set onto the first m computational basis states.

        This mirrors the high-level idea used by PennyLane's Superposition template
        while keeping the implementation local to this repository.
        """
        if not basis_states:
            return {}

        lengths = {len(state) for state in basis_states}
        if len(lengths) != 1:
            raise ValueError('All basis states must have the same length.')

        seen: set[tuple[int, ...]] = set()
        normalized_states: list[tuple[int, ...]] = []
        for state in basis_states:
            tuple_state = tuple(int(bit) for bit in state)
            if any(bit not in (0, 1) for bit in tuple_state):
                raise ValueError('Basis-state entries must be binary.')
            if tuple_state in seen:
                raise ValueError('Basis states must be unique.')
            seen.add(tuple_state)
            normalized_states.append(tuple_state)

        m = len(normalized_states)
        length = len(normalized_states[0])
        basis_ints = [int(''.join(str(bit) for bit in state), 2) for state in normalized_states]

        state_map: dict[tuple[int, ...], tuple[int, ...]] = {}
        unmapped_states: list[tuple[int, ...]] = []
        unmapped_ints = {index: None for index in range(m)}

        for basis_int, state in zip(basis_ints, normalized_states):
            if basis_int < m:
                state_map[state] = state
                unmapped_ints.pop(basis_int, None)
            else:
                unmapped_states.append(state)

        for state, mapped_int in zip(unmapped_states, unmapped_ints):
            state_map[state] = tuple(int(bit) for bit in f'{mapped_int:0{length}b}')

        return state_map


    def _index_to_bits(index: int, num_qubits: int) -> tuple[int, ...]:
        """Return one computational basis index as an n-bit tuple."""
        return tuple(int(bit) for bit in format(index, f'0{num_qubits}b'))


    def _bits_to_index(bits: tuple[int, ...]) -> int:
        """Convert one computational basis bit string into its integer index."""
        return int(''.join(str(bit) for bit in bits), 2)


    def _extract_sparse_superposition(
        state_vector: np.ndarray,
        tol: float = 1e-12,
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """Extract non-zero coefficients and their computational basis states."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        if state_vector.ndim != 1:
            raise ValueError('state_vector must be one-dimensional.')
        if state_vector.size == 0 or state_vector.size & (state_vector.size - 1):
            raise ValueError('state_vector length must be a non-zero power of 2.')

        num_qubits = state_vector.size.bit_length() - 1
        support_indices = [index for index, amplitude in enumerate(state_vector) if abs(amplitude) > tol]
        if not support_indices:
            raise ValueError('state_vector must contain at least one non-zero amplitude.')

        coeffs = np.asarray([state_vector[index] for index in support_indices], dtype=np.complex128)
        bases = [SuperpositionAlgorithm._index_to_bits(index, num_qubits) for index in support_indices]
        return coeffs, bases


    def _coefficient_register_qubits(num_terms: int) -> int:
        """Return the number of qubits needed to index the support terms."""
        if num_terms <= 0:
            raise ValueError('num_terms must be positive.')
        return int(math.ceil(math.log2(num_terms))) if num_terms > 1 else 0


    def _pad_coefficients(coeffs: np.ndarray, register_qubits: int) -> np.ndarray:
        """Pad the coefficient vector to the full coefficient-register dimension."""
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        dim = 1 << register_qubits
        padded = np.zeros(dim, dtype=np.complex128)
        padded[:coeffs.size] = coeffs
        return padded


    def _build_coefficient_stage_matrix(
        coeffs: np.ndarray,
        num_qubits: int,
        target_error: float,
    ) -> np.ndarray:
        """
        Build the exact first-stage unitary for the compact coefficient register.

        The first stage prepares ``sum_i c_i |i>`` on the smallest possible register and
        leaves the remaining wires in ``|0>``.
        """
        del target_error  # QR completion is exact; the public result still stores this value.

        register_qubits = SuperpositionAlgorithm._coefficient_register_qubits(int(coeffs.size))
        if register_qubits == 0:
            return np.eye(1 << num_qubits, dtype=np.complex128)

        coefficient_state = SuperpositionAlgorithm._pad_coefficients(coeffs, register_qubits)
        dim = int(coefficient_state.size)
        basis = np.eye(dim, dtype=np.complex128)
        basis[:, 0] = coefficient_state
        local_unitary, _ = np.linalg.qr(basis)

        overlap = np.vdot(coefficient_state, local_unitary[:, 0])
        if abs(overlap) > 1e-12:
            local_unitary[:, 0] = local_unitary[:, 0] * np.conj(overlap / abs(overlap))

        remaining_qubits = num_qubits - register_qubits
        return np.kron(np.eye(1 << remaining_qubits, dtype=np.complex128), local_unitary)


    def _build_prefix_to_support_permutation(
        basis_states: list[tuple[int, ...]],
    ) -> np.ndarray:
        """
        Build a full-system permutation that sends |j> to the j-th support basis state.

        The permutation acts non-trivially only on the union of the support set and the
        first ``m`` computational basis states.
        """
        if not basis_states:
            raise ValueError('basis_states must not be empty.')

        state_map = SuperpositionAlgorithm.order_states([list(state) for state in basis_states])
        num_qubits = len(basis_states[0])
        dim = 1 << num_qubits
        m = len(basis_states)
        support_set = set(basis_states)
        inverse_map = {mapped: source for source, mapped in state_map.items()}

        permutation = list(range(dim))
        for prefix_index in range(m):
            prefix_state = SuperpositionAlgorithm._index_to_bits(prefix_index, num_qubits)
            source_state = inverse_map[prefix_state]
            permutation[prefix_index] = SuperpositionAlgorithm._bits_to_index(source_state)

        for state in support_set:
            state_index = SuperpositionAlgorithm._bits_to_index(state)
            if state_index < m:
                continue
            permutation[state_index] = SuperpositionAlgorithm._bits_to_index(state_map[state])

        matrix = np.zeros((dim, dim), dtype=np.complex128)
        for input_index, output_index in enumerate(permutation):
            matrix[output_index, input_index] = 1.0

        return matrix


    def _permutation_operator(
        basis1: tuple[int, ...] | list[int],
        basis2: tuple[int, ...] | list[int],
        wires: list[int],
        work_wire: int,
    ) -> list[dict[str, object]]:
        """
        Return a basic-gate sequence that maps ``basis1`` to ``basis2``.

        The work wire is flagged when the system matches ``basis1``. Controlled-NOT
        gates then flip the differing target bits, and a final multi-controlled X
        uncomputes the work wire on ``basis2``.
        """
        basis1 = tuple(int(bit) for bit in basis1)
        basis2 = tuple(int(bit) for bit in basis2)

        if len(basis1) != len(basis2):
            raise ValueError('basis1 and basis2 must have the same length.')
        if len(basis1) != len(wires):
            raise ValueError('Basis-state length must match the number of wires.')
        if any(bit not in (0, 1) for bit in basis1 + basis2):
            raise ValueError('Basis-state entries must be binary.')
        if work_wire in wires:
            raise ValueError('work_wire must be different from the target wires.')

        gates: list[dict[str, object]] = [
            {
                'name': 'mcx',
                'controls': list(wires),
                'target': int(work_wire),
                'control_state': list(basis1),
            }
        ]

        for wire, bit1, bit2 in zip(wires, basis1, basis2):
            if bit1 != bit2:
                gates.append(
                    {
                        'name': 'cx',
                        'control': int(work_wire),
                        'target': int(wire),
                    }
                )

        gates.append(
            {
                'name': 'mcx',
                'controls': list(wires),
                'target': int(work_wire),
                'control_state': list(basis2),
            }
        )

        return gates


    def _ordered_coefficients_for_prefix_basis(
        coeffs: np.ndarray,
        basis_states: list[tuple[int, ...]],
    ) -> np.ndarray:
        """Reorder support coefficients so entry j corresponds to prefix basis state |j>."""
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        if coeffs.ndim != 1 or coeffs.size != len(basis_states):
            raise ValueError('coeffs must align one-to-one with basis_states.')

        state_map = SuperpositionAlgorithm.order_states([list(state) for state in basis_states])
        ordered = np.zeros_like(coeffs)
        for coeff, state in zip(coeffs, basis_states):
            prefix_state = state_map[state]
            prefix_index = SuperpositionAlgorithm._bits_to_index(prefix_state)
            ordered[prefix_index] = coeff
        return ordered


    def _build_superposition_unitary(
        state_vector: np.ndarray,
        num_qubits: int,
        target_error: float,
    ) -> np.ndarray:
        """Build the full emitted superposition unitary from the two decomposition stages."""
        coeffs, basis_states = SuperpositionAlgorithm._extract_sparse_superposition(state_vector)
        ordered_coeffs = SuperpositionAlgorithm._ordered_coefficients_for_prefix_basis(coeffs, basis_states)
        coefficient_stage = SuperpositionAlgorithm._build_coefficient_stage_matrix(
            ordered_coeffs,
            num_qubits,
            target_error,
        )
        permutation_stage = SuperpositionAlgorithm._build_prefix_to_support_permutation(basis_states)
        return np.asarray(permutation_stage @ coefficient_stage, dtype=np.complex128)


    def superposition_state_preparation_circuit(
        state_vector: np.ndarray,
        num_qubits: int,
        target_error: float = 1e-9,
        work_wire: int | None = None,
    ) -> Circuit:
        """
        Build a sparse-superposition circuit using a coefficient stage plus one permutation stage.

        The coefficient stage is prepared on the smallest possible index register, then a
        full-system permutation maps those prefix basis states onto the true support basis.
        """
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        expected_dim = 1 << num_qubits
        if state_vector.ndim != 1 or state_vector.size != expected_dim:
            raise ValueError(f'state_vector must have shape ({expected_dim},).')
        if work_wire is None:
            work_wire = num_qubits
        if work_wire < 0:
            raise ValueError('work_wire must be non-negative.')
        if work_wire < num_qubits:
            raise ValueError('work_wire must not overlap with the state-preparation wires.')

        coeffs, basis_states = SuperpositionAlgorithm._extract_sparse_superposition(state_vector)
        ordered_coeffs = SuperpositionAlgorithm._ordered_coefficients_for_prefix_basis(coeffs, basis_states)
        register_qubits = SuperpositionAlgorithm._coefficient_register_qubits(coeffs.size)
        coefficient_stage = SuperpositionAlgorithm._build_coefficient_stage_matrix(
            ordered_coeffs,
            num_qubits,
            target_error,
        )
        state_map = SuperpositionAlgorithm.order_states([list(state) for state in basis_states])
        wires = list(range(num_qubits))

        qc = Circuit(max(num_qubits, work_wire + 1), name='Quantum circuit for Superposition State Preparation')
        if register_qubits > 0:
            qc.unitary(
                coefficient_stage,
                list(range(num_qubits)),
            )

        for basis_state, prefix_state in state_map.items():
            if basis_state == prefix_state:
                continue

            for gate in SuperpositionAlgorithm._permutation_operator(prefix_state, basis_state, wires, work_wire):
                name = str(gate['name'])
                if name == 'mcx':
                    qc.mcx(
                        gate['controls'],
                        gate['target'],
                        control_state=gate['control_state'],
                    )
                elif name == 'cx':
                    qc.cx(gate['control'], gate['target'])
                else:
                    raise ValueError(f'Unsupported permutation gate {name!r}.')

        return qc


    def superposition_state_preparation_matrix(
        state_vector: np.ndarray,
        num_qubits: int,
        target_error: float = 1e-9,
    ) -> np.ndarray:
        """Return the exact dense unitary emitted by the custom superposition decomposition."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        expected_dim = 1 << num_qubits
        if state_vector.ndim != 1 or state_vector.size != expected_dim:
            raise ValueError(f'state_vector must have shape ({expected_dim},).')
        return SuperpositionAlgorithm._build_superposition_unitary(state_vector, num_qubits, target_error)


    class Superposition(StatePreparationResult):
        """
        Sparse-support superposition preparation inspired by PennyLane's high-level idea.

        Unlike direct state initialization, this implementation first prepares a compact
        coefficient register and then permutes the occupied basis states onto the target
        support. This makes the algorithm structurally different from the exact generic
        initialization used by the default base class.
        """

        def __init__(self, Psi: np.ndarray, target_qubits: int, target_error: float) -> None:
            self.coeffs = np.zeros(0, dtype=np.complex128)
            self.basis_states: tuple[tuple[int, ...], ...] = tuple()
            self.index_register_qubits = 0
            super().__init__('superposition', Psi, target_qubits, target_error)
            self._run()

        def _run(self) -> None:
            """Build the custom sparse-superposition circuit and cache its emitted unitary."""
            self.coeffs, basis_states = SuperpositionAlgorithm._extract_sparse_superposition(self.Psi, self.tol)
            self.basis_states = tuple(basis_states)
            self.index_register_qubits = SuperpositionAlgorithm._coefficient_register_qubits(self.coeffs.size)

            self._circuit = self._flatten_circuit(
                SuperpositionAlgorithm.superposition_state_preparation_circuit(
                    self.Psi,
                    self.target_qubits,
                    target_error=self.target_error,
                    work_wire=self.target_qubits,
                )
            )
            self._evolution_result = SuperpositionAlgorithm.superposition_state_preparation_matrix(
                self.Psi,
                self.target_qubits,
                target_error=self.target_error,
            )
            prepared_state = self._evolution_result[:, 0]
            self._total_error = SuperpositionAlgorithm._state_vector_error(self.Psi, prepared_state, self.tol)


def test(Psi=None, target_qubits: int = 3, target_error: float = 1e-6):
    """Test sparse-superposition state preparation with a default sparse state."""
    if Psi is None:
        Psi = np.zeros(1 << target_qubits, dtype=np.complex128)
        Psi[1] = 1 / np.sqrt(2)
        Psi[6] = 1j / np.sqrt(2)
    algo = SuperpositionAlgorithm(text_mode="legacy")
    return algo.run(Psi=Psi, target_qubits=int(target_qubits), target_error=float(target_error))


if __name__ == "__main__":
    target_qubits = 3  # [PARAM]
    Psi = None  # [PARAM]
    target_error = 1e-6  # [PARAM]
    test(Psi=Psi, target_qubits=target_qubits, target_error=target_error)
