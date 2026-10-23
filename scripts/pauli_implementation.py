# -*- coding: utf-8 -*-

"""Pauli-word state preparation aligned with PennyLane ArbitraryStatePreparation."""

from __future__ import annotations

import functools
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize
from unitarylab.core import Circuit
from unitarylab.library.pauli_operator.pauli_string_decomposition import (
    pauli_state_preparation_circuit,
    pauli_string_to_matrix,
    state_preparation_pauli_words,
)
_MAX_OPTIMIZATION_RESTARTS = 4
_MAX_OPTIMIZATION_ITERATIONS = 800
_GRADIENT_SHIFT = np.pi / 2.0

try:
    from ...algo_base import BaseAlgorithm
except ImportError:
    # 单独运行时，将上级目录加入 sys.path，使 base 模块可被找到
    import sys
    _algorithms_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _algorithms_dir not in sys.path:
        sys.path.insert(0, _algorithms_dir)
    from unitarylab_algorithms.algo_base import BaseAlgorithm


class PauliAlgorithm(BaseAlgorithm):
    """Standalone Pauli-word state-preparation algorithm module."""

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
        super().__init__("Pauli State Preparation Algorithm", "PAU", text_mode, algo_dir)

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
            "Method": "pauli",
            "Target qubits": target_qubits,
            "Target error": target_error,
            "State vector length": int(psi.size),
        })
        start_time = time.time()
        self.log("Stage 1: Validating and normalizing target state")
        result = PauliAlgorithm.Pauli(psi, int(target_qubits), float(target_error))
        self.log("Stage 2: Fitting PauliRot weights and building circuit")
        circuit = result.circuit
        self.log("Stage 3: Computing emitted unitary and preparation error")
        prepared_state = np.asarray(result.evolution_result, dtype=np.complex128)[:, 0]
        total_error = float(result.total_error)
        comp_time = time.time() - start_time
        is_success = total_error <= max(float(target_error), 1e-10)
        self.update_output({
            "Prepared state": prepared_state,
            "Total error": total_error,
            "Pauli words": len(result.pauli_words),
            "Weights": result.weights,
            "Computation time (s)": round(comp_time, 4),
        })
        self.status = "success" if is_success else "failed"
        self.summary = f"Pauli state preparation completed with error {total_error:.6e}."
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
        """Result container used by the Pauli state-preparation implementation."""

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

            psi = PauliAlgorithm._normalize_state_vector(self.Psi, self.tol)
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
                f"PauliAlgorithm.StatePreparationResult(method={self.method}, "
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
                self._total_error = PauliAlgorithm._state_vector_error(self.Psi, prepared_state, self.tol)
            return self._total_error

        @property
        def evolution_result(self) -> np.ndarray:
            if self._evolution_result is None:
                self._run()
            return np.asarray(self._evolution_result, dtype=np.complex128)


    @functools.lru_cache(maxsize=None)
    def _cached_pauli_matrices(num_wires: int) -> tuple[np.ndarray, ...]:
        """Cache dense Pauli matrices for repeated objective evaluations."""
        return tuple(
            np.asarray(pauli_string_to_matrix(pauli_word), dtype=np.complex128)
            for pauli_word in state_preparation_pauli_words(num_wires)
        )


    def _pauli_rotation_matrix(theta: float, pauli_matrix: np.ndarray) -> np.ndarray:
        """Return the dense matrix of one PennyLane-style PauliRot block."""
        dim = pauli_matrix.shape[0]
        identity = np.eye(dim, dtype=np.complex128)
        angle = float(theta)
        return np.cos(angle / 2.0) * identity - 1j * np.sin(angle / 2.0) * pauli_matrix


    def build_dense_pauli_state_preparation_matrix(
        weights: np.ndarray,
        num_wires: int,
    ) -> np.ndarray:
        """
        Build the dense unitary induced by PennyLane-style Pauli word rotations.

        The emitted circuit is still useful for drawing and downstream export, but
        the dense matrix is the reliable numerical reference for validation in the
        current backend environment.
        """
        if num_wires < 0:
            raise ValueError('num_wires must be non-negative.')

        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError('weights must be a one-dimensional array.')

        pauli_words = state_preparation_pauli_words(num_wires)
        if weights.size != len(pauli_words):
            raise ValueError(
                f'weights must have length {len(pauli_words)} for num_wires={num_wires}.'
            )

        dim = 1 << num_wires
        unitary = np.eye(dim, dtype=np.complex128)

        # Apply the PauliRot blocks in the same left-to-right order as PennyLane.
        for theta, pauli_matrix in zip(weights, PauliAlgorithm._cached_pauli_matrices(num_wires)):
            if abs(float(theta)) <= 1e-15:
                continue
            unitary = PauliAlgorithm._pauli_rotation_matrix(float(theta), pauli_matrix) @ unitary

        return unitary


    def _prepare_state(weights: np.ndarray, num_wires: int) -> np.ndarray:
        """Apply the PauliRot sequence to |0...0> and return the prepared state."""
        return PauliAlgorithm.build_dense_pauli_state_preparation_matrix(weights, num_wires)[:, 0]


    def _state_fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
        """Return |<reference|candidate>|^2 for normalized states."""
        overlap = np.vdot(reference, candidate)
        return float(np.abs(overlap) ** 2)


    def _make_objective(target_state: np.ndarray, num_wires: int):
        """Build cached objective and gradient callables for the optimizer."""
        cache: dict[bytes, float] = {}

        def fidelity(weights: np.ndarray) -> float:
            # Optimization revisits nearby points frequently, so cache on raw bytes.
            weights = np.ascontiguousarray(weights, dtype=np.float64)
            key = weights.tobytes()
            cached = cache.get(key)
            if cached is not None:
                return cached

            value = PauliAlgorithm._state_fidelity(target_state, PauliAlgorithm._prepare_state(weights, num_wires))
            cache[key] = value
            return value

        def objective(weights: np.ndarray) -> float:
            return 1.0 - fidelity(weights)

        def gradient(weights: np.ndarray) -> np.ndarray:
            weights = np.asarray(weights, dtype=np.float64)
            grad = np.empty_like(weights)

            for index in range(weights.size):
                plus = weights.copy()
                minus = weights.copy()
                plus[index] += _GRADIENT_SHIFT
                minus[index] -= _GRADIENT_SHIFT
                grad[index] = -0.5 * (fidelity(plus) - fidelity(minus))

            return grad

        return objective, gradient


    def _initial_guesses(num_params: int) -> tuple[np.ndarray, ...]:
        """Return deterministic multi-start guesses for the optimization."""
        rng = np.random.default_rng(7)
        guesses = [np.zeros(num_params, dtype=np.float64)]

        for scale in (0.25, 0.5, 1.0):
            guesses.append(rng.uniform(-np.pi, np.pi, size=num_params) * scale)

        return tuple(guesses)


    def find_weights_from_state(
        target_state: np.ndarray,
        num_wires: int,
        target_error: float,
    ) -> tuple[np.ndarray, float]:
        """Fit PennyLane-style ArbitraryStatePreparation weights for a target state."""
        pauli_words = state_preparation_pauli_words(num_wires)
        expected_params = len(pauli_words)
        if expected_params == 0:
            return np.zeros(0, dtype=np.float64), 0.0

        target_state = np.asarray(target_state, dtype=np.complex128)
        objective, gradient = PauliAlgorithm._make_objective(target_state, num_wires)

        best_weights = np.zeros(expected_params, dtype=np.float64)
        best_error = float('inf')

        for initial in PauliAlgorithm._initial_guesses(expected_params)[:_MAX_OPTIMIZATION_RESTARTS]:
            result = minimize(
                objective,
                initial,
                method='L-BFGS-B',
                jac=gradient,
                options={
                    'maxiter': _MAX_OPTIMIZATION_ITERATIONS,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'maxls': 50,
                },
            )

            candidate_weights = np.asarray(result.x, dtype=np.float64)
            candidate_state = PauliAlgorithm._prepare_state(candidate_weights, num_wires)

            # Align global phase before reporting the physically meaningful 2-norm error.
            idx = int(np.argmax(np.abs(target_state)))
            phase_diff = np.angle(target_state[idx]) - np.angle(candidate_state[idx])
            candidate_state_corrected = candidate_state * np.exp(1j * phase_diff)
            candidate_error = float(np.linalg.norm(target_state - candidate_state_corrected))

            if candidate_error < best_error:
                best_weights = candidate_weights
                best_error = candidate_error

            if candidate_error <= target_error:
                break

        return best_weights, best_error


    class Pauli(StatePreparationResult):
        """
        Pauli-word state preparation based on PennyLane's ArbitraryStatePreparation.

        The class numerically fits the PauliRot angles, emits the corresponding
        circuit, flattens any nested Pauli-rotation blocks, and computes
        `evolution_result` directly from the emitted circuit matrix.
        """

        def __init__(self, Psi: np.ndarray, target_qubits: int, target_error: float) -> None:
            """Initialize the Pauli state-preparation solver."""
            self.weights = np.zeros(0, dtype=np.float64)
            self.pauli_words: tuple[str, ...] = tuple()
            super().__init__('pauli', Psi, target_qubits, target_error)
            self._run()

        def _run(self) -> None:
            """Fit PauliRot weights, build the circuit, and cache result fields."""
            self.pauli_words = state_preparation_pauli_words(self.target_qubits)
            self.weights, _ = PauliAlgorithm.find_weights_from_state(
                self.Psi,
                self.target_qubits,
                self.target_error,
            )

            self._circuit = self._flatten_circuit(
                pauli_state_preparation_circuit(self.weights, self.target_qubits)
            )
            self._evolution_result = np.asarray(self._circuit.get_matrix(), dtype=np.complex128)
            prepared_state = self._evolution_result[:, 0]
            self._total_error = PauliAlgorithm._state_vector_error(self.Psi, prepared_state, self.tol)


def test(Psi=None, target_qubits: int = 1, target_error: float = 1e-6):
    """Test Pauli-word state preparation with a default one-qubit state."""
    if Psi is None:
        Psi = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    algo = PauliAlgorithm(text_mode="legacy")
    return algo.run(Psi=Psi, target_qubits=int(target_qubits), target_error=float(target_error))


if __name__ == "__main__":
    Psi = [1, 1j]  # [PARAM]
    target_qubits = 1  # [PARAM]
    target_error = 1e-6  # [PARAM]
    test(Psi=Psi, target_qubits=target_qubits, target_error=target_error)
