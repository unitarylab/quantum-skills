# -*- coding: utf-8 -*-

"""
Matrix Product State (MPS) Preparation

Implementation of MPS-based quantum state preparation following the decomposition
described in Eq. (23) of arXiv:2310.18410.

Given a matrix product state (MPS) representation of a quantum state, this module
constructs a quantum circuit that prepares the corresponding state by encoding each
MPS tensor into a unitary gate via QR decomposition.

The MPS is a list of N tensors [A^{(0)}, ..., A^{(N-1)}] with shapes:
  - First tensor:  (2, chi_0)          -- (physical, bond)
  - Intermediate:  (chi_{i-1}, 2, chi_i) -- (bond_left, physical, bond_right)
  - Last tensor:   (chi_{N-2}, 2)        -- (bond, physical)

All physical dimensions are 2 (qubit), and bond dimensions must be powers of two.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Dict, Optional

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


class MPSAlgorithm(BaseAlgorithm):
    """Standalone Matrix Product State preparation algorithm module."""

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
        super().__init__("MPS State Preparation Algorithm", "MPS", text_mode, algo_dir)

    def run(
        self,
        Psi,
        target_qubits: int,
        target_error: float = 1e-6,
        mps: Optional[list[np.ndarray]] = None,
        work_wires: Optional[list[int]] = None,
        right_canonicalize: bool = False,
        mps_max_bond_dim: Optional[int] = None,
        rng_seed: int = 42,
        backend='torch',
        device='cpu',
        dtype=np.complex128,
    ) -> Dict[str, Any]:
        psi = np.asarray(Psi, dtype=np.complex128)
        self.update_input({
            "Method": "mps",
            "Target qubits": target_qubits,
            "Target error": target_error,
            "State vector length": int(psi.size),
            "Right canonicalize": bool(right_canonicalize),
            "MPS max bond dim": mps_max_bond_dim,
        })
        start_time = time.time()
        self.log("Stage 1: Building or validating MPS tensors")
        result = MPSAlgorithm.MPS(
            psi,
            int(target_qubits),
            float(target_error),
            mps=mps,
            work_wires=work_wires,
            right_canonicalize=right_canonicalize,
            mps_max_bond_dim=mps_max_bond_dim,
            rng_seed=int(rng_seed),
        )
        self.log("Stage 2: Building MPS preparation circuit")
        circuit = result.circuit
        self.log("Stage 3: Computing emitted unitary and preparation error")
        prepared_state = np.asarray(result.evolution_result, dtype=np.complex128)[:, 0]
        total_error = float(result.total_error)
        comp_time = time.time() - start_time
        is_success = total_error <= max(float(target_error), 1e-10)
        self.update_output({
            "Prepared state": prepared_state,
            "Total error": total_error,
            "Work leakage": float(result.work_leakage),
            "MPS tensors": len(result.mps),
            "Computation time (s)": round(comp_time, 4),
        })
        self.status = "success" if is_success else "failed"
        self.summary = f"MPS state preparation completed with error {total_error:.6e}."
        self.log("Stage 4: Exporting circuit diagram")
        circuit_path = self.save_circuit(circuit)
        filename = self.save_txt()
        return self._build_return_dict(is_success, circuit_path, filename, circuit)



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
        """Result container used by the MPS state-preparation implementation."""

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

            psi = MPSAlgorithm._normalize_state_vector(self.Psi, self.tol)
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
                f"MPSAlgorithm.StatePreparationResult(method={self.method}, "
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
                self._run()
            return self._total_error

        @property
        def evolution_result(self) -> np.ndarray:
            if self._evolution_result is None:
                self._run()
            return np.asarray(self._evolution_result, dtype=np.complex128)

    def _bit_reversed_state_vector(state_vector: np.ndarray, num_qubits: int) -> np.ndarray:
        """Reorder amplitudes so that bit-reversed indices match the project's convention.

        The project uses a wire ordering where the first system qubit maps to
        the most significant bit (big-endian), whereas our circuit (little-endian)
        maps wire 0 to the least significant bit.  This function reverses the bits
        of each state index to convert between the two conventions.

        Args:
            state_vector: Input state vector of length ``2**num_qubits``.
            num_qubits: Number of qubits.

        Returns:
            Bit-reversed copy of the state vector.
        """
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        if num_qubits <= 1:
            return state_vector.copy()

        reordered = np.empty_like(state_vector)
        for index, amplitude in enumerate(state_vector):
            reversed_index = int(format(index, f'0{num_qubits}b')[::-1], 2)
            reordered[reversed_index] = amplitude
        return reordered


    def _is_power_of_two(value: int) -> bool:
        """Return whether ``value`` is a positive power of two."""
        return value > 0 and (value & (value - 1)) == 0


    def _complete_columns_to_unitary(columns: np.ndarray, rng_seed: int = 42) -> np.ndarray:
        """Complete orthonormal columns to a deterministic dense unitary."""
        columns = np.asarray(columns, dtype=np.complex128)
        if columns.ndim != 2:
            raise ValueError("columns must be a two-dimensional matrix.")

        d, k = columns.shape
        if k > d:
            raise ValueError("columns must not have more columns than rows.")

        if k == d:
            if not np.allclose(columns.conj().T @ columns, np.eye(d), atol=1e-10):
                raise ValueError("columns must be unitary when k == d.")
            return columns

        rng = np.random.RandomState(rng_seed)
        filler = rng.random((d, d - k)) + 1j * rng.random((d, d - k))
        augmented = np.hstack([columns, filler])
        q_matrix, r_matrix = np.linalg.qr(augmented)

        diag = np.diag(r_matrix)
        phase = np.ones_like(diag, dtype=np.complex128)
        mask = np.abs(diag) > 1e-14
        phase[mask] = diag[mask] / np.abs(diag[mask])
        q_matrix *= phase[np.newaxis, :]

        return np.asarray(q_matrix, dtype=np.complex128)


    def _validate_max_bond_dim(max_bond_dim: Optional[int]) -> Optional[int]:
        """Validate the optional MPS truncation bond dimension."""
        if max_bond_dim is None:
            return None
        if isinstance(max_bond_dim, bool) or not isinstance(max_bond_dim, (int, np.integer)):
            raise TypeError("mps_max_bond_dim must be an integer.")

        value = int(max_bond_dim)
        if not MPSAlgorithm._is_power_of_two(value):
            raise ValueError("mps_max_bond_dim must be a positive power of two.")
        return value


    def state_vector_to_mps(
        state_vector: np.ndarray,
        num_qubits: Optional[int] = None,
        max_bond_dim: Optional[int] = None,
        rng_seed: int = 42,
        tol: float = 1e-12,
    ) -> list[np.ndarray]:
        """Convert a state vector into a right-canonical MPS with power-of-two bonds."""
        state_vector = np.asarray(state_vector, dtype=np.complex128)
        if state_vector.ndim != 1:
            raise ValueError("state_vector must be one-dimensional.")
        if state_vector.size == 0 or not MPSAlgorithm._is_power_of_two(int(state_vector.size)):
            raise ValueError("state_vector length must be a non-zero power of 2.")

        inferred_qubits = int(state_vector.size).bit_length() - 1
        if num_qubits is None:
            num_qubits = inferred_qubits
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, (int, np.integer)):
            raise TypeError("num_qubits must be an integer.")
        num_qubits = int(num_qubits)
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative.")
        if state_vector.size != (1 << num_qubits):
            raise ValueError(
                f"state_vector length {state_vector.size} does not match 2**num_qubits."
            )

        max_bond_dim = MPSAlgorithm._validate_max_bond_dim(max_bond_dim)

        norm = float(np.linalg.norm(state_vector))
        if norm <= tol:
            raise ValueError("state_vector must not be the zero vector.")
        psi = np.asarray(state_vector / norm, dtype=np.complex128)

        if num_qubits == 0:
            return []
        if num_qubits == 1:
            return [MPSAlgorithm._complete_columns_to_unitary(psi.reshape(2, 1), rng_seed)]

        tensors: list[np.ndarray | None] = [None] * num_qubits
        right_bond = 1
        block = psi.reshape(1 << (num_qubits - 1), 2)

        for site in range(num_qubits - 1, 0, -1):
            left_dim = 1 << site
            block = block.reshape(left_dim, 2 * right_bond)
            u_matrix, singular_values, vh_matrix = np.linalg.svd(block, full_matrices=False)

            chi = len(singular_values)
            if max_bond_dim is not None:
                chi = min(chi, max_bond_dim)

            u_matrix = u_matrix[:, :chi]
            singular_values = singular_values[:chi]
            vh_matrix = vh_matrix[:chi, :]

            if site == num_qubits - 1:
                tensors[site] = vh_matrix.reshape(chi, 2)
            else:
                tensors[site] = vh_matrix.reshape(chi, 2, right_bond)

            block = u_matrix @ np.diag(singular_values)
            right_bond = chi

        represented_norm = float(np.linalg.norm(block))
        if represented_norm <= tol:
            raise ValueError("mps_max_bond_dim truncation removed all state weight.")
        tensors[0] = np.asarray(block / represented_norm, dtype=np.complex128)

        return [np.asarray(tensor, dtype=np.complex128) for tensor in tensors if tensor is not None]


    # ──────────────────────────────────────────────────────────────────────
    #  Validation
    # ──────────────────────────────────────────────────────────────────────

    def validate_mps_shape(mps: list[np.ndarray]) -> None:
        """Validate that the MPS tensor dimensions conform to the expected format.

        For a single-qubit state (one tensor), the tensor must have shape (2, 2),
        serving as both the first and last tensor.

        Args:
            mps: List of rank-2 and rank-3 tensors.

        Raises:
            AssertionError: If any dimension constraint is violated.
        """
        if not isinstance(mps, (list, tuple)) or not mps:
            raise ValueError("mps must be a non-empty sequence of tensors.")

        n_sites = len(mps)
        shape = np.shape(mps[0])
        assert len(shape) == 2, (
            f"The first tensor must have exactly 2 dimensions, got {len(shape)}."
        )
        dj0, dj2 = shape
        assert dj0 == 2, (
            f"The first dimension of the first tensor must be exactly 2, got {dj0}."
        )
        assert dj2 > 0 and (dj2 & (dj2 - 1)) == 0, (
            f"The second dimension of the first tensor must be a power of 2, got {dj2}."
        )

        if n_sites == 1:
            # Single tensor is both first and last: physical dim = 2 for both indices.
            assert shape[1] == 2, (
                f"A single-tensor MPS must have shape (2, 2), got {shape}."
            )
            return

        for i, array in enumerate(mps[1:-1], start=1):
            shape = np.shape(array)
            assert len(shape) == 3, (
                f"Tensor {i} must have exactly 3 dimensions, got {len(shape)}."
            )
            ndj0, ndj1, ndj2 = shape
            assert ndj1 == 2, (
                f"The second (physical) dimension of tensor {i} must be 2, got {ndj1}."
            )
            assert ndj0 > 0 and (ndj0 & (ndj0 - 1)) == 0, (
                f"The first dimension of tensor {i} must be a power of 2, got {ndj0}."
            )
            assert ndj2 > 0 and (ndj2 & (ndj2 - 1)) == 0, (
                f"The third dimension of tensor {i} must be a power of 2, got {ndj2}."
            )
            assert ndj0 == dj2, (
                f"Dimension mismatch: tensor {i}'s first dim ({ndj0}) does not match "
                f"previous tensor's third dim ({dj2})."
            )
            dj2 = ndj2

        shape = np.shape(mps[-1])
        assert len(shape) == 2, (
            f"The last tensor must have exactly 2 dimensions, got {len(shape)}."
        )
        ndj0, ndj1 = shape
        assert ndj1 == 2, (
            f"The second dimension of the last tensor must be 2, got {ndj1}."
        )
        assert ndj0 > 0 and (ndj0 & (ndj0 - 1)) == 0, (
            f"The first dimension of the last tensor must be a power of 2, got {ndj0}."
        )
        assert ndj0 == dj2, (
            f"Dimension mismatch: last tensor's first dim ({ndj0}) does not match "
            f"previous tensor's third dim ({dj2})."
        )


    # ──────────────────────────────────────────────────────────────────────
    #  Right-canonicalization
    # ──────────────────────────────────────────────────────────────────────

    def right_canonicalize_mps(mps: list[np.ndarray]) -> list[np.ndarray]:
        """Convert an MPS to right-canonical form via sequential SVD from right to left.

        A right-canonical MPS satisfies the orthonormality condition [Eq. (21) of
        arXiv:2310.18410]:

            sum_{d_{j,1}, d_{j,2}} A^{(j)}_{d_{j,0}, d_{j,1}, d_{j,2}}
            (A^{(j)}_{d'_{j,0}, d_{j,1}, d_{j,2}})^* = delta_{d_{j,0}, d'_{j,0}}

        Args:
            mps: List of MPS tensors.

        Returns:
            List of MPS tensors in right-canonical form with the same dimensions.
        """
        MPSAlgorithm.validate_mps_shape(mps)

        n_sites = len(mps)
        if n_sites == 1:
            return [np.copy(mps[0])]

        mps_cpy = [np.copy(A) for A in mps]

        # Quick check: if the MPS is already right-canonical, skip the SVD sweep.
        already_canonical = True
        for tensor in mps_cpy[1:]:
            if tensor.ndim == 2:
                contraction = tensor @ tensor.conj().T
            else:
                contraction = np.tensordot(tensor, tensor.conj(), axes=([1, 2], [1, 2]))
            if not np.allclose(contraction, np.eye(tensor.shape[0]), atol=1e-12):
                already_canonical = False
                break
        if already_canonical:
            return mps_cpy
        mps_cpy[0] = mps_cpy[0].reshape((1, *mps_cpy[0].shape))       # (1, 2, chi_0)
        mps_cpy[-1] = mps_cpy[-1].reshape((*mps_cpy[-1].shape, 1))    # (chi_{N-2}, 2, 1)

        d_shapes = []
        for tensor in mps_cpy[1:-1]:
            d_shapes.extend(tensor.shape)
        max_bond_dim = max(d_shapes) if d_shapes else None

        n_sites = len(mps_cpy)
        output_mps: list[np.ndarray | None] = [None] * n_sites

        for i in range(n_sites - 1, 0, -1):
            chi_left, d, chi_right = mps_cpy[i].shape
            input_matrix = mps_cpy[i].reshape(chi_left, d * chi_right)

            u_matrix, s_diag, vd_matrix = np.linalg.svd(input_matrix, full_matrices=False)

            chi_new = len(s_diag) if max_bond_dim is None else min(int(max_bond_dim), len(s_diag))
            u_matrix = u_matrix[:, :chi_new]
            s_diag = s_diag[:chi_new]
            vd_matrix = vd_matrix[:chi_new, :]

            output_mps[i] = vd_matrix.reshape(chi_new, d, chi_right)
            mps_cpy[i - 1] = np.tensordot(
                mps_cpy[i - 1], u_matrix @ np.diag(s_diag), axes=([2], [0])
            )

        first: np.ndarray = mps_cpy[0][0]
        last: np.ndarray = output_mps[-1]  # type: ignore[assignment]
        last = last[:, :, 0]
        output_mps[0] = first
        output_mps[-1] = last

        return output_mps  # type: ignore[return-value]


    # ──────────────────────────────────────────────────────────────────────
    #  Decomposition -- Eq. (23) of arXiv:2310.18410
    # ──────────────────────────────────────────────────────────────────────

    def _qr_unitary(columns: np.ndarray, rng_seed: int = 42) -> np.ndarray:
        """Complete a set of column vectors to a unitary via QR decomposition.

        Given a (d x k) matrix with k <= d, this function appends (d - k) random
        columns, performs QR, and returns the orthogonal Q matrix. The phase of each
        diagonal of R is absorbed into Q to enforce uniqueness (handles complex
        MPS tensors correctly).

        Args:
            columns: Initial columns of shape (d, k).
            rng_seed: Seed for the random column filler.

        Returns:
            A unitary matrix of shape (d, d).
        """
        return MPSAlgorithm._complete_columns_to_unitary(columns, rng_seed)


    def mps_preparation_decomposition(
        mps: list[np.ndarray],
        work_wires: Optional[list[int]] = None,
        right_canonicalize: bool = False,
        rng_seed: int = 42,
    ) -> list[np.ndarray]:
        """Decompose an MPS into a sequence of unitary gates per Eq. (23) of arXiv:2310.18410.

        Each site tensor is encoded into a unitary matrix that acts on the site's system
        qubit plus a set of auxiliary ``work_wires``. The number of work wires must be
        sufficient to accommodate the largest bond dimension: if the max bond dimension
        is ``2^k``, then at least ``k`` work wires are required.

        Args:
            mps: List of MPS tensors.
            work_wires: Auxiliary qubit indices.
            right_canonicalize: Whether to convert the MPS to right-canonical form first.
            rng_seed: Seed for the QR filler (deterministic by default).

        Returns:
            List of unitary matrices, one per MPS site.  Each matrix has size
            ``2**(len(work_wires) + 1)``.

        Raises:
            ValueError: If ``work_wires`` is ``None`` or insufficient.
        """
        MPSAlgorithm.validate_mps_shape(mps)

        if work_wires is None:
            raise ValueError(
                "MPS preparation decomposition requires `work_wires` to be specified."
            )

        max_bond_dim = max(t.shape[-1] for t in mps[:-1]) if len(mps) > 1 else 0
        if max_bond_dim > (1 << len(work_wires)):
            raise ValueError(
                f"Insufficient number of `work_wires`. Need at least "
                f"{int(np.ceil(np.log2(max_bond_dim)))} work wires, "
                f"got {len(work_wires)}."
            )

        mps_cpy = [np.copy(A) for A in mps]
        n_work = len(work_wires)
        n_total = n_work + 1

        if right_canonicalize:
            mps_cpy = MPSAlgorithm.right_canonicalize_mps(mps_cpy)

        if len(mps_cpy) == 1:
            # Single-site MPS.
            Ai = mps_cpy[0].reshape((1, *mps_cpy[0].shape))  # (1, 2, 2)
            if n_work == 0:
                # No work wires: build a 2x2 unitary directly.
                # The first column is Ai[:, 0], we QR-complete to a 2x2 unitary.
                init_col = Ai[0, :, 0]  # [A₀[0,0], A₀[1,0]], length 2
                filler = np.random.RandomState(rng_seed).random((2, 1))
                augmented = np.hstack([init_col.reshape(-1, 1), filler])
                Q, R = np.linalg.qr(augmented)
                diag = np.diag(R)
                phase = np.ones_like(diag, dtype=np.complex128)
                mask = np.abs(diag) > 1e-14
                phase[mask] = diag[mask] / np.abs(diag[mask])
                Q *= phase[np.newaxis, :]
                return [Q]
            columns = []
            half = 1 << n_work
            for col_data in Ai:
                vec = np.zeros(1 << n_total, dtype=np.complex128)
                block_size = col_data.shape[1]
                vec[:block_size] = col_data[0]
                vec[half:half + block_size] = col_data[1]
                columns.append(vec)
            vec_matrix = np.column_stack(columns)
            return [MPSAlgorithm._qr_unitary(vec_matrix, rng_seed)]

        mps_cpy[0] = mps_cpy[0].reshape((1, *mps_cpy[0].shape))       # (1, 2, chi_0)
        mps_cpy[-1] = mps_cpy[-1].reshape((*mps_cpy[-1].shape, 1))    # (chi_{N-2}, 2, 1)

        unitaries: list[np.ndarray] = []
        half = 1 << n_work

        for Ai in mps_cpy:
            columns = []
            for col_data in Ai:  # col_data has shape (2, chi_i)
                vec = np.zeros(1 << n_total, dtype=np.complex128)
                block_size = col_data.shape[1]
                vec[:block_size] = col_data[0]
                vec[half:half + block_size] = col_data[1]
                columns.append(vec)

            vec_matrix = np.column_stack(columns)
            unitary = MPSAlgorithm._qr_unitary(vec_matrix, rng_seed)
            unitaries.append(unitary)

        return unitaries


    # ──────────────────────────────────────────────────────────────────────
    #  Circuit builder
    # ──────────────────────────────────────────────────────────────────────

    def build_mps_circuit(
        unitaries: list[np.ndarray],
        system_wires: list[int],
        work_wires: list[int],
        num_qubits: int,
    ) -> Circuit:
        """Construct the full MPS-preparation circuit from the unitary list.

        Each unitary is applied as a custom gate on ``work_wires + [system_wire[i]]``.
        The Circuit uses little-endian convention (wire 0 = LSB of the state index),
        and the gate target list is reversed internally, so we supply the work wire
        first to obtain the intended action on (system, work) internally.

        Args:
            unitaries: List of unitary matrices from ``mps_preparation_decomposition``.
            system_wires: Target system qubit indices (one per site).
            work_wires: Auxiliary qubit indices.
            num_qubits: Total number of qubits in the circuit (system + work).

        Returns:
            A ``Circuit`` instance implementing the MPS preparation.
        """
        qc = Circuit(num_qubits)
        for i, U in enumerate(unitaries):
            # little-endian: target list is reversed internally.
            # Supplying [work, system] so that internally it acts on [system, work].
            qc.unitary(U, work_wires + [system_wires[i]])

        return qc


    # ──────────────────────────────────────────────────────────────────────
    #  Evolution matrix (manual, avoiding Circuit.get_matrix() overhead)
    # ──────────────────────────────────────────────────────────────────────

    def _apply_gate(
        state: np.ndarray,
        unitary: np.ndarray,
        affected_wires: list[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Apply a unitary gate on a subset of wires to a state vector.

        The state index uses little-endian convention: wire ``k`` is bit ``k``.
        The unitary's row/column index uses big-endian encoding within the
        gate's subspace: ``affected_wires[0]`` is the most significant bit.

        Args:
            state: Input state vector of length ``2**num_qubits``.
            unitary: Unitary matrix of size ``2**len(affected_wires)``.
            affected_wires: Wires the gate acts on (order matters).
            num_qubits: Total qubit count.

        Returns:
            Transformed state vector.
        """
        result = np.copy(state)
        n_affected = len(affected_wires)
        u_dim = 1 << n_affected

        # Precompute the mapping from U-index (big-endian) to global bit set.
        # u_idx = sum w_k * 2^{n_affected-1-k}  where w_k = value of affected_wires[k]
        # global_bits[u_idx] = set of bits (OR of 1<<affected_wires[k] for each k with w_k=1)
        all_qubits = list(range(num_qubits))
        unaffected = [q for q in all_qubits if q not in affected_wires]
        global_bits = np.zeros(u_dim, dtype=np.intp)
        for u_idx in range(u_dim):
            bits = 0
            for k in range(n_affected):
                if (u_idx >> (n_affected - 1 - k)) & 1:
                    bits |= 1 << affected_wires[k]
            global_bits[u_idx] = bits

        for prefix in range(1 << len(unaffected)):
            offset = 0
            for bit_pos, q in enumerate(unaffected):
                if (prefix >> bit_pos) & 1:
                    offset |= 1 << q

            subspace = np.empty(u_dim, dtype=np.complex128)
            for u_idx in range(u_dim):
                subspace[u_idx] = state[offset | global_bits[u_idx]]

            subspace = unitary @ subspace

            for u_idx in range(u_dim):
                result[offset | global_bits[u_idx]] = subspace[u_idx]

        return result


    def _build_evolution_matrix(
        unitaries: list[np.ndarray],
        system_wires: list[int],
        work_wires: list[int],
        num_qubits: int,
    ) -> np.ndarray:
        """Accumulate the full unitary of the MPS preparation circuit.

        The overall unitary is U_{N-1} ... U_1 U_0 (right-to-left: earlier gates
        are applied first, so later gates multiply on the left).

        Args:
            unitaries: List of unitaries from ``mps_preparation_decomposition``.
            system_wires: System qubit indices.
            work_wires: Auxiliary qubit indices.
            num_qubits: Total qubit count.

        Returns:
            Full ``2**num_qubits x 2**num_qubits`` unitary.
        """
        dim = 1 << num_qubits
        evolution = np.eye(dim, dtype=np.complex128)

        for i, U in enumerate(unitaries):
            # Wire order must match the U-matrix encoding: system wire is MSB,
            # work wires are LSBs.  This is the convention used in
            # mps_preparation_decomposition.
            affected = [system_wires[i]] + list(work_wires)
            new_evolution = np.zeros_like(evolution)
            for col in range(dim):
                new_evolution[:, col] = MPSAlgorithm._apply_gate(
                    evolution[:, col], U, affected, num_qubits
                )
            evolution = new_evolution

        return evolution


    # ──────────────────────────────────────────────────────────────────────
    #  Error computation
    # ──────────────────────────────────────────────────────────────────────

    def _phase_invariant_error(reference: np.ndarray, candidate: np.ndarray) -> float:
        """Return a global-phase-invariant state-vector error."""
        reference = np.asarray(reference, dtype=np.complex128)
        candidate = np.asarray(candidate, dtype=np.complex128)
        overlap = np.vdot(reference, candidate)
        if abs(overlap) > 1e-12:
            candidate = candidate * np.conj(overlap / abs(overlap))
        return float(np.linalg.norm(reference - candidate))


    def _required_work_qubits(mps: list[np.ndarray]) -> int:
        """Return how many work qubits are needed for the largest MPS bond."""
        MPSAlgorithm.validate_mps_shape(mps)
        if len(mps) <= 1:
            return 0

        max_bond_dim = max(int(tensor.shape[-1]) for tensor in mps[:-1])
        return int(np.ceil(np.log2(max_bond_dim)))


    def _extract_zero_work_system_state(
        full_state: np.ndarray,
        system_wires: list[int],
        work_wires: list[int],
        target_qubits: int,
        total_qubits: int,
    ) -> np.ndarray:
        """Extract the system state from the subspace where all work qubits are zero."""
        full_state = np.asarray(full_state, dtype=np.complex128)
        if not work_wires:
            return full_state.copy()

        work_mask = sum(1 << wire for wire in work_wires)
        system_state = np.zeros(1 << target_qubits, dtype=np.complex128)
        for global_index in range(1 << total_qubits):
            if (global_index & work_mask) != 0:
                continue

            system_index = 0
            for system_position, wire in enumerate(system_wires):
                if (global_index >> wire) & 1:
                    system_index |= 1 << system_position
            system_state[system_index] = full_state[global_index]

        return system_state


    def _complete_state_preparation_matrix(
        prepared_state: np.ndarray,
        rng_seed: int = 42,
        tol: float = 1e-12,
    ) -> np.ndarray:
        """Return a target-space unitary whose first column is ``prepared_state``."""
        prepared_state = np.asarray(prepared_state, dtype=np.complex128)
        norm = float(np.linalg.norm(prepared_state))
        if norm <= tol:
            raise ValueError("prepared_state must not be the zero vector.")

        first_column = np.asarray(prepared_state / norm, dtype=np.complex128).reshape(-1, 1)
        return MPSAlgorithm._complete_columns_to_unitary(first_column, rng_seed)


    # ──────────────────────────────────────────────────────────────────────
    #  Main algorithm class
    # ──────────────────────────────────────────────────────────────────────

    class MPS(StatePreparationResult):
        """State preparation via Matrix Product State decomposition.

        Implements the MPS-based state-preparation algorithm described in
        arXiv:2310.18410.  The MPS tensors are converted into a quantum circuit
        using the QR-based technique from Eq. (23), yielding one unitary gate
        per site acting on the system qubit and a shared set of auxiliary
        ``work_wires``.

        The class can either decompose ``Psi`` into an exact MPS automatically, or
        use pre-computed tensors supplied via the ``mps`` argument. The MPS must be
        a list of :math:`n` tensors with shapes:
          - First:  ``(2, chi_0)``
          - Intermediate (if any): ``(chi_{i-1}, 2, chi_i)``
          - Last: ``(chi_{n-2}, 2)``
        where all bond dimensions are powers of two.

        Args:
            Psi: Target state vector (used for error computation).
            target_qubits: Number of qubits.
            target_error: Precision target (for bookkeeping).
            mps: Optional list of MPS tensors.
            work_wires: Auxiliary qubit indices for the gate decomposition.
                If the maximum bond dimension is :math:`2^k`, at least ``k``
                work wires are needed.  If ``None`` (default), the lowest
                indices are used.
            right_canonicalize: Whether to right-canonicalize the MPS first.
            mps_max_bond_dim: Optional power-of-two cap for the automatic
                state-vector-to-MPS decomposition. ``None`` keeps the exact MPS.
            rng_seed: Seed used for deterministic QR completion.
        """

        def __init__(
            self,
            Psi: np.ndarray,
            target_qubits: int,
            target_error: float,
            mps: Optional[list[np.ndarray]] = None,
            work_wires: Optional[list[int]] = None,
            right_canonicalize: bool = False,
            mps_max_bond_dim: Optional[int] = None,
            rng_seed: int = 42,
        ) -> None:
            super().__init__("mps", Psi, target_qubits, target_error)

            self._rng_seed = int(rng_seed)
            if mps is None and target_qubits > 0:
                mps = MPSAlgorithm.state_vector_to_mps(
                    self.Psi,
                    num_qubits=target_qubits,
                    max_bond_dim=mps_max_bond_dim,
                    rng_seed=self._rng_seed,
                    tol=self.tol,
                )
                right_canonicalize = False
            elif mps is None:
                mps = []
            else:
                mps = [np.asarray(tensor, dtype=np.complex128) for tensor in mps]

            if target_qubits > 0:
                MPSAlgorithm.validate_mps_shape(mps)
                if len(mps) != target_qubits:
                    raise ValueError(
                        f"mps must contain {target_qubits} tensors for target_qubits={target_qubits}."
                    )

            self._mps = mps
            self._right_canonicalize = right_canonicalize
            self._unitaries: Optional[list[np.ndarray]] = None
            self._full_evolution_result: Optional[np.ndarray] = None
            self._work_leakage = 0.0

            if work_wires is not None:
                self._work_wires = list(work_wires)
                if len(set(self._work_wires)) != len(self._work_wires):
                    raise ValueError("work_wires must be unique.")
                if any(wire < 0 for wire in self._work_wires):
                    raise ValueError("work_wires must be non-negative.")
                if len(self._work_wires) < MPSAlgorithm._required_work_qubits(self._mps):
                    raise ValueError("work_wires does not contain enough auxiliary qubits.")
                used = set(self._work_wires)
                sys_wires: list[int] = []
                next_wire = 0
                while len(sys_wires) < target_qubits:
                    if next_wire not in used:
                        sys_wires.append(next_wire)
                    next_wire += 1
                self._system_wires = sys_wires
                active_wires = self._work_wires + self._system_wires
                self._total_qubits = max(active_wires) + 1 if active_wires else 0
            elif target_qubits > 0:
                n_work = MPSAlgorithm._required_work_qubits(self._mps)
                self._work_wires = list(range(n_work))
                self._system_wires = list(range(n_work, n_work + target_qubits))
                self._total_qubits = n_work + target_qubits
            else:
                self._work_wires = []
                self._system_wires = []
                self._total_qubits = 0

            self._run()

        def _run(self) -> None:
            """Build the MPS preparation circuit and compute the emitted unitary."""
            if self.target_qubits == 0:
                self._circuit = Circuit(0)
                self._evolution_result = np.eye(1, dtype=np.complex128)
                self._full_evolution_result = np.eye(1, dtype=np.complex128)
                self._total_error = 0.0
                self._work_leakage = 0.0
                return

            # -- Decompose MPS -> unitaries ----------------------------------------
            self._unitaries = MPSAlgorithm.mps_preparation_decomposition(
                self._mps,
                work_wires=self._work_wires,
                right_canonicalize=self._right_canonicalize,
                rng_seed=self._rng_seed,
            )

            # -- Build the circuit -------------------------------------------------
            self._circuit = self._flatten_circuit(
                MPSAlgorithm.build_mps_circuit(
                    self._unitaries,
                    self._system_wires,
                    self._work_wires,
                    self._total_qubits,
                )
            )

            # -- Compute the evolution matrix --------------------------------------
            self._full_evolution_result = MPSAlgorithm._build_evolution_matrix(
                self._unitaries,
                self._system_wires,
                self._work_wires,
                self._total_qubits,
            )

            # -- Error: compare prepared system state to target --------------------
            prepared_all = np.asarray(self._full_evolution_result, dtype=np.complex128)[:, 0]

            n_sys = self.target_qubits
            n_work = len(self._work_wires)

            if n_work == 0:
                system_state = prepared_all
            else:
                system_state = MPSAlgorithm._extract_zero_work_system_state(
                    prepared_all,
                    self._system_wires,
                    self._work_wires,
                    n_sys,
                    self._total_qubits,
                )

            # Report work-wire leakage: probability that work qubits are not all |0⟩.
            work_zero_prob = float(np.vdot(system_state, system_state).real)
            self._work_leakage = 1.0 - work_zero_prob

            system_state = MPSAlgorithm._bit_reversed_state_vector(system_state, n_sys)
            target = self.Psi[: 1 << n_sys]
            self._total_error = MPSAlgorithm._phase_invariant_error(target, system_state)
            self._evolution_result = MPSAlgorithm._complete_state_preparation_matrix(
                system_state,
                rng_seed=self._rng_seed,
                tol=self.tol,
            )

        @property
        def unitaries(self) -> Optional[list[np.ndarray]]:
            """The per-site unitary matrices from the MPS decomposition."""
            return self._unitaries

        @property
        def mps(self) -> list[np.ndarray]:
            """The MPS tensors used."""
            return self._mps

        @property
        def work_leakage(self) -> float:
            """Probability that the work (auxiliary) qubits are not all in ``|0⟩``.

            A correctly prepared state should have all work qubits in ``|0⟩``
            after the circuit, so this value should be near zero.  Non-zero
            leakage indicates incorrect wire ordering, missing canonicalization,
            or an error in the MPS decomposition.
            """
            return self._work_leakage

        @property
        def full_evolution_result(self) -> np.ndarray:
            """Return the full emitted unitary on system plus work qubits."""
            if self._full_evolution_result is None:
                self._run()
            return np.asarray(self._full_evolution_result, dtype=np.complex128)


    # ──────────────────────────────────────────────────────────────────────
    #  Convenience functions
    # ──────────────────────────────────────────────────────────────────────

    def mps_preparation_circuit(
        mps: list[np.ndarray],
        system_wires: Optional[list[int]] = None,
        work_wires: Optional[list[int]] = None,
        right_canonicalize: bool = False,
    ) -> Circuit:
        """Construct a quantum circuit that prepares the state given by MPS tensors.

        Args:
            mps: List of MPS tensors.
            system_wires: System qubit indices.  Defaults to ``[0, ..., N-1]``.
            work_wires: Auxiliary qubit indices.  Required for gate-based decomposition.
            right_canonicalize: Whether to right-canonicalize the MPS first.

        Returns:
            A ``Circuit`` preparing the target state.
        """
        n_sites = len(mps)
        if system_wires is None:
            system_wires = list(range(n_sites))
        if work_wires is None:
            raise ValueError("`work_wires` must be provided for gate-based decomposition.")

        unitaries = MPSAlgorithm.mps_preparation_decomposition(mps, work_wires, right_canonicalize)
        active_wires = list(system_wires) + list(work_wires)
        total_qubits = max(active_wires) + 1 if active_wires else 0
        return MPSAlgorithm.build_mps_circuit(unitaries, system_wires, work_wires, total_qubits)


    def mps_preparation_matrix(
        mps: list[np.ndarray],
        system_wires: Optional[list[int]] = None,
        work_wires: Optional[list[int]] = None,
        right_canonicalize: bool = False,
    ) -> np.ndarray:
        """Return the exact unitary emitted by the MPS preparation circuit.

        Args:
            mps: List of MPS tensors.
            system_wires: System qubit indices.  Defaults to ``[0, ..., N-1]``.
            work_wires: Auxiliary qubit indices.
            right_canonicalize: Whether to right-canonicalize the MPS first.

        Returns:
            The full ``2**num_qubits x 2**num_qubits`` unitary.
        """
        n_sites = len(mps)
        if system_wires is None:
            system_wires = list(range(n_sites))
        if work_wires is None:
            raise ValueError("`work_wires` must be provided for gate-based decomposition.")

        unitaries = MPSAlgorithm.mps_preparation_decomposition(mps, work_wires, right_canonicalize)
        active_wires = list(system_wires) + list(work_wires)
        total_qubits = max(active_wires) + 1 if active_wires else 0
        return MPSAlgorithm._build_evolution_matrix(unitaries, system_wires, work_wires, total_qubits)


def test(Psi=None, target_qubits: int = 2, target_error: float = 1e-6):
    """Test MPS state preparation with a default Bell-like state."""
    if Psi is None:
        Psi = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    algo = MPSAlgorithm(text_mode="legacy")
    return algo.run(Psi=Psi, target_qubits=int(target_qubits), target_error=float(target_error))


if __name__ == "__main__":
    Psi = [1, 0, 0, 1]  # [PARAM]
    target_qubits = 2  # [PARAM]
    target_error = 1e-6  # [PARAM]
    test(Psi=Psi, target_qubits=target_qubits, target_error=target_error)
