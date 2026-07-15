"""Prepare quantum states with a Matrix Product State decomposition.

The target is normalized and padded, decomposed from right to left into a
right-canonical MPS, and optionally truncated at each bond.  Each tensor row is
embedded as an isometry column and QR-completed to a local unitary acting on an
ordered work register and one system qubit.

Local unitaries are applied from the first tensor to the last to build both a
UnitaryLab circuit and a dense system-plus-work evolution.  The all-zero-work
component of ``full_evolution[:, 0]`` is extracted without normalization and
bit-reversed once into user amplitude order.  Its norm determines work leakage
and its phase-aligned distance from the padded target determines total error.
The returned prepared state is the normalized direction of this projection.

For more than one site, tensor shapes are ``(2, chi_right)``,
``(chi_left, 2, chi_right)``, and ``(chi_left, 2)``.  A one-site MPS has shape
``(2, 2)``.  The implementation requires NumPy and ``unitarylab.core.Circuit``.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from unitarylab.core import Circuit


ComplexArray = NDArray[np.complex128]


def mps_state_preparation(
    Psi: ArrayLike,
    target_qubits: int,
    *,
    target_error: float = 1e-6,
    mps: list[ArrayLike] | None = None,
    work_wires: list[int] | None = None,
    right_canonicalize: bool = False,
    mps_max_bond_dim: int | None = None,
    rng_seed: int = 42,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> dict[str, Any]:
    """Prepare ``Psi`` through an MPS circuit and dense evolution.

    ``Psi`` is converted to ``complex128``, validated, normalized, and padded
    with trailing zeros.  If ``mps`` is absent, a right-to-left SVD creates a
    right-canonical chain with the optional power-of-two bond cap.  Supplied
    tensors are shape/bond validated but are not automatically normalized;
    they are right-canonicalized only when ``right_canonicalize=True``.

    ``work_wires`` is an ordered register.  If omitted, it is
    ``[0, ..., required_work_qubits-1]``.  System wires are the first increasing
    non-work indices.  Every local matrix acts on
    ``work_wires + [system_wires[site]]`` in left-to-right tensor order.

    ``backend``, ``device``, and ``dtype`` are compatibility parameters and do
    not affect the explicit NumPy construction, which uses ``complex128``.

    ``Zero-work projection`` is the unnormalized projection after its single
    conversion to user order.  ``Work leakage`` and ``Total error`` use this
    unnormalized vector.  ``Prepared state`` is its normalized direction,
    obtained as the first column of a completed target-space unitary.
    """
    del backend, device, dtype
    start_time = time.time()

    num_qubits = _validate_nonboolean_integer(
        target_qubits, "target_qubits", minimum=1
    )
    error_threshold = _validate_positive_finite_real(
        target_error, "target_error"
    )
    seed = _validate_nonboolean_integer(rng_seed, "rng_seed")
    if not isinstance(right_canonicalize, (bool, np.bool_)):
        raise TypeError("right_canonicalize must be a boolean.")
    max_bond = _validate_max_bond_dim(mps_max_bond_dim)

    target = _normalize_and_pad_target(Psi, num_qubits, tol=1e-12)

    if mps is None:
        tensors = _state_vector_to_mps(
            target,
            num_qubits,
            max_bond_dim=max_bond,
            rng_seed=seed,
        )
        _validate_mps_shape(tensors, num_qubits)
    else:
        if not isinstance(mps, list):
            raise TypeError("mps must be a list of tensors or None.")
        tensors = [np.asarray(tensor, dtype=np.complex128) for tensor in mps]
        _validate_mps_shape(tensors, num_qubits)
        supplied_state = _contract_mps_to_state(tensors, num_qubits)
        if float(np.linalg.norm(supplied_state)) <= 1e-12:
            raise ValueError("the supplied MPS contracts to a near-zero state.")
        if bool(right_canonicalize):
            before = supplied_state
            tensors = _right_canonicalize_mps(tensors)
            _validate_mps_shape(tensors, num_qubits)
            after = _contract_mps_to_state(tensors, num_qubits)
            if not np.allclose(before, after, atol=1e-10, rtol=1e-10):
                raise ValueError(
                    "right canonicalization changed the represented state."
                )

    # Contract once to reject an empty represented state; preparation still
    # proceeds through the local-unitary evolution below.
    diagnostic_contraction = _contract_mps_to_state(tensors, num_qubits)
    if float(np.linalg.norm(diagnostic_contraction)) <= 1e-12:
        raise ValueError("the used MPS contracts to a near-zero state.")

    required_work = _required_work_qubits(tensors)
    ordered_work_wires = _validate_or_create_work_wires(
        work_wires, required_work
    )
    system_wires = _system_wires(ordered_work_wires, num_qubits)
    if set(system_wires) & set(ordered_work_wires):
        raise ValueError("system_wires and work_wires must not overlap.")

    local_unitaries = _mps_local_unitaries(
        tensors, len(ordered_work_wires), seed
    )
    total_qubits = max(ordered_work_wires + system_wires) + 1
    circuit = _build_mps_circuit(
        local_unitaries,
        system_wires,
        ordered_work_wires,
        total_qubits,
    )
    full_evolution = _build_evolution_matrix(
        local_unitaries,
        system_wires,
        ordered_work_wires,
        total_qubits,
    )

    # Apply the full evolution to the all-zero input state.
    full_state = np.asarray(full_evolution[:, 0], dtype=np.complex128)
    internal_projection = _extract_zero_work_system_state(
        full_state,
        system_wires,
        ordered_work_wires,
        num_qubits,
        total_qubits,
    )
    success_probability = float(np.vdot(internal_projection, internal_projection).real)
    work_leakage = 1.0 - success_probability

    # Exactly one bit reversal occurs here.  Before conversion, projection bit
    # position i follows system_wires[i] (tensor/site 0 is the least-significant
    # extracted bit).  Afterwards, indices match the user's amplitude order.
    user_projection = _bit_reversed_state_vector(
        internal_projection, num_qubits
    )

    total_error = _phase_invariant_error(target, user_projection)
    target_evolution = _complete_state_preparation_matrix(
        user_projection, rng_seed=seed, tol=1e-12
    )
    prepared_state = np.asarray(target_evolution[:, 0], dtype=np.complex128)

    return {
        "status": "ok"
        if total_error <= max(error_threshold, 1e-10)
        else "failed",
        "Prepared state": prepared_state,
        "Zero-work projection": user_projection,
        "Total error": float(total_error),
        "Work leakage": float(work_leakage),
        "MPS tensors": len(tensors),
        "Local unitaries": local_unitaries,
        "system_wires": system_wires,
        "work_wires": ordered_work_wires,
        "full_evolution": full_evolution,
        "Computation time (s)": round(time.time() - start_time, 4),
        "circuit": circuit,
    }


def _validate_nonboolean_integer(
    value: Any, name: str, minimum: int | None = None
) -> int:
    """Validate an integer parameter while rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _validate_positive_finite_real(value: Any, name: str) -> float:
    """Validate a positive finite real parameter, rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _validate_max_bond_dim(value: int | None) -> int | None:
    """Validate the optional positive power-of-two automatic-SVD cap."""
    if value is None:
        return None
    result = _validate_nonboolean_integer(value, "mps_max_bond_dim", minimum=1)
    if not _is_power_of_two(result):
        raise ValueError("mps_max_bond_dim must be a positive power of two.")
    return result


def _normalize_and_pad_target(
    state: ArrayLike, num_qubits: int, tol: float
) -> ComplexArray:
    """Validate, normalize, and trailing-zero-pad the user target."""
    target = np.asarray(state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("Psi must be a one-dimensional state vector.")
    if target.size == 0:
        raise ValueError("Psi must not be empty.")
    if not np.all(np.isfinite(target)):
        raise ValueError("Psi entries must be finite.")
    norm = float(np.linalg.norm(target))
    if norm <= tol:
        raise ValueError("Psi must not be the zero vector.")
    target = np.ascontiguousarray(target / norm)

    dimension = 1 << num_qubits
    if target.size > dimension:
        raise ValueError(
            f"State vector length {target.size} exceeds target Hilbert "
            f"dimension {dimension}."
        )
    if target.size == dimension:
        return target

    padded = np.zeros(dimension, dtype=np.complex128)
    padded[: target.size] = target
    warnings.warn(
        f"State vector length {target.size} is smaller than "
        f"2**target_qubits; padded to {dimension}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return np.ascontiguousarray(padded)


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def _state_vector_to_mps(
    state: ArrayLike,
    num_qubits: int,
    max_bond_dim: int | None = None,
    rng_seed: int = 42,
) -> list[ComplexArray]:
    """Create a right-canonical MPS by a right-to-left SVD.

    At each cut, singular values are absorbed into the left block.  The rows
    retained from ``Vh`` become the current tensor in
    ``(bond_left, physical, bond_right)`` order.  The first block is normalized
    after any truncation.
    """
    max_bond_dim = _validate_max_bond_dim(max_bond_dim)
    seed = _validate_nonboolean_integer(rng_seed, "rng_seed")
    state = np.asarray(state, dtype=np.complex128)
    expected = 1 << num_qubits
    if state.ndim != 1 or state.size != expected:
        raise ValueError(f"state must have shape ({expected},).")
    if not np.all(np.isfinite(state)):
        raise ValueError("state entries must be finite.")

    if num_qubits == 1:
        return [
            _complete_columns_to_unitary(
                state.reshape(2, 1), rng_seed=seed
            )
        ]

    tensors: list[ComplexArray | None] = [None] * num_qubits
    right_bond = 1
    block = state.reshape(1 << (num_qubits - 1), 2)

    for site in range(num_qubits - 1, 0, -1):
        block = block.reshape(1 << site, 2 * right_bond)
        left, singular_values, right = np.linalg.svd(
            block, full_matrices=False
        )
        retained = len(singular_values)
        if max_bond_dim is not None:
            retained = min(retained, max_bond_dim)

        left = left[:, :retained]
        singular_values = singular_values[:retained]
        right = right[:retained, :]
        if site == num_qubits - 1:
            tensors[site] = right.reshape(retained, 2)
        else:
            tensors[site] = right.reshape(retained, 2, right_bond)

        block = left @ np.diag(singular_values)
        right_bond = retained

    represented_norm = float(np.linalg.norm(block))
    if represented_norm <= 1e-12:
        raise ValueError("bond truncation removed all state weight.")
    tensors[0] = block / represented_norm
    result = [np.asarray(tensor, dtype=np.complex128) for tensor in tensors]
    _validate_mps_shape(result, num_qubits)
    return result


def _validate_mps_shape(
    tensors: list[ComplexArray], target_qubits: int
) -> None:
    """Validate tensor count, values, ranks, physical legs, and bonds."""
    if len(tensors) != target_qubits:
        raise ValueError(
            f"MPS must contain exactly {target_qubits} tensors; "
            f"got {len(tensors)}."
        )
    if not tensors:
        raise ValueError("MPS tensor list must be non-empty.")
    for index, tensor in enumerate(tensors):
        if tensor.size == 0:
            raise ValueError(f"tensor {index} must be non-empty.")
        if not np.all(np.isfinite(tensor)):
            raise ValueError(f"tensor {index} contains non-finite entries.")

    if target_qubits == 1:
        if tensors[0].shape != (2, 2):
            raise ValueError(
                f"one-site MPS must have shape (2, 2); got "
                f"{tensors[0].shape}."
            )
        return

    first = tensors[0]
    if first.ndim != 2 or first.shape[0] != 2:
        raise ValueError(
            f"first tensor must have shape (2, chi_right); got {first.shape}."
        )
    _validate_explicit_bond(first.shape[1], "first right bond")
    previous_right = first.shape[1]

    for site in range(1, target_qubits - 1):
        tensor = tensors[site]
        if tensor.ndim != 3 or tensor.shape[1] != 2:
            raise ValueError(
                f"interior tensor {site} must have shape "
                f"(chi_left, 2, chi_right); got {tensor.shape}."
            )
        if tensor.shape[0] != previous_right:
            raise ValueError(
                f"bond mismatch before tensor {site}: expected "
                f"{previous_right}, got {tensor.shape[0]}."
            )
        _validate_explicit_bond(tensor.shape[0], f"tensor {site} left bond")
        _validate_explicit_bond(tensor.shape[2], f"tensor {site} right bond")
        previous_right = tensor.shape[2]

    last = tensors[-1]
    if last.ndim != 2 or last.shape[1] != 2:
        raise ValueError(
            f"last tensor must have shape (chi_left, 2); got {last.shape}."
        )
    if last.shape[0] != previous_right:
        raise ValueError(
            f"bond mismatch before last tensor: expected {previous_right}, "
            f"got {last.shape[0]}."
        )
    _validate_explicit_bond(last.shape[0], "last left bond")


def _validate_explicit_bond(bond: int, name: str) -> None:
    """Require an explicit bond to be a positive power of two."""
    if not _is_power_of_two(int(bond)):
        raise ValueError(f"{name}={bond} must be a positive power of two.")


def _contract_mps_to_state(
    tensors: list[ComplexArray], num_qubits: int
) -> ComplexArray:
    """Contract an MPS in user amplitude order for validation checks."""
    if len(tensors) != num_qubits:
        raise ValueError("tensor count does not match num_qubits.")
    if num_qubits == 1:
        return np.asarray(tensors[0][:, 0], dtype=np.complex128).copy()

    contracted = np.asarray(tensors[0], dtype=np.complex128)
    for tensor in tensors[1:-1]:
        contracted = np.tensordot(contracted, tensor, axes=([-1], [0]))
    contracted = np.tensordot(contracted, tensors[-1], axes=([-1], [0]))
    return np.asarray(contracted.reshape(1 << num_qubits), dtype=np.complex128)


def _right_canonicalize_mps(
    tensors: list[ComplexArray],
) -> list[ComplexArray]:
    """Right-canonicalize right-to-left without changing the MPS state.

    For site ``i``, ``M`` has shape
    ``(chi_left, physical * chi_right)``.  QR of ``M.T`` gives
    ``M = R.T @ Q.T``.  ``Q.T`` replaces the right tensor and has orthonormal
    rows; ``R.T`` is absorbed into the right-bond leg of the tensor to its
    left.  Thus ``Q.T`` satisfies the required row-isometry condition.

    A one-site tensor is copied unchanged and its first column is validated
    during local-unitary construction.
    """
    if len(tensors) == 1:
        return [np.array(tensors[0], dtype=np.complex128, copy=True)]

    result = [np.array(tensor, dtype=np.complex128, copy=True) for tensor in tensors]
    for site in range(len(result) - 1, 0, -1):
        right_tensor = result[site]
        left_bond = right_tensor.shape[0]
        matrix = right_tensor.reshape(left_bond, -1)
        if left_bond > matrix.shape[1]:
            raise ValueError(
                f"tensor {site} cannot be right-canonical: left bond "
                f"{left_bond} exceeds physical-right dimension "
                f"{matrix.shape[1]}."
            )

        q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
        canonical_matrix = q_matrix.T
        gauge = r_matrix.T
        result[site] = canonical_matrix.reshape(right_tensor.shape)

        left_tensor = result[site - 1]
        if site - 1 == 0:
            result[site - 1] = left_tensor @ gauge
        else:
            result[site - 1] = np.tensordot(
                left_tensor, gauge, axes=([2], [0])
            )

    return [np.asarray(tensor, dtype=np.complex128) for tensor in result]


def _required_work_qubits(tensors: list[ComplexArray]) -> int:
    """Return ``ceil(log2(max explicit bond))``, or zero for one tensor."""
    if len(tensors) == 1:
        return 0
    largest_bond = max(tensor.shape[-1] for tensor in tensors[:-1])
    return int(math.ceil(math.log2(largest_bond)))


def _validate_or_create_work_wires(
    work_wires: list[int] | None, required: int
) -> list[int]:
    """Validate the ordered work register or create the default register."""
    if work_wires is None:
        return list(range(required))
    if not isinstance(work_wires, list):
        raise TypeError("work_wires must be a list of integers or None.")
    validated = [
        _validate_nonboolean_integer(wire, f"work_wires[{index}]", minimum=0)
        for index, wire in enumerate(work_wires)
    ]
    if len(set(validated)) != len(validated):
        raise ValueError("work_wires must be unique.")
    if len(validated) < required:
        raise ValueError(
            f"work_wires provides {len(validated)} wires, but {required} "
            "are required by the largest MPS bond."
        )
    return validated


def _system_wires(work_wires: list[int], target_qubits: int) -> list[int]:
    """Return the first increasing indices not occupied by work wires."""
    occupied = set(work_wires)
    result: list[int] = []
    candidate = 0
    while len(result) < target_qubits:
        if candidate not in occupied:
            result.append(candidate)
        candidate += 1
    return result


def _mps_local_unitaries(
    tensors: list[ComplexArray], work_qubits: int, rng_seed: int
) -> list[ComplexArray]:
    """Map tensor rows to isometry columns and QR-complete each matrix."""
    if len(tensors) == 1:
        first_column = tensors[0][:, 0].reshape(2, 1)
        return [_complete_columns_to_unitary(first_column, rng_seed)]

    ranked = [np.array(tensor, copy=True) for tensor in tensors]
    ranked[0] = ranked[0].reshape(1, *ranked[0].shape)
    ranked[-1] = ranked[-1].reshape(*ranked[-1].shape, 1)

    local_dimension = 1 << (work_qubits + 1)
    physical_one_offset = 1 << work_qubits
    unitaries: list[ComplexArray] = []
    for site, tensor in enumerate(ranked):
        columns: list[ComplexArray] = []
        for left_bond_slice in tensor:
            if left_bond_slice.shape[0] != 2:
                raise ValueError(
                    f"tensor {site} physical dimension must equal 2."
                )
            right_bond = left_bond_slice.shape[1]
            if right_bond > 1 << work_qubits:
                raise ValueError(
                    f"tensor {site} right bond {right_bond} exceeds work "
                    f"capacity {1 << work_qubits}."
                )
            column = np.zeros(local_dimension, dtype=np.complex128)
            column[:right_bond] = left_bond_slice[0]
            column[
                physical_one_offset : physical_one_offset + right_bond
            ] = left_bond_slice[1]
            columns.append(column)

        isometry = np.column_stack(columns)
        expected = np.eye(isometry.shape[1], dtype=np.complex128)
        if not np.allclose(
            isometry.conj().T @ isometry, expected, atol=1e-10
        ):
            raise ValueError(
                f"tensor {site} does not define orthonormal isometry columns."
            )
        unitaries.append(_complete_columns_to_unitary(isometry, rng_seed))
    return unitaries


def _complete_columns_to_unitary(
    columns: ArrayLike, rng_seed: int
) -> ComplexArray:
    """Preserve verified isometry columns and complete them by seeded QR."""
    columns = np.asarray(columns, dtype=np.complex128)
    if columns.ndim != 2 or columns.size == 0:
        raise ValueError("isometry columns must be a non-empty matrix.")
    if not np.all(np.isfinite(columns)):
        raise ValueError("isometry columns must be finite.")
    rows, count = columns.shape
    if count > rows:
        raise ValueError("an isometry cannot have more columns than rows.")
    if not np.allclose(
        columns.conj().T @ columns,
        np.eye(count, dtype=np.complex128),
        atol=1e-10,
    ):
        raise ValueError("MPS tensor does not define orthonormal columns.")
    if count == rows:
        return np.asarray(columns, dtype=np.complex128)

    # Seeded filler makes the orthogonal complement deterministic.  Absorbing
    # QR diagonal phases preserves the supplied leading columns.
    rng = np.random.RandomState(rng_seed)
    filler = rng.random((rows, rows - count))
    filler = filler + 1j * rng.random((rows, rows - count))
    q_matrix, r_matrix = np.linalg.qr(np.hstack([columns, filler]))
    diagonal = np.diag(r_matrix)
    phase = np.ones(rows, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 1e-14
    phase[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    unitary = np.asarray(
        q_matrix * phase[np.newaxis, :], dtype=np.complex128
    )
    if not np.allclose(unitary[:, :count], columns, atol=1e-10):
        raise ValueError("QR completion failed to preserve isometry columns.")
    if not np.allclose(
        unitary.conj().T @ unitary,
        np.eye(rows, dtype=np.complex128),
        atol=1e-10,
    ):
        raise ValueError("QR completion did not produce a unitary matrix.")
    return unitary


def _build_mps_circuit(
    local_unitaries: list[ComplexArray],
    system_wires: list[int],
    work_wires: list[int],
    total_qubits: int,
) -> Circuit:
    """Build the real UnitaryLab circuit in left-to-right tensor order."""
    circuit = Circuit(total_qubits, name="MPS State Preparation")
    for site, unitary in enumerate(local_unitaries):
        # Work wires first gives the local matrix (system, work) big-endian
        # order after UnitaryLab reverses the target list internally.
        circuit.unitary(unitary, work_wires + [system_wires[site]])
    return circuit


def _build_evolution_matrix(
    local_unitaries: list[ComplexArray],
    system_wires: list[int],
    work_wires: list[int],
    total_qubits: int,
) -> ComplexArray:
    """Build full evolution from the same matrices, wires, and order as circuit."""
    dimension = 1 << total_qubits
    evolution = np.eye(dimension, dtype=np.complex128)
    for site, unitary in enumerate(local_unitaries):
        local_wires = work_wires + [system_wires[site]]
        expanded = _expand_unitary(unitary, local_wires, total_qubits)
        evolution = expanded @ evolution
    return evolution


def _expand_unitary(
    local_unitary: ArrayLike,
    local_wires: list[int],
    total_qubits: int,
) -> ComplexArray:
    """Expand a local matrix whose index bits follow ``local_wires`` order."""
    local_unitary = np.asarray(local_unitary, dtype=np.complex128)
    expected_local = 1 << len(local_wires)
    if local_unitary.shape != (expected_local, expected_local):
        raise ValueError(
            f"local unitary must have shape ({expected_local}, "
            f"{expected_local}); got {local_unitary.shape}."
        )
    if len(set(local_wires)) != len(local_wires):
        raise ValueError("local_wires must be unique.")
    for index, wire in enumerate(local_wires):
        _validate_nonboolean_integer(
            wire, f"local_wires[{index}]", minimum=0
        )
        if wire >= total_qubits:
            raise ValueError("local wire exceeds total circuit width.")

    dimension = 1 << total_qubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    other_wires = [
        wire for wire in range(total_qubits) if wire not in set(local_wires)
    ]
    for local_row in range(expected_local):
        global_row_part = _spread_bits(local_row, local_wires)
        for local_column in range(expected_local):
            amplitude = local_unitary[local_row, local_column]
            if amplitude == 0:
                continue
            global_column_part = _spread_bits(local_column, local_wires)
            for other_index in range(1 << len(other_wires)):
                other_part = _spread_bits(other_index, other_wires)
                result[
                    global_row_part | other_part,
                    global_column_part | other_part,
                ] = amplitude
    return result


def _spread_bits(value: int, wires: list[int]) -> int:
    """Place bit position ``i`` of ``value`` on ``wires[i]``."""
    result = 0
    for position, wire in enumerate(wires):
        if (value >> position) & 1:
            result |= 1 << wire
    return result


def _extract_zero_work_system_state(
    full_state: ArrayLike,
    system_wires: list[int],
    work_wires: list[int],
    target_qubits: int,
    total_qubits: int,
) -> ComplexArray:
    """Return the unnormalized internal-order all-zero-work projection.

    Sparse labels can leave physical wire positions that are neither system nor
    work wires.  The emitted circuit never acts on them, so from ``|0...0>``
    they remain zero.  Extraction therefore addresses the unique basis index
    with the requested system bits and every non-system position equal to zero.
    """
    full_state = np.asarray(full_state, dtype=np.complex128)
    if full_state.shape != (1 << total_qubits,):
        raise ValueError("full_state has the wrong dimension.")
    if set(system_wires) & set(work_wires):
        raise ValueError("system_wires and work_wires must not overlap.")
    result = np.zeros(1 << target_qubits, dtype=np.complex128)
    for system_index in range(1 << target_qubits):
        full_index = _spread_bits(system_index, system_wires)
        result[system_index] = full_state[full_index]
    return result


def _bit_reversed_state_vector(
    state: ArrayLike, num_qubits: int
) -> ComplexArray:
    """Convert internal system-bit order to user amplitude order exactly once."""
    state = np.asarray(state, dtype=np.complex128)
    dimension = 1 << num_qubits
    if state.shape != (dimension,):
        raise ValueError(f"state must have shape ({dimension},).")
    result = np.empty_like(state)
    for index, amplitude in enumerate(state):
        reversed_index = int(format(index, f"0{num_qubits}b")[::-1], 2)
        result[reversed_index] = amplitude
    return result


def _complete_state_preparation_matrix(
    state: ArrayLike, rng_seed: int = 42, tol: float = 1e-12
) -> ComplexArray:
    """Complete the normalized direction of a nonzero projection to a unitary."""
    state = np.asarray(state, dtype=np.complex128)
    norm_squared = float(np.vdot(state, state).real)
    if norm_squared <= tol:
        raise ValueError("zero-work projection has near-zero norm.")
    first_column = (state / np.sqrt(norm_squared)).reshape(-1, 1)
    return _complete_columns_to_unitary(first_column, rng_seed)


def _phase_invariant_error(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Phase-align the unnormalized candidate without changing its norm."""
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes must match.")
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 1e-12:
        candidate = candidate * np.conj(overlap / abs(overlap))
    return float(np.linalg.norm(reference - candidate))


def _assert_raises(expected: type[BaseException], function: Any) -> None:
    """Assert that ``function`` raises ``expected``."""
    try:
        function()
    except expected:
        return
    except Exception as error:
        raise AssertionError(
            f"expected {expected.__name__}, got "
            f"{type(error).__name__}: {error}"
        ) from error
    raise AssertionError(f"expected {expected.__name__}, but none was raised")


def _run_tests() -> None:
    """Run deterministic validation of the preparation pipeline."""
    test_count = 0
    max_error = 0.0

    def run_case(
        state: ArrayLike,
        num_qubits: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal test_count, max_error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = mps_state_preparation(state, num_qubits, **kwargs)

        for unitary in result["Local unitaries"]:
            identity = np.eye(unitary.shape[0], dtype=np.complex128)
            local_gram = np.einsum("ki,kj->ij", unitary.conj(), unitary)
            assert np.allclose(local_gram, identity, atol=2e-10)

        full_evolution = result["full_evolution"]
        full_identity = np.eye(full_evolution.shape[0], dtype=np.complex128)
        gram = np.einsum(
            "ki,kj->ij", full_evolution.conj(), full_evolution
        )
        assert np.allclose(gram, full_identity, atol=2e-10)
        full_state = np.einsum(
            "ij,j->i", full_evolution, full_identity[:, 0]
        )
        assert np.allclose(full_state, full_evolution[:, 0], atol=1e-12)

        total_qubits = full_evolution.shape[0].bit_length() - 1
        internal_projection = _extract_zero_work_system_state(
            full_state,
            result["system_wires"],
            result["work_wires"],
            num_qubits,
            total_qubits,
        )
        expected_user_projection = _bit_reversed_state_vector(
            internal_projection, num_qubits
        )
        assert np.allclose(
            result["Zero-work projection"], expected_user_projection, atol=1e-12
        )
        expected_leakage = 1.0 - float(
            np.vdot(internal_projection, internal_projection).real
        )
        assert abs(result["Work leakage"] - expected_leakage) <= 1e-12

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            target = _normalize_and_pad_target(state, num_qubits, tol=1e-12)
        expected_error = _phase_invariant_error(
            target, result["Zero-work projection"]
        )
        assert abs(result["Total error"] - expected_error) <= 1e-12
        threshold = float(kwargs.get("target_error", 1e-6))
        expected_status = (
            "ok" if expected_error <= max(threshold, 1e-10) else "failed"
        )
        assert result["status"] == expected_status

        projection = result["Zero-work projection"]
        projection_norm = float(np.linalg.norm(projection))
        assert projection_norm > 1e-12
        assert np.allclose(
            result["Prepared state"],
            projection / projection_norm,
            atol=2e-10,
        )
        assert set(result["system_wires"]).isdisjoint(result["work_wires"])
        assert isinstance(result["circuit"], Circuit)

        # Replay the left-to-right local-unitary schedule and compare states.
        sequential_state = np.zeros(full_evolution.shape[0], dtype=np.complex128)
        sequential_state[0] = 1.0
        for site, unitary in enumerate(result["Local unitaries"]):
            expanded = _expand_unitary(
                unitary,
                result["work_wires"] + [result["system_wires"][site]],
                total_qubits,
            )
            sequential_state = np.einsum(
                "ij,j->i", expanded, sequential_state
            )
        assert np.allclose(sequential_state, full_state, atol=1e-12)

        test_count += 1
        max_error = max(max_error, float(result["Total error"]))
        return result

    # One qubit, product, GHZ, W, random real, and random complex states.
    one_qubit = run_case(np.array([1.0, 1.0j]), 1)
    assert one_qubit["work_wires"] == []

    # Every computational basis state through four qubits verifies the single
    # post-projection bit reversal and the returned user amplitude order.
    for num_qubits in range(1, 5):
        for basis_index in range(1 << num_qubits):
            basis_state = np.zeros(1 << num_qubits, dtype=np.complex128)
            basis_state[basis_index] = 1.0
            basis_result = run_case(basis_state, num_qubits)
            assert int(np.argmax(np.abs(basis_result["Prepared state"]))) == basis_index

    run_case(np.ones(8), 3, mps_max_bond_dim=2)
    ghz = np.zeros(8, dtype=np.complex128)
    ghz[[0, 7]] = 1.0
    ghz_result = run_case(ghz, 3, mps_max_bond_dim=2)
    assert ghz_result["MPS tensors"] == 3
    assert ghz_result["work_wires"] == [0]
    assert ghz_result["system_wires"] == [1, 2, 3]
    ghz_mps = _state_vector_to_mps(ghz / np.linalg.norm(ghz), 3, 2)
    for tensor in ghz_mps[1:]:
        matrix = tensor.reshape(tensor.shape[0], -1)
        assert np.allclose(
            matrix @ matrix.conj().T,
            np.eye(matrix.shape[0]),
            atol=1e-10,
        )
    ghz_diagnostic = _contract_mps_to_state(ghz_mps, 3)
    assert _phase_invariant_error(
        ghz_diagnostic, ghz_result["Zero-work projection"]
    ) <= 1e-10
    w_state = np.zeros(8, dtype=np.complex128)
    w_state[[1, 2, 4]] = 1.0
    run_case(w_state, 3, mps_max_bond_dim=4)

    rng = np.random.default_rng(20260715)
    for num_qubits in range(1, 5):
        dimension = 1 << num_qubits
        run_case(rng.normal(size=dimension), num_qubits)
        run_case(
            rng.normal(size=dimension) + 1j * rng.normal(size=dimension),
            num_qubits,
        )

    # Supplied MPS with unit bonds for |000>.
    supplied = [
        np.array([[1.0], [0.0]], dtype=np.complex128),
        np.array([[[1.0], [0.0]]], dtype=np.complex128),
        np.array([[1.0, 0.0]], dtype=np.complex128),
    ]
    supplied_result = run_case(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        3,
        mps=supplied,
    )
    assert supplied_result["work_wires"] == []
    mismatched_supplied = run_case(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        3,
        mps=supplied,
        target_error=1e-6,
    )
    assert mismatched_supplied["status"] == "failed"
    assert int(np.argmax(np.abs(mismatched_supplied["Prepared state"]))) == 0

    # Right canonicalization preserves a deliberately gauged MPS state.
    canonical_source = _state_vector_to_mps(ghz / np.linalg.norm(ghz), 3, 2)
    gauge = np.array([[1.2, 0.3j], [0.1, 0.9]], dtype=np.complex128)
    gauged = [np.array(tensor, copy=True) for tensor in canonical_source]
    gauged[0] = gauged[0] @ gauge
    gauged[1] = np.tensordot(np.linalg.inv(gauge), gauged[1], axes=([1], [0]))
    before = _contract_mps_to_state(gauged, 3)
    recanonicalized = _right_canonicalize_mps(gauged)
    after = _contract_mps_to_state(recanonicalized, 3)
    assert np.allclose(before, after, atol=1e-10)
    for tensor in recanonicalized[1:]:
        matrix = tensor.reshape(tensor.shape[0], -1)
        assert np.allclose(
            matrix @ matrix.conj().T,
            np.eye(matrix.shape[0]),
            atol=1e-10,
        )
    run_case(ghz, 3, mps=gauged, right_canonicalize=True)
    _assert_raises(
        ValueError,
        lambda: mps_state_preparation(
            ghz, 3, mps=gauged, right_canonicalize=False
        ),
    )

    # Bond truncation is a target approximation, not work leakage.
    entangled = rng.normal(size=16) + 1j * rng.normal(size=16)
    truncated_mps = _state_vector_to_mps(
        entangled / np.linalg.norm(entangled), 4, max_bond_dim=2
    )
    assert max(tensor.shape[-1] for tensor in truncated_mps[:-1]) <= 2
    truncated = run_case(
        entangled, 4, mps_max_bond_dim=2, target_error=1e-12
    )
    assert np.isfinite(truncated["Total error"])

    # Sparse custom work placement controls system wires and circuit width.
    sparse = run_case(ghz, 3, mps_max_bond_dim=2, work_wires=[4])
    assert sparse["work_wires"] == [4]
    assert sparse["system_wires"] == [0, 1, 2]
    assert sparse["full_evolution"].shape == (32, 32)

    # Padding, zero amplitudes, and global phase use user amplitude order.
    padded = run_case(np.array([1.0, 0.0, 2.0j]), 3)
    assert np.sum(np.abs(padded["Prepared state"][3:]) ** 2) <= 1e-20
    phased_target = np.exp(0.713j) * np.array([1.0, 2.0j, 0.0, -1.0])
    run_case(phased_target, 2)

    # Verify leakage using a state with nonzero work-register probability.
    leaky_column = np.array(
        [np.sqrt(0.75), 0.5, 0.0, 0.0], dtype=np.complex128
    ).reshape(4, 1)
    leaky_local = _complete_columns_to_unitary(leaky_column, 42)
    leaky_evolution = _build_evolution_matrix(
        [leaky_local], [1], [0], total_qubits=2
    )
    synthetic_full = leaky_evolution[:, 0]
    synthetic_projection = _extract_zero_work_system_state(
        synthetic_full, [1], [0], 1, 2
    )
    assert abs(
        (1.0 - float(np.vdot(synthetic_projection, synthetic_projection).real))
        - 0.25
    ) <= 1e-12

    # Insufficient/illegal work wires and public integer/real validation.
    invalid_calls: list[tuple[type[BaseException], Any]] = [
        (ValueError, lambda: mps_state_preparation(ghz, 3, work_wires=[])),
        (ValueError, lambda: mps_state_preparation(ghz, 3, work_wires=[0, 0])),
        (ValueError, lambda: mps_state_preparation(ghz, 3, work_wires=[-1])),
        (TypeError, lambda: mps_state_preparation(ghz, 3, work_wires=[True])),
        (TypeError, lambda: mps_state_preparation(ghz, 3, work_wires=(0,))),
        (TypeError, lambda: mps_state_preparation([1, 0], True)),
        (TypeError, lambda: mps_state_preparation([1, 0], 1.0)),
        (ValueError, lambda: mps_state_preparation([1, 0], 0)),
        (ValueError, lambda: mps_state_preparation([1, 0], 1, target_error=np.inf)),
        (ValueError, lambda: mps_state_preparation([1, 0], 1, target_error=np.nan)),
        (TypeError, lambda: mps_state_preparation([1, 0], 1, target_error=True)),
        (TypeError, lambda: mps_state_preparation([1, 0], 1, mps_max_bond_dim=True)),
        (ValueError, lambda: mps_state_preparation([1, 0], 1, mps_max_bond_dim=3)),
        (TypeError, lambda: mps_state_preparation([1, 0], 1, rng_seed=True)),
        (ValueError, lambda: mps_state_preparation([], 1)),
        (ValueError, lambda: mps_state_preparation(np.eye(2), 1)),
        (ValueError, lambda: mps_state_preparation([np.nan, 1], 1)),
        (ValueError, lambda: mps_state_preparation([0, 0], 1)),
        (ValueError, lambda: mps_state_preparation(np.ones(5), 2)),
    ]
    for expected, function in invalid_calls:
        _assert_raises(expected, function)

    # Invalid supplied tensor count, shape, physical leg, bonds, and values.
    bad_shape = [np.ones((2, 1)), np.ones((1, 2, 1)), np.ones((1, 2, 1))]
    bad_physical = [np.ones((2, 1)), np.ones((1, 3, 1)), np.ones((1, 2))]
    bad_bond = [np.ones((2, 2)), np.ones((1, 2, 1)), np.ones((1, 2))]
    bad_power = [np.ones((2, 3)), np.ones((3, 2, 1)), np.ones((1, 2))]
    bad_finite = [tensor.copy() for tensor in supplied]
    bad_finite[1][0, 0, 0] = np.nan
    empty_tensor = [tensor.copy() for tensor in supplied]
    empty_tensor[1] = np.empty((1, 2, 0), dtype=np.complex128)
    zero_chain = [np.zeros_like(tensor) for tensor in supplied]
    for tensors in [
        supplied[:2],
        bad_shape,
        bad_physical,
        bad_bond,
        bad_power,
        bad_finite,
        empty_tensor,
        zero_chain,
    ]:
        _assert_raises(
            ValueError,
            lambda tensors=tensors: mps_state_preparation(
                np.r_[1.0, np.zeros(7)], 3, mps=tensors
            ),
        )

    # QR must reject a non-isometry instead of silently changing it.
    _assert_raises(
        ValueError,
        lambda: _complete_columns_to_unitary(
            np.array([[1.0], [1.0]], dtype=np.complex128), 42
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _complete_state_preparation_matrix(
            np.zeros(4, dtype=np.complex128), rng_seed=42
        ),
    )
    preserved_columns = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    completed = _complete_columns_to_unitary(preserved_columns, 42)
    assert np.allclose(completed[:, :2], preserved_columns, atol=1e-10)

    print(
        f"{test_count} preparation cases passed; "
        f"maximum Total error = {max_error:.3e}"
    )


if __name__ == "__main__":
    _run_tests()
