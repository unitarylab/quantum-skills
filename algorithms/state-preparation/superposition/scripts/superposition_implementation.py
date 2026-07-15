"""Sparse superposition state preparation.

The target is normalized and trailing-zero padded, then amplitudes with
``abs(amplitude) > 1e-12`` are retained.  Their coefficients are prepared on
compact prefix basis states by a QR-completed unitary.  A full permutation
maps those prefixes to the retained MSB-first computational-basis tuples.
The target-space evolution is ``permutation_stage @ coefficient_stage``.

The returned prepared state is the first column of that dense target-space
evolution.  The returned circuit separately represents the coefficient gate
and prefix-to-support exchanges on the system register plus one clean work
wire; its Hilbert-space dimension is therefore larger than that of the
returned state.
"""

from __future__ import annotations

import math
import time
import warnings
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from unitarylab.core import Circuit


SUPPORT_THRESHOLD = 1e-12
PHASE_TOLERANCE = 1e-12

ComplexVector = NDArray[np.complex128]
BasisState = tuple[int, ...]


def superposition_state_preparation(
    Psi: ArrayLike,
    target_qubits: int,
    *,
    target_error: float = 1e-6,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> dict[str, Any]:
    """Prepare a normalized sparse-support target state.

    Parameters
    ----------
    Psi:
        Non-empty, finite, one-dimensional amplitudes.  The vector is
        normalized before trailing-zero padding.
    target_qubits:
        Number of system qubits; a non-boolean integer of at least one.
    target_error:
        Positive finite success threshold.  It affects only ``status`` and
        never changes the fixed support threshold.
    backend, device, dtype:
        Accepted compatibility parameters.  The dense calculation always
        uses NumPy ``complex128``.

    Returns
    -------
    dict
        ``status``, ``Prepared state``, ``Total error``, ``Support size``,
        ``Index register qubits``, ``Computation time (s)``, and ``circuit``.
        The circuit uses system wires ``0..target_qubits-1`` and the default
        clean work wire ``target_qubits``.
    """
    del backend, device, dtype
    started = time.perf_counter()

    num_qubits = _validate_target_qubits(target_qubits)
    error_threshold = _validate_target_error(target_error)
    target = _normalize_and_pad(Psi, num_qubits)

    coefficients, basis_states = _extract_sparse_support(target)
    state_map = _order_states(basis_states)
    ordered_coefficients = _ordered_coefficients_for_prefix_basis(
        coefficients,
        basis_states,
        state_map,
    )
    coefficient_stage, index_qubits = _build_coefficient_stage_matrix(
        ordered_coefficients,
        num_qubits,
    )
    permutation_stage = _build_prefix_to_support_permutation(
        basis_states,
        state_map,
    )
    # Some BLAS/LAPACK combinations leave floating-point status flags set
    # after QR even though the following finite product is valid.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        evolution = np.asarray(
            permutation_stage @ coefficient_stage,
            dtype=np.complex128,
        )
    prepared_state = np.asarray(evolution[:, 0], dtype=np.complex128)
    total_error = _state_vector_error(target, prepared_state)
    circuit = _build_superposition_circuit(
        coefficient_stage,
        basis_states,
        state_map,
        num_qubits,
        index_qubits,
        work_wire=num_qubits,
    )

    return {
        "status": (
            "ok"
            if total_error <= max(error_threshold, 1e-10)
            else "failed"
        ),
        "Prepared state": prepared_state,
        "Total error": float(total_error),
        "Support size": int(coefficients.size),
        "Index register qubits": int(index_qubits),
        "Computation time (s)": round(time.perf_counter() - started, 4),
        "circuit": circuit,
    }


def _validate_target_qubits(target_qubits: int) -> int:
    if isinstance(target_qubits, (bool, np.bool_)) or not isinstance(
        target_qubits, (int, np.integer)
    ):
        raise TypeError("target_qubits must be a non-boolean integer")
    value = int(target_qubits)
    if value < 1:
        raise ValueError("target_qubits must be at least 1")
    return value


def _validate_target_error(target_error: float) -> float:
    if isinstance(target_error, (bool, np.bool_)) or not isinstance(
        target_error, Real
    ):
        raise TypeError("target_error must be a non-boolean real number")
    value = float(target_error)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("target_error must be a positive finite number")
    return value


def _normalize_and_pad(Psi: ArrayLike, target_qubits: int) -> ComplexVector:
    psi = np.asarray(Psi, dtype=np.complex128)
    if psi.ndim != 1 or psi.size == 0:
        raise ValueError("Psi must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(psi)):
        raise ValueError("Psi entries must be finite")

    norm = float(np.linalg.norm(psi))
    if norm <= SUPPORT_THRESHOLD:
        raise ValueError("Psi norm must exceed 1e-12")
    psi = psi / norm

    dimension = 1 << target_qubits
    if psi.size > dimension:
        raise ValueError("Psi exceeds the target Hilbert-space dimension")
    if psi.size < dimension:
        warnings.warn(
            f"Psi length {psi.size} is smaller than 2**target_qubits="
            f"{dimension}; trailing zeros were appended.",
            RuntimeWarning,
            stacklevel=2,
        )
    target = np.zeros(dimension, dtype=np.complex128)
    target[: psi.size] = psi
    return target


def _index_to_bits(index: int, num_qubits: int) -> BasisState:
    """Convert an integer index to an MSB-first computational-basis tuple."""
    if isinstance(index, (bool, np.bool_)) or not isinstance(
        index, (int, np.integer)
    ):
        raise TypeError("index must be a non-boolean integer")
    if isinstance(num_qubits, (bool, np.bool_)) or not isinstance(
        num_qubits, (int, np.integer)
    ):
        raise TypeError("num_qubits must be a non-boolean integer")
    width = int(num_qubits)
    value = int(index)
    if width < 1:
        raise ValueError("num_qubits must be at least 1")
    if value < 0 or value >= 1 << width:
        raise ValueError("index is outside the computational-basis range")
    return tuple(int(bit) for bit in format(value, f"0{width}b"))


def _bits_to_index(bits: BasisState) -> int:
    """Convert a non-empty MSB-first binary tuple to its integer index."""
    state = tuple(bits)
    if not state:
        raise ValueError("basis state must be non-empty")
    if any(
        isinstance(bit, (bool, np.bool_))
        or not isinstance(bit, (int, np.integer))
        or int(bit) not in (0, 1)
        for bit in state
    ):
        raise ValueError("basis-state entries must be integer zeros or ones")
    return int("".join(str(int(bit)) for bit in state), 2)


def _extract_sparse_support(
    state: ComplexVector,
    threshold: float = SUPPORT_THRESHOLD,
) -> tuple[ComplexVector, list[BasisState]]:
    """Retain amplitudes satisfying the strict post-normalization threshold."""
    vector = np.asarray(state, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("state must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("state entries must be finite")
    if vector.size & (vector.size - 1):
        raise ValueError("state length must be a power of two")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a non-boolean real number")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or threshold_value < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    num_qubits = vector.size.bit_length() - 1
    indices = [
        index
        for index, amplitude in enumerate(vector)
        if abs(amplitude) > threshold_value
    ]
    if not indices:
        raise ValueError("state must contain at least one retained amplitude")
    coefficients = np.asarray(
        [vector[index] for index in indices],
        dtype=np.complex128,
    )
    basis_states = [_index_to_bits(index, num_qubits) for index in indices]
    return coefficients, basis_states


def _validate_basis_states(basis_states: list[BasisState]) -> list[BasisState]:
    try:
        states = [tuple(state) for state in basis_states]
    except TypeError as error:
        raise TypeError("each basis state must be an iterable of bits") from error
    if not states:
        raise ValueError("basis_states must be non-empty")
    width = len(states[0])
    if width < 1 or any(len(state) != width for state in states):
        raise ValueError("basis states must have one common positive width")
    for state in states:
        _bits_to_index(state)
    if len(set(states)) != len(states):
        raise ValueError("basis states must be unique")
    return states


def _order_states(basis_states: list[BasisState]) -> dict[BasisState, BasisState]:
    """Map each retained support tuple to one prefix basis tuple."""
    states = _validate_basis_states(basis_states)
    support_size = len(states)
    num_qubits = len(states[0])
    if support_size > 1 << num_qubits:
        raise ValueError("support exceeds the basis-state dimension")

    state_map: dict[BasisState, BasisState] = {}
    outside_prefix: list[BasisState] = []
    unused_prefixes = {index: None for index in range(support_size)}

    for state in states:
        state_index = _bits_to_index(state)
        if state_index < support_size:
            state_map[state] = state
            unused_prefixes.pop(state_index, None)
        else:
            outside_prefix.append(state)

    for state, prefix_index in zip(outside_prefix, unused_prefixes):
        state_map[state] = _index_to_bits(prefix_index, num_qubits)
    if len(state_map) != support_size:
        raise ValueError("could not assign every support state to a prefix")
    return state_map


def _ordered_coefficients_for_prefix_basis(
    coefficients: ComplexVector,
    basis_states: list[BasisState],
    state_map: dict[BasisState, BasisState],
) -> ComplexVector:
    """Place each retained coefficient at its assigned prefix index."""
    states = _validate_basis_states(basis_states)
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 1 or coeffs.size != len(states):
        raise ValueError("coefficients and basis_states must have equal lengths")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must be finite")
    if set(state_map) != set(states):
        raise ValueError("state_map must map every basis state exactly once")

    ordered = np.zeros_like(coeffs)
    used_prefixes: set[int] = set()
    for coefficient, state in zip(coeffs, states):
        prefix = tuple(state_map[state])
        if len(prefix) != len(state):
            raise ValueError("mapped prefixes must match the basis-state width")
        prefix_index = _bits_to_index(prefix)
        if prefix_index >= coeffs.size or prefix_index in used_prefixes:
            raise ValueError("state_map must assign distinct prefix basis states")
        ordered[prefix_index] = coefficient
        used_prefixes.add(prefix_index)
    return ordered


def _coefficient_register_qubits(support_size: int) -> int:
    if isinstance(support_size, (bool, np.bool_)) or not isinstance(
        support_size, (int, np.integer)
    ):
        raise TypeError("support_size must be a non-boolean integer")
    size = int(support_size)
    if size < 1:
        raise ValueError("support_size must be at least 1")
    return 0 if size == 1 else math.ceil(math.log2(size))


def _build_coefficient_stage_matrix(
    ordered_coefficients: ComplexVector,
    num_qubits: int,
) -> tuple[NDArray[np.complex128], int]:
    """QR-complete compact coefficients and embed them on low-order wires."""
    width = _validate_target_qubits(num_qubits)
    coefficients = np.asarray(ordered_coefficients, dtype=np.complex128)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("ordered_coefficients must be a non-empty vector")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("ordered_coefficients must be finite")
    if coefficients.size > 1 << width:
        raise ValueError("coefficient support exceeds the target dimension")

    register_qubits = _coefficient_register_qubits(coefficients.size)
    full_dimension = 1 << width
    if register_qubits == 0:
        return np.eye(full_dimension, dtype=np.complex128), 0

    local_dimension = 1 << register_qubits
    coefficient_state = np.zeros(local_dimension, dtype=np.complex128)
    coefficient_state[: coefficients.size] = coefficients
    local_unitary = _qr_complete_first_column(coefficient_state)

    remaining_qubits = width - register_qubits
    embedded = np.kron(
        np.eye(1 << remaining_qubits, dtype=np.complex128),
        local_unitary,
    )
    return np.asarray(embedded, dtype=np.complex128), register_qubits


def _qr_complete_first_column(column: ComplexVector) -> NDArray[np.complex128]:
    """QR-complete a nonzero vector and restore its first-column phase."""
    vector = np.asarray(column, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("column must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("column must be finite")
    if float(np.linalg.norm(vector)) <= SUPPORT_THRESHOLD:
        raise ValueError("column norm must exceed 1e-12")

    qr_input = np.eye(vector.size, dtype=np.complex128)
    qr_input[:, 0] = vector
    unitary, _ = np.linalg.qr(qr_input)
    overlap = np.vdot(vector, unitary[:, 0])
    if abs(overlap) > PHASE_TOLERANCE:
        unitary[:, 0] *= np.conj(overlap / abs(overlap))
    return np.asarray(unitary, dtype=np.complex128)


def _build_prefix_to_support_permutation(
    basis_states: list[BasisState],
    state_map: dict[BasisState, BasisState],
) -> NDArray[np.complex128]:
    """Build ``P[output,input]=1`` so prefixes map to support states."""
    states = _validate_basis_states(basis_states)
    num_qubits = len(states[0])
    dimension = 1 << num_qubits
    support_size = len(states)
    if set(state_map) != set(states):
        raise ValueError("state_map must map every basis state exactly once")

    inverse_map: dict[BasisState, BasisState] = {}
    for support_state, prefix_state_raw in state_map.items():
        prefix_state = tuple(prefix_state_raw)
        if len(prefix_state) != num_qubits:
            raise ValueError("mapped prefixes must match the basis-state width")
        prefix_index = _bits_to_index(prefix_state)
        if prefix_index >= support_size:
            raise ValueError("mapped state is outside the prefix range")
        if prefix_state in inverse_map:
            raise ValueError("mapped prefix states must be unique")
        inverse_map[prefix_state] = support_state

    permutation = list(range(dimension))
    for prefix_index in range(support_size):
        prefix_state = _index_to_bits(prefix_index, num_qubits)
        if prefix_state not in inverse_map:
            raise ValueError("state_map does not cover every prefix state")
        permutation[prefix_index] = _bits_to_index(inverse_map[prefix_state])

    for support_state in states:
        support_index = _bits_to_index(support_state)
        if support_index >= support_size:
            permutation[support_index] = _bits_to_index(state_map[support_state])

    if sorted(permutation) != list(range(dimension)):
        raise ValueError("prefix-to-support mapping is not a permutation")
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for input_index, output_index in enumerate(permutation):
        matrix[output_index, input_index] = 1.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram = matrix.conj().T @ matrix
    if not np.allclose(gram, np.eye(dimension, dtype=np.complex128)):
        raise ValueError("prefix-to-support matrix is not unitary")
    return matrix


def _append_support_exchange(
    circuit: Circuit,
    prefix_state: BasisState,
    support_state: BasisState,
    system_wires: list[int],
    work_wire: int,
) -> None:
    """Exchange one prefix/support pair through a clean work wire."""
    # Support tuples are MSB-first, while ascending system wires address the
    # tuple from its least-significant end. Reverse only at the circuit-wire
    # boundary; dense target-space indices remain MSB-first.
    prefix_wire_bits = tuple(reversed(prefix_state))
    support_wire_bits = tuple(reversed(support_state))
    circuit.mcx(
        system_wires,
        work_wire,
        control_state=list(prefix_wire_bits),
    )
    for wire, prefix_bit, support_bit in zip(
        system_wires,
        prefix_wire_bits,
        support_wire_bits,
    ):
        if prefix_bit != support_bit:
            circuit.cx(work_wire, wire)
    circuit.mcx(
        system_wires,
        work_wire,
        control_state=list(support_wire_bits),
    )


def _build_superposition_circuit(
    coefficient_stage: NDArray[np.complex128],
    basis_states: list[BasisState],
    state_map: dict[BasisState, BasisState],
    num_qubits: int,
    index_qubits: int,
    *,
    work_wire: int,
) -> Circuit:
    """Build the coefficient gate and support exchanges on one work wire."""
    width = _validate_target_qubits(num_qubits)
    if isinstance(index_qubits, (bool, np.bool_)) or not isinstance(
        index_qubits, (int, np.integer)
    ):
        raise TypeError("index_qubits must be a non-boolean integer")
    compact_width = int(index_qubits)
    if compact_width < 0 or compact_width > width:
        raise ValueError("index_qubits must lie between zero and num_qubits")
    if isinstance(work_wire, (bool, np.bool_)) or not isinstance(
        work_wire, (int, np.integer)
    ):
        raise TypeError("work_wire must be a non-boolean integer")
    work = int(work_wire)
    if work < 0 or work < width:
        raise ValueError("work_wire must be non-negative and not overlap system wires")

    matrix = np.asarray(coefficient_stage, dtype=np.complex128)
    dimension = 1 << width
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("coefficient_stage has an invalid shape or entries")
    states = _validate_basis_states(basis_states)
    if len(states[0]) != width:
        raise ValueError("basis-state width must equal num_qubits")

    system_wires = list(range(width))
    circuit = Circuit(
        max(width, work + 1),
        name="Sparse Superposition State Preparation",
    )
    if compact_width > 0:
        circuit.unitary(matrix, system_wires)
    for support_state, prefix_state in state_map.items():
        if support_state != prefix_state:
            _append_support_exchange(
                circuit,
                tuple(prefix_state),
                tuple(support_state),
                system_wires,
                work,
            )
    return circuit


def _state_vector_error(
    reference: ComplexVector,
    candidate: ComplexVector,
    tolerance: float = PHASE_TOLERANCE,
) -> float:
    """Return overlap-aligned, global-phase-invariant L2 error."""
    target = np.asarray(reference, dtype=np.complex128)
    prepared = np.asarray(candidate, dtype=np.complex128)
    if target.shape != prepared.shape or target.ndim != 1:
        raise ValueError("reference and candidate must be equal-length vectors")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prepared)):
        raise ValueError("reference and candidate must be finite")
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a non-boolean real number")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    overlap = np.vdot(target, prepared)
    if abs(overlap) > tolerance_value:
        prepared = prepared * np.conj(overlap / abs(overlap))
    return float(np.linalg.norm(target - prepared))


def _run_tests() -> None:
    """Execute returned circuits and validate them against independent targets."""
    passed = 0
    total = 0
    maximum_phase_error = 0.0
    minimum_fidelity = 1.0
    maximum_work_leakage = 0.0

    # SKILL.md gives 1e-12 as its overlap/numerical tolerance but does not
    # define a separate work-wire leakage tolerance. This test uses the
    # documented numerical tolerance without changing the public algorithm.
    work_leakage_tolerance = PHASE_TOLERANCE

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
        else:
            print(f"FAIL: {name}: {detail}")

    def expect(name: str, exception: type[Exception], function: Any) -> None:
        try:
            function()
        except exception:
            check(name, True)
        except Exception as error:
            check(name, False, f"raised {type(error).__name__}: {error}")
        else:
            check(name, False, "did not raise")

    def independent_expected(Psi: ArrayLike, num_qubits: int):
        """Build the test oracle without implementation-side helper calls."""
        raw = np.asarray(Psi, dtype=np.complex128)
        normalized = raw / np.linalg.norm(raw)
        target = np.zeros(1 << num_qubits, dtype=np.complex128)
        target[: normalized.size] = normalized
        retained_mask = np.abs(target) > SUPPORT_THRESHOLD
        expected = np.zeros_like(target)
        expected[retained_mask] = target[retained_mask]
        expected /= np.linalg.norm(expected)
        support_size = int(np.count_nonzero(retained_mask))
        index_qubits = (
            0 if support_size == 1 else math.ceil(math.log2(support_size))
        )
        return target, expected, support_size, index_qubits

    def independent_phase_metrics(reference, candidate):
        """Compute phase error and fidelity without _state_vector_error."""
        overlap = np.vdot(reference, candidate)
        aligned = np.asarray(candidate, dtype=np.complex128).copy()
        if abs(overlap) > PHASE_TOLERANCE:
            aligned *= np.conj(overlap / abs(overlap))
        return (
            float(np.linalg.norm(reference - aligned)),
            float(abs(overlap) ** 2),
        )

    def validate_circuit_case(
        name: str,
        Psi: ArrayLike,
        num_qubits: int,
        *,
        target_error: float = 1e-6,
        expect_padding_warning: bool = False,
        expected_support_size: int | None = None,
    ) -> None:
        nonlocal maximum_phase_error, minimum_fidelity, maximum_work_leakage
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = superposition_state_preparation(
                    Psi,
                    num_qubits,
                    target_error=target_error,
                )

            target, expected, support_size, index_qubits = independent_expected(
                Psi,
                num_qubits,
            )
            full_state = np.asarray(
                result["circuit"].execute(
                    backend="torch",
                    device="cpu",
                    dtype=np.complex128,
                ).state,
                dtype=np.complex128,
            ).reshape(-1)

            target_dimension = 1 << num_qubits
            if full_state.shape != (2 * target_dimension,):
                raise AssertionError(
                    f"unexpected circuit state shape {full_state.shape}"
                )

            # SKILL.md specifies system wires 0..n-1 and appended work wire n,
            # but does not separately document the flattened n+1-wire state
            # axis order. This extraction follows the documented appended-wire
            # construction: work=0 precedes work=1 in the full statevector.
            circuit_system_state = full_state[:target_dimension]
            work_one_state = full_state[target_dimension:]
            work_leakage = float(np.vdot(work_one_state, work_one_state).real)

            circuit_error, fidelity = independent_phase_metrics(
                expected,
                circuit_system_state,
            )
            dense_error, _ = independent_phase_metrics(
                expected,
                np.asarray(result["Prepared state"], dtype=np.complex128),
            )
            circuit_dense_error, _ = independent_phase_metrics(
                np.asarray(result["Prepared state"], dtype=np.complex128),
                circuit_system_state,
            )
            expected_total_error, _ = independent_phase_metrics(target, expected)
            expected_status = (
                "ok"
                if expected_total_error <= max(target_error, 1e-10)
                else "failed"
            )

            maximum_phase_error = max(maximum_phase_error, circuit_error)
            minimum_fidelity = min(minimum_fidelity, fidelity)
            maximum_work_leakage = max(maximum_work_leakage, work_leakage)

            failures = []
            if circuit_error > PHASE_TOLERANCE:
                failures.append(f"circuit phase error={circuit_error:.3e}")
            if dense_error > PHASE_TOLERANCE:
                failures.append(f"dense phase error={dense_error:.3e}")
            if circuit_dense_error > PHASE_TOLERANCE:
                failures.append(
                    f"circuit/dense disagreement={circuit_dense_error:.3e}"
                )
            if work_leakage > work_leakage_tolerance:
                failures.append(f"work leakage={work_leakage:.3e}")
            if result["Support size"] != support_size:
                failures.append("support size mismatch")
            if result["Index register qubits"] != index_qubits:
                failures.append("index-register width mismatch")
            if result["status"] != expected_status:
                failures.append("status mismatch")
            if abs(result["Total error"] - expected_total_error) > PHASE_TOLERANCE:
                failures.append("reported Total error mismatch")
            if expected_support_size is not None and support_size != expected_support_size:
                failures.append(
                    f"expected support {expected_support_size}, got {support_size}"
                )
            saw_padding_warning = any(
                item.category is RuntimeWarning
                and "trailing zeros were appended" in str(item.message)
                for item in caught
            )
            if saw_padding_warning != expect_padding_warning:
                failures.append("padding warning behavior mismatch")
            if not isinstance(result["circuit"], Circuit):
                failures.append("returned circuit has the wrong type")

            check(name, not failures, "; ".join(failures))
        except Exception as error:
            check(name, False, f"raised {type(error).__name__}: {error}")

    # Execute basis and deterministic sparse complex cases for every width
    # from one through five qubits, including non-contiguous support and
    # ordering-sensitive states outside the prefix range.
    rng = np.random.default_rng(2718)
    for num_qubits in range(1, 6):
        dimension = 1 << num_qubits
        for index in sorted({0, dimension // 2, dimension - 1}):
            basis = np.zeros(dimension, dtype=np.complex128)
            basis[index] = np.exp(0.37j)
            validate_circuit_case(
                f"basis-{num_qubits}-{index}",
                basis,
                num_qubits,
                expected_support_size=1,
            )

        support = sorted({0, dimension // 2, dimension - 1})
        sparse = np.zeros(dimension, dtype=np.complex128)
        sparse_values = rng.normal(size=len(support)) + 1j * rng.normal(
            size=len(support)
        )
        sparse[support] = sparse_values
        validate_circuit_case(
            f"sparse-complex-{num_qubits}",
            sparse,
            num_qubits,
            expected_support_size=len(support),
        )

    validate_circuit_case(
        "non-contiguous-support",
        np.array([1.0, 0.0, 0.0, 2.0j, 0.0, 0.0, -0.5, 0.0]),
        3,
        expected_support_size=3,
    )
    validate_circuit_case(
        "padding",
        np.array([1.0, 1.0j]),
        3,
        expect_padding_warning=True,
        expected_support_size=2,
    )
    validate_circuit_case(
        "endianness",
        np.array([0.0, 1.0, 0.0, 0.0, -2.0j, 0.0, 0.0, 0.0]),
        3,
        expected_support_size=2,
    )

    # The strict threshold is applied after normalization. A clearly greater
    # value is used instead of nextafter because normalization could move a
    # one-ULP difference back onto the boundary.
    at_boundary = np.array(
        [np.sqrt(1.0 - SUPPORT_THRESHOLD**2), SUPPORT_THRESHOLD, 0.0, 0.0],
        dtype=np.complex128,
    )
    above_boundary = np.array(
        [
            np.sqrt(1.0 - (2.0 * SUPPORT_THRESHOLD) ** 2),
            2.0 * SUPPORT_THRESHOLD,
            0.0,
            0.0,
        ],
        dtype=np.complex128,
    )
    validate_circuit_case(
        "threshold-equality",
        at_boundary,
        2,
        target_error=1e-14,
        expected_support_size=1,
    )
    validate_circuit_case(
        "threshold-greater",
        above_boundary,
        2,
        target_error=1e-14,
        expected_support_size=2,
    )

    invalid_cases = [
        ("bool qubits", TypeError, lambda: superposition_state_preparation([1], True)),
        ("zero qubits", ValueError, lambda: superposition_state_preparation([1], 0)),
        ("bool error", TypeError, lambda: superposition_state_preparation([1, 0], 1, target_error=True)),
        ("nan error", ValueError, lambda: superposition_state_preparation([1, 0], 1, target_error=np.nan)),
        ("inf error", ValueError, lambda: superposition_state_preparation([1, 0], 1, target_error=np.inf)),
        ("empty", ValueError, lambda: superposition_state_preparation([], 1)),
        ("matrix", ValueError, lambda: superposition_state_preparation(np.eye(2), 1)),
        ("zero state", ValueError, lambda: superposition_state_preparation([0, 0], 1)),
        ("nan state", ValueError, lambda: superposition_state_preparation([1, np.nan], 1)),
        ("inf state", ValueError, lambda: superposition_state_preparation([1, np.inf], 1)),
        ("oversized", ValueError, lambda: superposition_state_preparation([1, 0, 0], 1)),
        ("bool index", TypeError, lambda: _index_to_bits(True, 2)),
        ("bad bits", ValueError, lambda: _bits_to_index((0, 2))),
        ("duplicate support", ValueError, lambda: _order_states([(0, 0), (0, 0)])),
        (
            "overlapping work wire",
            ValueError,
            lambda: _build_superposition_circuit(
                np.eye(2, dtype=np.complex128),
                [(0,), (1,)],
                {(0,): (0,), (1,): (1,)},
                1,
                1,
                work_wire=0,
            ),
        ),
    ]
    for name, exception, function in invalid_cases:
        expect(f"invalid-{name}", exception, function)

    print(f"Total tests: {total}")
    print(f"Passed tests: {passed}")
    print(f"Maximum phase-invariant error: {maximum_phase_error:.6e}")
    print(f"Minimum fidelity: {minimum_fidelity:.12f}")
    print(f"Maximum work-qubit leakage: {maximum_work_leakage:.6e}")
    print(
        "SKILL.md 未说明：工作比特泄漏的独立专用容差；"
        "测试采用其明确的 1e-12 数值容差。"
    )
    if passed != total:
        raise AssertionError(f"self-tests failed: {total - passed} of {total}")


if __name__ == "__main__":
    _run_tests()
