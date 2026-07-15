"""Recursive multiplexer preparation of a complex quantum state.

The input is normalized, padded with trailing zeros, and bit-reversed once for
the internal wire schedule.  A depth-first binary probability tree loads
magnitudes with mixed-control RY, CRY, and MCRY gates.  Basis-selective P, CP,
and MCP gates then load the phases of the internally ordered amplitudes.

The explicit UnitaryLab circuit and dense NumPy evolution consume equivalent
magnitude and phase schedules.  ``Prepared state`` is the first column of the
dense evolution in the user's amplitude order.  ``Total error`` is its
global-phase-invariant Euclidean distance from the normalized padded target.
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
MagnitudeGate = dict[str, Any]


def multiplexer_state_preparation(
    Psi: ArrayLike,
    target_qubits: int,
    *,
    target_error: float = 1e-6,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> dict[str, Any]:
    """Prepare ``Psi`` with a recursive magnitude tree and selective phases.

    ``Psi`` must be a finite, nonempty one-dimensional vector with norm greater
    than ``1e-12``.  It is converted to ``complex128``, normalized, and padded
    to ``2**target_qubits``.  ``target_qubits`` is a non-boolean integer at
    least one, and ``target_error`` is a positive finite real number.

    ``backend``, ``device``, and ``dtype`` are compatibility parameters and do
    not affect this explicit NumPy construction, which uses ``complex128``.

    The returned dictionary contains ``status``, ``Prepared state``,
    ``Total error``, ``Computation time (s)``, and the real UnitaryLab
    ``circuit``.  ``magnitude_gates``, ``phase_records``, and ``evolution`` are
    inspection data for the tree schedule, selective phases, and dense unitary.
    Zero-angle tree records remain in ``magnitude_gates`` but are not emitted.
    """
    del backend, device, dtype
    start_time = time.time()

    num_qubits = _validate_integer(target_qubits, "target_qubits", minimum=1)
    error_threshold = _validate_positive_finite_real(
        target_error, "target_error"
    )
    target = _normalize_and_pad(Psi, num_qubits, tol=1e-12)

    # This is the only index-order conversion in the preparation path.
    ordered_state = _bit_reversed_state_vector(target, num_qubits)
    magnitude_gates = _multiplexer_gate_spec(np.abs(ordered_state))
    phase_records = [
        (basis_index, float(angle))
        for basis_index, angle in enumerate(np.angle(ordered_state))
        if abs(float(angle)) > 1e-15
    ]

    circuit = _build_multiplexer_circuit(
        magnitude_gates, phase_records, num_qubits
    )
    evolution = _build_multiplexer_dense_matrix(
        magnitude_gates, phase_records, num_qubits
    )
    prepared_state = np.asarray(evolution[:, 0], dtype=np.complex128)
    total_error = _state_vector_error(target, prepared_state)

    return {
        "status": "ok"
        if total_error <= max(error_threshold, 1e-10)
        else "failed",
        "Prepared state": prepared_state,
        "Total error": float(total_error),
        "Computation time (s)": round(time.time() - start_time, 4),
        "circuit": circuit,
        "magnitude_gates": magnitude_gates,
        "phase_records": phase_records,
        "evolution": evolution,
    }


def _validate_integer(value: Any, name: str, minimum: int | None = None) -> int:
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
    """Validate a positive finite real parameter while rejecting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _normalize_and_pad(
    state: ArrayLike, num_qubits: int, tol: float
) -> ComplexArray:
    """Validate, normalize, and trailing-zero-pad a target vector."""
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


def _bit_reversed_state_vector(
    state: ArrayLike, num_qubits: int
) -> ComplexArray:
    """Reverse amplitude-index bits for the internal recursive wire order."""
    num_qubits = _validate_integer(num_qubits, "num_qubits", minimum=1)
    state = np.asarray(state, dtype=np.complex128)
    dimension = 1 << num_qubits
    if state.shape != (dimension,):
        raise ValueError(f"state must have shape ({dimension},).")
    if not np.all(np.isfinite(state)):
        raise ValueError("state entries must be finite.")

    reordered = np.empty_like(state)
    for index, amplitude in enumerate(state):
        reversed_index = int(format(index, f"0{num_qubits}b")[::-1], 2)
        reordered[reversed_index] = amplitude
    return reordered


def _multiplexer_gate_spec(amplitudes: ArrayLike) -> list[MagnitudeGate]:
    """Build the depth-first binary probability-tree rotation schedule.

    At level ``level``, the target is wire ``level`` and controls are
    ``[0, ..., level-1]``.  The traversal prefix is passed directly as the
    mixed control values in the same order.
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError("amplitudes must be a nonempty one-dimensional vector.")
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be finite.")
    if np.any(amplitudes < 0.0):
        raise ValueError("amplitudes must be non-negative magnitudes.")
    size = int(amplitudes.size)
    if size & (size - 1):
        raise ValueError("amplitude length must be a power of two.")

    probabilities = amplitudes**2
    prefix_sum = np.concatenate(([0.0], np.cumsum(probabilities)))
    num_qubits = size.bit_length() - 1
    gates: list[MagnitudeGate] = []

    def split(
        level: int,
        start: int,
        length: int,
        branch_bits: tuple[int, ...],
    ) -> None:
        if level == num_qubits:
            return

        half = length // 2
        left_probability = float(
            prefix_sum[start + half] - prefix_sum[start]
        )
        right_probability = float(
            prefix_sum[start + length] - prefix_sum[start + half]
        )
        left_norm = math.sqrt(max(left_probability, 0.0))
        right_norm = math.sqrt(max(right_probability, 0.0))
        angle = (
            0.0
            if left_norm + right_norm <= 1e-15
            else 2.0 * math.atan2(right_norm, left_norm)
        )
        gates.append(
            {
                "angle": float(angle),
                "target": level,
                "controls": list(range(level)),
                "control_values": list(branch_bits),
                "level": level,
                "start": start,
                "length": length,
            }
        )
        split(level + 1, start, half, branch_bits + (0,))
        split(level + 1, start + half, half, branch_bits + (1,))

    split(0, 0, size, ())
    return gates


def _append_controlled_ry(
    circuit: Circuit,
    angle: float,
    target: int,
    controls: list[int],
    control_values: list[int],
) -> None:
    """Emit RY, CRY, or MCRY for one nonzero probability-tree split."""
    if abs(angle) <= 1e-15:
        return
    if not controls:
        circuit.ry(angle, target)
    elif len(controls) == 1:
        circuit.cry(angle, controls[0], target, control_values)
    else:
        circuit.mcry(angle, controls, target, control_values)


def _append_controlled_phase(
    circuit: Circuit,
    angle: float,
    target: int,
    controls: list[int],
    control_values: list[int],
) -> None:
    """Emit P, CP, or MCP for one selected control pattern."""
    if not controls:
        circuit.p(angle, target)
    elif len(controls) == 1:
        circuit.cp(angle, controls[0], target, control_values)
    else:
        circuit.mcp(angle, controls, target, control_values)


def _append_basis_state_phase(
    circuit: Circuit,
    basis_index: int,
    phase_angle: float,
    num_qubits: int,
) -> None:
    """Apply ``exp(i*phase_angle)`` to one internal basis state."""
    if abs(phase_angle) <= 1e-15:
        return
    _validate_basis_index(basis_index, num_qubits)
    bits = [int(bit) for bit in format(basis_index, f"0{num_qubits}b")]
    target = num_qubits - 1
    controls = list(range(target))
    control_values = bits[:-1]

    # Phase primitives activate on target |1>; flip around the phase operation
    # when the selected target bit is zero.
    target_is_zero = bits[-1] == 0
    if target_is_zero:
        circuit.x(target)
    _append_controlled_phase(
        circuit, phase_angle, target, controls, control_values
    )
    if target_is_zero:
        circuit.x(target)


def _build_multiplexer_circuit(
    magnitude_gates: list[MagnitudeGate],
    phase_records: list[tuple[int, float]],
    num_qubits: int,
) -> Circuit:
    """Build the explicit controlled-gate multiplexer circuit."""
    circuit = Circuit(
        num_qubits, name="Multiplexer State Preparation"
    )
    for gate in magnitude_gates:
        _append_controlled_ry(
            circuit,
            float(gate["angle"]),
            int(gate["target"]),
            list(gate["controls"]),
            list(gate["control_values"]),
        )
    for basis_index, phase_angle in phase_records:
        _append_basis_state_phase(
            circuit, basis_index, phase_angle, num_qubits
        )
    return circuit


def _build_multiplexer_dense_matrix(
    magnitude_gates: list[MagnitudeGate],
    phase_records: list[tuple[int, float]],
    num_qubits: int,
) -> ComplexArray:
    """Build a dense unitary from the magnitude and phase schedules."""
    evolution = np.eye(1 << num_qubits, dtype=np.complex128)
    for gate in magnitude_gates:
        angle = float(gate["angle"])
        if abs(angle) <= 1e-15:
            continue
        evolution = _apply_controlled_ry_dense(
            evolution,
            angle,
            int(gate["target"]),
            list(gate["controls"]),
            list(gate["control_values"]),
        )
    for basis_index, phase_angle in phase_records:
        evolution = _apply_basis_state_phase_dense(
            evolution, basis_index, phase_angle, num_qubits
        )
    return evolution


def _apply_controlled_ry_dense(
    evolution: ArrayLike,
    angle: float,
    target: int,
    controls: list[int],
    control_values: list[int],
) -> ComplexArray:
    """Left-multiply a dense evolution by a mixed-control RY gate."""
    evolution, num_qubits = _validate_evolution(evolution)
    half_angle = 0.5 * float(angle)
    rotation = np.array(
        [
            [math.cos(half_angle), -math.sin(half_angle)],
            [math.sin(half_angle), math.cos(half_angle)],
        ],
        dtype=np.complex128,
    )
    gate = _controlled_single_qubit_matrix(
        rotation, target, controls, control_values, num_qubits
    )
    return np.asarray(gate @ evolution, dtype=np.complex128)


def _controlled_single_qubit_matrix(
    matrix_2x2: ArrayLike,
    target: int,
    controls: list[int],
    control_values: list[int],
    num_qubits: int,
) -> ComplexArray:
    """Build a dense mixed-control single-qubit gate matrix."""
    num_qubits = _validate_integer(num_qubits, "num_qubits", minimum=1)
    matrix_2x2 = np.asarray(matrix_2x2, dtype=np.complex128)
    if matrix_2x2.shape != (2, 2) or not np.all(np.isfinite(matrix_2x2)):
        raise ValueError("matrix_2x2 must be a finite 2 by 2 matrix.")
    target = _validate_wire(target, "target", num_qubits)
    if not isinstance(controls, list) or not isinstance(control_values, list):
        raise TypeError("controls and control_values must be lists.")
    validated_controls = [
        _validate_wire(wire, f"controls[{index}]", num_qubits)
        for index, wire in enumerate(controls)
    ]
    if len(set(validated_controls)) != len(validated_controls):
        raise ValueError("controls must be unique.")
    if target in validated_controls:
        raise ValueError("target must not appear in controls.")
    if len(control_values) != len(validated_controls):
        raise ValueError("control_values length must match controls length.")
    validated_values = [
        _validate_control_value(value, index)
        for index, value in enumerate(control_values)
    ]

    dimension = 1 << num_qubits
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    control_mask = 0
    control_pattern = 0
    for control, value in zip(validated_controls, validated_values):
        control_mask |= 1 << control
        if value:
            control_pattern |= 1 << control

    for column in range(dimension):
        if column & control_mask != control_pattern:
            result[column, column] = 1.0
            continue
        target_value = (column >> target) & 1
        for output_value in range(2):
            row = (column & ~(1 << target)) | (output_value << target)
            result[row, column] = matrix_2x2[output_value, target_value]
    return result


def _validate_wire(wire: Any, name: str, num_qubits: int) -> int:
    """Validate a wire index against a register width."""
    result = _validate_integer(wire, name, minimum=0)
    if result >= num_qubits:
        raise ValueError(f"{name}={result} is outside [0, {num_qubits}).")
    return result


def _validate_control_value(value: Any, index: int) -> int:
    """Validate one mixed-control bit."""
    result = _validate_integer(value, f"control_values[{index}]", minimum=0)
    if result not in (0, 1):
        raise ValueError("control values must be 0 or 1.")
    return result


def _validate_evolution(evolution: ArrayLike) -> tuple[ComplexArray, int]:
    """Validate a finite square evolution with power-of-two dimension."""
    matrix = np.asarray(evolution, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("evolution must be a square matrix.")
    dimension = matrix.shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("evolution dimension must be a power of two >= 2.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("evolution entries must be finite.")
    return matrix, dimension.bit_length() - 1


def _validate_basis_index(basis_index: Any, num_qubits: int) -> int:
    """Validate an internal computational-basis index."""
    num_qubits = _validate_integer(num_qubits, "num_qubits", minimum=1)
    index = _validate_integer(basis_index, "basis_index", minimum=0)
    if index >= 1 << num_qubits:
        raise ValueError("basis_index exceeds the register dimension.")
    return index


def _apply_basis_state_phase_dense(
    evolution: ArrayLike,
    basis_index: int,
    phase_angle: float,
    num_qubits: int,
) -> ComplexArray:
    """Left-multiply by a phase on one internally ordered basis state."""
    evolution, evolution_qubits = _validate_evolution(evolution)
    num_qubits = _validate_integer(num_qubits, "num_qubits", minimum=1)
    if evolution_qubits != num_qubits:
        raise ValueError("evolution dimension does not match num_qubits.")
    basis_index = _validate_basis_index(basis_index, num_qubits)
    phase_angle = float(phase_angle)
    if not np.isfinite(phase_angle):
        raise ValueError("phase_angle must be finite.")

    bits = [int(bit) for bit in format(basis_index, f"0{num_qubits}b")]
    target = num_qubits - 1
    controls = list(range(target))
    control_values = bits[:-1]
    phase = np.array(
        [[1.0, 0.0], [0.0, np.exp(1j * phase_angle)]],
        dtype=np.complex128,
    )
    if bits[-1] == 0:
        x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        phase = x_gate @ phase @ x_gate
    gate = _controlled_single_qubit_matrix(
        phase, target, controls, control_values, num_qubits
    )
    return np.asarray(gate @ evolution, dtype=np.complex128)


def _state_vector_error(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return global-phase-invariant Euclidean error without renormalizing."""
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("states must be one-dimensional vectors.")
    if reference.shape != candidate.shape:
        raise ValueError("state shapes must match.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("state entries must be finite.")
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
    """Run deterministic validation of the multiplexer preparation path."""
    case_count = 0
    max_error = 0.0

    def run_case(
        state: ArrayLike, num_qubits: int, **kwargs: Any
    ) -> dict[str, Any]:
        nonlocal case_count, max_error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = multiplexer_state_preparation(
                state, num_qubits, **kwargs
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            target = _normalize_and_pad(state, num_qubits, tol=1e-12)
        prepared = result["Prepared state"]
        assert isinstance(result["circuit"], Circuit)
        assert np.allclose(prepared, result["evolution"][:, 0], atol=1e-12)
        gram = np.einsum(
            "ki,kj->ij", result["evolution"].conj(), result["evolution"]
        )
        assert np.allclose(
            gram, np.eye(1 << num_qubits, dtype=np.complex128), atol=2e-12
        )
        expected_error = _state_vector_error(target, prepared)
        assert abs(result["Total error"] - expected_error) <= 1e-14
        threshold = float(kwargs.get("target_error", 1e-6))
        expected_status = (
            "ok" if expected_error <= max(threshold, 1e-10) else "failed"
        )
        assert result["status"] == expected_status
        assert abs(np.linalg.norm(prepared) - 1.0) <= 2e-12

        case_count += 1
        max_error = max(max_error, float(result["Total error"]))
        return result

    # Every basis state verifies the internal conversion and user-order return.
    for num_qubits in range(1, 5):
        for basis_index in range(1 << num_qubits):
            state = np.zeros(1 << num_qubits, dtype=np.complex128)
            state[basis_index] = 1.0
            result = run_case(state, num_qubits)
            assert int(np.argmax(np.abs(result["Prepared state"]))) == basis_index

    # Non-equal magnitudes, complex phases, zeros, and padded input.
    run_case(np.array([1.0, 2.0j, -3.0, 0.5j]), 2)
    run_case(np.array([1.0, 0.0, 0.0, 1.0]), 2)
    padded = run_case(np.array([1.0, 1.0j, 0.0]), 3)
    assert np.sum(np.abs(padded["Prepared state"][3:]) ** 2) <= 1e-24
    run_case(np.exp(0.731j) * np.array([1.0, 2.0j, -1.0, 0.0]), 2)
    run_case(np.array([3.0, -4.0j, 2.0, 1.0j]), 2)

    # Deterministic random real and complex states, including the requested
    # random complex regression.
    rng = np.random.default_rng(20260715)
    for num_qubits in range(1, 6):
        dimension = 1 << num_qubits
        run_case(rng.normal(size=dimension), num_qubits)
        run_case(
            rng.normal(size=dimension) + 1j * rng.normal(size=dimension),
            num_qubits,
        )

    # Tree records keep traversal prefixes in control-wire order.
    tree = _multiplexer_gate_spec(np.ones(8) / np.sqrt(8))
    assert len(tree) == 7
    for gate in tree:
        level = int(gate["level"])
        assert gate["target"] == level
        assert gate["controls"] == list(range(level))
        assert len(gate["control_values"]) == level

    zero_tree = _multiplexer_gate_spec(np.array([1.0, 0.0, 0.0, 0.0]))
    assert any(abs(float(gate["angle"])) <= 1e-15 for gate in zero_tree)

    # Phase records contain only non-negligible internally ordered phases.
    phase_case = run_case(np.array([1.0, 1.0j, -1.0, -1.0j]), 2)
    assert all(abs(angle) > 1e-15 for _, angle in phase_case["phase_records"])

    invalid_calls: list[tuple[type[BaseException], Any]] = [
        (ValueError, lambda: multiplexer_state_preparation([], 1)),
        (ValueError, lambda: multiplexer_state_preparation(np.eye(2), 1)),
        (ValueError, lambda: multiplexer_state_preparation([np.nan, 1], 1)),
        (ValueError, lambda: multiplexer_state_preparation([0, 0], 1)),
        (ValueError, lambda: multiplexer_state_preparation(np.ones(5), 2)),
        (ValueError, lambda: multiplexer_state_preparation([1], 0)),
        (TypeError, lambda: multiplexer_state_preparation([1], True)),
        (TypeError, lambda: multiplexer_state_preparation([1, 0], 1.0)),
        (TypeError, lambda: multiplexer_state_preparation([1, 0], 1, target_error=True)),
        (ValueError, lambda: multiplexer_state_preparation([1, 0], 1, target_error=0)),
        (ValueError, lambda: multiplexer_state_preparation([1, 0], 1, target_error=np.inf)),
    ]
    for expected, function in invalid_calls:
        _assert_raises(expected, function)

    _assert_raises(ValueError, lambda: _multiplexer_gate_spec([]))
    _assert_raises(ValueError, lambda: _multiplexer_gate_spec([1.0, -1.0]))
    _assert_raises(ValueError, lambda: _multiplexer_gate_spec([1.0, np.nan]))
    _assert_raises(ValueError, lambda: _multiplexer_gate_spec(np.ones(3)))
    _assert_raises(
        ValueError,
        lambda: _controlled_single_qubit_matrix(
            np.eye(2), 1, [0], [], 2
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _controlled_single_qubit_matrix(
            np.eye(2), 1, [0, 0], [0, 1], 2
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _controlled_single_qubit_matrix(
            np.eye(2), 1, [1], [1], 2
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _apply_basis_state_phase_dense(np.eye(4), 4, 0.5, 2),
    )

    print(
        f"{case_count} multiplexer preparation cases passed; "
        f"maximum phase-invariant error = {max_error:.3e}"
    )


if __name__ == "__main__":
    _run_tests()
