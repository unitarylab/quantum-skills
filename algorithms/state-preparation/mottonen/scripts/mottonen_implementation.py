"""Standalone reference for UnitaryLab's Möttönen state preparation.

This file mirrors the gate schedule in the authoritative implementation at
``unitarylab_algorithms/state_preparation/mottonen/algorithm.py``.  It
normalizes and trailing-zero-pads the user's amplitudes, applies exactly one
internal amplitude-index bit reversal, emits the source's Gray-code RY/RZ/CX
operation sequence, and builds both the UnitaryLab circuit and dense evolution
from that same sequence.

``Prepared state`` is ``evolution[:, 0]`` in the user's original amplitude
index order; it is not bit-reversed on return.  The dense matrix is intended
only for small-system validation because its memory cost is exponential.
Production code should import ``MottonenAlgorithm`` from
``unitarylab_algorithms`` rather than importing this reference module.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from unitarylab.core import Circuit


Operation = dict[str, float | int | str]
Layer = dict[str, Any]


def mottonen_state_preparation(
    Psi: ArrayLike,
    target_qubits: int,
    *,
    target_error: float = 1e-6,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> dict[str, Any]:
    """Prepare ``Psi`` with the source-compatible elementary-gate schedule.

    Parameters
    ----------
    Psi:
        A non-empty, finite, one-dimensional vector with norm greater than
        ``1e-12``.  It is converted to ``complex128`` and normalized.
    target_qubits:
        A non-boolean integer at least one.  ``Psi`` may be shorter than
        ``2**target_qubits``; trailing zeros are then appended with a
        ``RuntimeWarning``.
    target_error:
        Positive finite tolerance.  Status is ``"ok"`` exactly when the
        source phase-invariant error is no greater than
        ``max(target_error, 1e-10)``.
    backend, device, dtype:
        API-compatibility parameters.  This reference path, like the formal
        NumPy construction, does not use them.  A non-default value emits a
        ``RuntimeWarning``; targets and matrices remain ``complex128``.

    Returns
    -------
    dict
        ``status``, ``Prepared state``, ``Total error``,
        ``Computation time (s)``, ``operations``, ``layers``, ``evolution``,
        ``circuit``, and ``RZ enabled``.  ``operations`` is the actual
        Gray-code RY/RZ/CX sequence consumed by both ``circuit`` and
        ``evolution``.  ``circuit`` is a real ``unitarylab.core.Circuit``.
        The diagnostic fields beyond the formal public result dictionary are
        reference-only conveniences.
    """
    start_time = time.time()
    _warn_for_ignored_api_arguments(backend, device, dtype)

    if isinstance(target_qubits, bool) or not isinstance(
        target_qubits, (int, np.integer)
    ):
        raise TypeError("target_qubits must be an integer.")
    if target_qubits < 1:
        raise ValueError("target_qubits must be at least 1.")

    target_error = float(target_error)
    if not np.isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be a positive finite number.")

    target = _normalize_and_pad(Psi, int(target_qubits), tol=1e-12)
    operations, layers, rz_enabled = _mottonen_operations(
        target, int(target_qubits)
    )
    circuit = _build_mottonen_circuit(operations, int(target_qubits))
    evolution = _build_evolution_matrix(operations, int(target_qubits))
    prepared_state = np.asarray(evolution[:, 0], dtype=np.complex128)
    total_error = _phase_invariant_error(target, prepared_state)

    return {
        "status": "ok"
        if total_error <= max(target_error, 1e-10)
        else "failed",
        "Prepared state": prepared_state,
        "Total error": float(total_error),
        "Computation time (s)": round(time.time() - start_time, 4),
        "operations": operations,
        "layers": layers,
        "evolution": evolution,
        "circuit": circuit,
        "RZ enabled": rz_enabled,
    }


def _warn_for_ignored_api_arguments(
    backend: str, device: str, dtype: type
) -> None:
    """Warn when an ignored compatibility argument differs from its default."""
    if backend != "torch":
        warnings.warn(
            f"backend={backend!r} is accepted for API compatibility but ignored; "
            "dense NumPy construction is always used.",
            RuntimeWarning,
            stacklevel=3,
        )
    if device != "cpu":
        warnings.warn(
            f"device={device!r} is accepted for API compatibility but ignored.",
            RuntimeWarning,
            stacklevel=3,
        )
    try:
        is_default_dtype = np.dtype(dtype) == np.dtype(np.complex128)
    except TypeError:
        is_default_dtype = False
    if not is_default_dtype:
        warnings.warn(
            f"dtype={dtype!r} is accepted for API compatibility but ignored; "
            "complex128 is always used.",
            RuntimeWarning,
            stacklevel=3,
        )


def _normalize_and_pad(
    state: ArrayLike, num_qubits: int, tol: float
) -> NDArray[np.complex128]:
    """Apply the formal source's validation, normalization, and tail padding."""
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
    psi = np.ascontiguousarray(psi / norm)

    padded_dim = 1 << num_qubits
    if psi.size > padded_dim:
        raise ValueError(
            f"State vector length {psi.size} exceeds target Hilbert "
            f"dimension {padded_dim}."
        )
    if psi.size == padded_dim:
        return psi

    padded = np.zeros(padded_dim, dtype=np.complex128)
    padded[: psi.size] = psi
    warnings.warn(
        f"State vector length {psi.size} is smaller than 2**target_qubits; "
        f"padded to {padded_dim}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return np.ascontiguousarray(padded)


def _bit_reversed_state_vector(
    state_vector: NDArray[np.complex128], num_qubits: int
) -> NDArray[np.complex128]:
    """Apply the repository's single internal amplitude-index conversion."""
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    expected_dim = 1 << num_qubits
    if state_vector.ndim != 1 or state_vector.size != expected_dim:
        raise ValueError(f"state_vector must have shape ({expected_dim},).")
    if num_qubits <= 1:
        return state_vector.copy()

    reordered = np.empty_like(state_vector)
    for index, amplitude in enumerate(state_vector):
        reversed_index = int(format(index, f"0{num_qubits}b")[::-1], 2)
        reordered[reversed_index] = amplitude
    return reordered


def _gray_code(num_bits: int) -> NDArray[np.int64]:
    """Return the reflected Gray sequence used by the formal source."""
    if num_bits < 0:
        raise ValueError("num_bits must be non-negative.")
    return np.array(
        [index ^ (index >> 1) for index in range(1 << num_bits)],
        dtype=np.int64,
    )


def _compute_theta(alphas: ArrayLike) -> NDArray[np.float64]:
    """Transform branch angles into the formal Gray-code ladder angles."""
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a non-empty one-dimensional vector.")
    size = int(alphas.size)
    if size & (size - 1):
        raise ValueError("alphas length must be a power of 2.")

    gray = _gray_code(size.bit_length() - 1)
    theta = np.zeros(size, dtype=np.float64)
    scale = 1.0 / float(size)
    for index, gray_word in enumerate(gray):
        accumulator = 0.0
        for branch, alpha in enumerate(alphas):
            parity = ((branch & int(gray_word)).bit_count()) & 1
            accumulator += -float(alpha) if parity else float(alpha)
        theta[index] = scale * accumulator
    return theta


def _compute_alpha_y(
    amplitudes: ArrayLike, num_qubits: int, k: int
) -> NDArray[np.float64]:
    """Compute the source amplitude-split angles for level ``k``."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    num_pairs = 1 << (num_qubits - k)
    half_block = 1 << (k - 1)
    full_block = 1 << k
    alpha_y = np.zeros(num_pairs, dtype=np.float64)

    for branch in range(num_pairs):
        start = branch * full_block
        numerator = float(
            np.sum(amplitudes[start + half_block : start + full_block] ** 2)
        )
        denominator = float(np.sum(amplitudes[start : start + full_block] ** 2))
        if denominator <= 1e-15:
            alpha_y[branch] = 0.0
            continue
        ratio = float(np.clip(numerator / denominator, 0.0, 1.0))
        alpha_y[branch] = float(2.0 * np.arcsin(np.sqrt(ratio)))
    return alpha_y


def _compute_alpha_z(
    phases: ArrayLike, num_qubits: int, k: int
) -> NDArray[np.float64]:
    """Compute the source principal-phase difference angles for level ``k``."""
    phases = np.asarray(phases, dtype=np.float64)
    num_pairs = 1 << (num_qubits - k)
    block = 1 << (k - 1)
    alpha_z = np.zeros(num_pairs, dtype=np.float64)

    for branch in range(num_pairs):
        start0 = (2 * branch) * block
        start1 = (2 * branch + 1) * block
        phase_diff = (
            phases[start1 : start1 + block] - phases[start0 : start0 + block]
        )
        alpha_z[branch] = float(np.sum(phase_diff) / block)
    return alpha_z


def _validate_wire(wire: int, num_qubits: int, name: str) -> int:
    """Return a validated non-boolean wire index."""
    if isinstance(wire, bool) or not isinstance(wire, (int, np.integer)):
        raise TypeError(f"{name} must be an integer wire index.")
    wire = int(wire)
    if wire < 0 or wire >= num_qubits:
        raise ValueError(
            f"{name}={wire} is outside the legal range [0, {num_qubits})."
        )
    return wire


def _uniform_rotation_operations(
    axis: str,
    alphas: ArrayLike,
    control_wires: list[int],
    target_wire: int,
) -> list[Operation]:
    """Emit one uniformly controlled rotation as source-ordered gates.

    The rotation at each Gray position is followed by the CX selected by the
    cyclic transition to the next Gray word.  Controlled ladders retain every
    CX even when a rotation angle is omitted at the ``1e-15`` gate threshold.
    """
    if axis not in {"ry", "rz"}:
        raise ValueError("axis must be 'ry' or 'rz'.")
    alphas = np.asarray(alphas, dtype=np.float64)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a non-empty one-dimensional vector.")
    expected = 1 << len(control_wires)
    if alphas.size != expected:
        raise ValueError(
            f"len(alphas) must equal 2**len(control_wires)={expected}; "
            f"got {alphas.size}."
        )
    if len(set(control_wires)) != len(control_wires):
        raise ValueError("control_wires must not contain duplicates.")
    if target_wire in control_wires:
        raise ValueError("target_wire must not appear in control_wires.")

    if not control_wires:
        angle = float(alphas[0])
        if abs(angle) <= 1e-15:
            return []
        return [{"name": axis, "angle": angle, "target": int(target_wire)}]

    theta = _compute_theta(alphas)
    gray = _gray_code(len(control_wires))
    operations: list[Operation] = []
    for index, angle in enumerate(theta):
        angle = float(angle)
        if abs(angle) > 1e-15:
            operations.append(
                {"name": axis, "angle": angle, "target": int(target_wire)}
            )
        changed = int(gray[index] ^ gray[(index + 1) % len(gray)])
        changed_bit = changed.bit_length() - 1
        operations.append(
            {
                "name": "cx",
                "control": int(control_wires[changed_bit]),
                "target": int(target_wire),
            }
        )
    return operations


def _mottonen_operations(
    state_vector: ArrayLike, num_qubits: int
) -> tuple[list[Operation], list[Layer], bool]:
    """Create the exact RY-then-conditional-RZ source operation schedule.

    This function is the only place that applies bit reversal, and it applies
    it exactly once.  Builders consume its output without further reordering.
    """
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    expected_dim = 1 << num_qubits
    if state_vector.ndim != 1 or state_vector.size != expected_dim:
        raise ValueError(f"state_vector must have shape ({expected_dim},).")

    ordered_state = _bit_reversed_state_vector(state_vector, num_qubits)
    amplitudes = np.abs(ordered_state)
    phases = np.angle(ordered_state)
    wires_reversed = list(range(num_qubits))[::-1]
    operations: list[Operation] = []
    layers: list[Layer] = []

    for k in range(num_qubits, 0, -1):
        controls = list(wires_reversed[k:])
        target = int(wires_reversed[k - 1])
        alpha = _compute_alpha_y(amplitudes, num_qubits, k)
        layer_operations = _uniform_rotation_operations(
            "ry", alpha, controls, target
        )
        operations.extend(layer_operations)
        layers.append(
            {
                "axis": "ry",
                "k": k,
                "target": target,
                "controls": controls,
                "alpha": alpha,
                "operations": layer_operations,
            }
        )

    rz_enabled = not np.allclose(phases, 0.0, atol=1e-15)
    if rz_enabled:
        for k in range(num_qubits, 0, -1):
            controls = list(wires_reversed[k:])
            target = int(wires_reversed[k - 1])
            alpha = _compute_alpha_z(phases, num_qubits, k)
            layer_operations = _uniform_rotation_operations(
                "rz", alpha, controls, target
            )
            operations.extend(layer_operations)
            layers.append(
                {
                    "axis": "rz",
                    "k": k,
                    "target": target,
                    "controls": controls,
                    "alpha": alpha,
                    "operations": layer_operations,
                }
            )
    return operations, layers, rz_enabled


def _ry_matrix(angle: float) -> NDArray[np.complex128]:
    """Return the single-qubit RY matrix used by the formal source."""
    half_angle = 0.5 * float(angle)
    return np.array(
        [
            [np.cos(half_angle), -np.sin(half_angle)],
            [np.sin(half_angle), np.cos(half_angle)],
        ],
        dtype=np.complex128,
    )


def _rz_matrix(angle: float) -> NDArray[np.complex128]:
    """Return the single-qubit RZ matrix used by the formal source."""
    half_angle = 0.5 * float(angle)
    return np.array(
        [
            [np.exp(-1j * half_angle), 0.0],
            [0.0, np.exp(1j * half_angle)],
        ],
        dtype=np.complex128,
    )


def _apply_single_qubit_gate(
    evolution: NDArray[np.complex128],
    matrix_2x2: NDArray[np.complex128],
    target_wire: int,
) -> NDArray[np.complex128]:
    """Left-multiply ``evolution`` by a one-qubit gate on ``target_wire``."""
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


def _apply_cnot(
    evolution: NDArray[np.complex128], control_wire: int, target_wire: int
) -> NDArray[np.complex128]:
    """Left-multiply ``evolution`` by the source's CNOT permutation."""
    dim = evolution.shape[0]
    permutation = np.arange(dim, dtype=np.int64)
    for basis_index in range(dim):
        if (basis_index >> control_wire) & 1:
            permutation[basis_index] = basis_index ^ (1 << target_wire)
    return np.asarray(evolution[permutation, :], dtype=np.complex128)


def _build_evolution_matrix(
    operations: list[Operation], num_qubits: int
) -> NDArray[np.complex128]:
    """Build the dense unitary by consuming the explicit operation sequence."""
    evolution = np.eye(1 << num_qubits, dtype=np.complex128)
    for operation in operations:
        name = str(operation["name"])
        if name == "ry":
            evolution = _apply_single_qubit_gate(
                evolution,
                _ry_matrix(float(operation["angle"])),
                int(operation["target"]),
            )
        elif name == "rz":
            evolution = _apply_single_qubit_gate(
                evolution,
                _rz_matrix(float(operation["angle"])),
                int(operation["target"]),
            )
        elif name == "cx":
            evolution = _apply_cnot(
                evolution,
                int(operation["control"]),
                int(operation["target"]),
            )
        else:
            raise ValueError(f"Unsupported gate operation {name!r}.")
    return evolution


def _build_mottonen_circuit(
    operations: list[Operation], num_qubits: int
) -> Circuit:
    """Build a real UnitaryLab circuit from the explicit operation sequence."""
    circuit = Circuit(
        num_qubits, name="Quantum circuit for Mottonen State Preparation"
    )
    for operation in operations:
        name = str(operation["name"])
        if name == "ry":
            circuit.ry(float(operation["angle"]), int(operation["target"]))
        elif name == "rz":
            circuit.rz(float(operation["angle"]), int(operation["target"]))
        elif name == "cx":
            circuit.cx(int(operation["control"]), int(operation["target"]))
        else:
            raise ValueError(f"Unsupported gate operation {name!r}.")
    return circuit


def _uniformly_controlled_rotation_matrix(
    axis: str,
    alpha: ArrayLike,
    controls: list[int],
    target: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Build a dense uniformly controlled RY or RZ matrix for layer checks.

    This diagnostic helper is not used to construct the returned evolution;
    the main path uses the genuine Gray-code operation sequence.  Control bit
    zero corresponds to ``controls[0]`` when indexing ``alpha``.
    """
    if axis not in {"ry", "rz"}:
        raise ValueError("axis must be 'ry' or 'rz'.")
    if isinstance(num_qubits, bool) or not isinstance(
        num_qubits, (int, np.integer)
    ):
        raise TypeError("num_qubits must be an integer.")
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1.")

    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 1:
        raise ValueError("alpha must be a one-dimensional vector.")
    expected = 1 << len(controls)
    if alpha.size != expected:
        raise ValueError(
            f"len(alpha) must equal 2**len(controls)={expected}; "
            f"got {alpha.size}."
        )

    target = _validate_wire(target, int(num_qubits), "target")
    validated_controls = [
        _validate_wire(wire, int(num_qubits), f"controls[{index}]")
        for index, wire in enumerate(controls)
    ]
    if len(set(validated_controls)) != len(validated_controls):
        raise ValueError("controls must not contain duplicates.")
    if target in validated_controls:
        raise ValueError("target must not appear in controls.")

    dim = 1 << int(num_qubits)
    result = np.zeros((dim, dim), dtype=np.complex128)
    target_mask = 1 << target
    for column in range(dim):
        control_pattern = 0
        for index, control in enumerate(validated_controls):
            if (column >> control) & 1:
                control_pattern |= 1 << index
        angle = float(alpha[control_pattern])
        one_qubit = _ry_matrix(angle) if axis == "ry" else _rz_matrix(angle)
        target_value = (column >> target) & 1
        for output_value in range(2):
            row = (column & ~target_mask) | (output_value << target)
            result[row, column] = one_qubit[output_value, target_value]
    return result


def _phase_invariant_error(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return the formal source's global-phase-invariant Euclidean error."""
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 1e-12:
        candidate = candidate * np.conj(overlap / abs(overlap))
    return float(np.linalg.norm(reference - candidate))


def validation_metrics(
    reference: ArrayLike, candidate: ArrayLike
) -> dict[str, float]:
    """Return normalized diagnostic metrics without changing ``Total error``."""
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if reference.ndim != 1 or candidate.ndim != 1 or reference.size == 0:
        raise ValueError("states must be non-empty one-dimensional vectors.")
    if reference.shape != candidate.shape:
        raise ValueError("state shapes must match.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("state entries must be finite.")
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    if reference_norm <= 1e-15 or candidate_norm <= 1e-15:
        raise ValueError("states must have non-zero norm.")

    reference_n = reference / reference_norm
    candidate_n = candidate / candidate_norm
    overlap = np.vdot(reference_n, candidate_n)
    aligned = candidate_n.copy()
    if abs(overlap) > 1e-15:
        aligned *= np.conj(overlap / abs(overlap))
    return {
        "norm_error": abs(candidate_norm - 1.0),
        "phase_invariant_error": float(np.linalg.norm(reference_n - aligned)),
        "fidelity": float(abs(overlap) ** 2),
    }


def _assert_raises(expected: type[BaseException], function: Any) -> None:
    """Minimal exception assertion used by the dependency-light self-tests."""
    try:
        function()
    except expected:
        return
    except Exception as error:
        raise AssertionError(
            f"expected {expected.__name__}, got {type(error).__name__}: {error}"
        ) from error
    raise AssertionError(f"expected {expected.__name__}, but no exception was raised")


def _run_tests() -> None:
    """Run deterministic 1--5 qubit contract and formal-source comparisons."""
    import importlib.util
    import sys
    from pathlib import Path

    formal_path = (
        Path(__file__).resolve().parents[5]
        / "unitarylab_algorithms"
        / "state_preparation"
        / "mottonen"
        / "algorithm.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_formal_mottonen_algorithm", formal_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load formal source at {formal_path}")
    formal_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = formal_module
    spec.loader.exec_module(formal_module)
    MottonenAlgorithm = formal_module.MottonenAlgorithm

    max_error = 0.0
    case_count = 0

    def run_case(state: ArrayLike, num_qubits: int) -> dict[str, Any]:
        nonlocal max_error, case_count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = mottonen_state_preparation(state, num_qubits)
        assert result["status"] == "ok"
        assert isinstance(result["circuit"], Circuit)
        assert len(result["circuit"].data().data()) == len(result["operations"])
        gram = np.einsum(
            "ki,kj->ij",
            result["evolution"].conj(),
            result["evolution"],
        )
        assert np.allclose(gram, np.eye(1 << num_qubits), atol=2e-12)
        max_error = max(max_error, float(result["Total error"]))
        case_count += 1
        return result

    # Every computational basis state checks user-facing return order.
    for num_qubits in range(1, 6):
        for basis_index in range(1 << num_qubits):
            state = np.zeros(1 << num_qubits, dtype=np.complex128)
            state[basis_index] = 1.0
            result = run_case(state, num_qubits)
            assert int(np.argmax(np.abs(result["Prepared state"]))) == basis_index

    # Multiple deterministic real and complex random states at every size.
    rng = np.random.default_rng(20260715)
    comparison_cases: list[tuple[NDArray[np.complex128], int]] = []
    for num_qubits in range(1, 6):
        dim = 1 << num_qubits
        for _ in range(3):
            real_state = rng.normal(size=dim).astype(np.complex128)
            complex_state = (
                rng.normal(size=dim) + 1j * rng.normal(size=dim)
            ).astype(np.complex128)
            run_case(real_state, num_qubits)
            complex_result = run_case(complex_state, num_qubits)
            comparison_cases.append((complex_state, num_qubits))
            assert complex_result["RZ enabled"]

    # Non-normalized and global-phase inputs preserve the source contract.
    nonnormalized = np.array([3.0, -4.0j, 2.0, 1.0j])
    run_case(nonnormalized, 2)
    base = np.array([1.0, 2.0j, -3.0, 4.0j], dtype=np.complex128)
    phased = np.exp(0.731j) * base
    phased_result = run_case(phased, 2)
    target = phased / np.linalg.norm(phased)
    assert _phase_invariant_error(target, phased_result["Prepared state"]) <= 1e-10

    # Zero amplitudes, trailing padding, and leakage into the padded region.
    sparse_short = np.array([1.0, 0.0, -2.0j, 0.0, 3.0])
    padded_result = run_case(sparse_short, 3)
    leakage = float(np.sum(np.abs(padded_result["Prepared state"][5:]) ** 2))
    assert leakage <= 1e-24

    # Preserve the original n=3 layer-order and angle-count checks.
    n3_layers = padded_result["layers"]
    ry_layers = [layer for layer in n3_layers if layer["axis"] == "ry"]
    assert [layer["k"] for layer in ry_layers] == [3, 2, 1]
    for layer in n3_layers:
        assert len(layer["alpha"]) == 1 << len(layer["controls"])

    # The retained dense helper has strict axis/shape/wire validation.
    valid_alpha = np.array([0.1, 0.2])
    _assert_raises(
        ValueError,
        lambda: _uniformly_controlled_rotation_matrix(
            "rx", valid_alpha, [0], 1, 2
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _uniformly_controlled_rotation_matrix("ry", [0.1], [0], 1, 2),
    )
    _assert_raises(
        ValueError,
        lambda: _uniformly_controlled_rotation_matrix("ry", valid_alpha, [2], 1, 2),
    )
    _assert_raises(
        ValueError,
        lambda: _uniformly_controlled_rotation_matrix("ry", valid_alpha, [0], 2, 2),
    )
    _assert_raises(
        ValueError,
        lambda: _uniformly_controlled_rotation_matrix(
            "rz", np.arange(4.0), [0, 0], 1, 2
        ),
    )
    _assert_raises(
        ValueError,
        lambda: _uniformly_controlled_rotation_matrix("rz", valid_alpha, [1], 1, 2),
    )

    # General public-input rejection and ignored-parameter warnings.
    for expected, function in [
        (ValueError, lambda: mottonen_state_preparation([], 1)),
        (ValueError, lambda: mottonen_state_preparation(np.eye(2), 1)),
        (ValueError, lambda: mottonen_state_preparation([np.nan, 1], 1)),
        (ValueError, lambda: mottonen_state_preparation([0, 0], 1)),
        (ValueError, lambda: mottonen_state_preparation(np.ones(5), 2)),
        (ValueError, lambda: mottonen_state_preparation([1], 0)),
        (TypeError, lambda: mottonen_state_preparation([1], True)),
        (TypeError, lambda: mottonen_state_preparation([1], 1.0)),
        (ValueError, lambda: mottonen_state_preparation([1], 1, target_error=0)),
    ]:
        _assert_raises(expected, function)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mottonen_state_preparation(
            [1, 0], 1, backend="ignored", device="ignored", dtype=np.complex64
        )
    assert len(caught) == 3

    # Compare operation order, Prepared state, Total error, and status with
    # the formal implementation on deterministic cases at every size.
    comparison_cases.extend(
        [
            (nonnormalized, 2),
            (phased, 2),
            (sparse_short, 3),
        ]
    )
    for state, num_qubits in comparison_cases:
        reference = run_case(state, num_qubits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            formal = MottonenAlgorithm.Mottonen(
                np.asarray(state, dtype=np.complex128), num_qubits, 1e-6
            )
        formal_prepared = np.asarray(formal.evolution_result)[:, 0]
        formal_error = float(formal.total_error)
        formal_status = "ok" if formal_error <= max(1e-6, 1e-10) else "failed"
        formal_operations = MottonenAlgorithm._mottonen_operations(
            formal.Psi, num_qubits
        )
        assert reference["operations"] == formal_operations
        assert np.allclose(reference["Prepared state"], formal_prepared, atol=1e-14)
        assert abs(reference["Total error"] - formal_error) <= 1e-15
        assert reference["status"] == formal_status

    print(
        f"{case_count} deterministic cases passed for 1--5 qubits; "
        f"maximum phase-invariant error = {max_error:.3e}"
    )


if __name__ == "__main__":
    _run_tests()
