"""Variational state preparation with a fixed ordered Pauli-word sequence.

The target is normalized and padded, then one angle per Pauli word is fitted by
deterministic multi-start L-BFGS-B minimization of infidelity.  Each rotation is

``R_P(theta) = cos(theta/2) I - i sin(theta/2) P``.

Rotations are left-multiplied in listed order, so the first Pauli word acts
first on the all-zero state.  The returned state is ``evolution[:, 0]``.
Infidelity drives optimization.  Restart candidates use the prescribed
largest-target-component phase for ranking and early stopping; final success
uses a separately recomputed overlap-phase-invariant L2 error.
"""

from __future__ import annotations

import math
import time
import warnings
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from unitarylab.core import Circuit

try:
    from scipy.optimize import minimize
except ImportError as error:  # pragma: no cover - environment dependent
    minimize = None
    _SCIPY_IMPORT_ERROR: ImportError | None = error
else:
    _SCIPY_IMPORT_ERROR = None

try:
    from unitarylab.library.pauli_operator.pauli_string_decomposition import (
        pauli_state_preparation_circuit,
        pauli_string_to_matrix,
        state_preparation_pauli_words,
    )
except ImportError as error:  # pragma: no cover - environment dependent
    pauli_state_preparation_circuit = None
    pauli_string_to_matrix = None
    state_preparation_pauli_words = None
    _PAULI_HELPER_IMPORT_ERROR: ImportError | None = error
else:
    _PAULI_HELPER_IMPORT_ERROR = None


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def pauli_state_preparation(
    Psi: ArrayLike,
    target_qubits: int,
    *,
    target_error: float = 1e-6,
    backend: str = "torch",
    device: str = "cpu",
    dtype: type = np.complex128,
) -> dict[str, Any]:
    """Fit the fixed Pauli-word ansatz to ``Psi``.

    ``Psi`` must be a finite, nonempty one-dimensional vector with norm greater
    than ``1e-12``.  It is converted to ``complex128``, normalized, and padded
    with trailing zeros.  ``target_qubits`` is a non-boolean integer at least
    one, and ``target_error`` is a positive finite real number.

    ``backend``, ``device``, and ``dtype`` are compatibility parameters and do
    not affect the dense NumPy optimization.

    The return dictionary contains ``status``, ``Prepared state``,
    ``Total error``, ``Pauli words`` (the sequence length), ``Weights``,
    ``Computation time (s)``, and the UnitaryLab ``circuit``.
    """
    del backend, device, dtype
    _require_dependencies()
    start_time = time.time()

    num_qubits = _validate_integer(target_qubits, "target_qubits", minimum=1)
    error_threshold = _validate_positive_finite_real(
        target_error, "target_error"
    )
    target = _normalize_and_pad(Psi, num_qubits, tol=1e-12)

    pauli_words = _get_pauli_words(num_qubits)
    pauli_matrices = _cached_pauli_matrices(pauli_words, num_qubits)
    weights, _ = _find_weights_from_state(
        target, pauli_matrices, error_threshold
    )

    circuit = _build_pauli_circuit(weights, num_qubits)
    evolution = build_dense_pauli_state_preparation_matrix(
        weights, pauli_matrices
    )
    prepared_state = np.asarray(evolution[:, 0], dtype=np.complex128)
    total_error = _state_vector_error(target, prepared_state)

    return {
        "status": "ok"
        if total_error <= max(error_threshold, 1e-10)
        else "failed",
        "Prepared state": prepared_state,
        "Total error": float(total_error),
        "Pauli words": len(pauli_words),
        "Weights": np.asarray(weights, dtype=np.float64),
        "Computation time (s)": round(time.time() - start_time, 4),
        "circuit": circuit,
    }


def _require_dependencies() -> None:
    """Raise a clear error when required optimizer or Pauli helpers are absent."""
    _require_optimizer()
    _require_pauli_helpers()


def _require_optimizer() -> None:
    """Require SciPy's L-BFGS-B entry point."""
    if _SCIPY_IMPORT_ERROR is not None:
        raise ImportError("scipy.optimize.minimize is required.") from _SCIPY_IMPORT_ERROR


def _require_pauli_helpers() -> None:
    """Require the fixed-word, matrix, and circuit helper functions."""
    if _PAULI_HELPER_IMPORT_ERROR is not None:
        raise ImportError(
            "UnitaryLab must provide state_preparation_pauli_words, "
            "pauli_string_to_matrix, and pauli_state_preparation_circuit."
        ) from _PAULI_HELPER_IMPORT_ERROR


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
    """Validate a positive finite real value while rejecting booleans."""
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
    """Validate, normalize, and trailing-zero-pad a target state."""
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


def _get_pauli_words(num_qubits: int) -> tuple[str, ...]:
    """Return the ordered Pauli words supplied by UnitaryLab."""
    _require_pauli_helpers()
    num_qubits = _validate_integer(num_qubits, "num_qubits", minimum=1)
    words = tuple(state_preparation_pauli_words(num_qubits))
    if not words:
        raise ValueError("state_preparation_pauli_words returned no words.")
    for index, word in enumerate(words):
        if not isinstance(word, str) or len(word) != num_qubits:
            raise ValueError(
                f"Pauli word {index} must be a string of length {num_qubits}."
            )
        if any(letter not in "IXYZ" for letter in word):
            raise ValueError(f"Pauli word {index} contains an invalid letter.")
    return words


@lru_cache(maxsize=None)
def _cached_pauli_matrices(
    pauli_words: tuple[str, ...], num_qubits: int
) -> tuple[ComplexArray, ...]:
    """Cache dense matrices in the exact Pauli-word order."""
    _require_pauli_helpers()
    dimension = 1 << num_qubits
    matrices: list[ComplexArray] = []
    for index, word in enumerate(pauli_words):
        matrix = np.asarray(pauli_string_to_matrix(word), dtype=np.complex128)
        if matrix.shape != (dimension, dimension):
            raise ValueError(
                f"Pauli matrix {index} has shape {matrix.shape}; expected "
                f"({dimension}, {dimension})."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Pauli matrix {index} contains non-finite entries.")
        matrices.append(matrix)
    return tuple(matrices)


def _pauli_rotation_matrix(theta: float, pauli_matrix: ArrayLike) -> ComplexArray:
    """Return ``exp(-i*theta*P/2)`` for a dense Pauli matrix ``P``."""
    pauli_matrix = np.asarray(pauli_matrix, dtype=np.complex128)
    if (
        pauli_matrix.ndim != 2
        or pauli_matrix.shape[0] != pauli_matrix.shape[1]
        or not np.all(np.isfinite(pauli_matrix))
    ):
        raise ValueError("pauli_matrix must be a finite square matrix.")
    theta = float(theta)
    if not np.isfinite(theta):
        raise ValueError("theta must be finite.")
    dimension = pauli_matrix.shape[0]
    return (
        math.cos(theta / 2.0) * np.eye(dimension, dtype=np.complex128)
        - 1j * math.sin(theta / 2.0) * pauli_matrix
    )


def build_dense_pauli_state_preparation_matrix(
    weights: ArrayLike, pauli_matrices: tuple[ComplexArray, ...]
) -> ComplexArray:
    """Build the ordered dense Pauli-rotation product.

    Each update is ``unitary = rotation @ unitary``.  Therefore the first
    listed rotation acts first on ``|0...0>``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be a finite one-dimensional vector.")
    if len(weights) != len(pauli_matrices):
        raise ValueError("weights and pauli_matrices lengths must match.")
    if not pauli_matrices:
        raise ValueError("pauli_matrices must not be empty.")
    dimension = pauli_matrices[0].shape[0]
    unitary = np.eye(dimension, dtype=np.complex128)
    for angle, matrix in zip(weights, pauli_matrices):
        if matrix.shape != (dimension, dimension):
            raise ValueError("all Pauli matrices must have the same shape.")
        if abs(float(angle)) > 1e-15:
            unitary = _pauli_rotation_matrix(float(angle), matrix) @ unitary
    return np.asarray(unitary, dtype=np.complex128)


def _prepare_state(
    weights: ArrayLike, pauli_matrices: tuple[ComplexArray, ...]
) -> ComplexArray:
    """Apply the ordered Pauli rotations to the all-zero state."""
    evolution = build_dense_pauli_state_preparation_matrix(
        weights, pauli_matrices
    )
    return np.asarray(evolution[:, 0], dtype=np.complex128)


def _state_fidelity(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return ``abs(vdot(reference, candidate))**2``."""
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("states must be one-dimensional with matching shapes.")
    return float(abs(np.vdot(reference, candidate)) ** 2)


def _make_objective(
    target: ArrayLike,
    pauli_matrices: tuple[ComplexArray, ...],
) -> tuple[Callable[[RealArray], float], Callable[[RealArray], RealArray]]:
    """Build infidelity and its parameter-shift gradient."""
    target = np.asarray(target, dtype=np.complex128)
    parameter_count = len(pauli_matrices)
    fidelity_cache: dict[bytes, float] = {}

    def fidelity(weights: RealArray) -> float:
        contiguous = np.ascontiguousarray(weights, dtype=np.float64)
        key = contiguous.tobytes()
        if key not in fidelity_cache:
            fidelity_cache[key] = _state_fidelity(
                target, _prepare_state(contiguous, pauli_matrices)
            )
        return fidelity_cache[key]

    def loss(weights: RealArray) -> float:
        return 1.0 - fidelity(weights)

    def gradient(weights: RealArray) -> RealArray:
        weights = np.asarray(weights, dtype=np.float64)
        result = np.empty(parameter_count, dtype=np.float64)
        for index in range(parameter_count):
            plus = weights.copy()
            minus = weights.copy()
            plus[index] += math.pi / 2.0
            minus[index] -= math.pi / 2.0
            result[index] = -0.5 * (fidelity(plus) - fidelity(minus))
        return result

    return loss, gradient


def _initial_guesses(parameter_count: int) -> list[RealArray]:
    """Return the four deterministic optimizer starts."""
    parameter_count = _validate_integer(
        parameter_count, "parameter_count", minimum=1
    )
    rng = np.random.default_rng(7)
    guesses = [np.zeros(parameter_count, dtype=np.float64)]
    for scale in (0.25, 0.5, 1.0):
        guesses.append(
            rng.uniform(-math.pi, math.pi, size=parameter_count) * scale
        )
    return guesses


def _candidate_phase_error(target: ComplexArray, candidate: ComplexArray) -> float:
    """Rank a candidate using the target's largest-magnitude component."""
    pivot = int(np.argmax(np.abs(target)))
    phase = np.angle(target[pivot]) - np.angle(candidate[pivot])
    aligned = candidate * np.exp(1j * phase)
    return float(np.linalg.norm(target - aligned))


def _find_weights_from_state(
    target: ComplexArray,
    pauli_matrices: tuple[ComplexArray, ...],
    target_error: float,
) -> tuple[RealArray, float]:
    """Fit angles with up to four deterministic L-BFGS-B starts."""
    _require_optimizer()
    loss, gradient = _make_objective(target, pauli_matrices)
    best_weights: RealArray | None = None
    best_error = float("inf")

    for initial in _initial_guesses(len(pauli_matrices)):
        result = minimize(
            loss,
            initial,
            method="L-BFGS-B",
            jac=gradient,
            options={
                "maxiter": 800,
                "ftol": 1e-15,
                "gtol": 1e-10,
                "maxls": 50,
            },
        )
        candidate_weights = np.asarray(result.x, dtype=np.float64)
        candidate_state = _prepare_state(candidate_weights, pauli_matrices)
        candidate_error = _candidate_phase_error(target, candidate_state)
        if candidate_error < best_error:
            best_error = candidate_error
            best_weights = candidate_weights.copy()
        if candidate_error <= target_error:
            break

    if best_weights is None:
        best_weights = np.zeros(len(pauli_matrices), dtype=np.float64)
    return best_weights, float(best_error)


def _build_pauli_circuit(weights: RealArray, num_qubits: int) -> Circuit:
    """Build the UnitaryLab Pauli state-preparation circuit."""
    _require_pauli_helpers()
    circuit = pauli_state_preparation_circuit(
        np.asarray(weights, dtype=np.float64), num_qubits
    )
    if not isinstance(circuit, Circuit):
        raise TypeError("pauli_state_preparation_circuit must return Circuit.")
    return circuit


def _state_vector_error(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return overlap-phase-invariant L2 error without renormalizing."""
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
    """Run deterministic numerical and integration checks."""
    # These checks do not require the optional Pauli-word helpers.
    x_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    matrices = (x_matrix, z_matrix)
    weights = np.array([0.37, -0.29], dtype=np.float64)
    evolution = build_dense_pauli_state_preparation_matrix(weights, matrices)
    expected = (
        _pauli_rotation_matrix(weights[1], z_matrix)
        @ _pauli_rotation_matrix(weights[0], x_matrix)
    )
    assert np.allclose(evolution, expected, atol=1e-12)
    assert np.allclose(
        evolution.conj().T @ evolution, np.eye(2), atol=1e-12
    )
    prepared = _prepare_state(weights, matrices)
    assert np.allclose(prepared, evolution[:, 0])

    target = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    fidelity = _state_fidelity(target, prepared)
    assert abs(fidelity - abs(np.vdot(target, prepared)) ** 2) <= 1e-15
    total_error = _state_vector_error(target, prepared)
    overlap = np.vdot(target, prepared)
    aligned = prepared.copy()
    if abs(overlap) > 1e-12:
        aligned *= np.conj(overlap / abs(overlap))
    assert abs(total_error - np.linalg.norm(target - aligned)) <= 1e-15
    target_error = 1e-6
    status = "ok" if total_error <= max(target_error, 1e-10) else "failed"
    assert (status == "ok") == (
        total_error <= max(target_error, 1e-10)
    )

    loss, gradient = _make_objective(target, matrices)
    analytic = gradient(weights)
    finite_difference = np.empty_like(weights)
    epsilon = 1e-6
    for index in range(len(weights)):
        plus = weights.copy()
        minus = weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (loss(plus) - loss(minus)) / (2 * epsilon)
    assert np.allclose(analytic, finite_difference, atol=1e-6)

    if _SCIPY_IMPORT_ERROR is None:
        fitted_weights, candidate_error = _find_weights_from_state(
            target, matrices, target_error=1e-3
        )
        fitted_state = _prepare_state(fitted_weights, matrices)
        assert abs(
            candidate_error - _candidate_phase_error(target, fitted_state)
        ) <= 1e-12

    if _PAULI_HELPER_IMPORT_ERROR is not None:
        _assert_raises(
            ImportError,
            lambda: pauli_state_preparation([1.0, 0.0], 1),
        )
        print(
            "Pauli helper dependency mismatch detected; "
            "order/unitary/state/gradient/candidate checks passed."
        )
        return
    if _SCIPY_IMPORT_ERROR is not None:
        _assert_raises(
            ImportError,
            lambda: pauli_state_preparation([1.0, 0.0], 1),
        )
        print("SciPy dependency mismatch detected; dense checks passed.")
        return

    words_one = _get_pauli_words(1)
    words_two = _get_pauli_words(2)
    assert words_one == tuple(state_preparation_pauli_words(1))
    assert words_two == tuple(state_preparation_pauli_words(2))
    matrices_one = _cached_pauli_matrices(words_one, 1)
    assert matrices_one is _cached_pauli_matrices(words_one, 1)

    # A nonzero first angle alone must apply the first listed rotation.
    first_only = np.zeros(len(words_one), dtype=np.float64)
    first_only[0] = 0.41
    first_evolution = build_dense_pauli_state_preparation_matrix(
        first_only, matrices_one
    )
    assert np.allclose(
        first_evolution,
        _pauli_rotation_matrix(first_only[0], matrices_one[0]),
        atol=1e-12,
    )

    case_count = 0

    def run_case(
        state: ArrayLike, num_qubits: int, target_error: float
    ) -> dict[str, Any]:
        nonlocal case_count
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pauli_state_preparation(
                state,
                num_qubits,
                target_error=target_error,
            )
            padded = _normalize_and_pad(state, num_qubits, tol=1e-12)

        assert isinstance(result["circuit"], Circuit)
        assert len(result["Weights"]) == result["Pauli words"]
        assert result["Pauli words"] == len(
            state_preparation_pauli_words(num_qubits)
        )
        assert np.all(np.isfinite(result["Weights"]))
        words = _get_pauli_words(num_qubits)
        matrices = _cached_pauli_matrices(words, num_qubits)
        dense_evolution = build_dense_pauli_state_preparation_matrix(
            result["Weights"], matrices
        )
        assert np.allclose(
            result["Prepared state"], dense_evolution[:, 0], atol=1e-12
        )
        gram = np.einsum(
            "ki,kj->ij", dense_evolution.conj(), dense_evolution
        )
        assert np.allclose(
            gram, np.eye(1 << num_qubits, dtype=np.complex128), atol=2e-12
        )
        expected_fidelity = _state_fidelity(padded, result["Prepared state"])
        expected_infidelity = 1.0 - expected_fidelity
        assert np.isfinite(expected_fidelity)
        assert np.isfinite(expected_infidelity)
        expected_error = _state_vector_error(padded, result["Prepared state"])
        assert abs(result["Total error"] - expected_error) <= 1e-12
        expected_status = (
            "ok"
            if expected_error <= max(target_error, 1e-10)
            else "failed"
        )
        assert result["status"] == expected_status
        assert "Fidelity" not in result
        assert "Infidelity" not in result
        assert "Fit candidate error" not in result
        assert "evolution" not in result
        case_count += 1
        return result

    run_case(np.array([1.0, 0.0]), 1, 1e-6)
    run_case(np.array([1.0, 1.0j]), 1, 1e-4)
    run_case(np.array([1.0, 2.0j]), 2, 1e-3)
    run_case(np.array([1.0, 0.0, 0.0, 1.0j]), 2, 1e-2)
    rng = np.random.default_rng(20260715)
    run_case(
        rng.normal(size=4) + 1j * rng.normal(size=4),
        2,
        1e-1,
    )

    invalid_calls: list[tuple[type[BaseException], Any]] = [
        (ValueError, lambda: pauli_state_preparation([], 1)),
        (ValueError, lambda: pauli_state_preparation(np.eye(2), 1)),
        (ValueError, lambda: pauli_state_preparation([np.nan, 1], 1)),
        (ValueError, lambda: pauli_state_preparation([0, 0], 1)),
        (ValueError, lambda: pauli_state_preparation(np.ones(5), 2)),
        (ValueError, lambda: pauli_state_preparation([1], 0)),
        (TypeError, lambda: pauli_state_preparation([1], True)),
        (TypeError, lambda: pauli_state_preparation([1, 0], 1.0)),
        (TypeError, lambda: pauli_state_preparation([1, 0], 1, target_error=True)),
        (ValueError, lambda: pauli_state_preparation([1, 0], 1, target_error=0)),
    ]
    for expected_exception, function in invalid_calls:
        _assert_raises(expected_exception, function)

    print(f"{case_count} Pauli preparation integration cases passed.")


if __name__ == "__main__":
    _run_tests()
