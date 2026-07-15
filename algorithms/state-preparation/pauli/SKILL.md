---
name: pauli
description: 使用固定 Pauli-word 旋转序列，通过保真度优化近似制备目标态，并用全局相位不变的 L2 误差验证结果。
---

# Pauli-Word State Preparation

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab Pauli-word state-preparation algorithm.

Given a target state, the implementation obtains the repository's fixed ordered Pauli-word ansatz, fits one rotation angle per word by deterministic multi-start L-BFGS-B optimization of infidelity, emits the matching Pauli-rotation circuit, and reports a global-phase-invariant state-vector error.

Use this skill when you need to:

- Approximate a target state with the repository's trainable Pauli-word rotation sequence.
- Explain or inspect the fixed ansatz, ordered matrix product, fidelity loss, parameter-shift gradient, or deterministic restart policy.
- Diagnose dependency, Pauli-order, gradient, optimizer, expressibility, convergence, global-phase, or return-field issues.
- Modify or independently implement the method while preserving the repository's Pauli-word source and optimization contract.

When using this skill:

- **Explanation:** Explain the variational ansatz, objective, gradient, convergence limitations, global-phase validation, and exponential dense-matrix cost. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code that imports `PauliAlgorithm` from `unitarylab_algorithms`. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the documented one-qubit example first. Verify the Pauli-word dependency, parameter count, objective/gradient agreement, returned state, `"ok"` or `"failed"` status, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the formal Pauli-word sequence, rotation product order, deterministic guesses, optimizer options, candidate selection, emitted circuit, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/pauli_implementation.py` as reference-only material. The latter generates its own Pauli words and uses different phase alignment, so it is not source-equivalent. The formal source at `unitarylab_algorithms/state_preparation/pauli/algorithm.py` is authoritative.
- **Validation:** Validate a small deterministic state with fidelity, infidelity, source `Total error`, finite weights, and gradient checks. Report dependency compatibility, optimizer convergence, backend assumptions, ansatz limitations, and scale limits.

## Overview

1. Convert `Psi` to `complex128`, validate and normalize it, then append zeros to dimension `2**target_qubits` when shorter.
2. Obtain the exact Pauli-word tuple from `state_preparation_pauli_words(target_qubits)` and cache each dense Pauli matrix.
3. Build the infidelity objective `1-|<Psi|prepared(weights)>|**2` and its parameter-shift gradient.
4. Run up to four deterministic L-BFGS-B starts and retain the candidate with the smallest phase-aligned L2 error, stopping early when it reaches `target_error`.
5. Build and flatten the formal Pauli state-preparation circuit, obtain its dense matrix, and extract the first column.
6. Compute the final overlap-phase-invariant error and return the best weights, state, circuit, and artifacts.

## Prerequisites

- Pauli strings, Pauli rotations, state fidelity, parameter-shift gradients, numerical optimization, `numpy`, `scipy`, and UnitaryLab `Circuit` plus its Pauli state-preparation helpers.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import PauliAlgorithm

psi = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
result = PauliAlgorithm().run(
    Psi=psi,
    target_qubits=1,
    target_error=1e-6,
)
print(result["status"], result["Pauli words"], result["Total error"])
assert result["status"] in {"ok", "failed"}
```

The current repository checkout must provide `state_preparation_pauli_words`, `pauli_string_to_matrix`, and `pauli_state_preparation_circuit` from `unitarylab.library.pauli_operator.pauli_string_decomposition`. If the installed/local UnitaryLab version lacks either state-preparation symbol, importing the formal algorithm fails before execution; report the dependency mismatch rather than replacing the word sequence heuristically.

## Core Parameters Explained

| Parameter | Type | Default | Actual contract |
|---|---|---:|---|
| `Psi` | array-like | required | Non-empty, one-dimensional, finite, nonzero target; normalized before padding. |
| `target_qubits` | integer | required | Register width; use a non-boolean integer `>=1` for the end-to-end public path. |
| `target_error` | float | `1e-6` | Positive phase-aligned L2 target and final success threshold. |
| `backend` | string | `'torch'` | Accepted but not used to choose the dense optimization path. |
| `device` | string | `'cpu'` | Accepted but unused by the current implementation. |
| `dtype` | dtype | `np.complex128` | Accepted while input and dense calculations use `complex128`. |

`run()` applies `int(target_qubits)` and `float(target_error)` before result validation; generated interfaces should reject booleans and coercible non-integers explicitly. `Psi` norm must exceed `1e-12`; `len(Psi)<=2**target_qubits`; shorter input is trailing-zero padded with `RuntimeWarning`; and `target_error>0`. Although the result validator accepts zero, the complete helper/circuit path is not a supported `Circuit(0)` workflow, so require `target_qubits>=1`. Validation, import, optimization, and circuit exceptions propagate rather than returning a dictionary.

The optimizer uses at most four starts: an all-zero vector followed by three vectors generated from `np.random.default_rng(7)` with scales `0.25`, `0.5`, and `1.0`. Each L-BFGS-B call uses `maxiter=800`, `ftol=1e-15`, `gtol=1e-10`, and `maxls=50`.

## Return Fields

| Key | Type | Meaning |
|---|---|---|
| `status` | `str` | `"ok"` if `Total error<=max(target_error,1e-10)`, otherwise `"failed"`. |
| `Prepared state` | `numpy.ndarray` | First column of the emitted flattened circuit matrix. |
| `Total error` | `float` | Final overlap-phase-invariant L2 error with tolerance `1e-12`. |
| `Pauli words` | `int` | Number of formal Pauli words, not the word tuple itself. |
| `Weights` | `numpy.ndarray` | Fitted angles in the exact formal Pauli-word order. |
| `Computation time (s)` | `float` | Wall time rounded to four decimal places. |
| `circuit_path` | `str` | Saved flattened-circuit SVG path. |
| `plot` | `list[dict]` | Saved result-file descriptors from the base class. |
| `circuit` | `Circuit` | Flattened emitted Pauli-rotation circuit. |

Numerical non-convergence normally returns the best candidate with `status="failed"`; validation, import, or execution failures raise and produce no result dictionary.

## Implementation Architecture

| Stage | Source symbol | Role |
|---|---|---|
| Normalize and pad | `_normalize_state_vector`, `StatePreparationResult.__post_init__` | Enforce target/error/dimension rules. |
| Obtain ansatz | `state_preparation_pauli_words`, `_cached_pauli_matrices` | Define exact words, order, and dense matrices. |
| Apply rotations | `_pauli_rotation_matrix`, `build_dense_pauli_state_preparation_matrix` | Build `exp(-i*theta*P/2)` and ordered unitary product. |
| Evaluate state | `_prepare_state`, `_state_fidelity` | Apply the dense sequence to `|0...0>`. |
| Optimize | `_make_objective`, `_initial_guesses`, `find_weights_from_state` | Fit angles with cached fidelity and shifted gradient. |
| Emit circuit | `pauli_state_preparation_circuit`, `Pauli._run` | Construct, flatten, and obtain the circuit matrix. |
| Validate and return | `_state_vector_error`, `run`, `_build_return_dict` | Compute final error, save artifacts, and package fields. |

Data flow: `Psi -> normalize/pad -> formal Pauli words/matrices -> fidelity objective + shifted gradient -> deterministic L-BFGS-B restarts -> best weights -> formal circuit matrix -> Prepared state/Total error -> return dictionary`.

## Understanding the Key Quantum Components

For each formal Pauli word `P`, the source uses `R_P(theta)=cos(theta/2)I-i*sin(theta/2)P`. Starting from identity, it updates `unitary=R_P(theta)@unitary`, so the first listed rotation acts first on `|0...0>` and the final matrix is the reversed left product of the listed operations. Angles with absolute value `<=1e-15` are skipped in the dense builder.

The optimization loss is infidelity, not returned state error. `_make_objective` returns `L=1-F`, where `F=|<target|prepared>|**2`. For each parameter, the gradient uses shifts `±pi/2` and returns `-0.5*(F_plus-F_minus)=dL/dtheta`.

`find_weights_from_state` ranks restart candidates with a unit-modulus phase determined from the target's largest-magnitude component. Final `Total error` is recomputed independently by `_state_vector_error` from the inner-product phase. Do not replace either with an arbitrary component ratio, and do not equate infidelity with L2 error.

## Theory-to-Code Mapping

| Theory concept | Exact source symbol |
|---|---|
| Formal ansatz words/order | `state_preparation_pauli_words` |
| Pauli word matrix | `pauli_string_to_matrix`, `_cached_pauli_matrices` |
| `exp(-i theta P/2)` | `_pauli_rotation_matrix` |
| Ordered dense product | `build_dense_pauli_state_preparation_matrix` |
| Prepared variational state | `_prepare_state` |
| Fidelity and infidelity | `_state_fidelity`, `_make_objective` |
| Parameter-shift loss gradient | `_make_objective.gradient` |
| Deterministic restarts | `_initial_guesses`, `find_weights_from_state` |
| Formal circuit emission | `pauli_state_preparation_circuit` |
| Returned phase-invariant error | `_state_vector_error` |

## Mathematical Deep Dive

Because `P**2=I`,

`exp(-i*theta*P/2)=cos(theta/2)I-i*sin(theta/2)P`.

For normalized target `u` and prepared state `v(theta)`,

`F(theta)=|vdot(u,v(theta))|**2`, and `L(theta)=1-F(theta)`.

With shift `s=pi/2`, the source computes

`dL/dtheta_i=-0.5*(F(theta_i+s)-F(theta_i-s))`.

The optimizer minimizes `L`, but candidate early stopping compares phase-aligned L2 error with `target_error`. Final success separately compares returned `Total error` with `max(target_error,1e-10)`. These quantities are related for nearby normalized states but are not interchangeable.

## Hands-On Example

```python
prepared = np.asarray(result["Prepared state"], dtype=np.complex128)
target = psi / np.linalg.norm(psi)
overlap = np.vdot(target, prepared)
aligned = prepared.copy()
if abs(overlap) > 1e-12:
    aligned *= np.conj(overlap / abs(overlap))

fidelity = float(abs(overlap) ** 2)
infidelity = 1.0 - fidelity
phase_error = float(np.linalg.norm(target - aligned))

assert np.all(np.isfinite(result["Weights"]))
assert np.isfinite(result["Total error"])
assert abs(result["Total error"] - phase_error) <= 1e-10
print("status:", result["status"])
print("fidelity:", fidelity, "infidelity:", infidelity)
print("phase-aligned L2 error:", phase_error)
```

Test at least basis states, the documented one-qubit complex state, non-equal-magnitude complex states, padding, zero amplitudes, deterministic targets, and invalid empty/non-1D/non-finite/zero/oversized inputs, `target_qubits=0`, and non-positive `target_error`. Numerically compare the analytic shifted gradient with a central finite-difference gradient on a tiny case. Accept either `"ok"` or `"failed"` for valid optimization runs unless the selected target and environment are verified to converge at the requested threshold.

## Minimal Manual Implementation

**Implementation level: Variational optimization plus elementary-gate construction.** This implementation does not import or call `PauliAlgorithm`, `state_preparation_pauli_words`, `pauli_string_to_matrix`, `pauli_state_preparation_circuit`, or any other algorithm-library helper. It generates the fixed Pauli-word ansatz recursively, fits its angles with the source's deterministic parameter-shift optimization, and decomposes every Pauli rotation into `H`, `P`, `CX`, `RZ`, and `S` circuit primitives.

```python
import numpy as np
from scipy.optimize import minimize
from unitarylab.core import Circuit


_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _state_preparation_words(num_qubits):
    """Reproduce the fixed PennyLane-style Pauli-word ordering."""
    if num_qubits == 1:
        return ("X", "Y")
    previous = _state_preparation_words(num_qubits - 1)
    identity_tail = "I" * (num_qubits - 1)
    return (
        ("X" + identity_tail, "Y" + identity_tail)
        + tuple("I" + word for word in previous)
        + tuple("X" + word for word in previous)
    )


def _pauli_word_matrix(word):
    """Map the leftmost character to wire 0, the least-significant qubit."""
    matrices = [_PAULI[letter] for letter in reversed(word)]
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return np.asarray(result, dtype=np.complex128)


def _prepared_state(weights, pauli_matrices):
    dimension = pauli_matrices[0].shape[0]
    state = np.zeros(dimension, dtype=np.complex128)
    state[0] = 1.0
    for angle, pauli_matrix in zip(weights, pauli_matrices):
        half = float(angle) / 2.0
        state = (
            np.cos(half) * state
            - 1j * np.sin(half) * (pauli_matrix @ state)
        )
    return state


def _phase_invariant_error(reference, candidate):
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 1e-12:
        candidate = candidate * np.conj(overlap / abs(overlap))
    return float(np.linalg.norm(reference - candidate))


def _fit_pauli_angles(target, pauli_matrices, target_error):
    """Fit the fixed ansatz with deterministic parameter-shift L-BFGS-B."""
    parameter_count = len(pauli_matrices)
    fidelity_cache = {}

    def fidelity(weights):
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        key = weights.tobytes()
        if key not in fidelity_cache:
            state = _prepared_state(weights, pauli_matrices)
            fidelity_cache[key] = float(abs(np.vdot(target, state)) ** 2)
        return fidelity_cache[key]

    def loss(weights):
        return 1.0 - fidelity(weights)

    def gradient(weights):
        weights = np.asarray(weights, dtype=np.float64)
        result = np.empty_like(weights)
        for index in range(parameter_count):
            plus = weights.copy()
            minus = weights.copy()
            plus[index] += np.pi / 2.0
            minus[index] -= np.pi / 2.0
            result[index] = -0.5 * (fidelity(plus) - fidelity(minus))
        return result

    rng = np.random.default_rng(7)
    initial_guesses = [np.zeros(parameter_count, dtype=np.float64)]
    for scale in (0.25, 0.5, 1.0):
        guess = rng.uniform(-np.pi, np.pi, size=parameter_count) * scale
        initial_guesses.append(guess)

    best_weights = initial_guesses[0]
    best_error = float("inf")
    for initial in initial_guesses:
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
        candidate = _prepared_state(candidate_weights, pauli_matrices)

        # Preserve the source's largest-target-component phase convention for
        # ranking restart candidates.
        pivot = int(np.argmax(np.abs(target)))
        phase = np.angle(target[pivot]) - np.angle(candidate[pivot])
        candidate_error = float(
            np.linalg.norm(target - candidate * np.exp(1j * phase))
        )
        if candidate_error < best_error:
            best_weights = candidate_weights
            best_error = candidate_error
        if candidate_error <= target_error:
            break

    return best_weights, best_error


def _append_pauli_rotation(circuit, angle, pauli_word):
    """Decompose exp(-i*angle*P/2) into elementary circuit gates."""
    # Change every active Pauli axis to Z.
    for wire, letter in enumerate(pauli_word):
        if letter == "X":
            circuit.h(wire)
        elif letter == "Y":
            circuit.p(-np.pi / 2.0, wire)  # S-dagger
            circuit.h(wire)

    active_wires = [
        wire for wire, letter in enumerate(pauli_word) if letter != "I"
    ]
    for index in range(len(active_wires) - 1):
        circuit.cx(active_wires[index], active_wires[index + 1])
    circuit.rz(float(angle), active_wires[-1])
    for index in reversed(range(len(active_wires) - 1)):
        circuit.cx(active_wires[index], active_wires[index + 1])

    # Undo the basis change.
    for wire, letter in reversed(list(enumerate(pauli_word))):
        if letter == "X":
            circuit.h(wire)
        elif letter == "Y":
            circuit.h(wire)
            circuit.s(wire)


def manual_pauli_circuit(Psi, target_qubits, *, target_error=1e-6):
    if isinstance(target_qubits, bool) or not isinstance(
        target_qubits, (int, np.integer)
    ):
        raise TypeError("target_qubits must be an integer")
    if target_qubits < 1:
        raise ValueError("target_qubits must be at least 1")
    if not np.isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be a positive finite number")

    psi = np.asarray(Psi, dtype=np.complex128)
    if psi.ndim != 1 or psi.size == 0:
        raise ValueError("Psi must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(psi)):
        raise ValueError("Psi entries must be finite")
    norm = float(np.linalg.norm(psi))
    if norm <= 1e-12:
        raise ValueError("Psi must not be the zero vector")
    psi = psi / norm

    dimension = 1 << target_qubits
    if psi.size > dimension:
        raise ValueError("Psi exceeds the target Hilbert-space dimension")
    target = np.zeros(dimension, dtype=np.complex128)
    target[:psi.size] = psi

    pauli_words = _state_preparation_words(target_qubits)
    pauli_matrices = tuple(
        _pauli_word_matrix(word) for word in pauli_words
    )
    weights, fit_error = _fit_pauli_angles(
        target,
        pauli_matrices,
        target_error,
    )

    circuit = Circuit(
        target_qubits,
        name="Manual Pauli State Preparation",
    )
    for angle, pauli_word in zip(weights, pauli_words):
        if abs(float(angle)) > 1e-15:
            _append_pauli_rotation(circuit, float(angle), pauli_word)

    prepared_state = _prepared_state(weights, pauli_matrices)
    total_error = _phase_invariant_error(target, prepared_state)
    fidelity = float(abs(np.vdot(target, prepared_state)) ** 2)

    return {
        "status": "ok" if total_error <= max(target_error, 1e-10) else "failed",
        "Prepared state": prepared_state,
        "Total error": float(total_error),
        "Fidelity": float(fidelity),
        "Infidelity": float(1.0 - fidelity),
        "Fit candidate error": float(fit_error),
        "Pauli words": tuple(pauli_words),
        "Weights": np.asarray(weights, dtype=np.float64),
        "circuit": circuit,
    }


psi = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
result = manual_pauli_circuit(psi, target_qubits=1)
circuit = result["circuit"]
```

The function returns the elementary-gate circuit, fitted weights, fixed Pauli-word sequence, prepared state, fidelity, and phase-invariant error. It omits only timing, logging, artifact saving, and circuit flattening—the circuit is already emitted without nested Pauli-rotation blocks. Because this method is variational, a valid run can still return `"failed"` when the fixed ansatz or optimizer does not reach `target_error`.

## Debugging Tips

1. Confirm that all three formal Pauli helper imports exist before diagnosing optimization or circuit behavior.
2. Check `len(weights)==len(state_preparation_pauli_words(target_qubits))` and preserve that exact order.
3. Compare shifted analytic gradients with finite differences before changing L-BFGS-B settings.
4. Diagnose fidelity, infidelity, candidate-selection L2 error, and final `Total error` as distinct quantities.
5. A valid run may return `"failed"` when the fixed ansatz or optimizer misses `target_error`; this is not the same as an exception.
6. Use only unit-modulus global-phase alignment and inspect the emitted circuit matrix before blaming optimization.
