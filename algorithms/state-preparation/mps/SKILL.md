---
name: mps
description: Loads states with a Matrix Product State representation when low-entanglement structure can reduce the preparation cost. It covers state-to-MPS decomposition, bond-dimension truncation, canonicalization, QR-based unitary completion, work-qubit encoding, leakage, and phase-invariant validation in UnitaryLab.
---

# Matrix Product State Preparation

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab Matrix Product State (MPS) preparation algorithm.

Given a target vector or compatible MPS tensors, the implementation creates or validates a right-canonical tensor chain, embeds each local isometry into a QR-completed unitary, schedules those unitaries on system and work wires, and evaluates the all-zero work projection, leakage, and phase-invariant target error.

Use this skill when you need to:

- Prepare a state from supplied MPS tensors or from an automatically generated state-vector decomposition.
- Explore bond-dimension truncation, tensor shapes, right canonicalization, work-qubit encoding, or local isometry completion.
- Diagnose tensor-order, bond, work-wire, leakage, truncation, QR-completion, or output-state errors.
- Modify or independently implement the algorithm while preserving the repository's tensor and wire contracts.

When using this skill:

- **Explanation:** Explain the tensor representation, assumptions, canonical form, work-register embedding, approximation sources, and dense-simulation limits. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code that imports `MPSAlgorithm` from `unitarylab_algorithms`. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run a small product or GHZ-like state first. Compare MPS shapes, canonicalization, system/work wires, projection norm, leakage, returned `"ok"` status, and numerical error before changing code.
- **Modification or reimplementation:** Follow the implementation architecture, algorithm contract, and theory-to-code mapping. Preserve tensor ranks, right-to-left decomposition, bond padding, QR phase correction, wire order, zero-work projection, and return fields.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/mps_implementation.py` as reference-only material. The latter is an educational implementation and is not source-equivalent; the formal source at `unitarylab_algorithms/state_preparation/mps/algorithm.py` is authoritative.
- **Validation:** Validate product, GHZ, W, supplied-MPS, truncated, sparse-work-wire, padded, zero-amplitude, and deterministic random cases. Report projection norm, work leakage, conditional fidelity, phase-invariant error, and dependency/scale limitations separately.

## Overview

1. Convert `Psi` to `complex128`, validate and normalize it, then append zeros to dimension `2**target_qubits` when shorter.
2. Build a right-canonical MPS by a right-to-left SVD, optionally capping each retained bond, or validate caller-supplied tensors.
3. Determine required work qubits from the largest explicit power-of-two bond and establish non-overlapping system/work wires.
4. Convert each right-canonical tensor into zero-padded isometry columns and complete them to local unitaries with deterministic QR.
5. Build the circuit and full system-plus-work evolution using the same local unitaries and wire schedule.
6. Extract the unnormalized all-zero work projection, bit-reverse it once to user order, compute leakage and phase-invariant error, and expose a normalized target-space `Prepared state`.

## Prerequisites

- MPS tensor notation, SVD and QR, canonical forms, bond dimensions, state-vector indexing, `numpy`, and UnitaryLab `Circuit`.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import MPSAlgorithm

psi = np.zeros(8, dtype=np.complex128)
psi[[0, 7]] = 1 / np.sqrt(2)
result = MPSAlgorithm().run(
    Psi=psi,
    target_qubits=3,
    target_error=1e-6,
    mps_max_bond_dim=2,
    rng_seed=42,
)
print(result["status"], result["Work leakage"], result["Total error"])
assert result["status"] == "ok"
```

## Core Parameters Explained

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `Psi` | array-like | required | Target amplitudes; normalized and trailing-zero padded internally. |
| `target_qubits` | integer | required | Number of system qubits; use a non-boolean integer `>=1` for the end-to-end path. |
| `target_error` | float | `1e-6` | Positive threshold; success requires `Total error <= max(target_error,1e-10)`. |
| `mps` | `list[numpy.ndarray]` or `None` | `None` | Optional supplied tensor chain; otherwise generated from padded `Psi`. |
| `work_wires` | `list[int]` or `None` | `None` | Ordered non-negative unique auxiliary wires; must provide enough bond capacity. |
| `right_canonicalize` | bool | `False` | Right-canonicalize supplied tensors before synthesis. Auto-generated tensors are already right-canonical. |
| `mps_max_bond_dim` | integer or `None` | `None` | Positive power-of-two cap used only during automatic SVD decomposition. |
| `rng_seed` | integer | `42` | Deterministic random seed used in QR completion. |
| `backend`, `device`, `dtype` | backend settings | `'torch'`, `'cpu'`, `np.complex128` | Accepted by `run()` but not used by the explicit NumPy construction. |

`run()` applies `int()`/`float()` coercion before the result object performs stricter validation. Do not rely on coercion of booleans, fractions, or numeric strings in generated interfaces.

## Input Validation and Boundary Behavior

- `Psi` must be non-empty, one-dimensional, finite, and have norm greater than `1e-12`. It is normalized before trailing-zero padding.
- `len(Psi)` must not exceed `2**target_qubits`; a shorter vector emits `RuntimeWarning` and is padded.
- Use `target_qubits>=1`. Although the result validator only rejects negatives and contains a zero-qubit branch, that branch constructs `Circuit(0)`, which the current repository rejects.
- `target_error` must be positive. `mps_max_bond_dim`, when supplied, must be a positive power-of-two integer.
- For `target_qubits>0`, an MPS must be nonempty, contain exactly `target_qubits` tensors, satisfy the tensor-shape contract, have matching adjacent bonds, and use positive power-of-two explicit bonds.
- Supplied tensors are converted to `complex128` but are only shape/bond validated; they are not automatically normalized and the validator does not explicitly reject non-finite entries. Callers must provide finite tensors representing the intended normalized state.
- `work_wires` must be unique and non-negative, and its count must be at least `_required_work_qubits(mps)`. Sparse wire indices are allowed and determine circuit width.
- A near-zero all-zero-work projection causes `_complete_state_preparation_matrix` to raise instead of returning a result.
- Validation and construction exceptions propagate rather than producing a result dictionary.

## Return Fields

| Key | Type | Meaning |
|---|---|---|
| `status` | `str` | `"ok"` when the final threshold is met, otherwise `"failed"`. |
| `Prepared state` | `numpy.ndarray` | First column of a target-space unitary completed from the normalized zero-work projection. |
| `Total error` | `float` | Phase-invariant error between padded target and the unnormalized user-order zero-work projection. |
| `Work leakage` | `float` | `1-||zero_work_projection||**2`; not explicitly clipped to `[0,1]`. |
| `MPS tensors` | `int` | Number of tensors in the used MPS chain. |
| `Computation time (s)` | `float` | Wall time rounded to four decimal places. |
| `circuit_path` | `str` | Saved SVG path for the full system-plus-work circuit. |
| `plot` | `list[dict]` | Saved result-file descriptors from the base class. |
| `circuit` | `Circuit` | Flattened full system-plus-work circuit. |

`Prepared state` is normalized by target-space unitary completion, while `Total error` and `Work leakage` are computed from the unnormalized projection. Do not use `Prepared state` to reconstruct leakage.

## Implementation Architecture

| Stage | Source symbol | Role |
|---|---|---|
| Validate target | `_normalize_state_vector`, `StatePreparationResult.__post_init__` | Normalize, dimension-check, and pad `Psi`. |
| Build tensors | `state_vector_to_mps` | Right-to-left SVD with optional bond truncation. |
| Validate/canonicalize | `validate_mps_shape`, `right_canonicalize_mps` | Enforce shapes/bonds and prepare supplied tensors. |
| Determine wires | `_required_work_qubits`, `MPS.__init__` | Allocate ordered work wires and disjoint system wires. |
| Create unitaries | `mps_preparation_decomposition`, `_complete_columns_to_unitary` | Embed tensor slices and perform seeded QR completion. |
| Build outputs | `build_mps_circuit`, `_build_evolution_matrix` | Apply the same local unitaries to the same wires. |
| Project and validate | `_extract_zero_work_system_state`, `_bit_reversed_state_vector`, `_phase_invariant_error` | Compute leakage and user-order error. |
| Expose state | `_complete_state_preparation_matrix`, `run` | Build normalized target-space reference and package return fields. |

Data flow: `Psi/supplied MPS -> normalize/pad -> right-canonical tensors -> bond/work layout -> QR-completed local unitaries -> full circuit/evolution -> zero-work projection -> user-order error/leakage -> return dictionary`.

## Algorithm Contract

For `n>1`, use exactly `n` tensors:

| Position | Shape | Index order |
|---|---|---|
| First | `(2,chi_0)` | `(physical,bond_right)` |
| Interior | `(chi_left,2,chi_right)` | `(bond_left,physical,bond_right)` |
| Last | `(chi_left,2)` | `(bond_left,physical)` |

A one-site MPS is one `(2,2)` tensor; neither `(2,)` nor `(1,2,1)` is accepted. Adjacent explicit bonds must match and every explicit bond must be a positive power of two.

`state_vector_to_mps` sweeps from the last site to the first and returns right-canonical tensors. For an interior tensor `A`, the source checks rows of `A.reshape(chi_left,2*chi_right)` via `M @ M.conj().T = I`. Do not relabel this as left canonicalization or natural-reshape column orthogonality.

For synthesis, boundary ranks become `(1,2,chi)` and `(chi,2,1)`. Each left-bond slice has natural `(physical,bond_right)` order and is copied into a zero-padded vector with physical index as the local most-significant bit and bond/work index as lower bits. Do not introduce an unverified transpose.

`_required_work_qubits` returns `ceil(log2(max explicit bond))`, or zero for one tensor. When omitted, work wires are `[0,...,k-1]` and system wires follow. When supplied, their list order is significant; system wires are the first increasing non-work indices. Circuit width is `max(work_wires+system_wires)+1`, so sparse work-wire labels may create unused physical positions.

The full emitted state is projected onto all work wires equal to zero. This projection is not conditionally normalized before `Work leakage` or `Total error`. It is bit-reversed exactly once before comparison with padded `Psi`; the separate target-space `Prepared state` is completed from its normalized direction.

## Understanding the Key Quantum Components

Right canonicalization makes each tensor slice an isometry from the left bond into physical-plus-right-bond space. Zero padding embeds smaller bonds into a fixed binary work register. Seeded QR supplies orthogonal complement columns without changing the intended isometry columns when their orthonormality contract holds.

Work leakage measures probability outside the all-zero auxiliary subspace. Conditional fidelity describes the direction of the nonzero projection after normalization, while unconditioned overlap and `Total error` retain projection weight. High conditional fidelity must not hide high leakage.

## Theory-to-Code Mapping

| Theory concept | Exact source symbol |
|---|---|
| State-vector normalization/padding | `_normalize_state_vector`, `StatePreparationResult.__post_init__` |
| Right-to-left MPS decomposition | `state_vector_to_mps` |
| Tensor and bond contract | `validate_mps_shape` |
| Right canonicalization | `right_canonicalize_mps` |
| Work-qubit requirement | `_required_work_qubits` |
| Tensor-to-isometry embedding | `mps_preparation_decomposition` |
| Deterministic unitary completion | `_complete_columns_to_unitary`, `_qr_unitary` |
| Circuit scheduling | `build_mps_circuit` |
| Full dense evolution | `_build_evolution_matrix` |
| Zero-work projection | `_extract_zero_work_system_state` |
| User-order conversion | `_bit_reversed_state_vector` |
| Leakage and reported error | `MPS._run`, `_phase_invariant_error` |

## Mathematical Deep Dive

For a right-canonical interior tensor, the source invariant is

`sum_(physical,right) A[left,physical,right] * conj(A[left_prime,physical,right]) = delta[left,left_prime]`.

If `v` is the unnormalized all-zero-work projection, then

- `success_probability = real(vdot(v,v))`;
- `Work leakage = 1-success_probability`;
- conditional state `v_c=v/||v||` is diagnostic only;
- conditional fidelity is `abs(vdot(target,v_c))**2`;
- unconditioned overlap is `abs(vdot(target,v))**2`;
- `Total error` phase-aligns `v` without changing its norm.

At each truncated SVD cut, at most `mps_max_bond_dim` singular values are retained. The omitted squared singular values describe cut-local discarded weight, but the implementation does not return that quantity. Bond truncation, work leakage, QR/scheduling error, and final `Total error` are distinct.

For rectangular isometry columns `C`, QR completion must satisfy both `U.conj().T@U=I` and `U[:,:C.shape[1]]≈C`. The source absorbs each nonzero diagonal phase of `R` into the corresponding `Q` column to preserve this invariant.

## Minimal Manual Implementation

**Implementation level: Circuit-level construction from MPS tensors.** This implementation does not import or call `MPSAlgorithm`, a state-preparation routine, or any algorithm-library helper. It performs the right-to-left SVD, embeds every right-canonical tensor as isometry columns, completes each local isometry to a unitary with QR, and appends those local matrix gates directly to a UnitaryLab `Circuit`.

```python
import numpy as np
from unitarylab.core import Circuit


def _power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _complete_columns_to_unitary(columns, rng_seed):
    """Keep the isometry columns and construct their orthogonal complement."""
    columns = np.asarray(columns, dtype=np.complex128)
    rows, count = columns.shape
    if count > rows:
        raise ValueError("an isometry cannot have more columns than rows")
    if not np.allclose(
        columns.conj().T @ columns,
        np.eye(count),
        atol=1e-10,
    ):
        raise ValueError("MPS tensor does not define orthonormal columns")
    if count == rows:
        return columns

    rng = np.random.RandomState(rng_seed)
    filler = rng.random((rows, rows - count))
    filler = filler + 1j * rng.random((rows, rows - count))
    q_matrix, r_matrix = np.linalg.qr(np.hstack([columns, filler]))

    # Absorb QR diagonal phases so the leading columns remain exactly the
    # supplied MPS isometry, rather than phase-shifted versions of it.
    diagonal = np.diag(r_matrix)
    phase = np.ones(rows, dtype=np.complex128)
    nonzero = np.abs(diagonal) > 1e-14
    phase[nonzero] = diagonal[nonzero] / np.abs(diagonal[nonzero])
    return np.asarray(q_matrix * phase[np.newaxis, :], dtype=np.complex128)


def _state_vector_to_mps(
    state,
    num_qubits,
    max_bond_dim=None,
    rng_seed=42,
):
    """Use a right-to-left SVD sweep to create a right-canonical MPS."""
    if max_bond_dim is not None:
        if isinstance(max_bond_dim, bool) or not isinstance(
            max_bond_dim, (int, np.integer)
        ):
            raise TypeError("max_bond_dim must be an integer")
        if not _power_of_two(int(max_bond_dim)):
            raise ValueError("max_bond_dim must be a positive power of two")

    if num_qubits == 1:
        # The one-site source contract uses a complete (2, 2) tensor.
        return [
            _complete_columns_to_unitary(
                state.reshape(2, 1),
                rng_seed=rng_seed,
            )
        ]

    tensors = [None] * num_qubits
    right_bond = 1
    block = state.reshape(1 << (num_qubits - 1), 2)

    for site in range(num_qubits - 1, 0, -1):
        block = block.reshape(1 << site, 2 * right_bond)
        left, singular_values, right = np.linalg.svd(
            block,
            full_matrices=False,
        )
        bond = len(singular_values)
        if max_bond_dim is not None:
            bond = min(bond, int(max_bond_dim))

        left = left[:, :bond]
        singular_values = singular_values[:bond]
        right = right[:bond, :]
        if site == num_qubits - 1:
            tensors[site] = right.reshape(bond, 2)
        else:
            tensors[site] = right.reshape(bond, 2, right_bond)

        block = left @ np.diag(singular_values)
        right_bond = bond

    represented_norm = float(np.linalg.norm(block))
    if represented_norm <= 1e-12:
        raise ValueError("bond truncation removed all state weight")
    tensors[0] = block / represented_norm
    return [np.asarray(tensor, dtype=np.complex128) for tensor in tensors]


def _required_work_qubits(mps):
    if len(mps) == 1:
        return 0
    largest_bond = max(tensor.shape[-1] for tensor in mps[:-1])
    return int(np.ceil(np.log2(largest_bond)))


def _mps_local_unitaries(mps, work_qubits, rng_seed):
    """Embed tensor rows as columns of one system-plus-work unitary per site."""
    if len(mps) == 1:
        first_column = mps[0][:, 0].reshape(2, 1)
        return [_complete_columns_to_unitary(first_column, rng_seed)]

    tensors = [np.array(tensor, copy=True) for tensor in mps]
    tensors[0] = tensors[0].reshape(1, *tensors[0].shape)
    tensors[-1] = tensors[-1].reshape(*tensors[-1].shape, 1)

    local_dimension = 1 << (work_qubits + 1)
    physical_one_offset = 1 << work_qubits
    unitaries = []

    for tensor in tensors:
        columns = []
        for left_bond_slice in tensor:
            # local index = physical_bit * 2**work_qubits + right_bond
            column = np.zeros(local_dimension, dtype=np.complex128)
            right_bond = left_bond_slice.shape[1]
            column[:right_bond] = left_bond_slice[0]
            column[
                physical_one_offset:physical_one_offset + right_bond
            ] = left_bond_slice[1]
            columns.append(column)

        isometry = np.column_stack(columns)
        unitaries.append(
            _complete_columns_to_unitary(isometry, rng_seed)
        )
    return unitaries


def manual_mps_circuit(
    Psi,
    target_qubits,
    *,
    max_bond_dim=None,
    rng_seed=42,
):
    """Build an MPS preparation circuit without calling an algorithm library."""
    if isinstance(target_qubits, bool) or not isinstance(
        target_qubits, (int, np.integer)
    ):
        raise TypeError("target_qubits must be an integer")
    if target_qubits < 1:
        raise ValueError("target_qubits must be at least 1")

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

    mps = _state_vector_to_mps(
        target,
        target_qubits,
        max_bond_dim,
        rng_seed,
    )
    work_qubit_count = _required_work_qubits(mps)
    work_wires = list(range(work_qubit_count))
    system_wires = list(
        range(work_qubit_count, work_qubit_count + target_qubits)
    )
    unitaries = _mps_local_unitaries(
        mps,
        work_qubit_count,
        rng_seed,
    )

    circuit = Circuit(
        work_qubit_count + target_qubits,
        name="Manual MPS State Preparation",
    )
    for site, unitary in enumerate(unitaries):
        # UnitaryLab reverses this target list internally. Supplying work wires
        # first makes the local matrix use (system, work) big-endian ordering.
        circuit.unitary(unitary, work_wires + [system_wires[site]])

    return {
        "circuit": circuit,
        "mps": mps,
        "local_unitaries": unitaries,
        "system_wires": system_wires,
        "work_wires": work_wires,
    }


psi = np.zeros(8, dtype=np.complex128)
psi[[0, 7]] = 1 / np.sqrt(2)
result = manual_mps_circuit(psi, target_qubits=3, max_bond_dim=2)
circuit = result["circuit"]
```

The function returns the constructed circuit together with the intermediate MPS tensors, local unitaries, and wire layout so each synthesis stage can be inspected. Dense simulation, zero-work projection, leakage/error calculation, logging, artifact saving, caller-supplied tensors, and custom work-wire placement are intentionally outside this minimal automatic circuit-building core.

## Hands-On Example

For every successful case, report separately:

```python
target = np.asarray(target, dtype=np.complex128)
projection = np.asarray(zero_work_projection, dtype=np.complex128)
target /= np.linalg.norm(target)
projection_norm = float(np.linalg.norm(projection))
work_leakage = 1.0 - projection_norm**2
conditional = projection / projection_norm
overlap = np.vdot(target, conditional)
aligned = conditional.copy()
if abs(overlap) > 1e-12:
    aligned *= np.conj(overlap / abs(overlap))
conditional_fidelity = float(abs(overlap)**2)
conditional_error = float(np.linalg.norm(target - aligned))
```

Do not use a non-unit complex ratio. Also compare the source's unnormalized projection error to returned `Total error`. Test product, GHZ, W, random low-bond, truncated entangled, padded, supplied-MPS, one-site, sparse-work-wire, and deterministic random cases. Reject empty/non-finite/zero targets, `target_qubits=0`, invalid bond caps, malformed or mismatched tensors, non-power-of-two bonds, duplicate/negative/insufficient work wires, and near-zero work projections. For QR completion, assert preservation of the input isometry columns and full unitarity.

### Minimal result checks

```python
assert result["status"] == "ok"
assert result["Total error"] <= max(1e-6, 1e-10)
assert abs(result["Work leakage"]) <= 1e-10
assert result["MPS tensors"] == 3
```

## Debugging Tips

1. Check target normalization/padding, tensor ranks, adjacent bonds, and power-of-two dimensions first.
2. Check right-canonical row orthogonality in the exact source reshape; do not substitute a left-canonical convention.
3. Check work-wire count, order, sparse labels, system-wire disjointness, and local physical/work bit layout.
4. Verify QR completion preserves the supplied isometry columns as well as full unitarity.
5. Keep unnormalized projection error, work leakage, conditional fidelity, and bond truncation conceptually separate.
6. Inspect the single post-projection bit reversal before changing gate order or normalizing the projection.
