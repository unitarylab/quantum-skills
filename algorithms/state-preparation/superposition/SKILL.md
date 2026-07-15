---
name: superposition
description: Prepares a normalized state with sparse computational-basis support using compact coefficient preparation followed by a support permutation. The current implementation is exact up to floating-point error but materializes dense matrices.
---

# Sparse Superposition State Preparation

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab sparse Superposition state-preparation algorithm.

Given a normalized or normalizable target with computational-basis support above the source threshold, the implementation prepares its retained coefficients on a compact conceptual index register, embeds that stage into the full target space, and permutes prefix basis states onto the requested support.

Use this skill when you need to:

- Prepare or study a target with small computational-basis support.
- Explain or inspect support extraction, compact coefficient width, QR completion, support ordering, permutation orientation, or work-wire decomposition.
- Diagnose thresholding, coefficient order, permutation bijection, matrix-product order, padding, state-order, or global-phase errors.
- Modify or independently implement the algorithm while preserving the repository's support and permutation contracts.

When using this skill:

- **Explanation:** Explain the sparse-support idea, threshold behavior, coefficient and permutation stages, assumptions, and current dense-memory limits. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code that imports `SuperpositionAlgorithm` from `unitarylab_algorithms`. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented sparse complex state first. Compare retained support, ordered coefficients, permutation columns, `P @ Uc`, returned `"ok"` status, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve normalization-before-padding, strict support threshold, MSB-first basis tuples, coefficient reordering, permutation orientation, work-wire constraint, and return fields.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/superposition_implementation.py` as reference-only material. The latter builds `P_full` but multiplies a different identity variable and uses different tolerances, so it is not source-equivalent. The formal source at `unitarylab_algorithms/state_preparation/Superposition/algorithm.py` is authoritative.
- **Validation:** Validate basis, sparse complex, threshold-edge, padded, ordering-sensitive, zero-amplitude, and deterministic random sparse states. Report support size, index width, norm error, phase-invariant error, fidelity, work-wire assumptions, and dense scale limits.

## Overview

1. Convert `Psi` to `complex128`, validate and normalize it, then append zeros to dimension `2**target_qubits` when shorter.
2. Retain amplitudes satisfying the strict source predicate `abs(amplitude)>1e-12` and record their MSB-first computational-basis tuples.
3. Map support states onto the first `m` prefix basis states and reorder coefficients to match those prefixes.
4. Pad the coefficient vector to `2**ceil(log2(m))`, QR-complete it, and embed that local unitary into the full target Hilbert space.
5. Build a full bijective permutation and form the target-space dense unitary `permutation_stage @ coefficient_stage`; separately decompose the support mapping into a circuit using one non-overlapping work wire.
6. Return the target-space matrix's first column, retained support metadata, and global-phase-invariant error against normalized padded `Psi`.

## Prerequisites

- Sparse state vectors, computational-basis indexing, QR unitary completion, permutation matrices, mixed-control gates, `numpy`, and UnitaryLab `Circuit`.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import SuperpositionAlgorithm

psi = np.array([1, 0, 1j, 0], dtype=np.complex128) / np.sqrt(2)
result = SuperpositionAlgorithm().run(
    Psi=psi,
    target_qubits=2,
    target_error=1e-6,
)
print(result["status"], result["Support size"], result["Total error"])
assert result["status"] == "ok"
```

The source package directory is capitalized as `Superposition`; preserve this import casing on case-sensitive systems. The Skill directory remains lowercase.

## Core Parameters Explained

| Parameter | Type | Default | Actual contract |
|---|---|---:|---|
| `Psi` | array-like | required | Non-empty, one-dimensional, finite, nonzero target; normalized before padding. |
| `target_qubits` | integer | required | Target-state width; use a non-boolean integer `>=1` for the working end-to-end path. |
| `target_error` | float | `1e-6` | Positive final success threshold; it does not control sparse support extraction. |
| `backend` | string | `'torch'` | Accepted but not used to choose the dense construction path. |
| `device` | string | `'cpu'` | Accepted but unused by this implementation. |
| `dtype` | dtype | `np.complex128` | Accepted while input and dense references use `complex128`. |

`run()` applies `int(target_qubits)` and `float(target_error)` before result validation; generated interfaces should reject booleans and coercible non-integers explicitly. `Psi` norm must exceed `1e-12`; `len(Psi)<=2**target_qubits`; shorter vectors are trailing-zero padded with `RuntimeWarning`; and `target_error>0`. Use `target_qubits>=1`: the current zero-qubit path fails during coefficient/permutation matrix composition even though its initial validator accepts zero. Exceptions propagate rather than returning a dictionary.

Support is extracted only after normalization and padding with the fixed strict condition `abs(amplitude)>1e-12`. Amplitudes at or below that threshold are omitted regardless of `target_error`; because final error is measured against the original normalized padded target, omitted weight can make a valid run return `"failed"`.

## Return Fields

| Key | Type | Meaning |
|---|---|---|
| `status` | `str` | `"ok"` if `Total error<=max(target_error,1e-10)`, otherwise `"failed"`. |
| `Prepared state` | `numpy.ndarray` | First column of the `2**n` target-space dense unitary. |
| `Total error` | `float` | Global-phase-invariant L2 error with overlap tolerance `1e-12`. |
| `Support size` | `int` | Number of amplitudes retained by the strict support predicate. |
| `Index register qubits` | `int` | `ceil(log2(Support size))`, or zero for one retained term. |
| `Computation time (s)` | `float` | Wall time rounded to four decimal places. |
| `circuit_path` | `str` | Saved SVG path for the decomposed circuit. |
| `plot` | `list[dict]` | Saved result-file descriptors from the base class. |
| `circuit` | `Circuit` | Flattened circuit, normally on `target_qubits+1` wires because the default work wire is `target_qubits`. |

`Prepared state` is not extracted from the returned circuit matrix: it comes from `superposition_state_preparation_matrix`, a separate target-space dense construction. Do not assume its dimension equals the full returned circuit's Hilbert dimension.

## Implementation Architecture

| Stage | Source symbol | Role |
|---|---|---|
| Normalize and pad | `_normalize_state_vector`, `StatePreparationResult.__post_init__` | Enforce target/error/dimension rules. |
| Extract support | `_extract_sparse_superposition`, `_index_to_bits` | Apply `1e-12` threshold and build MSB-first basis tuples. |
| Order support | `order_states`, `_ordered_coefficients_for_prefix_basis` | Map retained states to prefix states and reorder coefficients. |
| Prepare coefficients | `_coefficient_register_qubits`, `_pad_coefficients`, `_build_coefficient_stage_matrix` | Build and embed the QR-completed coefficient stage. |
| Build permutation | `_build_prefix_to_support_permutation` | Complete prefix-to-support mapping to a full bijection. |
| Build dense state | `_build_superposition_unitary`, `superposition_state_preparation_matrix` | Form `P @ Uc` in target space. |
| Build circuit | `_permutation_operator`, `superposition_state_preparation_circuit` | Decompose support mapping with MCX/CX and a work wire. |
| Validate and return | `_state_vector_error`, `Superposition._run`, `_build_return_dict` | Compute error, metadata, artifacts, and public fields. |

Data flow: `Psi -> normalize/pad -> strict support -> prefix ordering -> compact coefficients -> QR coefficient stage -> prefix-to-support permutation -> P @ Uc / decomposed circuit -> Prepared state/error -> return dictionary`.

## Understanding the Key Quantum Components

For retained support `x_j` and ordered coefficients `c_j`, the coefficient stage prepares a prefix-index state and the permutation sends each prefix to the corresponding support basis state. `order_states` first preserves support states already inside the prefix range, then pairs support states outside it with still-unmapped prefixes. `_ordered_coefficients_for_prefix_basis` is required because support enumeration order does not always equal prefix order.

The permutation matrix uses rows as outputs and columns as inputs: `P[output_index,input_index]=1`, hence `P|input>=|output>`. The full target-space order is `P @ Uc`; reversing multiplication changes the prepared state. `_build_prefix_to_support_permutation` completes the partial mapping to a bijection and must produce one unit entry in every row and column.

The conceptual coefficient width is `r=ceil(log2(m))`, but the current implementation embeds its unitary into an `n`-qubit dense matrix and materializes a full `2**n by 2**n` permutation. Sparse support therefore does not imply sparse memory or gate complexity in this implementation.

The emitted circuit uses `work_wire=target_qubits` by default and creates `Circuit(max(target_qubits,work_wire+1))`, normally `n+1` wires. The work wire must be non-negative and must not overlap state wires (`work_wire>=num_qubits`).

## Theory-to-Code Mapping

| Theory concept | Exact source symbol |
|---|---|
| Normalization and trailing padding | `_normalize_state_vector`, `StatePreparationResult.__post_init__` |
| Strict sparse support | `_extract_sparse_superposition` |
| Integer/MSB-first basis conversion | `_index_to_bits`, `_bits_to_index` |
| Prefix mapping | `order_states` |
| Compact width and coefficient padding | `_coefficient_register_qubits`, `_pad_coefficients` |
| QR coefficient unitary | `_build_coefficient_stage_matrix` |
| Coefficient-prefix alignment | `_ordered_coefficients_for_prefix_basis` |
| Full support permutation | `_build_prefix_to_support_permutation` |
| Dense product `P @ Uc` | `_build_superposition_unitary` |
| Circuit permutation gates | `_permutation_operator`, `superposition_state_preparation_circuit` |
| Reported error | `_state_vector_error` |

## Mathematical Deep Dive

Let the normalized padded target be `psi` and retained support be `S={i:abs(psi_i)>1e-12}` with `m=|S|`. The conceptual coefficient width is `r=0` for `m=1`, otherwise `ceil(log2(m))`. Coefficients are zero-padded to dimension `2**r` after reordering to prefix indices.

The QR basis has the coefficient state as its first column. After QR, the source computes `overlap=vdot(coefficient_state,Q[:,0])` and, when `abs(overlap)>1e-12`, multiplies the first column by `conj(overlap/abs(overlap))`. This is a unit-modulus phase correction, not arbitrary scaling.

If thresholding removes nonzero weight, QR normalizes the retained coefficient direction while validation still uses the original padded target. Consequently, thresholding error is real and `target_error` cannot recover omitted amplitudes.

For target `u` and prepared `v`, `_state_vector_error` phase-aligns `v` with `conj(vdot(u,v)/abs(vdot(u,v)))` only when overlap exceeds `1e-12`, then reports `||u-v||`.

## Hands-On Example

```python
prepared = np.asarray(result["Prepared state"], dtype=np.complex128)
target = psi / np.linalg.norm(psi)
overlap = np.vdot(target, prepared)
aligned = prepared.copy()
if abs(overlap) > 1e-12:
    aligned *= np.conj(overlap / abs(overlap))

norm_error = abs(np.linalg.norm(prepared) - 1.0)
fidelity = float(abs(overlap) ** 2)
phase_error = float(np.linalg.norm(target - aligned))

assert result["status"] == "ok"
assert result["Support size"] == 2
assert result["Index register qubits"] == 1
assert result["Total error"] <= max(1e-6, 1e-10)
assert norm_error <= 1e-12
assert fidelity >= 1 - 1e-12
assert phase_error <= 1e-12
```

Also test a single basis state, support already inside and outside the prefix range, non-power-of-two support size, complex non-equal coefficients, input padding, threshold-edge amplitudes, deterministic random sparse states, permutation unitarity/bijection, and invalid empty/non-1D/non-finite/zero/oversized inputs, `target_qubits=0`, and non-positive `target_error`.

## Minimal Manual Implementation

**Implementation level: Sparse-support synthesis from circuit primitives.** This implementation does not import or call `SuperpositionAlgorithm`, a state-preparation routine, or any algorithm-library helper. It extracts the sparse support, constructs the compact coefficient unitary with QR, completes the prefix-to-support permutation, and emits the circuit directly with a coefficient matrix gate plus `MCX/CX` permutation gates.

```python
import numpy as np
from unitarylab.core import Circuit


def _index_to_bits(index, num_qubits):
    return tuple(int(bit) for bit in format(index, f"0{num_qubits}b"))


def _bits_to_index(bits):
    return int("".join(str(bit) for bit in bits), 2)


def _extract_sparse_support(state, threshold=1e-12):
    """Apply the source's strict post-normalization support predicate."""
    num_qubits = state.size.bit_length() - 1
    support_indices = [
        index
        for index, amplitude in enumerate(state)
        if abs(amplitude) > threshold
    ]
    if not support_indices:
        raise ValueError("state must contain at least one retained amplitude")
    coefficients = np.asarray(
        [state[index] for index in support_indices],
        dtype=np.complex128,
    )
    basis_states = [
        _index_to_bits(index, num_qubits) for index in support_indices
    ]
    return coefficients, basis_states


def _order_states(basis_states):
    """Assign every retained support state to one prefix state |0>,...,|m-1>."""
    normalized = [tuple(int(bit) for bit in state) for state in basis_states]
    if not normalized:
        return {}
    if len(set(normalized)) != len(normalized):
        raise ValueError("basis states must be unique")
    if any(bit not in (0, 1) for state in normalized for bit in state):
        raise ValueError("basis-state entries must be binary")

    support_size = len(normalized)
    num_qubits = len(normalized[0])
    state_map = {}
    outside_prefix = []
    unused_prefixes = {index: None for index in range(support_size)}

    # Preserve support states already located inside the prefix range.
    for state in normalized:
        state_index = _bits_to_index(state)
        if state_index < support_size:
            state_map[state] = state
            unused_prefixes.pop(state_index, None)
        else:
            outside_prefix.append(state)

    # Pair remaining support states with remaining prefixes in insertion order.
    for state, prefix_index in zip(outside_prefix, unused_prefixes):
        state_map[state] = _index_to_bits(prefix_index, num_qubits)
    return state_map


def _ordered_coefficients(coefficients, basis_states, state_map):
    ordered = np.zeros_like(coefficients)
    for coefficient, state in zip(coefficients, basis_states):
        prefix_index = _bits_to_index(state_map[state])
        ordered[prefix_index] = coefficient
    return ordered


def _coefficient_stage(ordered_coefficients, num_qubits):
    """QR-complete the compact coefficient state and embed it into n qubits."""
    support_size = ordered_coefficients.size
    register_qubits = (
        int(np.ceil(np.log2(support_size))) if support_size > 1 else 0
    )
    full_dimension = 1 << num_qubits
    if register_qubits == 0:
        return np.eye(full_dimension, dtype=np.complex128), 0

    local_dimension = 1 << register_qubits
    coefficient_state = np.zeros(local_dimension, dtype=np.complex128)
    coefficient_state[:support_size] = ordered_coefficients

    qr_input = np.eye(local_dimension, dtype=np.complex128)
    qr_input[:, 0] = coefficient_state
    local_unitary, _ = np.linalg.qr(qr_input)

    # Restore the intended first-column phase after QR.
    overlap = np.vdot(coefficient_state, local_unitary[:, 0])
    if abs(overlap) > 1e-12:
        local_unitary[:, 0] *= np.conj(overlap / abs(overlap))

    remaining_qubits = num_qubits - register_qubits
    embedded = np.kron(
        np.eye(1 << remaining_qubits, dtype=np.complex128),
        local_unitary,
    )
    return np.asarray(embedded, dtype=np.complex128), register_qubits


def _prefix_to_support_permutation(basis_states, state_map):
    """Build P with P[output,input]=1 so P|prefix>=|support>."""
    num_qubits = len(basis_states[0])
    dimension = 1 << num_qubits
    support_size = len(basis_states)
    inverse_map = {prefix: support for support, prefix in state_map.items()}
    permutation = list(range(dimension))

    for prefix_index in range(support_size):
        prefix_state = _index_to_bits(prefix_index, num_qubits)
        support_state = inverse_map[prefix_state]
        permutation[prefix_index] = _bits_to_index(support_state)

    # Complete every prefix/support exchange to a full bijection.
    for support_state in basis_states:
        support_index = _bits_to_index(support_state)
        if support_index >= support_size:
            permutation[support_index] = _bits_to_index(
                state_map[support_state]
            )

    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for input_index, output_index in enumerate(permutation):
        matrix[output_index, input_index] = 1.0
    if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension)):
        raise ValueError("prefix-to-support mapping is not a permutation")
    return matrix


def _append_support_exchange(
    circuit,
    prefix_state,
    support_state,
    system_wires,
    work_wire,
):
    """Exchange one prefix/support pair using a clean work qubit."""
    circuit.mcx(
        system_wires,
        work_wire,
        control_state=list(prefix_state),
    )
    for wire, prefix_bit, support_bit in zip(
        system_wires,
        prefix_state,
        support_state,
    ):
        if prefix_bit != support_bit:
            circuit.cx(work_wire, wire)
    circuit.mcx(
        system_wires,
        work_wire,
        control_state=list(support_state),
    )


def _phase_invariant_error(reference, candidate):
    overlap = np.vdot(reference, candidate)
    if abs(overlap) > 1e-12:
        candidate = candidate * np.conj(overlap / abs(overlap))
    return float(np.linalg.norm(reference - candidate))


def manual_superposition_circuit(Psi, target_qubits, *, target_error=1e-6):
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

    coefficients, basis_states = _extract_sparse_support(target)
    state_map = _order_states(basis_states)
    ordered_coefficients = _ordered_coefficients(
        coefficients,
        basis_states,
        state_map,
    )
    coefficient_stage, index_qubits = _coefficient_stage(
        ordered_coefficients,
        target_qubits,
    )
    permutation_stage = _prefix_to_support_permutation(
        basis_states,
        state_map,
    )
    evolution = np.asarray(
        permutation_stage @ coefficient_stage,
        dtype=np.complex128,
    )

    system_wires = list(range(target_qubits))
    work_wire = target_qubits
    circuit = Circuit(
        target_qubits + 1,
        name="Manual Superposition State Preparation",
    )
    if index_qubits > 0:
        circuit.unitary(coefficient_stage, system_wires)

    # order_states maps support -> prefix, while the circuit must apply
    # prefix -> support.
    for support_state, prefix_state in state_map.items():
        if support_state != prefix_state:
            _append_support_exchange(
                circuit,
                prefix_state,
                support_state,
                system_wires,
                work_wire,
            )

    prepared_state = np.asarray(evolution[:, 0], dtype=np.complex128)
    total_error = _phase_invariant_error(target, prepared_state)

    return {
        "status": "ok" if total_error <= max(target_error, 1e-10) else "failed",
        "Prepared state": prepared_state,
        "Total error": float(total_error),
        "Support size": int(len(coefficients)),
        "Index register qubits": int(index_qubits),
        "coefficients": coefficients,
        "basis_states": tuple(basis_states),
        "ordered_coefficients": ordered_coefficients,
        "coefficient_stage": coefficient_stage,
        "permutation_stage": permutation_stage,
        "evolution": evolution,
        "circuit": circuit,
    }


psi = np.array([1, 0, 1j, 0], dtype=np.complex128) / np.sqrt(2)
result = manual_superposition_circuit(psi, target_qubits=2)
circuit = result["circuit"]
```

The function returns the circuit, target-space evolution, support metadata, coefficient/permutation stages, prepared state, and error. It omits only logging, timing, flattening, warning emission, and artifact saving. The returned circuit has one extra work wire, while `Prepared state` and `evolution` remain in the `target_qubits` state space, matching the formal implementation's contract.

## Debugging Tips

1. Check normalization-before-padding and the strict `abs(amplitude)>1e-12` support predicate first.
2. Verify each retained coefficient is moved to the prefix assigned by `order_states`; do not assume support enumeration already has prefix order.
3. Assert the permutation has one `1` per row and column and maps each prefix column to its support row.
4. Preserve `permutation_stage @ coefficient_stage`; reversing the product prepares a different state.
5. Distinguish the `2**n` target-space `Prepared state` from the normally `n+1`-wire returned circuit.
6. Use unit-modulus overlap phase alignment, and report thresholding error separately from numerical QR or permutation error.
