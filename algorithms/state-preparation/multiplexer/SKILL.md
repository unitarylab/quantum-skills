---
name: multiplexer
description: Provides repository-faithful multiplexer state preparation through normalization, zero padding, one internal bit reversal, binary magnitude loading, basis-selective phase loading, and the actual run() contract.
---

# Multiplexer State Preparation

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab Multiplexer state-preparation algorithm.

Given a nonzero complex target vector and register width, the implementation normalizes and pads the target, performs one repository-specific bit reversal, recursively loads magnitudes with RY/CRY/MCRY gates, and loads complex phases with basis-state-selective phase gates.

Use this skill when you need to:

- Prepare a general small complex state with the repository's recursive binary probability-tree construction.
- Explain or inspect branch probabilities, traversal prefixes, mixed control states, basis-selective phase loading, or wire ordering.
- Diagnose amplitude-tree, control-value, phase, padding, endianness, or global-phase errors.
- Modify or independently implement the algorithm while preserving the formal gate schedule and return contract.

When using this skill:

- **Explanation:** Explain the recursive probability split, control-state convention, phase-loading stage, assumptions, and dense-simulation limits. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code that imports `MultiplexerAlgorithm` from `unitarylab_algorithms`. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented complex-state example first. Compare branch gates, traversal prefixes, phase gates, returned `"ok"` status, user-order state, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve normalization-before-padding, one internal bit reversal, prefix/control order, zero-mass handling, basis-state phase selection, output order, and return fields.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/multiplexer_implementation.py` as reference-only material. The latter reverses control-state and validation order differently and is not source-equivalent; the formal source at `unitarylab_algorithms/state_preparation/multiplexer/algorithm.py` is authoritative.
- **Validation:** Validate basis, ordering-sensitive, complex, zero-amplitude, zero-probability-branch, padded, and deterministic random states. Report norm error, phase-invariant Euclidean error, fidelity, dependency assumptions, and dense scale limits.

## Overview

1. Convert `Psi` to `complex128`, validate and normalize it, then append zeros to dimension `2**target_qubits` when shorter.
2. Bit-reverse the normalized padded vector once for the recursive repository schedule.
3. Build a depth-first binary tree of left/right probability splits and emit RY/CRY/MCRY gates for nonzero angles.
4. Iterate every internally ordered basis amplitude and emit a phase operation when `abs(phase_angle)>1e-15`.
5. Build the explicit UnitaryLab circuit and dense evolution from equivalent magnitude and phase schedules.
6. Return the dense matrix's first column in user-facing order and compute its global-phase-invariant error against padded `Psi`.

## Prerequisites

- Binary probability trees, controlled rotations with mixed control values, basis-state-selective phase gates, state-vector indexing, `numpy`, and UnitaryLab `Circuit`.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import MultiplexerAlgorithm

psi = np.array([1, 1j, 1, -1j], dtype=np.complex128) / 2
result = MultiplexerAlgorithm().run(
    Psi=psi,
    target_qubits=2,
    target_error=1e-6,
)
print(result["status"], result["Total error"])
assert result["status"] == "ok"
```

## Core Parameters Explained

| Parameter | Type | Default | Actual contract |
|---|---|---:|---|
| `Psi` | array-like | required | Non-empty, one-dimensional, finite, nonzero target; normalized before padding. |
| `target_qubits` | integer | required | Register width; use a non-boolean integer `>=1` for the end-to-end public path. |
| `target_error` | float | `1e-6` | Positive threshold; success requires `Total error<=max(target_error,1e-10)`. |
| `backend` | string | `'torch'` | Accepted but not used to select the explicit NumPy synthesis path. |
| `device` | string | `'cpu'` | Accepted but unused by this implementation. |
| `dtype` | dtype | `np.complex128` | Accepted, while target and dense calculations are explicitly `complex128`. |

`run()` applies `int(target_qubits)` and `float(target_error)` before result-object validation. Generated interfaces should still reject booleans and coercible non-integers rather than rely on truncation or parsing. `Psi` norm must exceed `1e-12`; `len(Psi)` must not exceed `2**target_qubits`; shorter input is trailing-zero padded with a `RuntimeWarning`; and `target_error` must be positive. Although the result validator accepts zero, the circuit builder calls `Circuit(0)`, which this repository rejects, so the usable public contract is `target_qubits>=1`. Exceptions propagate instead of returning a result dictionary.

## Return Fields

| Key | Type | Meaning |
|---|---|---|
| `status` | `str` | `"ok"` when the numerical threshold is met, otherwise `"failed"`. |
| `Prepared state` | `numpy.ndarray` | Dense emitted unitary's first column in normalized padded user order. |
| `Total error` | `float` | Global-phase-invariant Euclidean error using overlap tolerance `1e-12`. |
| `Computation time (s)` | `float` | Wall time rounded to four decimal places. |
| `circuit_path` | `str` | Saved SVG circuit path. |
| `plot` | `list[dict]` | Saved result-file descriptors from the base class. |
| `circuit` | `Circuit` | Flattened explicit multiplexer circuit. |

## Implementation Architecture

| Stage | Source symbol | Role |
|---|---|---|
| Normalize and pad | `_normalize_state_vector`, `StatePreparationResult.__post_init__` | Enforce target/error/dimension rules. |
| Reorder | `_bit_reversed_state_vector` | Adapt user indexing once for recursive wire order. |
| Build magnitude tree | `_multiplexer_gate_spec` | Produce depth-first split gates and branch control values. |
| Emit magnitude gates | `_apply_controlled_ry` | Select RY, CRY, or MCRY and omit negligible angles. |
| Emit phases | `_apply_basis_state_phase`, `_apply_controlled_phase` | Select one ordered basis state with mixed controls and apply its phase. |
| Build circuit | `_build_multiplexer_gate_circuit` | Emit explicit UnitaryLab primitives. |
| Build dense reference | `_build_multiplexer_dense_matrix` | Reproduce the same magnitude/phase schedule in matrix form. |
| Validate and return | `_state_vector_error`, `run`, `_build_return_dict` | Compare user-order state, save artifacts, package fields. |

Data flow: `Psi -> normalize/pad -> one bit reversal -> probability-tree gates + basis phases -> circuit/dense evolution -> user-order Prepared state -> phase-invariant error`.

## Understanding the Key Quantum Components

At recursion level `level`, the target is wire `level`; controls are wires `[0,...,level-1]`. The recursive `controls` tuple stores traversal branch bits in exactly that same order, so `control_values[q]` belongs to control wire `q`. Never reverse the tuple or derive it from little-endian integer bit extraction.

For a node with lower and upper subtree norms `left_norm` and `right_norm`, the split angle is `2*arctan2(right_norm,left_norm)`. If their sum is `<=1e-15`, the source stores zero; `_apply_controlled_ry` omits any gate whose absolute angle is `<=1e-15`. The depth-first specification may therefore contain zero-angle records that do not become circuit gates.

Phase loading iterates `np.angle(ordered_state)`. `_apply_basis_state_phase` formats `basis_index` as an `n`-bit most-significant-bit-first string, uses the last wire as phase target, uses preceding bits as ordered mixed control values, and temporarily flips the target when its selected bit is zero. Zero amplitudes have NumPy phase zero and need no gate.

The input target uses ordinary NumPy indexing; NumPy itself defines no quantum endianness. The sole bit reversal is a repository-specific adaptation for this recursive schedule. Both formal builders independently perform that conversion at their input boundary, and both outputs are already in user-facing order. Do not reverse `Prepared state` before validation.

## Theory-to-Code Mapping

| Theory concept | Exact source symbol |
|---|---|
| Target normalization and trailing padding | `_normalize_state_vector`, `StatePreparationResult.__post_init__` |
| Repository recursive ordering | `_bit_reversed_state_vector` |
| Binary probability tree | `_multiplexer_gate_spec` |
| RY/CRY/MCRY selection | `_apply_controlled_ry` |
| Basis-selective complex phase | `_apply_basis_state_phase`, `_apply_controlled_phase` |
| Dense controlled gate action | `_apply_controlled_single_qubit_matrix` |
| Dense basis phase | `_apply_basis_state_phase_dense` |
| Circuit and matrix outputs | `_build_multiplexer_gate_circuit`, `_build_multiplexer_dense_matrix` |
| Reported state error | `_state_vector_error` |

## Mathematical Deep Dive

For normalized ordered magnitudes `a`, let `p_i=a_i**2` and prefix sums `S_t=sum_{i<t}p_i`. At a node covering `[start,start+length)`, `half=length/2` and

`left_norm=sqrt(S[start+half]-S[start])`,

`right_norm=sqrt(S[start+length]-S[start+half])`,

`theta=2*arctan2(right_norm,left_norm)` when their sum exceeds `1e-15`, otherwise zero.

Magnitude gates alone prepare only `abs(ordered_state)`. Arbitrary complex preparation requires every non-negligible `angle(ordered_state[index])` to be applied to exactly that basis state.

For reference `u` and candidate `v`, the source computes `overlap=vdot(u,v)` and, if `abs(overlap)>1e-12`, replaces `v` by `v*conj(overlap/abs(overlap))`. It then reports `||u-v||`. The factor has unit modulus; do not use a component-wise or arbitrary complex ratio.

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
assert result["Total error"] <= max(1e-6, 1e-10)
assert norm_error <= 1e-12
assert fidelity >= 1 - 1e-12
assert phase_error <= 1e-12
```

Also test a computational-basis state, an ordering-sensitive `n=3` state, non-power-of-two input requiring padding, a non-equal-magnitude complex state, zero-amplitude and zero-probability branches, a deterministic random complex state, and invalid empty/non-1D/non-finite/zero/oversized inputs, `target_qubits=0`, and non-positive `target_error`.

## Minimal Manual Implementation

**Implementation level: Circuit-level construction from controlled gates.** This implementation does not import or call `MultiplexerAlgorithm`, a state-preparation routine, or any algorithm-library helper. It builds the binary probability tree directly, emits its mixed-control `RY/CRY/MCRY` gates, and then emits basis-selective `P/CP/MCP` phase gates using only UnitaryLab circuit primitives.

```python
import numpy as np
from unitarylab.core import Circuit


def _bit_reverse_state(state, num_qubits):
    """Perform the repository's one required amplitude-index conversion."""
    reordered = np.empty_like(state)
    for index, amplitude in enumerate(state):
        bits = format(index, f"0{num_qubits}b")
        reordered[int(bits[::-1], 2)] = amplitude
    return reordered


def _magnitude_gate_spec(amplitudes):
    """Create the depth-first binary probability-tree rotation schedule."""
    probabilities = np.asarray(amplitudes, dtype=np.float64) ** 2
    prefix_sum = np.concatenate(([0.0], np.cumsum(probabilities)))
    num_qubits = probabilities.size.bit_length() - 1
    gates = []

    def split(level, start, length, branch_bits):
        if level == num_qubits:
            return

        half = length // 2
        left_probability = prefix_sum[start + half] - prefix_sum[start]
        right_probability = (
            prefix_sum[start + length] - prefix_sum[start + half]
        )
        left_norm = np.sqrt(max(float(left_probability), 0.0))
        right_norm = np.sqrt(max(float(right_probability), 0.0))
        if left_norm + right_norm <= 1e-15:
            angle = 0.0
        else:
            angle = 2.0 * np.arctan2(right_norm, left_norm)

        gates.append({
            "angle": float(angle),
            "target": level,
            # branch_bits[q] controls wire q; do not reverse this tuple.
            "controls": list(range(level)),
            "control_values": list(branch_bits),
        })
        split(level + 1, start, half, branch_bits + (0,))
        split(level + 1, start + half, half, branch_bits + (1,))

    split(0, 0, probabilities.size, ())
    return gates


def _append_controlled_ry(circuit, angle, target, controls, values):
    """Choose the elementary rotation primitive for one probability-tree node."""
    if abs(angle) <= 1e-15:
        return
    if not controls:
        circuit.ry(angle, target)
    elif len(controls) == 1:
        circuit.cry(angle, controls[0], target, values)
    else:
        circuit.mcry(angle, controls, target, values)


def _append_controlled_phase(circuit, angle, target, controls, values):
    if not controls:
        circuit.p(angle, target)
    elif len(controls) == 1:
        circuit.cp(angle, controls[0], target, values)
    else:
        circuit.mcp(angle, controls, target, values)


def _append_basis_state_phase(circuit, basis_index, angle, num_qubits):
    """Apply ``exp(i*angle)`` to exactly one computational-basis state."""
    if abs(angle) <= 1e-15:
        return

    # format() gives the source schedule's most-significant-bit-first pattern.
    bits = [int(bit) for bit in format(basis_index, f"0{num_qubits}b")]
    target = num_qubits - 1
    controls = list(range(target))
    control_values = bits[:-1]

    # A phase primitive activates when its target is |1>. Temporarily flip a
    # selected |0> target, apply the mixed-control phase, and flip it back.
    target_is_zero = bits[-1] == 0
    if target_is_zero:
        circuit.x(target)
    _append_controlled_phase(
        circuit,
        angle,
        target,
        controls,
        control_values,
    )
    if target_is_zero:
        circuit.x(target)


def manual_multiplexer_circuit(Psi, target_qubits):
    """Prepare ``Psi`` with an explicit multiplexer gate construction."""
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
    padded = np.zeros(dimension, dtype=np.complex128)
    padded[:psi.size] = psi

    # The recursive schedule consumes exactly one bit-reversed copy.
    ordered = _bit_reverse_state(padded, target_qubits)
    magnitude_gates = _magnitude_gate_spec(np.abs(ordered))

    circuit = Circuit(
        target_qubits,
        name="Manual Multiplexer State Preparation",
    )
    for gate in magnitude_gates:
        _append_controlled_ry(
            circuit,
            gate["angle"],
            gate["target"],
            gate["controls"],
            gate["control_values"],
        )

    phase_records = []
    for basis_index, phase_angle in enumerate(np.angle(ordered)):
        phase_angle = float(phase_angle)
        if abs(phase_angle) > 1e-15:
            phase_records.append((basis_index, phase_angle))
        _append_basis_state_phase(
            circuit,
            basis_index,
            phase_angle,
            target_qubits,
        )

    return {
        "circuit": circuit,
        "magnitude_gates": magnitude_gates,
        "phase_records": phase_records,
    }


psi = np.array([1, 1j, 1, -1j], dtype=np.complex128) / 2
result = manual_multiplexer_circuit(psi, target_qubits=2)
circuit = result["circuit"]
```

The function returns the constructed circuit plus the magnitude-tree and phase records so the recursive schedule can be inspected. Dense simulation, error measurement, logging, timing, flattening, warning emission, and artifact saving are intentionally outside this minimal circuit-building core. Both magnitude and phase stages are required for arbitrary complex states.

## Debugging Tips

1. Check normalization-before-padding, `target_qubits>=1`, and the single internal bit reversal first.
2. Pass each depth-first traversal prefix directly to controls `[0,...,level-1]`; never reverse `control_values`.
3. Distinguish zero-mass tree records from emitted gates: zero angles remain in the specification but are skipped by circuit helpers.
4. If magnitudes are correct but phases are wrong, inspect basis-index bit strings, the last-wire target flip, and mixed phase controls.
5. Compare `Prepared state` directly with padded user-order `Psi` using a unit-modulus overlap phase; do not reverse or rescale either state.
6. Treat the dense matrix as a reconstruction of the same schedule, not an independent synthesis oracle.
