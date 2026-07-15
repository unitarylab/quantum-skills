---
name: mottonen
description: Prepares an arbitrary small complex quantum state using the Möttönen decomposition. This skill explains the amplitude and phase rotations, Gray-code uniformly controlled RY/RZ ladders, bit-order conversion, and phase-invariant validation used by UnitaryLab.
---

# Möttönen State Preparation

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the UnitaryLab Möttönen arbitrary complex state-preparation algorithm.

Given a nonzero complex amplitude vector `Psi` and a target register width, the implementation normalizes and pads the vector, computes hierarchical amplitude and phase rotations, and emits uniformly controlled RY/RZ ladders using the repository's Gray-code and wire-order conventions.

Use this skill when you need to:

- Prepare a general small complex state deterministically with `MottonenAlgorithm`.
- Explain or inspect Möttönen amplitude splits, phase reconstruction, Gray-code ladders, target/control mapping, or repository-specific bit reversal.
- Diagnose probability, phase, endianness, global-phase, padding, or angle-count errors in a Möttönen implementation.
- Modify or independently implement the algorithm while preserving the formal repository contract.

When using this skill:

- **Explanation:** Explain the algorithm, assumptions, mathematical model, ordering conventions, and exponential state-loading cost. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code that imports `MottonenAlgorithm` from `unitarylab_algorithms`. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented complex-state example first. Compare the observed state, returned `"ok"` status, error, fidelity, layer mappings, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture, algorithm contract, and theory-to-code mapping. Preserve input validation, parameter schema, one internal bit reversal, RY and conditional RZ stages, exact Gray-code schedule, output order, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and `scripts/mottonen_implementation.py` as reference-only material for troubleshooting, API comparison, and validation. The formal source at `unitarylab_algorithms/state_preparation/mottonen/algorithm.py` remains authoritative.
- **Validation:** Validate with small deterministic basis, ordering-sensitive, complex, zero-amplitude, padded, and random states. Report norm error, phase-invariant Euclidean error, fidelity, dependency assumptions, and dense-simulation scale limits.

## Overview

1. Convert `Psi` to `complex128`, validate and normalize it, then append zeros to dimension `2**target_qubits` when necessary.
2. Apply one internal bit reversal so the recursive decomposition matches the repository wire schedule.
3. For `k=n,...,1`, compute amplitude angles and emit uniformly controlled RY ladders.
4. If the source phase test is satisfied, compute phase angles for `k=n,...,1` and emit uniformly controlled RZ ladders.
5. Build the UnitaryLab circuit and dense evolution matrix from the same explicit operation list.
6. Return the matrix's first column in user-facing amplitude order and compute the global-phase-invariant error against the normalized padded target.

## Prerequisites

- Controlled RY/RZ rotations, reflected Gray code, state-vector indexing, `numpy`, and UnitaryLab `Circuit`.

## Reference Implementation Example

```python
import numpy as np
from unitarylab_algorithms import MottonenAlgorithm

psi = np.array([1, 1j, 1, -1j], dtype=np.complex128) / 2
result = MottonenAlgorithm().run(
    Psi=psi,
    target_qubits=2,
    target_error=1e-6,
)
print(result["status"], result["Total error"])
assert result["status"] == "ok"
```

## Core Parameters Explained

| Parameter | Type | Default | Meaning |
|---|---|---:|---|
| `Psi` | array-like | required | Target amplitudes; converted to a one-dimensional `complex128` vector. |
| `target_qubits` | integer | required | Target register width; use a non-boolean integer `>=1` for the end-to-end public path. |
| `target_error` | float | `1e-6` | Positive success tolerance; success requires `Total error <= max(target_error, 1e-10)`. |
| `backend` | string | `'torch'` | Accepted by `run()` but not used by the explicit NumPy circuit/matrix construction. |
| `device` | string | `'cpu'` | Accepted but not used by this implementation path. |
| `dtype` | dtype | `np.complex128` | Accepted but the target and dense references are explicitly converted to `complex128`. |

`run()` applies `int(target_qubits)` and `float(target_error)` before constructing the result object. Callers should not rely on permissive coercion of booleans, fractional values, or numeric strings; independent implementations must enforce the table's public contract.

## Input Validation and Boundary Behavior

- `Psi` must be non-empty, one-dimensional, finite, and have norm greater than the result tolerance `1e-12`; it is normalized before padding.
- `len(Psi)` must not exceed `2**target_qubits`. Shorter vectors are padded with trailing complex zeros and emit `RuntimeWarning`.
- The result object rejects booleans and non-integer `target_qubits` and rejects negative values. The public `run()` coercion occurs first, but the usable end-to-end contract is `target_qubits >=1` because `_build_mottonen_circuit` calls `Circuit(target_qubits)` and this repository rejects `Circuit(0)`.
- `target_error` and the internal tolerance must be positive.
- A zero-probability RY block uses `alpha_y=0` when its denominator is `<=1e-15`; no division is attempted.
- Validation and construction exceptions propagate; they are not converted into a return dictionary.

## Return Fields

| Key | Type | Meaning |
|---|---|---|
| `status` | `str` | `"ok"` when the numerical threshold is met, otherwise `"failed"`. |
| `Prepared state` | `numpy.ndarray` | `evolution_result[:,0]` in normalized, padded, user-facing amplitude-index order. |
| `Total error` | `float` | Source global-phase-invariant Euclidean error between padded `Psi` and `Prepared state`. |
| `Computation time (s)` | `float` | Wall time rounded to four decimal places. |
| `circuit_path` | `str` | Saved SVG circuit path. |
| `plot` | `list[dict]` | Saved result-file descriptors produced by the base class. |
| `circuit` | `Circuit` | Flattened emitted UnitaryLab circuit. |

## Implementation Architecture

| Stage | Source symbol | Role |
|---|---|---|
| Validate and pad | `_normalize_state_vector`, `StatePreparationResult.__post_init__` | Normalize `Psi`, enforce dimension/error rules, append zeros. |
| Reorder once | `_bit_reversed_state_vector` in `_mottonen_operations` | Adapt user indexing to the recursive repository schedule. |
| Compute angles | `_compute_alpha_y`, `_compute_alpha_z` | Produce one uniformly controlled angle vector per level. |
| Gray transform | `_gray_code`, `_compute_theta` | Convert desired branch angles to ladder rotation angles. |
| Schedule gates | `_uniform_rotation_operations`, `_mottonen_operations` | Emit RY/RZ and CX operations in exact order. |
| Build outputs | `_build_mottonen_circuit`, `_build_evolution_matrix` | Consume the same operation list for circuit and dense references. |
| Validate and return | `_phase_invariant_error`, `run`, `_build_return_dict` | Compute error, save artifacts, and return public fields. |

Data flow: `Psi -> normalize/pad -> one bit reversal -> alpha_y/alpha_z -> Gray ladders -> operations -> circuit/evolution -> user-order Prepared state -> phase-invariant error`.

## Algorithm Contract

For `n` qubits, set `wires_reversed=[n-1,...,0]`. Visit levels `k=n,...,1` for all RY ladders and, conditionally, all RZ ladders:

- `target = wires_reversed[k-1]`.
- `controls = wires_reversed[k:]` in that exact order.
- `len(controls)=n-k` and `len(alpha_y)=len(alpha_z)=2**len(controls)`.
- Reject a mismatched angle vector; never truncate, broadcast, zip-shortest, or silently discard angles.
- Use reflected Gray codes `g[i]=i^(i>>1)` for `i=0,...,2**len(controls)-1`.
- After ladder rotation `i`, compute `changed=g[i]^g[(i+1)%len(g)]`; its single set-bit position selects `controls[changed.bit_length()-1]` for the CX. Include the final cyclic transition.
- Every rotation acts on `target`; every CX uses the selected control and the same target.

For `n=3`, both axes use:

| `k` | `target` | ordered `controls` | required angle count |
|---:|---:|---|---:|
| 3 | 0 | `[]` | 1 |
| 2 | 1 | `[0]` | 2 |
| 1 | 2 | `[1,0]` | 4 |

The input and padded target use ordinary NumPy array indexing; NumPy itself specifies no quantum endianness. Bit reversal is a repository-specific requirement of this recursive decomposition. `_mottonen_operations` performs it exactly once before amplitude and phase extraction. The circuit/matrix builders do not reverse again, and `Prepared state` is already returned in user-facing order. Do not reverse either vector before validation.

## Understanding the Key Quantum Components

RY levels distribute probability mass between recursively paired halves. RZ levels reproduce relative phases. `_compute_theta` is the signed Walsh transform ordered by reflected Gray code, allowing adjacent branch configurations to differ in one control bit. The dense matrix is a reconstruction of the emitted operation schedule, not an independently synthesized oracle.

The entire RZ stage is entered only when the source condition `not np.allclose(phases, 0.0, atol=1e-15)` is true. Inside a ladder, a no-control rotation is omitted for `abs(angle)<=1e-15`, and a controlled rotation is emitted only for `abs(angle)>1e-15`; the controlled ladder's CX transitions remain scheduled.

## Theory-to-Code Mapping

| Theory concept | Exact source symbol |
|---|---|
| Normalized padded target | `_normalize_state_vector`, `StatePreparationResult.__post_init__` |
| Recursive ordering adaptation | `_bit_reversed_state_vector` |
| Amplitude split angles | `_compute_alpha_y` |
| Relative phase angles | `_compute_alpha_z` |
| Reflected Gray sequence | `_gray_code` |
| Signed ladder transform | `_compute_theta` |
| Uniformly controlled ladder | `_uniform_rotation_operations` |
| Complete RY then conditional RZ schedule | `_mottonen_operations` |
| User-order emitted state | `_build_evolution_matrix(... )[:,0]` |
| Reported error | `_phase_invariant_error` |

## Mathematical Deep Dive

For already bit-reversed amplitudes `a`, level `k`, full block size `2**k`, and half size `h=2**(k-1)`, define

`P1 = sum(abs(a[j*2**k+h : (j+1)*2**k])**2)` and `P = sum(abs(a[j*2**k : (j+1)*2**k])**2)`.

When `P>1e-15`, the source uses `alpha_y[j]=2*arcsin(sqrt(clip(P1/P,0,1)))`; otherwise it uses zero.

For `phi=np.angle(ordered_state)`, `m=2**(n-k)`, and `h=2**(k-1)`, the exact phase definition is

`alpha_z[j]=(1/h)*sum_{r=0}^{h-1}(phi[(2*j+1)*h+r]-phi[(2*j)*h+r])`, for `j=0,...,m-1`.

`np.angle` supplies principal phases; the source performs no additional unwrapping. Zero amplitudes therefore contribute the finite NumPy value `angle(0)=0`; do not divide by amplitude or invent a replacement phase.

For `c=len(controls)`, `g_i=i^(i>>1)`, and `N=2**c`, the ladder angle is

`theta[i]=(1/N)*sum_j (-1)**popcount(j & g_i) * alpha[j]`.

## Minimal Manual Implementation

**Implementation level: Circuit-level construction from elementary gates.** The implementation below does not import or call `MottonenAlgorithm`, a state-preparation routine, or any algorithm-library helper. It derives the Möttönen angles directly from the input amplitudes and phases, converts each uniformly controlled rotation with the reflected Gray-code transform, and appends only elementary `RY`, `RZ`, and `CX` gates to a UnitaryLab `Circuit`.

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


def _alpha_y(amplitudes, num_qubits, k):
    """Angles that distribute probability between the two halves of each block."""
    pair_count = 1 << (num_qubits - k)
    half = 1 << (k - 1)
    full = 1 << k
    alpha = np.zeros(pair_count, dtype=np.float64)

    for j in range(pair_count):
        start = j * full
        p1 = np.sum(amplitudes[start + half:start + full] ** 2)
        total = np.sum(amplitudes[start:start + full] ** 2)
        if total > 1e-15:
            alpha[j] = 2.0 * np.arcsin(
                np.sqrt(np.clip(p1 / total, 0.0, 1.0))
            )
    return alpha


def _alpha_z(phases, num_qubits, k):
    """Angles that reconstruct the relative phase of each paired block."""
    pair_count = 1 << (num_qubits - k)
    half = 1 << (k - 1)
    alpha = np.zeros(pair_count, dtype=np.float64)

    for j in range(pair_count):
        left = phases[(2 * j) * half:(2 * j + 1) * half]
        right = phases[(2 * j + 1) * half:(2 * j + 2) * half]
        alpha[j] = np.sum(right - left) / half
    return alpha


def _gray_code(num_bits):
    return [index ^ (index >> 1) for index in range(1 << num_bits)]


def _gray_angles(alpha):
    """Signed Walsh transform in reflected Gray-code order."""
    size = len(alpha)
    if size == 0 or size & (size - 1):
        raise ValueError("angle-vector length must be a non-zero power of two")

    theta = np.zeros(size, dtype=np.float64)
    gray = _gray_code(size.bit_length() - 1)
    for i, gray_word in enumerate(gray):
        for j, angle in enumerate(alpha):
            sign = -1.0 if (j & gray_word).bit_count() & 1 else 1.0
            theta[i] += sign * angle / size
    return theta


def _append_uniform_rotation(circuit, axis, alpha, controls, target):
    """Decompose one uniformly controlled rotation into rotations and CX gates."""
    expected = 1 << len(controls)
    if len(alpha) != expected:
        raise ValueError(f"expected {expected} angles, got {len(alpha)}")

    rotate = circuit.ry if axis == "ry" else circuit.rz
    if not controls:
        if abs(alpha[0]) > 1e-15:
            rotate(float(alpha[0]), target)
        return

    theta = _gray_angles(alpha)
    gray = _gray_code(len(controls))
    for i, angle in enumerate(theta):
        if abs(angle) > 1e-15:
            rotate(float(angle), target)

        # Adjacent reflected Gray words differ in exactly one bit. The final
        # cyclic transition is also required by the decomposition.
        changed = gray[i] ^ gray[(i + 1) % len(gray)]
        changed_bit = changed.bit_length() - 1
        circuit.cx(controls[changed_bit], target)


def manual_mottonen_circuit(Psi, target_qubits):
    """Prepare ``Psi`` using an explicit Möttönen RY/RZ/CX decomposition."""
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

    # Reverse amplitude indices once, then use the source implementation's
    # level-to-wire mapping: k=n,...,1 and wires=[n-1,...,0].
    ordered = _bit_reverse_state(padded, target_qubits)
    amplitudes = np.abs(ordered)
    phases = np.angle(ordered)
    wires = list(range(target_qubits - 1, -1, -1))
    circuit = Circuit(
        target_qubits,
        name="Manual Mottonen State Preparation",
    )

    # Amplitude preparation.
    for k in range(target_qubits, 0, -1):
        _append_uniform_rotation(
            circuit,
            "ry",
            _alpha_y(amplitudes, target_qubits, k),
            controls=wires[k:],
            target=wires[k - 1],
        )

    # Relative-phase preparation. A real non-negative target needs no RZ stage.
    if not np.allclose(phases, 0.0, atol=1e-15):
        for k in range(target_qubits, 0, -1):
            _append_uniform_rotation(
                circuit,
                "rz",
                _alpha_z(phases, target_qubits, k),
                controls=wires[k:],
                target=wires[k - 1],
            )

    return circuit


psi = np.array([1, 1j, 1, -1j], dtype=np.complex128) / 2
circuit = manual_mottonen_circuit(psi, target_qubits=2)
```

This example constructs the preparation circuit itself and returns the `Circuit`; simulation, error measurement, logging, and artifact saving are intentionally outside this minimal circuit-building core. Keep both the amplitude `RY` stage and the conditional phase `RZ` stage—an amplitude-only loader does not implement arbitrary complex-state preparation.

## Hands-On Example

The returned `Total error` uses `_phase_invariant_error`, which aligns only when `abs(overlap)>1e-12`. For additional normalized diagnostics, check both vectors independently and use only a unit-modulus phase factor:

```python
def validation_metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if reference.ndim != 1 or candidate.ndim != 1 or reference.size == 0:
        raise ValueError("states must be non-empty 1-D vectors")
    if reference.shape != candidate.shape:
        raise ValueError("state shapes must match")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("state entries must be finite")
    nr, nc = float(np.linalg.norm(reference)), float(np.linalg.norm(candidate))
    if nr <= 1e-15 or nc <= 1e-15:
        raise ValueError("states must have non-zero norm")
    norm_error = abs(nc - 1.0)
    reference_n, candidate_n = reference / nr, candidate / nc
    overlap = np.vdot(reference_n, candidate_n)
    aligned = candidate_n.copy()
    if abs(overlap) > 1e-15:
        aligned *= np.conj(overlap / abs(overlap))
    return {
        "norm_error": norm_error,
        "phase_invariant_error": float(np.linalg.norm(reference_n - aligned)),
        "fidelity": float(abs(overlap) ** 2),
    }
```

Never use a non-unit complex ratio to scale `candidate`. Test computational-basis states, an ordering-sensitive basis state, a non-equal-magnitude complex state, zero amplitudes and zero-probability blocks, trailing-zero padding, a deterministic random complex state, every `n=3` layer invariant, and invalid empty/non-1D/non-finite/zero-norm/oversized inputs, `target_qubits=0`, and non-positive `target_error`.

### Minimal result checks

```python
assert result["status"] == "ok"
assert result["Total error"] <= max(1e-6, 1e-10)
metrics = validation_metrics(psi, result["Prepared state"])
assert metrics["fidelity"] >= 1 - 1e-12
```

## Debugging Tips

1. Check normalization, padding, `target_qubits>=1`, and angle-array lengths first.
2. Wrong probabilities indicate an RY block, target/control mapping, Gray transform, or extra/missing bit reversal problem.
3. Correct probabilities with wrong complex amplitudes indicate the conditional RZ stage, phase indexing, or RZ sign.
4. Inspect `g[i]^g[i+1]` and its mapping into the ordered controls, including the final cyclic transition.
5. Compare user-order vectors with unit-modulus phase alignment; inspect norm error, phase-invariant error, and fidelity separately.
