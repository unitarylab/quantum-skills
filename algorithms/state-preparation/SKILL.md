---
name: state-preparation
description: A router for UnitaryLab quantum state-preparation algorithms, including sparse-support, uniformly controlled-rotation, tensor-network, recursive multiplexer, and variational Pauli-word preparation. Use when selecting or comparing methods for loading target amplitude vectors into quantum circuits.
---

# Quantum State Preparation Algorithms

## Purpose

Route state-loading requests among the repository's sparse-support, uniformly controlled-rotation, tensor-network, recursive multiplexer, and variational Pauli-word implementations. This parent skill selects and compares methods; implementation details, formulas, wire-order rules, tolerances, and algorithm-specific validation belong to the selected leaf skill and its authoritative source.

## Routing Rules

Apply these rules in order of authority, not as a first-match list:

1. **Honor an explicit method request first.** If the user names Möttönen, Superposition, MPS, Multiplexer, or Pauli-word preparation, read that leaf even when another structural property also matches.
2. **Recognize supplied structured data next.** Explicit MPS tensors route to `./mps/SKILL.md`, unless the user explicitly selected another method.
3. **Use target structure and requirements to compare candidates.** Consider sparse computational-basis support, supplied or inferred bond dimension, deterministic versus variational preparation, permitted approximation/truncation error, available work qubits, and whether optimization is acceptable. Do not silently select the first applicable structured method.
4. **Default to Möttönen.** For a general complex target vector with no explicit method and no sufficiently identified sparse or low-bond-dimension structure, read `./mottonen/SKILL.md`. Treat it as the repository's default deterministic arbitrary-state route, without claiming an unproven complexity advantage.

When several candidates remain plausible, compare them and ask only for information that can change the choice, such as whether exact deterministic preparation is required, whether variational optimization is allowed, the available work-qubit budget, whether MPS tensors already exist, or whether amplitudes below a method-specific threshold may be discarded.

## Selection Guidance

| Situation or requirement | Route / comparison |
|---|---|
| User explicitly names a method | Read the named leaf; explicit choice overrides structural heuristics |
| General complex target, no special structure identified | `./mottonen/SKILL.md` (default deterministic route) |
| Small computational-basis support, or requested coefficient loading plus support permutation | `./superposition/SKILL.md`; the leaf states its threshold and truncation implications |
| MPS tensors supplied | `./mps/SKILL.md` first |
| Only “low entanglement” is stated | Compare MPS with Möttönen/Multiplexer using known bond dimension, work qubits, and allowed error; do not promise an efficient circuit automatically |
| Uniformly controlled RY/RZ, Gray-code ladders, arbitrary complex-state decomposition | `./mottonen/SKILL.md` |
| Recursive binary probability tree, RY/CRY/MCRY scheduling, basis-selective phase loading | `./multiplexer/SKILL.md`; this is not interchangeable with Möttönen merely because both use controlled rotations |
| Trainable Pauli-word rotations, fidelity-based optimization, variational preparation | `./pauli/SKILL.md`; normally optimization-dependent, not the default exact route |

The Superposition implementation currently materializes dense matrices, so sparse support does not by itself establish sparse gate-level or memory complexity. MPS suitability depends on the actual tensor/bond and work-wire constraints; low entanglement alone is not a guarantee. Pauli-word preparation uses a fixed ansatz and fidelity optimization; it should not be presented as a general deterministic state-preparation default.

## Available Leaf Skills

1. Sparse Superposition: `./superposition/SKILL.md`
2. Möttönen State Preparation: `./mottonen/SKILL.md`
3. Matrix Product State Preparation: `./mps/SKILL.md`
4. Multiplexer State Preparation: `./multiplexer/SKILL.md`
5. Pauli-Word State Preparation: `./pauli/SKILL.md`

All paths are relative to this file's directory and use the repository's lowercase directory names; preserve this casing for Linux environments. Before producing code or commands, read the final selected leaf skill and use its corresponding source as authoritative when the prose and source differ.

## Shared Validation Principles

Apply these rules before the selected leaf's algorithm-specific contract:

1. Treat the formal implementation under `unitarylab_algorithms/state_preparation/` as the sole authority for public parameters, normalization, padding, circuit construction, returned fields, tolerances, and failure behavior. Treat leaf `scripts/` files as reference material only unless they have been verified against that source.
2. Require `Psi` to be a non-empty, one-dimensional, finite complex vector with nonzero norm. Preserve each leaf's exact normalization, trailing-zero padding, sparse-threshold, truncation, and supplied-structure behavior.
3. Require a non-boolean integer `target_qubits >= 1` for the current end-to-end public algorithms. Their result containers may initially accept zero, but the current UnitaryLab `Circuit(0)` rejects zero qubits, so do not advertise a supported zero-qubit `run()` path. Require `len(Psi) <= 2**target_qubits` and a positive `target_error`.
4. The successful `run()` return dictionary uses `status == "ok"`; unsuccessful numerical completion uses `"failed"`. Do not confuse this public field with an algorithm object's internal status attribute. Validation and construction exceptions propagate instead of being converted into a result dictionary.
5. Validate target and prepared vectors in the order returned by the selected leaf. Apply bit reversal or another wire-order conversion only where that leaf's source explicitly places it; do not infer quantum endianness from NumPy array indexing and do not add a second conversion during validation.
6. When comparing state vectors up to global phase, normalize reference and candidate independently when the leaf's contract calls for normalized comparison, obtain the phase from `overlap = np.vdot(reference, candidate)`, and apply only the unit-modulus factor `np.conj(overlap / abs(overlap))` when the leaf's overlap tolerance is satisfied. Never scale a candidate by an arbitrary complex ratio.
7. Keep algorithm-specific metrics distinct. Möttönen and Multiplexer report phase-invariant state error; Superposition also has support-threshold effects; MPS separates zero-work projection error, work leakage, and possible bond truncation; Pauli optimizes infidelity but reports phase-aligned state error. Do not compare these values as though they were the same objective.
8. For a generated implementation or test, include computational-basis, ordering-sensitive, complex-amplitude, zero-amplitude, padding, invalid-input, and deterministic random cases where applicable. Add method-specific cases such as MPS bond/work-wire checks, Superposition support thresholds, Multiplexer zero-probability branches, Möttönen Gray-code layers, or Pauli optimization failure.

## Response Contract

1. State the selected method and why it matches the user's explicit request, target structure, exactness/approximation requirement, work-qubit budget, and optimization preference. If multiple methods are viable, compare them before selecting or ask a material clarifying question.
2. Read the selected leaf skill before generating code or commands. Do not reproduce its implementation details in this parent skill or invent capabilities absent from its source.
3. In comparisons, distinguish deterministic decompositions from variational optimization; identify auxiliary/work-qubit requirements, possible approximation, thresholding, leakage, or truncation error, whether the current implementation constructs dense matrices, and which target structure is actually exploited.
4. Preserve the selected leaf's public validation, normalization and padding behavior, qubit/wire order, repository-specific ordering conversions, matrix multiplication order, and returned-field/status contract.
5. Validate according to the selected leaf's exact error rule and tolerance. Where a leaf uses global-phase-invariant numerical state comparison, preserve that convention and use unit-modulus phase alignment only; do not imply that every algorithm optimizes or internally evaluates the same objective. In particular, do not conflate phase-aligned L2 validation with Möttönen/Multiplexer construction, Superposition support thresholding, MPS projection/leakage or truncation measures, or Pauli fidelity loss.
6. If a user supplies only a general complex target and no qualifying special structure or method preference, route to `./mottonen/SKILL.md` by default.
