---
name: state-preparation
description: Quantum state-preparation algorithms for loading target amplitude vectors into UnitaryLab circuits, including sparse Superposition, Möttönen decomposition, Matrix Product State preparation, recursive Multiplexer preparation, and variational Pauli-word preparation.
---

# Quantum State Preparation Algorithms

## Purpose
This file routes requests for loading target amplitude vectors into quantum circuits.

Use this category when the user asks to prepare a general complex state, exploit sparse support or tensor-network structure, use controlled-rotation decompositions, or optimize a trainable state-preparation circuit.

## Routing Rules

- If the user explicitly names Superposition, Möttönen, MPS, Multiplexer, or Pauli-word preparation:
  - Read the named leaf skill.
- If the user provides a general complex target state without a method or special structure:
  - Read `./mottonen/SKILL.md` as the default deterministic route.
- If the target has small computational-basis support or needs coefficient loading followed by a support permutation:
  - Read `./superposition/SKILL.md`.
- If the user supplies MPS tensors or a known low-bond-dimension representation:
  - Read `./mps/SKILL.md`.
- If the user asks for a recursive probability tree, controlled RY scheduling, or basis-selective phase loading:
  - Read `./multiplexer/SKILL.md`.
- If the user asks for trainable Pauli-word rotations, fidelity optimization, or variational preparation:
  - Read `./pauli/SKILL.md`.

## Available Leaf Skills

1. Sparse Superposition: `./superposition/SKILL.md`
2. Möttönen State Preparation: `./mottonen/SKILL.md`
3. Matrix Product State Preparation: `./mps/SKILL.md`
4. Multiplexer State Preparation: `./multiplexer/SKILL.md`
5. Pauli-Word State Preparation: `./pauli/SKILL.md`

## Response Contract

1. Identify the requested method or target structure and determine whether deterministic, approximate, tensor-network, or variational preparation is appropriate.
2. Default a general complex target with no special structure or method preference to `./mottonen/SKILL.md`.
3. Read the matching leaf skill before writing code or commands.
4. Keep method-specific APIs, ordering rules, validation tolerances, error metrics, and implementation details in the leaf skill.
