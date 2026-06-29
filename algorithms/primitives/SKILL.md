---
name: primitives
description: A collection of fundamental quantum computing primitives implemented using UnitaryLab, including basic gates, state preparation techniques, and measurement protocols. Provides efficient implementations and educational resources for understanding and utilizing quantum primitives in algorithm development.
---
# Quantum Primitives

## Purpose
This file routes requests for reusable quantum algorithm building blocks.

Use this category when the user asks about amplitude methods, phase estimation, Hadamard-based routines, or small foundational subroutines that are often embedded inside larger algorithms.

## Routing Rules

- If the user asks about Grover search, unstructured search, or finding a marked bit string:
  - Read `./grover/SKILL.md`
- If the user asks about general amplitude amplification, oracle amplification, or increasing marked-state probability from a custom state-preparation circuit:
  - Read `./amplitude-amplification/SKILL.md`
- If the user asks to estimate an unknown amplitude or success probability:
  - Read `./amplitude-estimation/SKILL.md`
- If the user asks to estimate eigenphases, eigenvalues from a unitary, or QPE as a subroutine:
  - Read `./quantum-phase-estimation/SKILL.md`
- If the user asks to estimate expectation values, overlaps, matrix elements, or controlled-unitary measurements:
  - Read `./hadamard-test/SKILL.md`
- If the user asks for the Hadamard layer, Walsh-Hadamard transform, or basis spreading over computational states:
  - Read `./hadamard-transform/SKILL.md`

## Available Leaf Skills

1. Grover Search: `./grover/SKILL.md`
2. Amplitude Amplification: `./amplitude-amplification/SKILL.md`
3. Amplitude Estimation: `./amplitude-estimation/SKILL.md`
4. Quantum Phase Estimation: `./quantum-phase-estimation/SKILL.md`
5. Hadamard Test: `./hadamard-test/SKILL.md`
6. Hadamard Transform: `./hadamard-transform/SKILL.md`

## Response Contract

1. Select the matching primitive from the routing rules.
2. Read the leaf skill before writing code or commands.
3. Keep this file as routing guidance; leave algorithm-specific implementation details to the leaf skill.
