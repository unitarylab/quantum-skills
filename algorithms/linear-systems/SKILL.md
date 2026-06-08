---
name: linear-systems
description: A set of quantum algorithms for solving linear systems of equations and related Fourier/signal-processing subroutines. This skill includes UnitaryLab implementations and educational resources for HHL, LCU, QFT, QSP, QSVT-QLSA, and VQLS.
---
# Quantum Linear Systems Algorithms

## Purpose
This file routes requests for quantum linear algebra algorithms and related Fourier or signal-processing subroutines.

Use this category when the user asks to solve linear systems, compare quantum linear-system solvers, build QFT/QSP components, or use QSVT-style linear algebra workflows.

## Routing Rules

- If the user asks for the HHL algorithm, phase-estimation-based linear solving, or textbook quantum linear systems:
  - Read `./hhl/SKILL.md`
- If the user asks about Linear Combination of Unitaries, block construction from weighted unitaries, or LCU-based solving:
  - Read `./lcu/SKILL.md`
- If the user asks about the Quantum Fourier Transform, inverse QFT, Fourier basis changes, or QFT as a subroutine:
  - Read `./quantum-fourier-transform/SKILL.md`
- If the user asks about Quantum Signal Processing, polynomial transformations, phase factors, or QSP circuits:
  - Read `./quantum-signal-processing/SKILL.md`
- If the user asks about QSVT-based quantum linear-system solving or singular-value transformation:
  - Read `./qsvt-qlsa/SKILL.md`
- If the user asks for a variational linear-system solver or NISQ-style Ax=b workflow:
  - Read `./vqls/SKILL.md`

## Available Leaf Skills

1. HHL Algorithm: `./hhl/SKILL.md`
2. LCU: `./lcu/SKILL.md`
3. Quantum Fourier Transform: `./quantum-fourier-transform/SKILL.md`
4. Quantum Signal Processing: `./quantum-signal-processing/SKILL.md`
5. QSVT QLSA: `./qsvt-qlsa/SKILL.md`
6. VQLS: `./vqls/SKILL.md`

## Response Contract

1. Decide whether the task is solving a linear system or using a supporting transform/subroutine.
2. Read the matching leaf skill before writing code or commands.
3. Keep solver-specific assumptions, parameters, and examples in the leaf skill.
