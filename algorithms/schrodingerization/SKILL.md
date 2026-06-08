---
name: schrodingerization
description: Quantum PDE solvers using Schrodingerization, transforming non-unitary PDEs into unitary Schrodinger-type dynamics for quantum simulation. Currently covers advection and heat equations in 1D/2D.
---

# Schrodingerization - Quantum PDE Solvers

## Purpose
This file routes requests for PDE-to-quantum workflows using Schrodingerization.

Use this category when the user asks to transform non-unitary PDE dynamics into unitary Schrodinger-type dynamics, compare supported PDE examples, or run educational PDE solver demos.

## Routing Rules

- If the user asks about 1D advection or transport dynamics:
  - Read `./advection-schrodingerization/SKILL.md`
- If the user asks about the 1D heat equation, diffusion, or forward heat dynamics:
  - Read `./heat-1d-schrodingerization/SKILL.md`
- If the user asks about the 2D heat equation or spatial diffusion on a 2D grid:
  - Read `./heat-2d-schrodingerization/SKILL.md`

## Available Leaf Skills

1. 1D Advection: `./advection-schrodingerization/SKILL.md`
2. 1D Heat Equation: `./heat-1d-schrodingerization/SKILL.md`
3. 2D Heat Equation: `./heat-2d-schrodingerization/SKILL.md`

## Missing or External Topics

- If the user asks about backward heat, first check whether `./backward-heat-1d-schrodingerization/SKILL.md` exists before routing there.
- If no matching leaf skill exists, answer conceptually or ask whether a new leaf skill should be created.

## Response Contract

1. Match the PDE type and dimension to a leaf skill.
2. Read the leaf skill before writing code or commands.
3. Keep derivations and implementation details in the leaf skill.
