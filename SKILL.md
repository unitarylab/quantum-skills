---
name: quantum-skills
description: Root entrypoint for the quantum-skills package. Use this skill to route requests to the correct simulator or algorithm sub-skill, enforce full skill-chain reading before coding, and avoid duplicating implementation details that are already defined in leaf skills.
---

# Quantum Skills Root Entrypoint

## Purpose
This file is only the package-level router.

Use it to choose one domain index:

- `./algorithms/SKILL.md`
- `./simulators/SKILL.md`

Detailed routing, setup, examples, and implementation guidance belong in those domain indexes or their child skills.

## Read Chain
Before writing code or commands:

1. Read this file.
2. Read either `./algorithms/SKILL.md` or `./simulators/SKILL.md`.
3. Continue to the relevant child skill when needed.

Never code from this root file alone.

## Routing

- Algorithm design, primitives, complexity, or implementations:
  - Read `./algorithms/SKILL.md`
- Simulator choice, runtime setup, backend comparison, or execution environment:
  - Read `./simulators/SKILL.md`

## Package Map

- `./simulators/SKILL.md`
- `./algorithms/SKILL.md`

## Maintenance
When adding a new sub-skill:

1. Add it to the nearest domain or category index.
2. Keep this file limited to root-level routing.
