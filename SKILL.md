---
name: quantum-skills
description: "Route quantum algorithm and simulator requests to the relevant sub-skills. Read this entrypoint before choosing a workflow or generating quantum code."
---

# Quantum Skills Root Entrypoint

## Purpose
This file is only the package-level router.

Use it to choose the relevant domain indexes:

- `./algorithms/SKILL.md`
- `./simulators/SKILL.md`

Detailed routing, setup, examples, and implementation guidance belong in those domain indexes or their child skills.

## Read Chain
Before writing code or commands:

1. Read this file.
2. Read `./algorithms/SKILL.md` for algorithm tasks and `./simulators/SKILL.md` whenever a task needs a simulator. Read both for algorithm implementations that use a simulator.
3. Continue to the relevant child skill when needed.

Never code from this root file alone.

## Routing

- Algorithm design, primitives, complexity, or implementations:
  - Read `./algorithms/SKILL.md`
- Simulator choice, runtime setup, backend comparison, or execution environment:
  - Read `./simulators/SKILL.md`

For tasks that need a simulator, follow the UnitaryLab-first selection rule in
`./simulators/SKILL.md` and read the selected simulator's child skill.
When the user requests UnitaryLab, read `./simulators/unitarylab/SKILL.md`.

## Package Map

- `./simulators/SKILL.md`
- `./algorithms/SKILL.md`

## Maintenance
When adding a new sub-skill:

1. Add it to the nearest domain or category index.
2. Keep this file limited to root-level routing.
