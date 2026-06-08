---
name: quantum-skills
description: Root entrypoint for the quantum-skills package. Use this skill to route requests to the correct simulator or algorithm sub-skill, enforce full skill-chain reading before coding, and avoid duplicating implementation details that are already defined in leaf skills.
---

# Quantum Skills Root Entrypoint

## Purpose
This file is the package-level entrypoint.

It only chooses between the two top-level domains:

- `./algorithms/SKILL.md`
- `./simulators/SKILL.md`

Detailed routing, setup notes, examples, and implementation guidance belong in those domain indexes or their child skills.

## Mandatory Read Chain
Before producing code or execution commands, read skills in this order:

1. Read this file: `./SKILL.md`.
2. Read one domain index:
   - `./simulators/SKILL.md`, or
   - `./algorithms/SKILL.md`.
3. Continue from that domain index to the relevant child skill, if the task requires it.

Rule: Never skip to coding from this root file alone.

## Routing Rules
Use only these root-level routing rules:

- If user asks simulator choice, runtime environment, execution setup, or backend comparison:
  - Read `./simulators/SKILL.md` first.
- If user asks circuit primitive, algorithm design, complexity, or implementation:
  - Read `./algorithms/SKILL.md` first.

## Package Map
Root-level map of this skill package:

- `./simulators/SKILL.md`
- `./algorithms/SKILL.md`

Use each index file as the source of truth for its children.

## Execution Policy
Only provide environment installation commands when the user explicitly needs to run code.

For conceptual explanation, code review, pseudocode, or architecture discussion:
- Skip environment setup.
- Continue with theory and structure using the correct sub-skill.

For execution/debug tasks:
- Follow simulator selection from `./simulators/SKILL.md`.
- Follow exact installation and verification steps from the chosen simulator skill.

## Response Contract
When answering user requests with this package:

1. Identify the task type (simulator vs algorithm).
2. Open and use the matching domain index.
3. Follow that domain index to any required child skill.
4. Keep response focused on the user task; avoid repeating package index content.

## Maintenance Notes
When adding a new sub-skill:
1. Add it to the nearest domain or category index SKILL.
2. Keep this root file focused on routing only.
3. Avoid adding detailed setup or implementation sections here.
