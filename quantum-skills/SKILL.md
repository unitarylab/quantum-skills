---
name: quantum-skills
description: Root entrypoint for the quantum-skills package. Use this skill to route requests to the correct simulator or algorithm sub-skill, enforce full skill-chain reading before coding, and avoid duplicating implementation details that are already defined in leaf skills.
---

# Quantum Skills Root Entrypoint

## Purpose
This file is the package-level index and routing guide.

It does three things:
1. Maps user intent to the correct sub-skill path.
2. Enforces a full read chain before writing code.

## Mandatory Read Chain
Before producing code or execution commands, read skills in this order:

1. Read this file: `./SKILL.md`.
2. Read one domain index:
   - `./simulators/SKILL.md`, or
   - `./algorithms/SKILL.md`.
3. Read one category index under algorithms (if algorithms path):
   - `./algorithms/primitives/SKILL.md`
   - `./algorithms/linear-systems/SKILL.md`
   - `./algorithms/cryptography/SKILL.md`
   - `./algorithms/hamiltonian-simulation/SKILL.md`
   - `./algorithms/schrodingerization/SKILL.md`
4. Read the leaf skill for the concrete task (for example `.../hhl/SKILL.md`, `.../shor/SKILL.md`, `.../heat-1d-schrodingerization/SKILL.md`).

Rule: Never skip to coding from this root file alone.

## Routing Rules
Use these intent-to-path rules:

- If user asks simulator choice, runtime environment, execution setup, or backend comparison:
  - Read `./simulators/SKILL.md` first.
- If user asks circuit primitive, algorithm design, complexity, or implementation:
  - Read `./algorithms/SKILL.md` first.

Then route to the correct branch:

- Grover-style search, QPE, Hadamard test/transform, amplitude methods:
  - `./algorithms/primitives/SKILL.md`
- Solve linear systems $Ax=b$, HHL, LCU, QSVT/QLSA, QSP-for-linear-systems:
  - `./algorithms/linear-systems/SKILL.md`
- Shor, discrete logarithm, Simon, cryptographic attacks:
  - `./algorithms/cryptography/SKILL.md`
- Time evolution, $e^{-iHt}$, Trotter/qDRIFT/Taylor/QSP simulation:
  - `./algorithms/hamiltonian-simulation/SKILL.md`
- PDE-to-quantum mapping or Schrodingerization workflows:
  - `./algorithms/schrodingerization/SKILL.md`

## Package Map
Top-level map of this skill package:

- `./simulators/SKILL.md`
  - `./simulators/unitarylab/SKILL.md`
  - `./simulators/qiskit/SKILL.md`
  - `./simulators/pennylane/SKILL.md`
- `./algorithms/SKILL.md`
  - `./algorithms/primitives/SKILL.md`
  - `./algorithms/linear-systems/SKILL.md`
  - `./algorithms/cryptography/SKILL.md`
  - `./algorithms/hamiltonian-simulation/SKILL.md`
  - `./algorithms/schrodingerization/SKILL.md`

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
2. Open and use the full read chain down to the leaf skill.
3. Base commands/code on leaf-level instructions.
4. Keep response focused on the user task; avoid repeating package index content.

## Quick Intent Router
- "Which simulator should I use?" -> `./simulators/SKILL.md`
- "Set up UnitaryLab/Qiskit/PennyLane" -> `./simulators/SKILL.md` then simulator leaf skill
- "Implement HHL/LCU/QSVT" -> `./algorithms/linear-systems/SKILL.md` then leaf skill
- "Run Shor/Simon/discrete log" -> `./algorithms/cryptography/SKILL.md` then leaf skill
- "Hamiltonian time evolution" -> `./algorithms/hamiltonian-simulation/SKILL.md` then leaf skill
- "Solve PDE with Schrodingerization" -> `./algorithms/schrodingerization/SKILL.md` then equation leaf skill

## Maintenance Notes
When adding a new sub-skill:
1. Add it to the nearest category index SKILL.
2. Keep this root file focused on routing only.
3. Avoid adding detailed setup or implementation sections here.
