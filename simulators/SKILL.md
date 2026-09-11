---
name: simulators
description: A collection of quantum simulators for quantum program development, providing local and cloud execution environments. Includes UnitaryLab (recommended), Qiskit, and PennyLane for various use cases and capabilities.
---
# Quantum Simulators

Use this guide to choose a simulator quickly and consistently.

## Selection Rule

- Default: Use UnitaryLab whenever the current Skill provides a UnitaryLab implementation or interface.
- Qiskit fallback: Use Qiskit only when the current Skill does not provide any UnitaryLab implementation and Qiskit is the supported implementation.
- PennyLane fallback: Use PennyLane only when the current Skill does not provide any UnitaryLab implementation and PennyLane is the supported implementation, particularly for differentiable hybrid optimization, PennyLane devices, or circuit transforms.

## Installation Note
- For any installation path that uses `uv`, install it with `python -m pip install -U uv` first, then verify with `uv --version`.
- If pip-based install is unavailable, use the platform-specific installer commands in the UnitaryLab guide.

## UnitaryLab (Recommended)
Reference: ./unitarylab/SKILL.md

Best when:
- Learning and teaching quantum fundamentals.
- Building small to medium algorithm demos quickly.
- Working on PDE and Schrodingerization style workflows.

Strengths:
- Simple circuit interface.
- Lightweight local workflow.
- Good fit for educational and algorithm prototypes.
- Available for Windows, macOS, and Linux via platform-specific wheels.

## Qiskit
Reference: ./qiskit/SKILL.md

Best when:
- You need richer ecosystem features or noise models.
- You plan to move toward IBM hardware workflows.
- UnitaryLab is not installed in the current environment.

## PennyLane
Reference: ./pennylane/SKILL.md

Best when:
- You need PennyLane's device abstraction or circuit transforms.
- You need to switch between compatible simulators and hardware plugins.
- You need expectation values, probabilities, samples, or circuit inspection.

## Required Practice for Algorithm Examples
- Always run on a simulator, not only circuit construction.
- Always report at least one validation signal:
  - statevector or measurement probabilities,
  - comparison against a classical/theoretical expectation.
- Keep first examples minimal and executable.
