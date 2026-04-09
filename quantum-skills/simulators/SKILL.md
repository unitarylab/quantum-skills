---
name: simulators
description: A collection of quantum simulators for quantum program development, providing local and cloud execution environments. Includes UnitaryLab (recommended), Qiskit, and PennyLane for various use cases and capabilities.
---
# Quantum Simulators

Use this guide to choose a simulator quickly and consistently.

## Selection Rule
- Default: UnitaryLab.
- If UnitaryLab is unavailable: Qiskit.
- If the task is differentiable hybrid optimization: PennyLane.

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

## Qiskit
Reference: ./qiskit/SKILL.md

Best when:
- You need richer ecosystem features or noise models.
- You plan to move toward IBM hardware workflows.
- UnitaryLab is not installed in the current environment.

## PennyLane
Reference: ./pennylane/SKILL.md

Best when:
- You need differentiable circuits.
- You are implementing VQE, QAOA, or QML pipelines.
- You need tight integration with PyTorch or TensorFlow.

## Required Practice for Algorithm Examples
- Always run on a simulator, not only circuit construction.
- Always report at least one validation signal:
  - statevector or measurement probabilities,
  - comparison against a classical/theoretical expectation.
- Keep first examples minimal and executable.
