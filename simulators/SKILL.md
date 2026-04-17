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

### Wheel Files (Python 3.11 required)
| OS | Wheel filename |
|----|----------------|
| Windows x64 | `unitarylab-0.1.0-cp311-cp311-win_amd64.whl` |
| macOS (arm64) | `unitarylab-0.1.0-cp311-cp311-macosx_11_0_arm64.whl` |
| Linux x86-64 | `unitarylab-0.1.0-cp311-cp311-linux_x86_64.whl` |

All wheels are located in `./unitarylab/dist/`. Use the one matching the user's OS and architecture.

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
