---
name: quantum-machine-learning
description: "Variational quantum algorithms for optimization and machine learning, plus two-dimensional transverse-field Ising dynamics, including QAOA for combinatorial optimization, VQE for ground-state energy estimation, Fermi-Hubbard VQE for lattice models, VQC for supervised learning, QCBM for generative modeling, and CVQNN for continuous-variable workflows. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---
# Quantum Machine Learning Algorithms

## Purpose
This file routes requests for variational and hybrid quantum-classical learning algorithms.

Use this category when the user asks about optimization with parameterized circuits, supervised classification, generative modeling, ground-state energy estimation, or continuous-variable quantum neural networks.

## Routing Rules

- If the user asks about combinatorial optimization, MaxCut-style objectives, or alternating cost/mixer layers:
  - Read `./qaoa/SKILL.md`
- If the user asks about generative modeling of bitstring distributions or Born-machine training:
  - Read `./qcbm/SKILL.md`
- If the user asks about supervised classification with a variational quantum circuit:
  - Read `./vqc/SKILL.md`
- If the user asks about ground-state energy, Hamiltonian expectation minimization, or chemistry-style variational solving:
  - Read `./vqe/SKILL.md`
- If the user asks about the Fermi-Hubbard model, Jordan-Wigner mapping, lattice-model VQE, or magnetic-moment measurement:
  - Read `./fermi-hubbard-vqe/SKILL.md`
- If the user asks about continuous-variable neural networks, photonic modes, or CV quantum layers:
  - Read `./cvqnn/SKILL.md`
- If the user asks about the two-dimensional transverse-field Ising model, Ising dynamics, local Trotter-Suzuki evolution, or the packaged `IsingAlgorithm`:
  - Read `./ising/SKILL.md`

## Available Leaf Skills

1. Quantum Approximate Optimization Algorithm: `./qaoa/SKILL.md`
2. Quantum Circuit Born Machine: `./qcbm/SKILL.md`
3. Variational Quantum Classifier: `./vqc/SKILL.md`
4. Variational Quantum Eigensolver: `./vqe/SKILL.md`
5. Continuous-Variable Quantum Neural Network: `./cvqnn/SKILL.md`
6. Fermi-Hubbard VQE: `./fermi-hubbard-vqe/SKILL.md`
7. 2D Transverse-Field Ising Dynamics: `./ising/SKILL.md`

## Response Contract

1. Identify the requested workflow: optimization, classification, generative modeling, energy minimization, continuous-variable modeling, or Ising dynamics.
2. Read the matching leaf skill before writing code or commands.
3. Keep model-specific APIs, training loops, and examples in the leaf skill.
