---
name: algorithms
description: "A top-level index of quantum algorithms centered on the UnitaryLab implementation, covering quantum primitives, linear systems, cryptography, Hamiltonian simulation, Schrodingerization, quantum machine learning, eigensolvers, gradients, and quantum error correction, with selected Qiskit, PennyLane, and Classiq examples included as reference extensions. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Quantum Algorithms

This directory collects the main quantum algorithm modules in UnitaryLab.
Each subdirectory contains its own `SKILL.md` with more detailed usage notes and implementation guidance.

## Note

Use the following workflow when handling `unitarylab_algorithms`:

1. Check whether you are only reading/editing docs or code.
	- If yes, do not install the package.
2. Install only when you are going to run code that imports `unitarylab_algorithms`, or when you see `No module named unitarylab_algorithms`.
3. Run:
	- `pip install unitarylab-algorithms`
4. Re-run the script or notebook cell and confirm the import works.

## Reference-Only Scripts Policy

Leaf algorithm skills often include a `scripts/` directory with two kinds of material: a small example that calls the corresponding `unitarylab_algorithms` class, and a direct implementation reference for the algorithm. Treat both as fallback references for troubleshooting, API comparison, and validation. They are not files to import from generated answers, and they should not become runtime dependencies.

For user-facing code generation, first write standalone code for the requested task. If that generated code fails, or if the required API/algorithm detail is ambiguous, inspect the scripts to understand the intended behavior and adapt the idea without importing or depending on the script files.

## 1. Quantum Primitives

Core building blocks for quantum algorithms, including phase estimation, amplitude amplification/estimation, Hadamard-based routines, and related primitives.

See reference: `./primitives/SKILL.md`

## 2. Quantum Linear Systems

Algorithms for solving linear systems on quantum hardware, including HHL, LCU, AQC, the basic single-qubit Quantum Signal Processing (QSP) demo, and QSVT-QLSA. Route QSP-based Hamiltonian simulation requests to `./hamiltonian-simulation/SKILL.md`.

See reference: `./linear-systems/SKILL.md`

## 3. Quantum Cryptography

Quantum algorithms with cryptographic relevance: Shor's factoring algorithm, discrete logarithm, and Simon's algorithm.

See reference: `./cryptography/SKILL.md`

## 4. Hamiltonian Simulation

Methods for simulating quantum Hamiltonians, including Trotter-Suzuki decomposition, QDrift randomized simulation, Taylor series simulation, and QSP-HS.

See reference: `./hamiltonian-simulation/SKILL.md`

## 5. Schrodingerization

PDE-to-quantum mapping via Schrodingerization, covering advection and 1D/2D heat equation examples.

See reference: `./schrodingerization/SKILL.md`

## 6. Quantum Machine Learning

Variational and hybrid quantum-classical learning algorithms, including VQE, VQC, QAOA, QCBM, CVQNN, and Fermi-Hubbard VQE.

See reference: `./quantum-machine-learning/SKILL.md`

## 7. Quantum State Preparation

Methods for loading target quantum states, including sparse superposition, Möttönen, MPS, multiplexer, and variational Pauli-word preparation.

See reference: `./state-preparation/SKILL.md`

## 8. Eigensolvers

Algorithms for computing eigenvalues and eigenstates of quantum operators, including exact classical diagonalization (NumPyEigensolver) and variational excited-state methods (VQD).

See reference: `./eigensolvers/SKILL.md`

## 9. Gradients

Quantum gradient and geometric tensor methods, including parameter-shift, finite-difference, linear-combination, SPSA, reverse-mode, and QFI.

See reference: `./gradients/SKILL.md`

## 10. Quantum Error Correction

Standalone PennyLane/NumPy educational material for quantum error-correcting code construction, including classical LDPC, CSS, and Hypergraph Product examples. This qLDPC material is Skill-local and is not a public `unitarylab_algorithms` implementation.

See reference: `./quantum-error-correction/SKILL.md`
