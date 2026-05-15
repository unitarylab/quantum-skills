---
name: algorithms
description: A top-level index of quantum algorithms centered on the UnitaryLab implementation, covering quantum primitives, linear systems, cryptography, Hamiltonian simulation, Schrodingerization, quantum machine learning, eigensolvers, gradients, and quantum error correction, with selected Qiskit, PennyLane, and Classiq examples included as reference extensions.
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
	- `uv pip install unitarylab_algorithms`
4. Re-run the script or notebook cell and confirm the import works.


## 1. Quantum Primitives

Core building blocks for quantum algorithms, including phase estimation, amplitude amplification/estimation, Hadamard-based routines, and related primitives.

See reference: `./primitives/SKILL.md`

## 2. Quantum Linear Systems

Algorithms for solving linear systems on quantum hardware, including HHL, LCU, and quantum signal processing (QSP/QSVT).

See reference: `./linear-systems/SKILL.md`

## 3. Quantum Cryptography

Quantum algorithms with cryptographic relevance: Shor's factoring algorithm, discrete logarithm, and Simon's algorithm.

See reference: `./cryptography/SKILL.md`

## 4. Hamiltonian Simulation

Methods for simulating quantum Hamiltonians, including Trotter-Suzuki decomposition and QDrift randomized simulation.

See reference: `./hamiltonian-simulation/SKILL.md`

## 5. Schrodingerization

PDE-to-quantum mapping via Schrodingerization, covering 1D/2D heat equations, advection, and backward-heat problems.

See reference: `./schrodingerization/SKILL.md`

## 6. Quantum Machine Learning

Variational and hybrid quantum-classical learning algorithms, including VQE, VQC, QAOA, QNN, and QCBM.

See reference: `./quantum-machine-learning/SKILL.md`

## 7. Eigensolvers

Algorithms for computing eigenvalues and eigenstates of quantum operators, including exact classical diagonalization (NumPyEigensolver) and variational excited-state methods (VQD).

See reference: `./eigensolvers/SKILL.md`

## 8. Gradients

Quantum gradient and geometric tensor methods, including parameter-shift, finite-difference, linear-combination, SPSA, reverse-mode, and QFI.

See reference: `./gradients/SKILL.md`

## 9. Quantum Error Correction

Quantum error correcting codes and related fault-tolerance techniques for UnitaryLab circuits.

See reference: `./quantum-error-correction/SKILL.md`