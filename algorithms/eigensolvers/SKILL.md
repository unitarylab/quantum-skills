---
name: eigensolvers
description: Quantum eigensolver algorithms for operator spectrum estimation, covering exact classical diagonalization with NumPyEigensolver and variational excited-state solving with VQD.
---

# Quantum Eigensolvers

## Purpose
This file is the category index and routing guide for eigensolver tasks.

Use this folder when the user wants to compute eigenvalues or eigenstates of a quantum operator, compare exact and variational eigensolver strategies, or implement an eigensolver workflow in Qiskit.

## Routing Rules
Choose the leaf skill by problem type:

- If the user wants exact eigenvalues, full diagonalization behavior, sparse or dense matrix backends, or auxiliary operator evaluation on exact eigenstates:
  - Read `./numyeigensolver/SKILL.md`
- If the user wants a variational workflow, low-lying excited states, ansatz-based optimization, fidelity penalties, or hybrid primitive-based execution:
  - Read `./vqd/SKILL.md`

## Selection Guide

### 1. NumPy Eigensolver
See reference: `./numyeigensolver/SKILL.md`

Use this path when:
- the operator is small enough for exact classical treatment,
- the user needs deterministic eigenpairs,
- the task involves dense or sparse matrix eigendecomposition,
- auxiliary operators should be evaluated exactly on returned eigenstates.

### 2. Variational Quantum Deflation
See reference: `./vqd/SKILL.md`

Use this path when:
- the user needs the ground state plus excited states,
- the workflow depends on a parameterized ansatz,
- the solution should use Qiskit primitives and a classical optimizer,
- overlap penalties or orthogonality constraints are part of the algorithm design.

## Quick Intent Router
- "Compute the lowest few exact eigenvalues" -> `./numyeigensolver/SKILL.md`
- "Diagonalize a SparsePauliOp exactly" -> `./numyeigensolver/SKILL.md`
- "Evaluate auxiliary observables on exact eigenstates" -> `./numyeigensolver/SKILL.md`
- "Find excited states with a variational ansatz" -> `./vqd/SKILL.md`
- "Implement VQD with estimator and fidelity primitives" -> `./vqd/SKILL.md`
- "Compare exact and variational eigensolvers" -> read this file first, then both leaf skills.

## Engineering Notes
- NumPyEigensolver is exact but scales poorly with qubit count because the operator dimension grows as $2^n$.
- VQD is more scalable in workflow design, but its result quality depends on the ansatz, optimizer, fidelity evaluation, and penalty tuning.
- Use NumPyEigensolver as a baseline when validating a VQD setup on small instances.

## Response Contract
When handling eigensolver requests:

1. Decide whether the task is exact classical diagonalization or variational excited-state estimation.
2. Read the corresponding leaf skill before producing code.
3. Keep category-level guidance here and leave implementation detail to the leaf skills.