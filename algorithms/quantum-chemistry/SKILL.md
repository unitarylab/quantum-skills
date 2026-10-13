---
name: quantum-chemistry
description: "Route molecular electronic-structure tasks to the appropriate UnitaryLab algorithm skill. Use for active-space molecular calculations and DMRG-to-quantum-circuit workflows."
---

# Quantum Chemistry

## Purpose

Use this category for molecular quantum-chemistry workflows that turn an XYZ geometry into an active-space electronic Hamiltonian and then solve, encode, or measure its state with classical tensor-network and quantum-circuit methods.

## Routing

- Molecular ground-state preparation with PySCF/OpenFermion electronic structure, TenPy DMRG, classical variational disentangling (CVD), UnitaryLab circuit evaluation, QWC measurement grouping, or OpenQASM export:
  - Read `./molecular-dmrg/SKILL.md`.

## Selection Notes

The current public `unitarylab_algorithms` quantum-chemistry implementation is Molecular DMRG. Route generic requests for the packaged molecular workflow to that leaf skill. Do not route unrelated VQE model problems here; for Fermi-Hubbard VQE, use `../quantum-machine-learning/fermi-hubbard-vqe/SKILL.md`.

## Response Contract

Before producing code or commands, read the Molecular DMRG leaf skill and use its documented public import, parameters, output fields, dependency requirements, and scale limitations. Keep category-level routing here and algorithm-specific details in the leaf skill.
