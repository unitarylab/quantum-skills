---
name: search
description: "Quantum search and oracle algorithms implemented by UnitaryLab, covering the coupled-oscillator Glued Trees quantum walk and the self-dual bent-function Hidden Shift algorithm. Use this category to route explanation, code generation, execution, debugging, validation, and source-guided reimplementation to the correct leaf skill."
---

# Quantum Search Algorithms

## Purpose

This file is the category-level router for the Search algorithms currently exported by `unitarylab_algorithms.search`:

- `GluedTreesAlgorithm`
- `HiddenShiftAlgorithm`

Use it to identify the requested problem and load the complete matching leaf skill. Do not write implementation code from this category file alone: parameter contracts, return fields, circuit construction, mathematical derivations, source mappings, and self-contained `Minimal Manual Implementation` code belong to the leaf skills.

## Scope

The two algorithms share a broad search/oracle theme but solve different problems and use different circuit structures:

| Algorithm | Problem | Quantum mechanism | Primary output |
|---|---|---|---|
| Glued Trees | Transport from the entrance root to the hidden exit of two randomly glued trees | Coupled-oscillator Hamiltonian embedding, Pauli evolution, and a Suzuki-Trotter time scan | Sampled exit-state probability versus time |
| Hidden Shift | Recover a binary translation of a fixed quadratic bent Boolean function | Shifted phase oracle, self-duality, and Hadamard interference | Recovered hidden bit string |

The category does not include every algorithm whose name contains “search.” In particular, Grover search and amplitude amplification remain quantum primitives.

## Routing Rules

### Glued Trees

Read `./glued-trees/SKILL.md` when the request involves:

- glued binary trees, entrance/exit vertices, or quantum-walk transport;
- graph adjacency or coupled-oscillator embeddings;
- `K = 3I - A`, Cholesky factors, or the block Hamiltonian `H = -[[0,B],[B†,0]]`;
- Pauli decomposition, `max_terms` cropping, or truncation error;
- first- or second-order Suzuki-Trotter evolution;
- time scans, exit-state samples, or `GluedTreesAlgorithm`;
- rebuilding the same algorithm from graph, Hamiltonian, and circuit primitives.

### Hidden Shift

Read `./hidden-shift/SKILL.md` when the request involves:

- a Boolean hidden shift or recovery of a secret binary translation;
- bent functions or the quadratic function `XOR_i x[2i]x[2i+1]`;
- phase oracles, `X_s U_f X_s`, or self-dual Walsh spectra;
- `H -> U_g -> H -> U_f -> H` interference;
- sample-versus-execute measurement behavior or output bit order;
- `HiddenShiftAlgorithm` or a basic-gate reconstruction of the same circuit.

### Route Elsewhere

- Grover search, marked-state amplification, or diffuser/oracle construction:
  - Read `../primitives/grover/SKILL.md`.
- General amplitude amplification without the packaged Search algorithms:
  - Read `../primitives/amplitude-amplification/SKILL.md`.
- Simon's hidden XOR mask and two-to-one promise:
  - Read `../cryptography/simon/SKILL.md`.
- Hamiltonian simulation as a general method rather than the Glued Trees workflow:
  - Read `../hamiltonian-simulation/SKILL.md`.

## Required Read Chain

Before writing code or commands for a Search algorithm:

1. Read the package root `../../SKILL.md` when it has not already been read.
2. Read the algorithms index `../SKILL.md` when it has not already been read.
3. Read this Search category file.
4. Read exactly one matching leaf skill:
   - `./glued-trees/SKILL.md`, or
   - `./hidden-shift/SKILL.md`.
5. Use the leaf's public API example for package calls.
6. Use the leaf's `Minimal Manual Implementation` when the user requests an independent implementation or reconstruction from circuit foundations.

Do not stop at this category page for code generation. It intentionally does not duplicate the leaf implementations.

## Available Leaf Skills

### 1. Glued Trees Quantum Walk

Reference: `./glued-trees/SKILL.md`

The leaf documents:

- graph construction and seeded random glue;
- adjacency, stiffness, Cholesky, and Hermitian embedding;
- Pauli decomposition and the exact cropping policy;
- circuit construction for both supported Suzuki orders;
- time parsing, execution, sampling, artifacts, and return fields;
- mathematical interpretation, theory-to-code mapping, and limitations;
- a self-contained minimal implementation built from graph/Hamiltonian/circuit primitives;
- a fuller independent implementation in `scripts/glued_trees_implementation.py`.

### 2. Boolean Hidden Shift

Reference: `./hidden-shift/SKILL.md`

The leaf documents:

- the implemented quadratic bent-function family;
- the pairwise-CZ phase oracle;
- the shifted oracle and complete Hadamard interference circuit;
- explicit versus seeded shifts and their precedence;
- sample and execute modes, measurement placement, and bit reversal;
- artifacts, return fields, mathematical derivation, and limitations;
- a self-contained minimal implementation constructed from `H`, `X`, `CZ`, and measurement operations;
- a fuller independent implementation in `scripts/hidden_shift_implementation.py`.

## Public API Boundary

For normal user-facing package code, use the classes exported by the `unitarylab-algorithms` package installed in the active Python environment:

```python
from unitarylab_algorithms import GluedTreesAlgorithm, HiddenShiftAlgorithm
```

Do not add the repository workspace to `sys.path` for API examples, and do not import a local checkout of `unitarylab_algorithms`. The leaf `scripts/algorithm.py` examples must resolve the installed package normally. Do not import files from a skill's `scripts/` directory into user applications; those scripts are reference and verification artifacts.

When the user asks for a manual implementation, do not replace the request with a call to the public algorithm class. Follow the selected leaf's `Minimal Manual Implementation` and construct the same algorithm from its underlying graph, Hamiltonian, oracle, or circuit components.

## Shared Interpretation Rules

- Treat `algorithm.py` as authoritative for callable signatures, defaults, validation, execution behavior, and returned fields.
- Treat `parameters.json` as the UI schema. A UI maximum or regex is not necessarily enforced by the Python API.
- Use the bilingual README files for intended theory and presentation, but do not let them override actual source behavior.
- Separate ideal query-complexity claims from classical simulator time and memory cost.
- Report sampling, Trotter, Pauli truncation, backend, and bit-order limitations when relevant.
- Do not infer algorithmic success from a process completing without exception. Apply the selected leaf's verification contract.

## Response Contract

1. Identify the exact search/oracle problem and select one leaf skill.
2. State whether the task is explanation, public API usage, execution, debugging, validation, or independent reconstruction.
3. Preserve the selected algorithm's parameter and return contract.
4. For a manual implementation, build the same algorithm from foundational operations rather than importing the packaged algorithm or the skill's reference script.
5. Verify the algorithm-specific observable:
   - Glued Trees: graph/Hamiltonian invariants and exit-state probabilities.
   - Hidden Shift: recovered string, count total, measurement mode, and bit order.
6. Report environment or dependency incompatibilities instead of silently changing source semantics.
