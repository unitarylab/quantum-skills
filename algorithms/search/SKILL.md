---
name: search
description: "Select a quantum search or oracle algorithm for graph transport, marked-vertex search, constrained-tree search, or hidden-shift recovery. Route Grover and general amplitude amplification to quantum primitives."
---

# Quantum Search Algorithms

## Purpose

This file routes the following Search workflows. Quantum Walk Search and Quantum Backtracking include complete local implementations and do not require the installed algorithm package:

- `GluedTreesAlgorithm`
- `HiddenShiftAlgorithm`
- Continuous-Time Quantum Walk Search (bundled `QuantumWalkSearchAlgorithm`)
- Quantum Backtracking (bundled `QuantumBacktrackingAlgorithm`)

Use it to identify the requested problem and load the complete matching leaf skill. Do not write implementation code from this category file alone: parameter contracts, return fields, circuit construction, mathematical derivations, source mappings, and self-contained `Minimal Manual Implementation` code belong to the leaf skills.

## Scope

These algorithms share a broad search/oracle theme but solve different problems and use different circuit structures:

| Algorithm | Problem | Quantum mechanism | Primary output |
|---|---|---|---|
| Glued Trees | Transport from the entrance root to the hidden exit of two randomly glued trees | Coupled-oscillator Hamiltonian embedding, Pauli evolution, and a Suzuki-Trotter time scan | Sampled exit-state probability versus time |
| Hidden Shift | Recover a binary translation of a fixed quadratic bent Boolean function | Shifted phase oracle, self-duality, and Hadamard interference | Recovered hidden bit string |
| Quantum Walk Search | Find one or more marked vertices in an undirected weighted graph | Graph hopping plus marked-projector Hamiltonian, uniform graph state and evolution | Simulated/exact marked probability, sampled counts and circuit evidence |
| Quantum Backtracking | Search binary CNF or accepted-path/rejected-prefix constraint trees | Reversible predicates, alternating reflections W=R_B R_A, QPE subtree detection | Detected path/assignment, phase sampling evidence and circuit resources |

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

### Continuous-Time Quantum Walk Search

Read `./quantum-walk-search/SKILL.md` when the request involves:

- complete/cycle/path/star or custom undirected weighted graphs;
- multiple marked vertices and H=-gamma*A-sum(|w><w|);
- uniform graph-state evolution, automatic spectral time selection, or padded vertex spaces;
- marked-vertex probability, exact spectral reference, seeded vertex samples or graph-search circuit evidence;
- explaining, running, debugging, modifying or independently rebuilding this continuous-time search variant.

Use the leaf's bundled implementation/example; this workflow does not require an installed algorithm class.
Distinguish marked-vertex search from Glued Trees entrance-to-exit transport.

### Quantum Backtracking

Read `./quantum-backtracking/SKILL.md` when the request involves:

- binary CNF, accepted paths, rejected prefixes or constrained backtracking trees;
- reversible accept/reject/child oracles and one-hot depth registers;
- alternating tree reflections R_A/R_B, W=R_B R_A and controlled-walk QPE;
- zero-phase subtree detection, threshold/shots/seed or recovery of a valid assignment;
- explaining, executing, debugging, modifying or independently rebuilding this gate-level variant.

Use the bundled implementation/example and diagnostic/--execute modes. Distinguish discrete tree-walk detection from continuous-time marked-vertex graph search; finite sampling is not an unsatisfiability proof.

### Route Elsewhere

- Grover search, marked-state amplification, or diffuser/oracle construction:
  - Read `../primitives/grover/SKILL.md`.
- General amplitude amplification without the packaged Search algorithms:
  - Read `../primitives/amplitude-amplification/SKILL.md`.
- Simon's hidden XOR mask and two-to-one promise:
  - Read `../cryptography/simon/SKILL.md`.
- Hamiltonian simulation as a general method rather than a graph-search/transport workflow:
  - Read `../hamiltonian-simulation/SKILL.md`.

## Required Read Chain

Before writing code or commands for a Search algorithm:

1. Read the package root `../../SKILL.md` when it has not already been read.
2. Read the algorithms index `../SKILL.md` when it has not already been read.
3. Read this Search category file.
4. Read exactly one matching leaf skill:
   - `./glued-trees/SKILL.md`, or
   - `./hidden-shift/SKILL.md`, or
   - `./quantum-walk-search/SKILL.md`, or
   - `./quantum-backtracking/SKILL.md`.
5. Follow the chosen leaf's entry contract: existing package examples for Glued Trees/Hidden Shift, bundled local implementation for Quantum Walk Search/Quantum Backtracking.
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

### 3. Continuous-Time Quantum Walk Search

Reference: `./quantum-walk-search/SKILL.md`

The leaf includes the complete graph-search implementation, local example, graph/marked validation,
Hamiltonian and spectral time selection, padding/bit-order conventions, backend/device/dtype forwarding,
probability/count/export contracts, theory-to-code mapping and an independent manual reconstruction.
Default script mode is diagnostic; `--execute` performs the documented small run.

### 4. Quantum Backtracking

Reference: `./quantum-backtracking/SKILL.md`

The complete leaf includes reversible predicates, tree registers/reflections, walk-QPE detection,
ordered branch recovery, seeded evidence, resources/exports, all local implementation and example
scripts, mathematical mapping and an independent small manual reconstruction. Default script
mode is diagnostic; --execute performs the documented small run.

## Public API Boundary

For Glued Trees and Hidden Shift package code, use the classes exported by the `unitarylab-algorithms` package installed in the active Python environment:

```python
from unitarylab_algorithms import GluedTreesAlgorithm, HiddenShiftAlgorithm
```

Do not add the repository workspace to `sys.path` for API examples, and do not import a local checkout of `unitarylab_algorithms`. The Glued Trees/Hidden Shift `scripts/algorithm.py` examples must resolve the installed package normally. Do not import files from a skill's `scripts/` directory into user applications; those scripts are reference and verification artifacts.

Quantum Walk Search and Quantum Backtracking are self-contained leaves: follow each leaf's local `scripts/algorithm.py` and its bundled `scripts/quantum_walk_search_implementation.py` or `scripts/quantum_backtracking_implementation.py` instructions. Neither imports `unitarylab_algorithms` or requires another checkout. This exception is limited to those two leaves.

When the user asks for a manual implementation, do not replace the request with a call to the public algorithm class. Follow the selected leaf's `Minimal Manual Implementation` and construct the same algorithm from its underlying graph, Hamiltonian, oracle, or circuit components.

## Shared Interpretation Rules

- Treat `algorithm.py` as authoritative for callable signatures, defaults, validation, execution behavior, and returned fields.
- For Quantum Backtracking, the bundled `scripts/quantum_backtracking_implementation.py` contains the authoritative computation/API; `scripts/algorithm.py` is its thin diagnostic/execute example.
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
   - Quantum Walk Search: graph/Hamiltonian invariants, exact versus simulated marked probabilities, padded probability and sampled-count totals.
   - Quantum Backtracking: predicate acceptance of returned paths, phase-count totals/strict threshold, fixed-seed evidence, register conventions and exports; exists=False is not an unsatisfiability proof.
6. Report environment or dependency incompatibilities instead of silently changing source semantics.
