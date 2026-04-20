---
name: pennylaneqldpc
description: A clear and practical skill guide for learning and running a PennyLane-based qLDPC tutorial, from classical LDPC basics to CSS and Hypergraph Product code construction.
---

# PennyLane qLDPC Skill

## Purpose
This skill helps you understand and run a qLDPC (quantum low-density parity-check) tutorial script in a simple, structured way.

It covers:
- Classical LDPC basics (parity-check matrix, Tanner graph, syndrome)
- CSS code construction and commutation checks
- Hypergraph Product (HGP) code construction
- Small utility functions for binary matrix rank and code dimension

## When to Use This Skill
Use this skill when you need to:
- Explain qLDPC ideas with a concrete Python example
- Validate CSS code commutation conditions
- Build and inspect small HGP code instances
- Create educational demos with NumPy + NetworkX + Matplotlib + PennyLane

Do not use this skill if you only need hardware execution workflows. This script is simulation and education focused.

## Prerequisites
- Python 3.10+ (3.11 recommended)
- Core packages:
  - numpy
  - networkx
  - matplotlib
  - pennylane

Install packages:

```bash
pip install numpy networkx matplotlib pennylane
```

## File Scope
- Main script: `script/algorithm.py`
- This skill document: `SKILL.md`

## Quick Start
Run the tutorial script directly:

```bash
python script/algorithm.py
```

Expected outcomes:
- A Tanner graph plot for a simple classical LDPC code
- Printed syndrome from an example error vector
- CSS code shape and commutation validation (`H_X H_Z^T = 0 mod 2`)
- Code dimension (`k`) calculation for a sample code
- HGP construction validation output

## Concept Map (Simple)
1. Classical LDPC:
- Define sparse parity-check matrix `H`
- Build Tanner graph (variable nodes and check nodes)
- Compute syndrome: `s = H e mod 2`

2. CSS construction:
- Use two binary parity-check matrices: `H_X`, `H_Z`
- Enforce commutation condition:
  - `H_X H_Z^T = 0 (mod 2)`

3. Hypergraph Product (HGP):
- Build qLDPC checks from two classical codes
- Produce sparse structured quantum checks with guaranteed CSS commutation

## Key Functions in `algorithm.py`
- `hamming_code(rank: int) -> np.ndarray`
  - Generates a Hamming parity-check matrix.

- `binary_matrix_rank(binary_matrix: np.ndarray) -> int`
  - Computes matrix rank over binary field `Z_2`.

- `rep_code(distance: int) -> np.ndarray`
  - Creates repetition code parity-check matrix.

- `hgp_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - Constructs HGP CSS matrices `H_X` and `H_Z`.

## Standard Workflow
1. Start from a small classical parity-check matrix `H`.
2. Visualize Tanner graph to understand sparsity and constraints.
3. Inject an example error vector and compute syndrome.
4. Build CSS matrices and verify commutation.
5. Compute code dimension from binary ranks.
6. Construct HGP matrices and verify consistency.

## Validation Checklist
Use this checklist after modifications:
- Script runs without import errors.
- Tanner graph renders correctly.
- Syndrome output shape and values are reasonable.
- CSS commutation checks return `True`.
- Computed code dimension matches expected theory.
- HGP commutation check returns `True`.

## Common Issues and Fixes
- `ModuleNotFoundError: pennylane`:
  - Install with `pip install pennylane`.

- Plot window does not appear:
  - Use a local GUI backend or save figures with `plt.savefig(...)`.

- Binary rank mismatch:
  - Confirm all matrices are binary (`0/1`) and operations are modulo 2.

## Extension Ideas
- Add belief-propagation decoding example for classical LDPC.
- Add random error channel simulation for CSS codes.
- Compare multiple HGP input code families.
- Export matrix sparsity statistics and scaling trends.

## Output Style Guidance
When using this skill for user-facing explanations:
- Keep language concise and practical.
- Use short steps and explicit equations.
- Explain why each matrix check matters.
- Prefer small reproducible code examples.

## Safety and Scope Notes
- This skill is educational and simulation-oriented.
- It does not claim hardware fault-tolerance guarantees.
- Performance results from tiny examples do not imply large-scale decoder performance.

## One-Line Summary
Use this skill to clearly explain and run a PennyLane-based qLDPC tutorial from LDPC foundations to CSS and HGP code construction, with practical validation steps.
