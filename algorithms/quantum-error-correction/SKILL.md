---
name: quantum-error-correction
description: "A clear and practical skill guide for learning and running a PennyLane-based qLDPC tutorial, from classical LDPC basics to CSS and Hypergraph Product code construction. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# PennyLane qLDPC Skill

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement the qLDPC (quantum low-density parity-check) tutorial and helper implementation in this folder.

It covers:
- Classical LDPC basics (parity-check matrix, Tanner graph, syndrome)
- CSS code construction and commutation checks
- Hypergraph Product (HGP) code construction
- Small utility functions for binary matrix rank and code dimension
- A class-based `QLDPCSolver` helper interface for reusable Agent-generated code

Use this skill when you need to:
- Explain qLDPC ideas with a concrete Python example
- Validate CSS code commutation conditions
- Build and inspect small HGP code instances
- Create educational demos with NumPy + NetworkX + Matplotlib + PennyLane

Do not use this skill if you only need hardware execution workflows. This script is simulation and education focused.

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.
- **Binary arithmetic:** Perform parity-check, syndrome, rank, and matrix operations over GF(2), not ordinary real-valued linear algebra.
- **CSS validation:** Verify `H_X H_Z^T = 0 mod 2` before treating the matrices as a valid CSS code.
- **HGP validation:** Verify matrix dimensions, block ordering, binary rank, and CSS commutation after constructing a Hypergraph Product code.
- **Educational scope:** Treat this workflow as a simulator-focused educational implementation, not a complete fault-tolerant hardware execution stack.

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
- Direct-run tutorial script: `scripts/algorithm.py`
- Reusable helper implementation: `scripts/qldpc_implementation.py`
- This skill document: `SKILL.md`

# Loom/Catalyst Reference Pattern


Use this pattern when explaining the Catalyst section:

```python
from catalyst import qjit, cond, measure, debug
from jax import random, numpy as jnp
import pennylane as qml

distance = 3
data_qubits = [0, 1, 2]
aux_qubits = [3, 4]
dev = qml.device("qrack.simulator", wires=5)

@qjit()
@qml.qnode(dev)
def repetition_code_memory(seed: int):
    random_qubit = random.randint(random.PRNGKey(seed), (1,), -1, distance)[0]

    @cond(random_qubit != -1)
    def apply_noise():
        debug.print("Applying noise to qubit: {}", random_qubit)
        qml.X(random_qubit)

    apply_noise()

    for i in range(distance - 1):
        qml.CNOT(wires=[data_qubits[i], aux_qubits[i]])
        qml.CNOT(wires=[data_qubits[i + 1], aux_qubits[i]])

    syndrome = [measure(q) for q in aux_qubits]

    @cond(jnp.logical_and(syndrome[0] == 0, syndrome[1] == 1))
    def fix_data_qubits():
        qml.X(data_qubits[2])

    @fix_data_qubits.else_if(jnp.logical_and(syndrome[0] == 1, syndrome[1] == 0))
    def fix_data_qubits():
        qml.X(data_qubits[0])

    @fix_data_qubits.else_if(jnp.logical_and(syndrome[0] == 1, syndrome[1] == 1))
    def fix_data_qubits():
        qml.X(data_qubits[1])

    fix_data_qubits()
    return [measure(q) for q in data_qubits], syndrome
```

Do not use this Catalyst snippet as the default qLDPC implementation path. For default qLDPC code generation, use `QLDPCSolver` and the matrix-construction functions above.

## Direct-Run Generation Contract (Mandatory)
When generating `scripts/algorithm.py` from this skill, the output MUST satisfy all items below without requiring manual edits.

1. Runtime behavior:
- The script must run with `python scripts/algorithm.py` in a properly prepared environment.
- The script must include `if __name__ == "__main__": main()`.
- The script must save the Tanner graph to file (for example `tanner_graph.png`) and must not depend on interactive `plt.show()`.

2. Required sections in execution order:
- Classical LDPC demo (matrix, Tanner graph, syndrome)
- CSS demo with explicit commutation check
- HGP demo with explicit commutation check
- Small PennyLane demo context (for example a tiny QNode)

3. Mathematical correctness checks:
- CSS check must print `True` for the built-in sample matrices.
- HGP check must print `True` for the built-in sample matrices.
- Code dimension computation must use binary rank over `Z_2`.

4. HGP construction must use the following standard form:

$$
H_X = [H_1 \otimes I_{n_2} \mid I_{r_1} \otimes H_2^T]
$$

$$
H_Z = [I_{n_1} \otimes H_2 \mid H_1^T \otimes I_{r_2}]
$$

where `H1` has shape `(r1, n1)` and `H2` has shape `(r2, n2)`.

5. Built-in CSS sample requirement:
- Use a commuting sample by default, e.g. `H_X = rep_code(3)` and `H_Z = [[1, 1, 1]]`.
- Avoid default samples that produce `False` commutation in the main demo.

## Overview

Quantum Low-Density Parity-Check (qLDPC) codes are a family of quantum error-correcting codes constructed from classical LDPC codes. This skill covers the full pipeline from classical LDPC concepts to CSS code construction and Hypergraph Product (HGP) codes, using a practical Python implementation based on NumPy, NetworkX, Matplotlib, and PennyLane.

The pipeline proceeds in four stages:

1. **Classical LDPC**: Define a sparse binary parity-check matrix $H$, visualize its Tanner graph (bipartite graph of variable nodes and check nodes), inject an error vector $e$, and compute the syndrome $s = H e \bmod 2$.
2. **CSS Construction**: Use two binary parity-check matrices $H_X$ and $H_Z$ to define a CSS quantum code, enforce the commutation condition $H_X H_Z^T = 0 \pmod{2}$, and estimate the number of encoded logical qubits via $k = n - \mathrm{rank}(H_X) - \mathrm{rank}(H_Z)$.
3. **Hypergraph Product (HGP)**: Construct a full qLDPC code from two classical codes $(H_1, H_2)$ using the tensor-product block form. HGP codes guarantee CSS commutation by construction and produce sparse quantum parity checks.
4. **PennyLane Integration**: A tiny QNode demo confirms the PennyLane environment is functional for follow-on quantum circuit work.

The skill provides both a direct-run tutorial script (`algorithm.py`) and a reusable helper (`qldpc_implementation.py`) with a `QLDPCSolver` class interface. A Loom/Catalyst reference tutorial demonstrates how to extend these ideas to repetition-code memory experiments with syndrome extraction, conditional correction, and decoder integration.

### Key Concepts

- **Parity-check matrix**: A binary matrix $H$ of shape $(r, n)$ where each row represents a parity check and each column a bit. Sparsity (low density) is the defining property of LDPC codes.
- **Tanner graph**: Bipartite graph with $n$ variable nodes and $r$ check nodes; an edge connects check $i$ to variable $j$ when $H_{ij} = 1$.
- **Syndrome**: $s = H e \bmod 2$ — the vector that encodes which parity checks are violated by error $e$.
- **CSS commutation**: $H_X H_Z^T = 0 \pmod{2}$ ensures the X and Z stabilizers mutually commute, a necessary condition for a valid quantum code.
- **Code dimension**: $k = n - \mathrm{rank}(H_X) - \mathrm{rank}(H_Z)$ gives the number of logical qubits encoded in $n$ physical qubits.
- **HGP construction**: Given classical codes $H_1 (r_1 \times n_1)$ and $H_2 (r_2 \times n_2)$, the quantum code has $n_1 n_2 + r_1 r_2$ physical qubits and dimension $k = k_1 \cdot k_2$ where $k_i = n_i - \mathrm{rank}(H_i)$.

## Key Functions in `algorithm.py`
- `hamming_code(rank: int) -> np.ndarray`
  - Generates a Hamming parity-check matrix.

- `binary_matrix_rank(binary_matrix: np.ndarray) -> int`
  - Computes matrix rank over binary field `Z_2`.

- `rep_code(distance: int) -> np.ndarray`
  - Creates repetition code parity-check matrix.

- `hgp_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - Constructs HGP CSS matrices `H_X` and `H_Z`.

## Key API in `qldpc_implementation.py`
- `compute_syndrome(h: np.ndarray, error: np.ndarray) -> np.ndarray`
  - Computes `s = H e mod 2`.

- `css_commutes(h_x: np.ndarray, h_z: np.ndarray) -> bool`
  - Checks `H_X H_Z^T = 0 mod 2`.

- `code_dimension(h_x: np.ndarray, h_z: np.ndarray) -> int`
  - Estimates CSS logical-qubit count as `k = n - rank(H_X) - rank(H_Z)`.

- `validate_css_code(h_x: np.ndarray, h_z: np.ndarray, label: str = "CSS") -> dict`
  - Prints and returns commutation, rank, stabilizer-count, and dimension metadata.

- `QLDPCSolver`
  - Class-based helper with `classical_demo(...)`, `css_demo(...)`, `build_hgp(...)`, `pennylane_demo()`, and `run_full_pipeline()`.

There is no repository-local `algo.run(...)` interface for this skill. If the user asks for class-style code, use `QLDPCSolver().run_full_pipeline()` or one of its specific methods.

## Core Parameters Explained

### `QLDPCSolver` (Constructor)

| Parameter | Type | Default | Description |
|---|---|---|---|
| *(none)* | — | — | Constructor takes no arguments; calls `_check_dependencies()` internally to verify optional packages. |

### `classical_demo(h, error)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h` | `np.ndarray \| None` | `hamming_code(3)` | Parity-check matrix of shape $(r, n)$. Defaults to the Hamming(3) code (shape $3 \times 7$). |
| `error` | `np.ndarray \| None` | `[1,0,1,0,0,0,0]` | Error vector of length $n$. Must be binary ($0/1$). |

### `css_demo(h_x, h_z)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h_x` | `np.ndarray \| None` | `rep_code(3)` | X-type stabilizer matrix of shape $(r_x, n)$. Defaults to repetition code with distance 3. |
| `h_z` | `np.ndarray \| None` | `[[1,1,1]]` | Z-type stabilizer matrix of shape $(r_z, n)$. Must have same column count as `h_x`. |

### `build_hgp(h1, h2)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h1` | `np.ndarray \| None` | `rep_code(3)` | First classical parity-check matrix of shape $(r_1, n_1)$. |
| `h2` | `np.ndarray \| None` | `rep_code(4)` | Second classical parity-check matrix of shape $(r_2, n_2)$. |

### `pennylane_demo()`

Takes no parameters. Runs a trivial 1-qubit QNode (`Hadamard` + `RY(θ)` → $\langle Z \rangle$) to verify the PennyLane environment.

### `run_full_pipeline()`

Takes no parameters. Runs all four stages sequentially: `classical_demo()` → `css_demo()` → `build_hgp()` → `pennylane_demo()`.

## Return Fields

### `classical_demo()` Returns

| Key | Type | Description |
|---|---|---|
| `h` | `np.ndarray` | The parity-check matrix used. |
| `error` | `np.ndarray` | The error vector. |
| `syndrome` | `np.ndarray` | Computed syndrome $s = H e \bmod 2$. |
| `tanner_graph_path` | `Path \| None` | Path to the saved Tanner graph PNG, or `None` if matplotlib/networkx unavailable. |

### `css_demo()` Returns

| Key | Type | Description |
|---|---|---|
| `commutes` | `bool` | Whether $H_X H_Z^T = 0 \pmod{2}$ holds. |
| `n_qubits` | `int` | Number of physical qubits $n$. |
| `rank_x` | `int` | Binary rank of $H_X$ over $\mathbb{Z}_2$. |
| `rank_z` | `int` | Binary rank of $H_Z$ over $\mathbb{Z}_2$. |
| `code_dimension` | `int` | Estimated logical qubit count $k = n - \mathrm{rank}(H_X) - \mathrm{rank}(H_Z)$. |
| `n_x_stabilizers` | `int` | Number of X-type stabilizer generators (rows of $H_X$). |
| `n_z_stabilizers` | `int` | Number of Z-type stabilizer generators (rows of $H_Z$). |

### `build_hgp()` Returns

| Key | Type | Description |
|---|---|---|
| `h1_shape` | `tuple` | Shape of the first classical code $(r_1, n_1)$. |
| `h2_shape` | `tuple` | Shape of the second classical code $(r_2, n_2)$. |
| `h_x_shape` | `tuple` | Shape of the constructed $H_X$ matrix. |
| `h_z_shape` | `tuple` | Shape of the constructed $H_Z$ matrix. |
| `n_physical_qubits` | `int` | Total physical qubits $n_1 n_2 + r_1 r_2$. |
| `n_x_stabilizers` | `int` | Number of X stabilizer rows. |
| `n_z_stabilizers` | `int` | Number of Z stabilizer rows. |
| `classical_k1` | `int` | Classical code dimension of $H_1$: $n_1 - \mathrm{rank}(H_1)$. |
| `classical_k2` | `int` | Classical code dimension of $H_2$: $n_2 - \mathrm{rank}(H_2)$. |
| `quantum_k` | `int` | Quantum code dimension (logical qubits). |
| `commutes` | `bool` | Whether the HGP CSS code satisfies the commutation condition. |
| `sparsity_x` | `float` | Density (fraction of ones) in $H_X$. |
| `sparsity_z` | `float` | Density (fraction of ones) in $H_Z$. |
| `h_x` | `np.ndarray` | The constructed $H_X$ matrix. |
| `h_z` | `np.ndarray` | The constructed $H_Z$ matrix. |

### `run_full_pipeline()` Returns

| Key | Type | Description |
|---|---|---|
| `classical` | `dict` | Return dict from `classical_demo()`. |
| `css` | `dict` | Return dict from `css_demo()`. |
| `hgp` | `dict` | Return dict from `build_hgp()`. |
| `pennylane_expectation` | `float \| None` | Expectation value from PennyLane demo, or `None` if unavailable. |

## Mathematical Deep Dive

### Classical LDPC Codes

A classical binary LDPC code is defined by a sparse parity-check matrix $H \in \{0,1\}^{r \times n}$ where each row defines a parity constraint on a subset of bits. For an error vector $e \in \{0,1\}^n$ (where $e_j = 1$ indicates a bit-flip on bit $j$), the syndrome is:

$$s = H e \bmod 2$$

A non-zero syndrome signals that errors have occurred. The Tanner graph representation makes the sparsity pattern visible: variable nodes (columns) connect to check nodes (rows) wherever $H_{ij} = 1$. LDPC codes are called "low-density" because each row and column of $H$ contains only $O(1)$ non-zero entries.

### CSS (Calderbank-Shor-Steane) Codes

CSS codes use two classical linear codes to define a quantum stabilizer code. Let $H_X$ (shape $r_x \times n$) define X-type stabilizers and $H_Z$ (shape $r_z \times n$) define Z-type stabilizers. The stabilizers mutually commute if and only if:

$$H_X H_Z^T = 0 \pmod{2}$$

This is the fundamental consistency condition for any CSS code. The number of encoded logical qubits is given by:

$$k = n - \mathrm{rank}_{\mathbb{Z}_2}(H_X) - \mathrm{rank}_{\mathbb{Z}_2}(H_Z)$$

where $\mathrm{rank}_{\mathbb{Z}_2}$ denotes the matrix rank computed over the binary field using Gaussian elimination modulo 2. The code can detect and correct errors whose syndrome patterns are distinguishable by the combined $H_X$ and $H_Z$ measurements.

### Hypergraph Product (HGP) Construction

The HGP construction takes two classical codes and produces a quantum CSS code with guaranteed commutation. Given classical parity-check matrices $H_1$ of shape $(r_1, n_1)$ and $H_2$ of shape $(r_2, n_2)$, the HGP CSS matrices are:

$$H_X = \left[ H_1 \otimes I_{n_2} \;\middle|\; I_{r_1} \otimes H_2^T \right]$$

$$H_Z = \left[ I_{n_1} \otimes H_2 \;\middle|\; H_1^T \otimes I_{r_2} \right]$$

where $\otimes$ is the Kronecker product (via `np.kron`). The dimensions work out as:

- $H_X$ has shape $(r_1 n_2, \; n_1 n_2 + r_1 r_2)$
- $H_Z$ has shape $(n_1 r_2, \; n_1 n_2 + r_1 r_2)$

The physical qubit count is $N = n_1 n_2 + r_1 r_2$, and the code dimension is:

$$K = k_1 \cdot k_2 = (n_1 - \mathrm{rank}(H_1)) \cdot (n_2 - \mathrm{rank}(H_2))$$

**Key property**: The CSS commutation condition $H_X H_Z^T = 0 \pmod{2}$ is satisfied by construction for any input classical codes $H_1, H_2$. This is because the block structure ensures all cross-terms cancel due to the mixed-product property of the Kronecker product, combined with the fact that $H_1 H_1^T$ and $H_2 H_2^T$ terms never appear simultaneously on opposite sides.

### Binary Rank Computation (Gaussian Elimination over $\mathbb{Z}_2$)

The rank of a binary matrix is computed by Gaussian elimination modulo 2. For each column, find a pivot row with a 1, swap to the current pivot position, then XOR the pivot row into all other rows that have a 1 in that column. The number of successful pivots is the rank. This differs from real-valued rank — for example, the repetition code matrix for distance $d$ has full $\mathbb{Z}_2$-rank of $d-1$ because its rows are linearly independent over $\mathbb{Z}_2$.

### Repetition Code and Quantum Memory

The quantum repetition code encodes one logical qubit into $d$ physical qubits as:

$$\alpha|0\rangle + \beta|1\rangle \rightarrow \alpha|0\cdots 0\rangle + \beta|1\cdots 1\rangle$$

It can correct up to $\lfloor(d-1)/2\rfloor$ bit-flip ($X$) errors using $d-1$ auxiliary qubits for syndrome extraction. The Loom/Catalyst reference tutorial extends this to a full memory experiment: encode → apply noise → extract syndrome → decode → correct → measure. However, the repetition code cannot detect phase-flip ($Z$) errors — this limitation motivates more advanced codes like the surface code and HGP codes.

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
- Script does not crash from matrix shape mismatch in `np.hstack` during HGP build.

## Debugging Tips
- `ModuleNotFoundError: pennylane`:
  - Install with `pip install pennylane`.

- Environment mismatch (packages installed but not found):
  - Activate the intended env first, e.g. `conda activate ceshiskill`.
  - Then run `python scripts/algorithm.py` from the same terminal.

- Plot window does not appear:
  - Use a local GUI backend or save figures with `plt.savefig(...)`.

- HGP `ValueError` during horizontal concatenation:
  - This usually means the HGP formula blocks were arranged incorrectly.
  - Re-check the mandatory HGP equations in this skill and verify row dimensions before `np.hstack`.

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
