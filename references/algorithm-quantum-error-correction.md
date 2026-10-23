
# Educational qLDPC, CSS, and Hypergraph Product Codes

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

This is an educational reference, with no corresponding class exported by `unitarylab_algorithms`. `../scripts/quantum_error_correction_algorithm.py` is tutorial material; `../scripts/quantum_error_correction_qldpc_implementation.py` defines the educational `QLDPCSolver` helpers. Those interfaces are not PennyLane SDK APIs or `unitarylab_algorithms` exports.

The optional repetition-code circuit follows the official [Loom/Catalyst tutorial](https://pennylane.ai/demos/tutorial_loom_catalyst), distinct from matrix-only qLDPC construction. The HGP theory is from [Tillich and Zémor](https://arxiv.org/abs/0903.0566).

## Overview


Quantum Low-Density Parity-Check (qLDPC) codes are a family of quantum error-correcting codes constructed from classical LDPC codes. This reference covers the full pipeline from classical LDPC concepts to CSS code construction and Hypergraph Product (HGP) codes, using a practical Python implementation based on NumPy, NetworkX, Matplotlib, and PennyLane.

The pipeline proceeds in four stages:

1. **Classical LDPC**: Define a sparse binary parity-check matrix $H$, visualize its Tanner graph (bipartite graph of variable nodes and check nodes), inject an error vector $e$, and compute the syndrome $s = H e \bmod 2$.
2. **CSS Construction**: Use two binary parity-check matrices $H_X$ and $H_Z$ to define a CSS quantum code, enforce the commutation condition $H_X H_Z^T = 0 \pmod{2}$, and estimate the number of encoded logical qubits via $k = n - \mathrm{rank}(H_X) - \mathrm{rank}(H_Z)$.
3. **Hypergraph Product (HGP)**: Construct CSS checks from two classical binary matrices using the tensor-product block form. Commutation holds by construction; sparse quantum checks require sparse seed matrices.
4. **PennyLane Integration**: A tiny QNode demo confirms the PennyLane environment is functional for follow-on quantum circuit work.

NumPy performs the binary matrix construction and rank calculations; NetworkX and Matplotlib provide graph visualization. The tiny PennyLane QNode is an environment/context example, not an implementation of qLDPC encoding, syndrome extraction, or decoding.

### Key Concepts

- **Parity-check matrix**: A binary matrix $H$ of shape $(r, n)$ where each row represents a parity check and each column a bit. Sparsity (low density) is the defining property of LDPC codes.
- **Tanner graph**: Bipartite graph with $n$ variable nodes and $r$ check nodes; an edge connects check $i$ to variable $j$ when $H_{ij} = 1$.
- **Syndrome**: $s = H e \bmod 2$ — the vector that encodes which parity checks are violated by error $e$.
- **CSS commutation**: $H_X H_Z^T = 0 \pmod{2}$ ensures the X and Z stabilizers mutually commute, a necessary condition for a valid quantum code.
- **Code dimension**: $k = n - \mathrm{rank}(H_X) - \mathrm{rank}(H_Z)$ gives the number of logical qubits encoded in $n$ physical qubits.
- **HGP construction**: For $H_i$ of shape $(r_i,n_i)$, the physical-qubit count is $n_1n_2+r_1r_2$ and logical-qubit count is $K=k_1k_2+k_1^Tk_2^T$, where $k_i=n_i-\operatorname{rank}_2(H_i)$ and $k_i^T=r_i-\operatorname{rank}_2(H_i)$. The shorter product formula applies when at least one transpose-code dimension is zero.

## Algorithm-Specific Rules


- **Binary arithmetic:** Perform parity-check, syndrome, rank, and matrix operations over GF(2), not ordinary real-valued linear algebra.
- **CSS validation:** Verify `H_X H_Z^T = 0 mod 2` before treating the matrices as a valid CSS code.
- **HGP validation:** Verify matrix dimensions, block ordering, binary rank, and CSS commutation after constructing a Hypergraph Product code.
- **Educational scope:** Treat this workflow as a simulator-focused educational implementation, not a complete fault-tolerant hardware execution stack.

The educational helper's `code_dimension()` clamps `n-rank_x-rank_z` to zero and does not itself reject noncommuting checks. Treat its logical-qubit interpretation as valid only after matching column counts and CSS commutation have been checked; clamping is not code validation.

### Prerequisites

- Python compatible with the selected dependencies; [PennyLane 0.45.1](https://pypi.org/project/PennyLane/0.45.1/) requires Python 3.11 or newer.
- Core packages:
  - numpy
  - networkx
  - matplotlib
  - pennylane


### Standalone Task Code Contract

For a requested full tutorial, use the four stages below. A request for a single construction or explanation only needs its relevant stage. Save a requested Tanner graph to a file such as `tanner_graph.png` so the example does not depend on an interactive plot window.

1. Full-tutorial sections in execution order:
- Classical LDPC demo (matrix, Tanner graph, syndrome)
- CSS demo with explicit commutation check
- HGP demo with explicit commutation check
- Small PennyLane demo context (for example a tiny QNode)

2. Mathematical correctness checks:
- CSS check must print `True` for the built-in sample matrices.
- HGP check must print `True` for the built-in sample matrices.
- Code dimension computation must use binary rank over `Z_2`.

3. HGP construction uses the following block form:

$$
H_X = [H_1 \otimes I_{n_2} \mid I_{r_1} \otimes H_2^T]
$$

$$
H_Z = [I_{n_1} \otimes H_2 \mid H_1^T \otimes I_{r_2}]
$$

where `H1` has shape `(r1, n1)` and `H2` has shape `(r2, n2)`.

4. Built-in CSS sample:
- Use a commuting sample by default, e.g. `H_X = rep_code(3)` and `H_Z = [[1, 1, 1]]`.
- Avoid default samples that produce `False` commutation in the main demo.

This commuting CSS sample has ranks 2 and 1 on 3 physical qubits, so it encodes `k=0` logical qubits. It is a commutation example, not an arbitrary logical-qubit memory. The default HGP pair `rep_code(3)` and `rep_code(4)` instead gives 18 physical qubits and one logical qubit.

### Expected Output

When execution is requested, the generated script prints the classical LDPC, CSS, HGP, and PennyLane demo stages. The built-in
CSS and HGP commutation checks should print `True`; the run also saves a Tanner graph
image such as `tanner_graph.png`.

## API Reference

### Main Interface

This reference documents educational qLDPC helper interfaces for independent reconstruction; it has no `unitarylab_algorithms` class. Define these helpers in generated task code rather than importing the bundled sources.

### Constructor Parameters

#### `QLDPCSolver` (Constructor)

| Parameter | Type | Default | Description |
|---|---|---|---|
| *(none)* | — | — | Constructor takes no arguments; calls `_check_dependencies()` internally to verify optional packages. |

### Public Functions

#### `classical_demo(h, error)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h` | `np.ndarray \| None` | `hamming_code(3)` | Parity-check matrix of shape $(r,n)$; default is the rank-3 matrix for the classical `[7,4,3]` Hamming code. |
| `error` | `np.ndarray \| None` | `[1,0,1,0,0,0,0]` | Error vector of length $n$. Must be binary ($0/1$). |

#### `css_demo(h_x, h_z)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h_x` | `np.ndarray \| None` | `rep_code(3)` | X-type stabilizer matrix of shape $(r_x, n)$. Defaults to repetition code with distance 3. |
| `h_z` | `np.ndarray \| None` | `[[1,1,1]]` | Z-type stabilizer matrix of shape $(r_z, n)$. Must have same column count as `h_x`. |

#### `build_hgp(h1, h2)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `h1` | `np.ndarray \| None` | `rep_code(3)` | First classical parity-check matrix of shape $(r_1, n_1)$. |
| `h2` | `np.ndarray \| None` | `rep_code(4)` | Second classical parity-check matrix of shape $(r_2, n_2)$. |

#### `pennylane_demo()`

Takes no parameters. Runs a trivial 1-qubit QNode (`Hadamard` + `RY(θ)` → $\langle Z \rangle$) to verify the PennyLane environment.

#### `run_full_pipeline()`

Takes no parameters. Runs all four stages sequentially: `classical_demo()` → `css_demo()` → `build_hgp()` → `pennylane_demo()`.

### Return Fields


#### `classical_demo()` Returns

| Key | Type | Description |
|---|---|---|
| `h` | `np.ndarray` | The parity-check matrix used. |
| `error` | `np.ndarray` | The error vector. |
| `syndrome` | `np.ndarray` | Computed syndrome $s = H e \bmod 2$. |
| `tanner_graph_path` | `Path \| None` | Path to the saved Tanner graph PNG, or `None` if matplotlib/networkx unavailable. |

#### `css_demo()` Returns

| Key | Type | Description |
|---|---|---|
| `commutes` | `bool` | Whether $H_X H_Z^T = 0 \pmod{2}$ holds. |
| `n_qubits` | `int` | Number of physical qubits $n$. |
| `rank_x` | `int` | Binary rank of $H_X$ over $\mathbb{Z}_2$. |
| `rank_z` | `int` | Binary rank of $H_Z$ over $\mathbb{Z}_2$. |
| `code_dimension` | `int` | Logical-qubit count $k=n-\operatorname{rank}_2(H_X)-\operatorname{rank}_2(H_Z)$ for a valid commuting CSS pair; the educational helper clamps its numeric result to zero. |
| `n_x_stabilizers` | `int` | Number of X-type stabilizer generators (rows of $H_X$). |
| `n_z_stabilizers` | `int` | Number of Z-type stabilizer generators (rows of $H_Z$). |

#### `build_hgp()` Returns

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

#### `run_full_pipeline()` Returns

| Key | Type | Description |
|---|---|---|
| `classical` | `dict` | Return dict from `classical_demo()`. |
| `css` | `dict` | Return dict from `css_demo()`. |
| `hgp` | `dict` | Return dict from `build_hgp()`. |
| `pennylane_expectation` | `float \| None` | Expectation value from PennyLane demo, or `None` if unavailable. |

## Execution Flow

1. Start from a small classical parity-check matrix `H`.
2. Visualize Tanner graph to understand sparsity and constraints.
3. Inject an example error vector and compute syndrome.
4. Build CSS matrices and verify commutation.
5. Compute code dimension from binary ranks.
6. Construct HGP matrices and verify consistency.

## Implementation Details

### Algorithm Functions

- `hamming_code(rank: int) -> np.ndarray`
  - Generates a Hamming parity-check matrix.

- `binary_matrix_rank(binary_matrix: np.ndarray) -> int`
  - Computes matrix rank over binary field `Z_2`.

- `rep_code(distance: int) -> np.ndarray`
  - Creates repetition code parity-check matrix.

- `hgp_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
  - Constructs HGP CSS matrices `H_X` and `H_Z`.

### Implementation API

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

There is no public `algo.run(...)` interface for this reference. If the user asks for class-style code, independently define `QLDPCSolver` in the generated task code, then use `QLDPCSolver().run_full_pipeline()` or one of its specific methods.

### Implementation Boundaries

`QLDPCSolver` and its NumPy/NetworkX/Matplotlib functions are educational interfaces. They do not provide a public `unitarylab_algorithms` implementation, a decoder, or a hardware fault-tolerance stack.

## Algorithm-Specific Validation

### Algorithm-Specific Checks

When validation is requested, inspect:
- Script runs without import errors.
- Tanner graph renders correctly.
- Syndrome output shape and values are reasonable.
- CSS commutation checks return `True`.
- Computed code dimension matches expected theory.
- HGP commutation check returns `True`.
- Script does not crash from matrix shape mismatch in `np.hstack` during HGP build.

## Mathematical Background


### Classical LDPC Codes

A classical binary LDPC code is defined by a sparse parity-check matrix $H \in \{0,1\}^{r \times n}$ where each row defines a parity constraint on a subset of bits. For an error vector $e \in \{0,1\}^n$ (where $e_j = 1$ indicates a bit-flip on bit $j$), the syndrome is:

$$s = H e \bmod 2$$

A nonzero syndrome detects an error outside the check matrix's kernel; a zero syndrome does not prove that the error is zero. The Tanner graph connects variable nodes (columns) to check nodes (rows) wherever $H_{ij}=1$. An LDPC family has bounded check and variable degrees as length grows; an arbitrary small Hamming example illustrates parity checks but does not establish that scaling property.

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

$$K=k_1k_2+k_1^Tk_2^T,\qquad k_i=n_i-\operatorname{rank}_2(H_i),\quad k_i^T=r_i-\operatorname{rank}_2(H_i).$$

**Key property:** Multiplication gives `H_X @ H_Z.T = (H1⊗H2.T) + (H1⊗H2.T) = 0 mod 2`. Thus compatible binary seed matrices commute by construction, including rank-deficient seeds. Low-density output additionally requires low-density seed families.

### Binary Rank Computation (Gaussian Elimination over $\mathbb{Z}_2$)

The rank of a binary matrix is computed by Gaussian elimination modulo 2. For each column, find a pivot row with a 1, swap to the current pivot position, then XOR the pivot row into all other rows that have a 1 in that column. The number of successful pivots is the rank. This differs from real-valued rank — for example, the repetition code matrix for distance $d$ has full $\mathbb{Z}_2$-rank of $d-1$ because its rows are linearly independent over $\mathbb{Z}_2$.

### Repetition Code and Quantum Memory

The quantum repetition code encodes one logical qubit into $d$ physical qubits as:

$$\alpha|0\rangle + \beta|1\rangle \rightarrow \alpha|0\cdots 0\rangle + \beta|1\cdots 1\rangle$$

It can correct up to $\lfloor(d-1)/2\rfloor$ bit-flip ($X$) errors using $d-1$ auxiliary qubits for syndrome extraction. The Loom/Catalyst reference tutorial extends this to a full memory experiment: encode → apply noise → extract syndrome → decode → correct → measure. However, the repetition code cannot detect phase-flip ($Z$) errors — this limitation motivates more advanced codes like the surface code and HGP codes.

## Practical Example

### Catalyst Reference Pattern

The official [Loom/Catalyst tutorial](https://pennylane.ai/demos/tutorial_loom_catalyst) uses `qrack.simulator`; the [PennyLane-Qrack device documentation](https://pennylane-qrack.readthedocs.io/en/stable/devices/simulator.html) documents `qjit` integration. This optional snippet requires compatible Catalyst, JAX, and PennyLane-Qrack packages in addition to PennyLane. `catalyst.measure` and `catalyst.cond` provide compiled mid-circuit measurement and control flow; see the [Catalyst quick start](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/quick_start.html).

The snippet starts from the default encoded logical zero, injects at most one X error, extracts one syndrome round, and applies a distance-three lookup correction. It does not prepare an arbitrary encoded input or implement the full Loom multi-round experiment.

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



## Troubleshooting

- `ModuleNotFoundError: pennylane`:
  - Verify that the PennyLane dependency is available in the active environment.

- Plot window does not appear:
  - Use a local GUI backend or save figures with `plt.savefig(...)`.

- HGP `ValueError` during horizontal concatenation:
  - This usually means the HGP formula blocks were arranged incorrectly.
  - Re-check the mandatory HGP equations in this reference and verify row dimensions before `np.hstack`.

- Binary rank mismatch:
  - Confirm all matrices are binary (`0/1`) and operations are modulo 2.

## Limitations

- This reference is educational and simulation-oriented.
- It does not claim hardware fault-tolerance guarantees.
- Performance results from tiny examples do not imply large-scale decoder performance.

Belief-propagation decoding, general CSS noise channels, and hardware fault-tolerance experiments are outside this educational interface.
