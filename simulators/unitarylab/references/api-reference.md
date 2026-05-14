# UnitaryLab API Reference

> This document lists all public interfaces exported by `__init__.py` files under `unitarylab`, `unitarylab/core`, `unitarylab/library`, and `unitarylab/algorithms`.

---

## Table of Contents

- [unitarylab (top-level)](#unitarylab-top-level)
- [unitarylab.core](#unitarylabcore)
  - [Circuit](#Circuit)
  - [Register](#register)
  - [ClassicalRegister](#classicalregister)
- [unitarylab.library](#unitarylablibrary)
  - [Quantum Fourier Transform (QFT / IQFT)](#quantum-fourier-transform-qft)
  - [Quantum Phase Estimation (QPE)](#quantum-phase-estimation-qpe)
  - [Linear Combination of Unitaries (LCU)](#linear-combination-of-unitaries-lcu)
  - [Quantum Signal Processing (QSP)](#quantum-signal-processing-qsp)
  - [Quantum Singular Value Transformation (QSVT)](#quantum-singular-value-transformation-qsvt)
  - [Block Encoding (block\_encode)](#block-encoding-block_encoding)
  - [Hamiltonian Simulation (hamiltonian\_simulation)](#hamiltonian-simulation-hamiltonian_simulation)
  - [Linear System Solver (solve)](#linear-system-solver-solve)
  - [Differential Operators (differential\_operator)](#differential-operators-differential_operator)
  - [Schrödingerization Solvers (schrodingerization)](#schrödingerization-solvers-schrodingerization)
  - [Equation Parser (equation\_parser)](#equation-parser-equation_parser)
- [unitarylab_algorithms](#unitarylabalgorithms)
  - [PDE Solvers (schrodingerization)](#pde-solvers-schrodingerization)
  - [Quantum Cryptology (cryptology)](#quantum-cryptology-cryptology)
  - [Hamiltonian Simulation (hamiltonian\_simulation)](#hamiltonian-simulation-hamiltonian_simulation)
  - [Fundamental Algorithms (fundamental\_algorithm)](#fundamental-algorithms-fundamental_algorithm)
  - [Linear Algebra (linear\_algebra)](#linear-algebra-linear_algebra)
  - [Quantum Machine Learning (quantum\_machine\_learning)](#quantum-machine-learning-quantum_machine_learning)

---

## unitarylab (top-level)

```python
from unitarylab import <symbol>
```

The top-level `unitarylab` package re-exports the most commonly used symbols for convenience.

| Symbol | Source | Description |
|--------|--------|-------------|
| `Circuit` | `unitarylab.core` | Quantum circuit container (see [Circuit](#Circuit)) |
| `Register` | `unitarylab.core` | Quantum register (see [Register](#register)) |
| `ClassicalRegister` | `unitarylab.core` | Classical register (see [ClassicalRegister](#classicalregister)) |
| `QFT` | `unitarylab.library` | n-qubit Quantum Fourier Transform circuit |
| `IQFT` | `unitarylab.library` | n-qubit Inverse Quantum Fourier Transform circuit |

---

## unitarylab.core

```python
from unitarylab.core import <symbol>
# or directly:
from unitarylab import Circuit, Register, ClassicalRegister
```

---

### Circuit

#### `Circuit(num_qubits, *classical_registers, name='Sequence')` or `Circuit(*registers)`

Quantum circuit container. Wraps the backend gate-sequence implementation and adds register management, circuit composition, visualization, and execution.

**Constructor overloads**

| Form | Description |
|------|-------------|
| `Circuit(n)` | Create a circuit with `n` qubits using a single auto-named quantum register `'q'` |
| `Circuit(n, cr1, cr2, ...)` | Same as above plus one or more `ClassicalRegister` objects |
| `Circuit(reg1, reg2, ...)` | Create a circuit from explicit `Register` / `ClassicalRegister` objects |

**Circuit management**

| Method | Return | Description |
|--------|--------|-------------|
| `get_num_qubits()` | int | Total qubit count |
| `get_backend_type()` | str | Active backend name |
| `update_name(name)` | — | Rename the circuit |
| `data()` | list | Raw gate data of the underlying backend circuit |
| `copy()` | `Circuit` | Shallow copy of the circuit |
| `add_register(register)` | — | Append a `Register` and assign global qubit indices |
| `add_classical_register(cl_register)` | — | Append a `ClassicalRegister` |

**Single-qubit gates** — `qubit` may be an int, `Register` index result, or list thereof.

| Method | Description |
|--------|-------------|
| `x(qubit)` | Pauli-X |
| `y(qubit)` | Pauli-Y |
| `z(qubit)` | Pauli-Z |
| `h(qubit)` | Hadamard |
| `s(qubit)` | S gate |
| `sdag(qubit)` | S† gate |
| `t(qubit)` | T gate |
| `tdag(qubit)` | T† gate |
| `sqrtx(qubit)` | √X gate |
| `sqrtxdag(qubit)` | √X† gate |
| `sqrty(qubit)` | √Y gate |
| `sqrtydag(qubit)` | √Y† gate |
| `i(qubit)` | Identity |
| `gp(angle)` | Global phase |

**Parameterized single-qubit gates**

| Method | Description |
|--------|-------------|
| `rx(angle, qubit)` | RX rotation |
| `ry(angle, qubit)` | RY rotation |
| `rz(angle, qubit)` | RZ rotation |
| `u1(lambda_, qubit)` | U1 gate |
| `u2(phi, lambda_, qubit)` | U2 gate |
| `u3(theta, phi, lambda_, qubit)` | U3 gate |
| `p(angle, qubit)` | Phase gate |

**Two-qubit gates** — `control_sequence` is an optional binary string specifying active control states (e.g. `'0'` for control-on-zero).

| Method | Description |
|--------|-------------|
| `swap(qubit1, qubit2)` | SWAP gate |
| `cnot(control, target, control_sequence=None)` | CNOT (CX) |
| `cx(control, target, control_sequence=None)` | CX (alias for CNOT) |
| `cy(control, target, control_sequence=None)` | CY |
| `cz(control, target, control_sequence=None)` | CZ |
| `ch(control, target, control_sequence=None)` | CH |
| `cs(control, target, control_sequence=None)` | CS |
| `cp(angle, control, target, control_sequence=None)` | Controlled-Phase |
| `crx(angle, control, target, control_sequence=None)` | Controlled-RX |
| `cry(angle, control, target, control_sequence=None)` | Controlled-RY |
| `crz(angle, control, target, control_sequence=None)` | Controlled-RZ |

**Multi-control gates** — `controls` is a list of control qubits.

| Method | Description |
|--------|-------------|
| `mcx(controls, target, control_sequence=None)` | Multi-controlled X |
| `mcy(controls, target, control_sequence=None)` | Multi-controlled Y |
| `mcz(controls, target, control_sequence=None)` | Multi-controlled Z |
| `mch(controls, target, control_sequence=None)` | Multi-controlled H |
| `mcrx(angle, controls, target, control_sequence=None)` | Multi-controlled RX |
| `mcry(angle, controls, target, control_sequence=None)` | Multi-controlled RY |
| `mcrz(angle, controls, target, control_sequence=None)` | Multi-controlled RZ |
| `mcp(angle, controls, target, control_sequence=None)` | Multi-controlled Phase |

**Custom unitary**

```python
qc.unitary(matrix, target, control=[], control_sequence=None)
```

Apply an arbitrary unitary matrix to `target` qubits. `matrix` must be a unitary ndarray of shape `(2^len(target), 2^len(target))`.

**Measurement**

```python
qc.measure(target, clbit)
```

Map `target` qubit(s) to `clbit` classical bit(s). Results are stored in the corresponding `ClassicalRegister.values` after `execute()`.

**Circuit composition**

| Method | Description |
|--------|-------------|
| `append(block, target, control=[], control_sequence=None)` | Append a sub-circuit at the end |
| `prepend(block, target, control=[], control_sequence=None)` | Prepend a sub-circuit at the beginning |
| `initialize(v, target, control=[], control_sequence=None)` | Prepare state vector `v` on `target` qubits (must be called before any gate on those qubits) |

**Circuit transformations**

| Method | Return | Description |
|--------|--------|-------------|
| `dagger()` | `Circuit` | Conjugate transpose |
| `inverse()` | `Circuit` | Inverse circuit |
| `reverse()` | `Circuit` | Gates in reverse order |
| `decompose(times=1)` | `Circuit` | Decompose composite gates |
| `repeat(times=1)` | `Circuit` | Repeat circuit `times` times |
| `control(num_ctrl_qubits=1, control_sequence=None)` | `Circuit` | Add `num_ctrl_qubits` control qubits |

**Execution**

| Method | Return | Description |
|--------|--------|-------------|
| `execute(initial_state=None)` | `ndarray` | Simulate the circuit. `initial_state` is a length-`2^n` complex vector; defaults to `|0…0⟩`. Returns the final statevector. |
| `get_matrix(m=0)` | `ndarray` | Extract the `2^m × 2^n` unitary matrix. Uses all qubits when `m ≤ 0`. |

**Visualization**

| Method | Return | Description |
|--------|--------|-------------|
| `draw(filename=None, title=None, style='dark', compact=True)` | Figure | Render circuit diagram. Saves to file when `filename` is provided. |
| `analyze(sections=None, show=True, qubit=None)` | `CircuitInfo` | Print/return circuit analysis (gate count, depth, qubit history, etc.). |

---

### Register

#### `Register(name, n_qubits)`

Quantum register — a named group of qubits with Python-style indexing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Register label |
| `n_qubits` | int | Number of qubits |

**Indexing** — Use the result directly as a gate target or control argument.

| Expression | Description |
|------------|-------------|
| `r[i]` | Single qubit at index `i` |
| `r[i:j]` | Slice of qubits |
| `r[[i, j, ...]]` | Arbitrary subset |

**Other methods**

| Method | Return | Description |
|--------|--------|-------------|
| `len(r)` | int | Number of qubits |

---

### ClassicalRegister

#### `ClassicalRegister(name, n_bits)`

Classical register for storing measurement results.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Register label |
| `n_bits` | int | Number of classical bits |

**Attributes**

| Attribute | Type | Description |
|-----------|------|-------------|
| `values` | list[int] | Measurement results; `-1` means unmeasured |
| `n_qubits` | int | Number of bits (alias for `n_bits`) |

**Indexing** — Same syntax as `Register`: `cr[i]`, `cr[i:j]`, `cr[[i, j]]`.

---

### State

#### `State(data, num_qubits=None)`

Quantum statevector class backed by PyTorch tensors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | int \| Tensor \| list \| ndarray | If int, creates a `2^n`-dimensional `|0…0⟩` state; otherwise interprets as a state vector (automatically normalized). |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `data` | `torch.Tensor` | Raw state vector (complex128, shape `(2^n,)`) |
| `num_qubits` | int | Number of qubits |
| `dim` | int | Hilbert space dimension (`2^n`) |
| `dtype` | dtype | Tensor dtype |

**Methods**

| Method | Return | Description |
|--------|--------|-------------|
| `norm()` | float | L2 norm of the state vector |
| `normalize()` | self | Normalize in-place |
| `inner_product(other)` | complex | ⟨self\|other⟩ |
| `tensor(other)` | `State` | Tensor product self ⊗ other |
| `expectation_value(matrix)` | complex | ⟨ψ\|O\|ψ⟩ for operator matrix |
| `probabilities()` | `Tensor` | Probability of every basis state |
| `probabilities_dict(target_indices, endian='little', threshold=1e-9)` | dict | Marginal probability dict `{bitstring: prob}` for selected qubits |
| `measure(target_indices, endian='little')` | str | Collapse measurement; returns outcome bitstring and updates internal state |
| `sample_counts(shots=1024)` | dict | Simulated measurement sampling: `{bitstring: count}` |
| `calculate_state(target_indices, endian='little', threshold=1e-5)` | dict | Detailed probability dict `{bitstring: {'prob': float, 'int': int}}` |

---

## unitarylab.library

```python
from unitarylab.library import <symbol>
```

---

### Differential Operators (differential_operator)

#### `CDiff(N, dx, order=1, scheme='central', boundary='dirichlet')`

Classical central finite-difference operator. Inherits from `ClassicalOperator`. Returns a sparse-matrix differential operator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `N` | int | Number of grid points |
| `dx` | float | Grid spacing |
| `order` | int | Derivative order (0–4) |
| `scheme` | str | Finite-difference scheme (only `'central'` supported) |
| `boundary` | str | Boundary condition: `'dirichlet'`, `'periodic'`, or `'neumann'` |

**Methods:**

| Method | Return | Description |
|--------|--------|-------------|
| `.get_matrix()` | `scipy.sparse.csc_matrix` | Return the difference matrix |
| `.data()` → `(A1, A2)` | `(ndarray, ndarray)` | Return Hermitian part H₁=(A+Aᵀ)/2 and anti-Hermitian part H₂=(A-Aᵀ)/2i |

---

#### `TDiff(n, dx, order=1, scheme='central', boundary='dirichlet', target=None)`

Trotter-decomposition-based quantum differential operator. Inherits from `TrotterOperator`. Used to build quantum circuits.

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | int | Number of spatial qubits (grid size Nx = 2ⁿ) |
| `dx` | float | Grid spacing |
| `order` | int | Derivative order (0–4) |
| `scheme` | str | Finite-difference scheme (only `'central'` supported) |
| `boundary` | str | Boundary condition: `'dirichlet'`, `'periodic'`, or `'neumann'` |
| `target` | list | Target qubit indices (default `range(n)`) |

**Methods:**

| Method | Return | Description |
|--------|--------|-------------|
| `.data()` → `(H1_func, H2_func)` | `(callable, callable)` | Return two callables, each taking a time step and returning a `Circuit` circuit |
| `.dagger()` | `TrotterOperator` | Return Hermitian conjugate (time reversal) |
| `op * scalar` | `TrotterOperator` | Scale time step by a scalar |

---

#### `ClassicalOperator(matrix=0)`

Base class for classical sparse-matrix operators. Supports addition, subtraction, and multiplication.

| Method | Description |
|--------|-------------|
| `.get_matrix()` | Return the operator matrix |
| `.data()` → `(A1, A2)` | Return H₁=(A+Aᵀ)/2 and H₂=(A-Aᵀ)/2i |

---

#### `TrotterOperator(H1_list, H2_list, theta_list, target_list)`

Base class for quantum Trotter operators. Manages multiple Trotter terms.

| Method | Description |
|--------|-------------|
| `.data()` → `(H1_func, H2_func)` | Return two callables: given a time step, produce a `Circuit` |
| `.dagger()` | Return the inverse operator |
| `op * scalar` / `scalar * op` | Scale time step |
| `op1 + op2` | Merge two Trotter operators |

---

### Quantum Fourier Transform (QFT)

#### `QFT(n)` → `Circuit`

Construct an n-qubit Quantum Fourier Transform circuit.

#### `IQFT(n)` → `Circuit`

Construct an n-qubit Inverse Quantum Fourier Transform circuit (dagger of QFT).

---

### Quantum Phase Estimation (QPE)

```python
from unitarylab.library import QPE
```

#### `QPE(U, d, prepare_target=None, return_circuit=False)`

Estimates the eigenphase φ of unitary `U` such that `U|ψ⟩ = e^{2πiφ}|ψ⟩`, using a `d`-qubit phase register (precision 1/2^d).

| Parameter | Type | Description |
|-----------|------|-------------|
| `U` | `Circuit` | Unitary circuit whose phase is estimated |
| `d` | int | Number of phase-register qubits; precision = `1/2^d` |
| `prepare_target` | `Circuit` \| None | Circuit preparing the eigenstate `|ψ⟩`; defaults to `|0⟩` |
| `return_circuit` | bool | If `True`, return only the constructed `Circuit` without executing |

**Returns** `(circuit, phi_est, probability)` — the QPE circuit, the estimated phase in `[0, 1)`, and its measurement probability.  
If `return_circuit=True`, returns only the `Circuit`.

---

### Linear Combination of Unitaries (LCU)

```python
from unitarylab.library import LCU
```

#### `LCU(decompositions)` → `Circuit`

Builds a quantum circuit implementing `A = Σⱼ αⱼ Uⱼ` using the LCU technique (PREPARE + SELECT + PREPARE†).

| Parameter | Type | Description |
|-----------|------|-------------|
| `decompositions` | list[tuple[Circuit, float]] | List of `(unitary_circuit, coefficient)` pairs; all circuits must have the same qubit count; coefficients must be finite and positive |

**Returns** a `Circuit` acting on ancilla + system qubits that block-encodes `A / ‖α‖₁`.

---

### Quantum Signal Processing (QSP)

```python
from unitarylab.library import QSP, QSP_hamiltonian_simulation
```

#### `QSP(U, n, m, coef, parity, eps=1e-12, maxiter=200, is_coef_cheby=False)` → `Circuit`

Builds a QSP circuit that applies a polynomial transformation to a block-encoded unitary.

| Parameter | Type | Description |
|-----------|------|-------------|
| `U` | `Circuit` | Block-encoding unitary circuit |
| `n` | int | Number of register (system) qubits |
| `m` | int | Number of auxiliary qubits |
| `coef` | array-like | Polynomial coefficients (monomial or Chebyshev basis) |
| `parity` | int | Parity of the target polynomial: `0` = even, `1` = odd |
| `eps` | float | Convergence tolerance for the phase solver |
| `maxiter` | int | Maximum phase-solver iterations |
| `is_coef_cheby` | bool | If `True`, `coef` is already in the Chebyshev basis |

**Returns** a `Circuit` on `n + m + 1` qubits.

---

#### `QSP_hamiltonian_simulation(U_H, n, alpha, m, t, epsilon, beta, flag)` → `Circuit`

Simulates `exp(−iHt)` or `exp(iHt)` via QSP, given a `(alpha, m, 0)`-block-encoding `U_H` of Hamiltonian `H`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `U_H` | `Circuit` | Block-encoding circuit of the Hamiltonian |
| `n` | int | Number of register (system) qubits |
| `alpha` | float | Block-encoding normalization factor |
| `m` | int | Number of auxiliary qubits in the block encoding |
| `t` | float | Evolution time |
| `epsilon` | float | Approximation error |
| `beta` | float | Normalization parameter for the output block-encoding |
| `flag` | int | Sign flag: `0` for `exp(−iHt)`, `1` for `exp(iHt)` |

**Returns** a `Circuit` that is a `(2/beta, m+2, epsilon)`-block-encoding of the time-evolution operator.

---

### Quantum Singular Value Transformation (QSVT)

```python
from unitarylab.library import QSVT
```

#### `QSVT(H, function, target_error=1e-6, block_encoding_method='nagy')` → `QSVTResult`

Applies a scalar function `f(H)` to a Hermitian matrix `H` using the QSVT framework. Complex-valued functions are handled by splitting into real and imaginary parts.

| Parameter | Type | Description |
|-----------|------|-------------|
| `H` | `np.ndarray` | Hermitian matrix |
| `function` | `Callable` | Scalar function `f(x)`; applied element-wise to singular values |
| `target_error` | float | Desired approximation error for polynomial fitting |
| `block_encoding_method` | str | Block-encoding backend: `'nagy'` (default) or `'fable'` |

**Returns** a `QSVTResult` object with `.circuit`, `.result_matrix`, and error metrics.

---

### Hamiltonian Simulation (hamiltonian_simulation)

```python
from unitarylab.library import hamiltonian_simulation
```

#### `hamiltonian_simulation(H, t, method='trotter', target_error=1e-6, **kwargs)` → `HamiltonianSimulationResult`

Unified interface for Hamiltonian time-evolution. Constructs a circuit approximating `exp(−iHt)` using the selected method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `H` | `np.ndarray` | Square Hermitian matrix (dimension must be a power of 2, or padded automatically) |
| `t` | float | Evolution time |
| `method` | str | Simulation algorithm (see table below) |
| `target_error` | float | Target precision (stored in result; some methods use it for step selection) |
| `**kwargs` | — | Method-specific arguments |

**Available methods:**

| `method` | Algorithm |
|----------|-----------|
| `'trotter'` (default) | First/second-order Trotter–Suzuki product formula |
| `'qdrift'` | Randomized qDRIFT channel |
| `'taylor'` | Taylor-series truncation |
| `'qsp'` or `'qsvt'` | QSP / QSVT block-encoding approach |
| `'cartan-lax'` | Cartan decomposition via Lax-pair |
| `'cartan-optimization'` | Cartan decomposition via optimization |

**Returns** a `HamiltonianSimulationResult` with `.circuit`, `.exact_matrix`, `.approx_matrix`, and `.error`.

---

### Linear System Solver (solve)

```python
from unitarylab.library import solve
```

#### `solve(A, b, method='hhl', **kwargs)` → `LinearSolverResult`

Quantum linear-system solver for `Ax = b`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `np.ndarray` | Coefficient matrix |
| `b` | `np.ndarray` | Right-hand-side vector |
| `method` | str | Solver algorithm (see table below) |
| `**kwargs` | — | Method-specific arguments |

**Available methods:**

| `method` | Algorithm | Extra kwargs |
|----------|-----------|-------------|
| `'hhl'` (default) | Harrow–Hassidim–Lloyd; requires Hermitian `A` | `d` (phase qubits, default 8), `t` (evolution time, auto if `None`) |
| `'qsvt'` or `'qsvt_qlsa'` | QSVT-based quantum linear algebra | `epsilon` (polynomial accuracy, default 0.01) |
| `'schro'` | Schrödingerization (auto select) | — |
| `'schro_trotter'` | Schrödingerization via Trotter circuit | — |
| `'schro_classical'` | Schrödingerization via matrix exponentiation | — |

**Returns** a `LinearSolverResult` with `.solution`, `.matrix`, `.rhs`, `.circuit`, `.scaling_factor`.

---

### Schrödingerization Solvers (schrodingerization)

#### `schro_classical(A, u0, T=1, na=5, R=4, order=1, point=1, b=None, scale_b=1)` → `ndarray`

Solve the Schrödingerization-lifted Schrödinger equation via matrix exponentiation. Returns `u(T)`.

Applicable to ODE/PDE: `du/dt = A u + b`

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | sparse matrix | System matrix |
| `u0` | ndarray | Initial condition vector |
| `T` | float | Final time |
| `na` | int | Auxiliary p-direction qubits (Nₐ = 2ⁿᵃ) |
| `R` | float | p-domain range [-πR, πR] |
| `order` | int | Smoothness order of the lifting function g(p) |
| `point` | int | Recovery point index (default 1) |
| `b` | ndarray \| None | Source term vector |
| `scale_b` | float | Source term scaling factor |

---

#### `schro_trotter(u0, H1=None, H2=None, Nt=1, na=3, R=4, order=1, point=0, b=None, theta=None)` → `(ndarray, Circuit)`

Solve the Schrödingerization-lifted Schrödinger equation via a Trotter quantum circuit. Returns `(u(T), quantum circuit)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `u0` | ndarray | Initial condition vector |
| `H1` | Circuit \| None | Circuit for Hermitian part H₁ = (A+Aᵀ)/2 |
| `H2` | Circuit \| None | Circuit for anti-Hermitian part H₂ = (A-Aᵀ)/2i |
| `Nt` | int | Number of Trotter time steps |
| `na` | int | Auxiliary p-direction qubits |
| `R` | float | p-domain range [-πR, πR] |
| `order` | int | Smoothness order of the lifting function g(p) |
| `point` | int | Recovery point index (≥0); `-1` sums over p≥0 |
| `b` | ndarray \| None | Source term vector |
| `theta` | float \| None | Source term strength scale (per-step time step × scale factor) |

---

#### `initial_schro_fp(p, order=1)` → `ndarray`

Construct the initial lifting function g(p) for the auxiliary p-direction in Schrödingerization.

| Parameter | Description |
|-----------|-------------|
| `p` | Discretized p-direction grid array |
| `order` | Smoothness order (higher → smoother) |

---

### Equation Parser (equation_parser)

#### `parse_equation(json_data)` → `Equation`

Parse a JSON configuration dictionary and construct a complete `Equation` object, including boundary conditions, initial values, discrete format, and solver settings.

**Common `Equation` object interface (`eq`):**

| Attribute / Method | Description |
|--------------------|-------------|
| `eq.get_common_coefficients()` | Returns `(L, T, nx, na, R, order, point, f0)` and other common parameters |
| `eq.boundary.type` | Boundary condition type string |
| `eq.initial` | Initial condition object |
| `eq.solver` | Solution method object |

**Other exported classes (rarely constructed directly):**

| Class | Description |
|-------|-------------|
| `BoundaryCondition` | Boundary condition object |
| `DiscreteFormat` | Discrete format object |
| `Equation` | Equation object |
| `InitialCondition` | Initial condition object |
| `Preprocessing` | Hamiltonian preprocessing object |
| `SolutionMethod` | Solution method object |

---

### Block Encoding (block_encoding)

#### `block_encode(matrix, method='fable', eps=1e-3, verbose=False)` → `BlockEncodingResult`

Block-encode a given matrix and return a result object containing the quantum circuit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix` | ndarray \| list | Input matrix |
| `method` | str | Encoding method: `'fable'` (default) or `'nagy'` |
| `eps` | float | Compression threshold for the FABLE method |
| `verbose` | bool | Whether to print details |

**`BlockEncodingResult` attributes:**

| Attribute | Description |
|-----------|-------------|
| `.circuit` | Quantum circuit encoding the matrix (`Circuit`) |
| `.alpha` | Normalization coefficient such that A/alpha is block-encoded |
| `.total_qubits` | Total number of qubits |
| `.target_qubits` | Number of target (system) qubits |
| `.method` | Encoding method used |
| `.eps` | FABLE compression threshold |
| `.get_encoded_matrix()` | Return the encoded matrix |
| `.get_unitary_matrix()` | Return the full unitary matrix |
| `.get_max_error()` | Return the maximum error norm of the encoding |

**Other exported classes:**

| Class | Description |
|-------|-------------|
| `FABLE` | Fast Approximate BLock-Encodings method |
| `Nagy` | Nagy block-encoding method |

---

## unitarylab_algorithms

```python
from unitarylab_algorithms import <symbol>
```

---

### PDE Solvers (schrodingerization)

All PDE algorithm classes inherit from `BaseAlgorithm` and are executed via `.run(eq)` or `.solve(eq)`.

| Class | Equation |
|-------|----------|
| `HeatEquationAlgorithm` | 1D heat equation |
| `Heat2dEquationAlgorithm` | 2D heat equation |
| `HeatVariableCoefficientEquationAlgorithm` | 1D variable-coefficient heat equation |
| `backHeatEquationAlgorithm` | 1D backward heat equation (ill-posed) |
| `backHeat2dEquationAlgorithm` | 2D backward heat equation |
| `AdvectionEquationAlgorithm` | 1D advection equation |
| `BurgersEquationAlgorithm` | 1D Burgers equation |
| `Burgers2DEquationAlgorithm` | 2D Burgers equation |
| `BlackScholesEquationAlgorithm` | 1D Black-Scholes equation |
| `ElasticWaveEquationAlgorithm` | 1D elastic wave equation |
| `ElasticWave2DEquationAlgorithm` | 2D elastic wave equation |
| `HelmholtzEquationAlgorithm` | 1D Helmholtz equation |
| `MaxwellEquationAlgorithm` | Maxwell equations |
| `MultiEllipticEquationAlgorithm` | Multiscale elliptic equation |
| `MultiTransportEquationAlgorithm` | Multiscale transport equation |
| `OUProcessEquationAlgorithm` | Ornstein-Uhlenbeck process |
| `SchrABCEquationAlgorithm` | Schrödinger equation with artificial boundary conditions |
| `TrafficFlowEquationAlgorithm` | Traffic flow equation |
| `HamiltonJacobiEquationAlgorithm` | Hamilton-Jacobi equation |
| `BaseAlgorithm` | Base class for all PDE algorithms |

---

### Quantum Cryptology (cryptology)

| Class | Algorithm |
|-------|-----------|
| `SimonAlgorithm` | Simon's algorithm (hidden subgroup problem) |
| `ShorAlgorithm` | Shor's factoring algorithm |
| `DiscreteLogAlgorithm` | Discrete logarithm algorithm |

---

### Hamiltonian Simulation (hamiltonian_simulation)

| Class | Algorithm |
|-------|-----------|
| `SuzukiTrotterAlgorithm` | Suzuki-Trotter product formula |
| `CartanDecompositionAlgorithm` | Cartan decomposition exact simulation |
| `CrankNicolsonAlgorithm` | Crank-Nicolson implicit time evolution |
| `QDriftAlgorithm` | qDRIFT randomized Trotter method |
| `TridiagonalSimulationAlgorithm` | Tridiagonal Hamiltonian simulation |

---

### Fundamental Algorithms (fundamental_algorithm)

| Class | Algorithm |
|-------|-----------|
| `AmplitudeAmplificationAlgorithm` | Amplitude amplification (generalized Grover) |
| `AmplitudeEstimationAlgorithm` | Amplitude estimation |
| `HadamardTransformAlgorithm` | Hadamard transform |
| `HadamardTestAlgorithm` | Hadamard test |
| `QPEAlgorithm` | Quantum Phase Estimation (QPE) |

---

### Linear Algebra (linear_algebra)

| Class | Algorithm |
|-------|-----------|
| `HHLAlgorithm` | HHL linear systems algorithm |
| `LCUAlgorithm` | Linear Combination of Unitaries (LCU) |
| `QSVTLinearSolverAlgorithm` | Quantum Singular Value Transformation linear solver (QSVT-QLSA) |
| `QSPAlgorithm` | Quantum Signal Processing (QSP) |

---

### Quantum Machine Learning (quantum_machine_learning)

| Class | Algorithm |
|-------|-----------|
| `QNNAlgorithm` | Quantum Neural Network (QNN) |
| `QCBMAlgorithm` | Quantum Circuit Born Machine (QCBM) |
| `CVQNNAlgorithm` | Continuous-Variable Quantum Neural Network (CV-QNN) |
| `VQCAlgorithm` | Variational Quantum Classifier (VQC) |
| `QAOAAlgorithm` | Quantum Approximate Optimization Algorithm (QAOA) |
| `VQEAlgorithm` | Variational Quantum Eigensolver (VQE) |
