# UnitaryLab Engine API Reference

> This document lists all public interfaces exported by `__init__.py` files under `engine`, `engine/core`, `engine/library`, and `engine/algorithms`.

---

## Table of Contents

- [engine (top-level)](#engine-top-level)
- [engine.core](#enginecore)
  - [GateSequence](#gatesequence)
  - [Register](#register)
  - [ClassicalRegister](#classicalregister)
  - [State](#state)
  - [state\_preparation](#state_preparation)
- [engine.library](#enginelibrary)
  - [Differential Operators (differential\_operator)](#differential-operators-differential_operator)
  - [Quantum Fourier Transform (QFT)](#quantum-fourier-transform-qft)
  - [Schrödingerization Solvers (schrodingerization)](#schrödingerization-solvers-schrodingerization)
  - [Equation Parser (equation\_parser)](#equation-parser-equation_parser)
  - [Block Encoding (block\_encoding)](#block-encoding-block_encoding)
- [engine.algorithms](#enginealgorithms)
  - [PDE Solvers (schrodingerization)](#pde-solvers-schrodingerization)
  - [Quantum Cryptology (cryptology)](#quantum-cryptology-cryptology)
  - [Hamiltonian Simulation (hamiltonian\_simulation)](#hamiltonian-simulation-hamiltonian_simulation)
  - [Fundamental Algorithms (fundamental\_algorithm)](#fundamental-algorithms-fundamental_algorithm)
  - [Linear Algebra (linear\_algebra)](#linear-algebra-linear_algebra)
  - [Quantum Machine Learning (quantum\_machine\_learning)](#quantum-machine-learning-quantum_machine_learning)

---

## engine (top-level)

```python
from engine import <symbol>
```

The top-level `engine` package re-exports the most commonly used symbols for convenience.

| Symbol | Source | Description |
|--------|--------|-------------|
| `set_backend` | `engine.backend` | Select the compute backend (`'torch'` or `'unitarylab'`) |
| `GateSequence` | `engine.core` | Quantum circuit container (see [GateSequence](#gatesequence)) |
| `Register` | `engine.core` | Quantum register (see [Register](#register)) |
| `ClassicalRegister` | `engine.core` | Classical register (see [ClassicalRegister](#classicalregister)) |
| `QFT` | `engine.library` | n-qubit Quantum Fourier Transform circuit |
| `IQFT` | `engine.library` | n-qubit Inverse Quantum Fourier Transform circuit |
| `state_preparation` | `engine.core` | State-preparation circuit builder (see [state\_preparation](#state_preparation)) |

---

#### `set_backend(backend='torch')`

Select the global compute backend. Must be called before constructing any circuit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `backend` | str | Backend name: `'torch'` (default) or `'unitarylab'` |

Raises `ValueError` if an unsupported backend is specified. Falls back to `'torch'` automatically when the requested backend is unavailable.

---

## engine.core

```python
from engine.core import <symbol>
# or directly:
from engine import GateSequence, Register, ClassicalRegister, state_preparation
```

---

### GateSequence

#### `GateSequence(num_qubits, *classical_registers, name='Sequence')` or `GateSequence(*registers)`

Quantum circuit container. Wraps the backend gate-sequence implementation and adds register management, circuit composition, visualization, and execution.

**Constructor overloads**

| Form | Description |
|------|-------------|
| `GateSequence(n)` | Create a circuit with `n` qubits using a single auto-named quantum register `'q'` |
| `GateSequence(n, cr1, cr2, ...)` | Same as above plus one or more `ClassicalRegister` objects |
| `GateSequence(reg1, reg2, ...)` | Create a circuit from explicit `Register` / `ClassicalRegister` objects |

**Circuit management**

| Method | Return | Description |
|--------|--------|-------------|
| `get_num_qubits()` | int | Total qubit count |
| `get_backend_type()` | str | Active backend name |
| `update_name(name)` | — | Rename the circuit |
| `data()` | list | Raw gate data of the underlying backend circuit |
| `copy()` | `GateSequence` | Shallow copy of the circuit |
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
| `dagger()` | `GateSequence` | Conjugate transpose |
| `inverse()` | `GateSequence` | Inverse circuit |
| `reverse()` | `GateSequence` | Gates in reverse order |
| `decompose(times=1)` | `GateSequence` | Decompose composite gates |
| `repeat(times=1)` | `GateSequence` | Repeat circuit `times` times |
| `control(num_ctrl_qubits=1, control_sequence=None)` | `GateSequence` | Add `num_ctrl_qubits` control qubits |

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

### state_preparation

#### `state_preparation(v, backend='torch')` → `GateSequence`

Build a quantum circuit that prepares the normalized state vector `|v⟩` using recursive amplitude encoding.

| Parameter | Type | Description |
|-----------|------|-------------|
| `v` | array-like | Normalized state vector; length must be a power of 2 |
| `backend` | str | Backend for the returned circuit (default `'torch'`) |

Raises `ValueError` if `v` is not unit-norm.

The returned `GateSequence` has `log2(len(v))` qubits and can be directly appended into a larger circuit via `qc.append(...)` or `qc.initialize(...)`.

---

## engine.library

```python
from engine.library import <symbol>
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
| `.data()` → `(H1_func, H2_func)` | `(callable, callable)` | Return two callables, each taking a time step and returning a `GateSequence` circuit |
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
| `.data()` → `(H1_func, H2_func)` | Return two callables: given a time step, produce a `GateSequence` |
| `.dagger()` | Return the inverse operator |
| `op * scalar` / `scalar * op` | Scale time step |
| `op1 + op2` | Merge two Trotter operators |

---

### Quantum Fourier Transform (QFT)

#### `QFT(n, backend='torch')` → `GateSequence`

Construct an n-qubit Quantum Fourier Transform circuit.

#### `IQFT(n, backend='torch')` → `GateSequence`

Construct an n-qubit Inverse Quantum Fourier Transform circuit (dagger of QFT).

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

#### `schro_trotter(u0, H1=None, H2=None, Nt=1, na=3, R=4, order=1, point=0, b=None, theta=None)` → `(ndarray, GateSequence)`

Solve the Schrödingerization-lifted Schrödinger equation via a Trotter quantum circuit. Returns `(u(T), quantum circuit)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `u0` | ndarray | Initial condition vector |
| `H1` | GateSequence \| None | Circuit for Hermitian part H₁ = (A+Aᵀ)/2 |
| `H2` | GateSequence \| None | Circuit for anti-Hermitian part H₂ = (A-Aᵀ)/2i |
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
| `.circuit` | Quantum circuit encoding the matrix (`GateSequence`) |
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

## engine.algorithms

```python
from engine.algorithms import <symbol>
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
