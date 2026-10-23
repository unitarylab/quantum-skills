# UnitaryLab API Reference

This document lists circuit, simulator, and `unitarylab.library` interfaces for the [documented UnitaryLab version](simulator-unitarylab.md#sources-and-version). It does not define the separate `unitarylab_algorithms` contract. Follow [SKILL.md](../SKILL.md) for task routing and execution permission; use the focused references for worked workflows.

---

## Table of Contents

- [unitarylab (top-level)](#unitarylab-top-level)
- [unitarylab.core](#unitarylabcore)
  - [Circuit](#circuit)
  - [Register](#register)
  - [ClassicalRegister](#classicalregister)
- [Execution backends and TensorNet](#execution-backends-and-tensornet)
- [Density matrices and noise](simulator-unitarylab-references-density-matrix-and-noise.md)
- [Transpilation](#transpilation)
- [Serialization and OpenQASM](#serialization-and-openqasm)
- [Compilation and hardware](simulator-unitarylab-references-compilation-and-hardware.md)
- [Molecular chemistry](simulator-unitarylab-references-chemistry.md)
- [Native QML](simulator-unitarylab-references-qml.md)
- [Classical Ising/QUBO tools](simulator-unitarylab-references-qubocore.md)
- [unitarylab.library](#unitarylablibrary)
  - [Differential Operators](#differential-operators-differential_operator)
  - [QFT / IQFT](#quantum-fourier-transform-qft)
  - [QPE](#quantum-phase-estimation-qpe)
  - [LCU](#linear-combination-of-unitaries-lcu)
  - [QSP](#quantum-signal-processing-qsp)
  - [QSVT](#quantum-singular-value-transformation-qsvt)
  - [Hamiltonian Simulation](#hamiltonian-simulation-hamiltonian_simulation)
  - [Linear System Solver](#linear-system-solver-solve)
  - [Schrödingerization](#schrödingerization-solvers-schrodingerization)
  - [Equation Parser](#equation-parser-equation_parser)
  - [Block Encoding](#block-encoding-block_encoding)
  - [Pauli Operators](#pauli-operators)
  - [PDE Algorithm Classes](#pde-algorithm-classes)
- [Package boundary](#package-boundary)

---

## unitarylab (top-level)

Import the following symbols from `unitarylab`. Compiler and hardware symbols load lazily.

| Symbol | Source | Description |
|--------|--------|-------------|
| `Circuit` | `unitarylab.core` | Quantum circuit container (see [Circuit](#circuit)) |
| `Register` | `unitarylab.core` | Quantum register (see [Register](#register)) |
| `ClassicalRegister` | `unitarylab.core` | Classical register (see [ClassicalRegister](#classicalregister)) |
| `GateSequencePacked` | `unitarylab.backend.gatesequence` | Native C++ packed gate store |
| `CompileConfig` | `unitarylab.qcompiler` | Validated compiler configuration (lazy export) |
| `CompilationResult` | `unitarylab.qcompiler` | Local compilation and mapping result (lazy export) |
| `HardwareJob` | `unitarylab.qcompiler` | Asynchronous hardware submission handle (lazy export) |
| `HardwareResult` | `unitarylab.qcompiler` | Provider-neutral hardware result (lazy export) |
| `QCompiler` | `unitarylab.qcompiler` | Compiler and hardware facade (lazy export) |


---

## unitarylab.core

Import `Circuit`, `Register`, and `ClassicalRegister` from `unitarylab.core` or the top-level package.

---

### Circuit

#### `Circuit(*args, name="Circuit", gate_storage="python", **kwargs)`

`Circuit` uses a flexible constructor. It parses different initialization patterns through `*args`.

Quantum circuit container. Wraps the backend gate-sequence implementation and adds register management, circuit composition, visualization, and execution.

**Constructor overloads**

| Form | Description |
|------|-------------|
| `Circuit(n)` | Create a circuit with `n` qubits and default quantum registers |
| `Circuit(state_vector)` | Create a circuit initialized from a normalized state vector |
| `Circuit(reg1, reg2, ...)` | Create a circuit from explicit `Register` / `ClassicalRegister` objects |

For `Circuit(state_vector)`, pass a normalized one-dimensional `np.ndarray` whose length is a power of two. The constructor does not accept a Python list as a state vector.

`gate_storage` selects `"python"` (default) or `"packed"` (requires the native C++ extension). Storage selection is independent of the execution backend; see [Gate Storage](simulator-unitarylab-references-circuitsbuild.md#gate-storage).

**Circuit management**

| Method | Return | Description |
|--------|--------|-------------|
| `get_num_qubits()` | int | Total qubit count |
| `get_num_clbits()` | int | Total classical-bit count |
| `update_name(name)` | — | Rename the circuit |
| `data()` | `GateSequence` or `GateSequencePacked` | Return the underlying gate sequence object |
| `gate_storage` | str property | Gate store: `"python"` or `"packed"` |
| `copy()` | `Circuit` | Create an independent circuit copy with copied registers, mappings, and gate sequence |
| `add_register(register)` | — | Append a `Register` and assign global qubit indices |
| `add_classical_register(cl_register)` | — | Append a `ClassicalRegister` |

**Single-qubit gates** — `target` may be an int, `Register` index result, or list thereof.

| Method | Description |
|--------|-------------|
| `x(target)` | Pauli-X |
| `y(target)` | Pauli-Y |
| `z(target)` | Pauli-Z |
| `h(target)` | Hadamard |
| `s(target)` | S gate |
| `sdag(target)` | S† gate |
| `t(target)` | T gate |
| `tdag(target)` | T† gate |
| `sqrtx(target)` | √X gate |
| `sqrtxdag(target)` | √X† gate |
| `sqrty(target)` | √Y gate |
| `sqrtydag(target)` | √Y† gate |
| `i(target)` | Identity |
| `gp(angle)` | Global phase |

**Parameterized single-qubit gates**

| Method | Description |
|--------|-------------|
| `rx(angle, target)` | RX rotation |
| `ry(angle, target)` | RY rotation |
| `rz(angle, target)` | RZ rotation |
| `u1(lmb, target)` | U1 gate |
| `u2(phi, lmb, target)` | U2 gate |
| `u3(theta, phi, lmb, target)` | U3 gate |
| `p(angle, target)` | Phase gate |

**Two-qubit gates** — `control_state` can be `None`, an integer, a binary string, or a list/tuple of 0/1 values. `None` defaults to all-1 controls.

| Method | Description |
|--------|-------------|
| `swap(target1, target2)` | SWAP gate |
| `rzz(angle, target1, target2)` | Two-qubit ZZ rotation |
| `cnot(control, target, control_state=None)` | CNOT (CX) |
| `cx(control, target, control_state=None)` | CX (alias for CNOT) |
| `cy(control, target, control_state=None)` | CY |
| `cz(control, target, control_state=None)` | CZ |
| `ch(control, target, control_state=None)` | CH |
| `cs(control, target, control_state=None)` | CS |
| `cp(angle, control, target, control_state=None)` | Controlled-Phase |
| `crx(angle, control, target, control_state=None)` | Controlled-RX |
| `cry(angle, control, target, control_state=None)` | Controlled-RY |
| `crz(angle, control, target, control_state=None)` | Controlled-RZ |

**Multi-control gates** — `controls` is a list of control qubits.

| Method | Description |
|--------|-------------|
| `mcx(controls, target, control_state=None)` | Multi-controlled X |
| `mcy(controls, target, control_state=None)` | Multi-controlled Y |
| `mcz(controls, target, control_state=None)` | Multi-controlled Z |
| `mch(controls, target, control_state=None)` | Multi-controlled H |
| `mcrx(angle, controls, target, control_state=None)` | Multi-controlled RX |
| `mcry(angle, controls, target, control_state=None)` | Multi-controlled RY |
| `mcrz(angle, controls, target, control_state=None)` | Multi-controlled RZ |
| `mcp(angle, controls, target, control_state=None)` | Multi-controlled Phase |

`control_state` string/list entries correspond positionally to the supplied control-qubit sequence. Integer inputs are converted to a padded binary string in that same order. This is distinct from displayed probability strings, where qubit 0 is rightmost; do not copy a displayed string to ascending controls without checking its order.

**Custom unitary**

```python
import numpy as np
from unitarylab import Circuit

matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
qc = Circuit(1)
qc.unitary(matrix, target=[0])

result = qc.execute(backend="numpy")
print(result.probabilities)
```

Apply an arbitrary unitary matrix to `target` qubits. `matrix` must be a unitary ndarray of shape `(2^len(target), 2^len(target))`.

**Measurement**

`measure(qubit, clbit)` maps qubit(s) to equal numbers of classical bit(s). Measurement results are available from the execution result, such as `result.classical_results_map`.
See the [circuit measurement workflow](simulator-unitarylab-references-circuitsbuild.md#circuit-measurement) for register construction and counts.

**Circuit composition**

| Method | Description |
|--------|-------------|
| `append(other, target, control=None, control_state=None)` | Append a sub-circuit at the end |
| `prepend(other, target, control=None, control_state=None)` | Prepend a sub-circuit at the beginning |
| `initialize(state, target, control=None, control_state=None)` | Prepare state vector `state` on `target` qubits (must be called before any gate on those qubits) |

**Circuit transformations**

| Method | Return | Description |
|--------|--------|-------------|
| `dagger()` | `Circuit` | Conjugate transpose |
| `inverse()` | `Circuit` | Inverse circuit |
| `reverse()` | `Circuit` | Return a new circuit with qubit indices mirrored |
| `decompose(n=1, name=None)` | `Circuit` | Decompose composite gates |
| `repeat(times)` | `Circuit` | Repeat circuit `times` times |
| `control(num_control_qubits, control_state=None)` | `Circuit` | Add `num_control_qubits` control qubits |
| `transpile(gates_to_unroll=None, basis="default")` | `Circuit` | Return a transpiled circuit |
| `optimize(optimization_level=1)` | `Circuit` | Return a rule-based optimized circuit |

Gate methods mutate the circuit and return `None`; call them on separate lines rather than chaining calls.

**Execution**

| Method | Return | Description |
|--------|--------|-------------|
| `execute(initial_state=None, backend="torch", device="cpu", dtype=np.complex128, shots=1, seed=42, backend_options=None)` | `ExecutionResult` | Execute the circuit; TensorNet returns a compatible `TensorNetExecutionResult` |
| `execute_density(initial_state=None, device="cpu", dtype=np.complex128, noise_model=None, shots=1, seed=42)` | `DensityMatrixResult` | Dense mixed-state execution with circuit measurement, final-shot state/classical snapshots, and aggregated counts |
| `get_matrix(m=None, backend="torch", dtype=np.complex128, device="cpu")` | `ndarray` | Compute the circuit matrix. If `m=None`, use all qubits. |
| `compile(provider="local", machine=None, token=None)` | `CompilationResult` | Compile and map without submitting a hardware task |
| `submit(provider, machine=None, shots=1024, token=None)` | `HardwareJob` | Submit without waiting; `.result()` queries once and returns `HardwareResult` |

Supported statevector backends are `torch`, `numpy`, `cpp`, and `tensornet`. `cpp` supports CPU and, with `unitarylab[cuda]`, NVIDIA GPU execution. Torch `device="gpu"` selects an available accelerator; Apple MPS requires complex64 and at most 16 qubits. `backend_options` is TensorNet-only and accepts `max_bond`, `cutoff`, and `routing` (`"auto"`, `"swap"`, or `"mpo"`). `shots` must be a positive integer; `seed` must be a non-negative integer or `None`. See the routed density and compiler references for their distinct contracts.

**Visualization**

| Method | Return | Description |
|--------|--------|-------------|
| `draw(filename=None, title=None, compact=True, output="mpl", **kwargs)` | backend-specific | Draw with Matplotlib (`mpl`), text (`text`), or LaTeX (`latex`); save when supported and `filename` is provided. |
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

**ExecutionResult properties**

`Circuit.execute(...)` returns an `ExecutionResult`, not a separate `State` object.

| Property | Type | Description |
|----------|------|-------------|
| `state` | `np.ndarray` | Final statevector, converted lazily from the backend state. |
| `backend_state` | backend object | Raw backend state, for example a torch tensor. |
| `classical_results_map` | `dict[int, int]` | Final-shot measured classical-bit values. |
| `shots` | int | Number of executions represented by the result. |
| `counts` | `dict[str, int]` | Aggregated measured bitstrings when circuit measurements are present. |
| `classical_registers` | `dict[str, list[int]]` | Classical values grouped by register name. |
| `num_qubits` | int | Number of qubits represented by the backend state. |
| `probabilities` | dict | Full computational-basis probability distribution. |

**ExecutionResult methods**

| Method | Return | Description |
|--------|--------|-------------|
| `probability(bitstring, qubits=None)` | float | Probability of one outcome. |
| `marginal_probabilities(qubits=None, threshold=0.0)` | dict | Probability distribution on selected qubits. |
| `sample(shots=1, qubits=None, seed=None)` | list[str] | Sample without collapsing the stored state. |
| `measure(qubits, seed=None)` | str | Projectively measure selected qubits and collapse the stored state. |
| `expectation(observable, qubits=None)` | complex | Return normalized observable expectation value. |

The returned `.state` is a read-only one-dimensional NumPy array converted from `backend_state` on each access. It describes the final shot when circuit measurements are present. For TensorNet, accessing `.state` contracts the MPS to a dense vector; prefer targeted queries when the request does not require full materialization.

Supported observables are full Pauli strings, Pauli strings with explicit qubits, a finite Hermitian 2×2 matrix on one qubit, and lists of weighted tuple/mapping terms. Pauli labels are limited to `I`, `X`, `Y`, and `Z`.
For example, a weighted term can be `(0.5, "ZZ", (0, 1))` or `{"coeff": -0.25, "pauli": "X", "qubits": (0,)}`. See the [result-query workflow](simulator-unitarylab-references-circuitsbuild.md#query-results) for a circuit example.

`counts` aggregates all shots. `classical_results_map` and `classical_registers` describe the final shot. Unmeasured classical bits use `#` in count keys and `-1` in register snapshots.

---

## Execution backends and TensorNet

| Backend | Device | Native state | Notes |
|---------|--------|--------------|-------|
| `torch` | `cpu`, `gpu` | PyTorch tensor | Default backend; GPU requires a compatible PyTorch installation. |
| `numpy` | `cpu` | NumPy array | Dense statevector execution. |
| `cpp` | `cpu` | NumPy-compatible result | Requires the packaged native `cppgates` extension. |
| `cpp` | `gpu` | NumPy-compatible result | CUDA 12 C++ backend; requires `unitarylab[cuda]`. |
| `tensornet` | `cpu` | `TensorNetState` | MPS execution with truncation and routing options. |

Use `TensorNetState` for an MPS input or native MPS result:

```python
import numpy as np
from unitarylab import Circuit
from unitarylab.backend.tensornet import TensorNetState

dense_state = np.array([1, 0, 0, 0], dtype=np.complex128)
mps_state = TensorNetState.from_statevector(dense_state, max_bond=64)

qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)
result = qc.execute(
    initial_state=mps_state,
    backend="tensornet",
    backend_options={"max_bond": 64, "cutoff": 1e-10, "routing": "auto"},
)

print(result.backend_state)
print(result.backend_state.to_statevector())
```

MPS site tensors use `(left_bond, 2, right_bond)` order, with the first tensor representing qubit 0. Explicit `max_bond` and `cutoff` execution options override values stored in the input state. Statevector backends do not implicitly contract MPS inputs; call `to_statevector()` first.

## Transpilation

Prefer the circuit method for ordinary use:

```python
from unitarylab import Circuit

qc = Circuit(3)
qc.h(0)
qc.mcx([0, 1], 2)
transpiled = qc.transpile(basis="default")
print(transpiled.draw(output="text"))
```

Advanced basis configuration is available through `GateBasis`, `DEFAULT_BASIS`, `GUODUN_BASIS`, and `Unroll`.

## Serialization and OpenQASM

```python
from unitarylab import Circuit

qc = Circuit(2)
qc.rx(0.5, 0)
qc.cx(0, 1)

qasm3 = qc.to_qasm()
qasm2 = qc.to_qasm2()
restored = Circuit.from_qasm(qasm3)

qc.to_qasm_file("circuit.qasm")
from_file = Circuit.from_qasm_file("circuit.qasm", gate_storage="python")

python_source = qc.to_python(variable_name="generated")
qc.to_python_file("generated.py", variable_name="generated")
cudaq_source = qc.to_python(framework="cudaq", variable_name="generated")
```

`from_qasm()` detects OpenQASM 2.0 or 3.0 from the header. Both QASM import methods accept
keyword-only `gate_storage="python"` or `"packed"`. `to_qasm()` and `to_qasm_file()` emit
OpenQASM 3.0; `to_qasm2()` emits OpenQASM 2.0. OpenQASM 2 parsing currently ignores measurements,
so use OpenQASM 3 when classical mappings must survive a round trip. For supported composite
gates, retry QASM export with `to_qasm(decompose=True, transpile=True)`.

| Python export API | Result |
|---|---|
| `to_python(*, framework="unitarylab", circuit_name="Unitarylab Circuit", variable_name="circuit")` | Source string for UnitaryLab or CUDA-Q |
| `to_python_file(filepath, *, framework="unitarylab", circuit_name="Unitarylab Circuit", variable_name="circuit")` | Resolved `pathlib.Path` of the written source |
| `to_cudaq()` | Non-parameterized `cudaq.Kernel`; requires optional CUDA-Q installation |

UnitaryLab Python export supports ordinary gates, measurement/register mappings, nested
controlled blocks, and custom unitaries, with generic `QuantumGate` reconstruction for internal
operations. CUDA-Q source generation requires no CUDA-Q installation, but running the source does.
CUDA-Q export supports arbitrary one-/two-qubit unitaries and either no measurement or complete,
ordered terminal measurement. See [CUDA-Q](simulator-unitarylab-references-circuitsbuild.md#cuda-q) for restrictions.


## unitarylab.library

Use the import path shown in each API section. Only the primary algorithm entry points are exported directly from `unitarylab.library`.

```python
from unitarylab.library import (
    QFT,
    IQFT,
    QPE,
    LCU,
    QSP,
    QSP_hamiltonian_simulation,
    QSVT,
    block_encode,
    hamiltonian_simulation,
    solve,
)
```

---

### Differential Operators (differential_operator)

```python
from unitarylab.library.equation import (
    CDiff,
    TDiff,
    ClassicalOperator,
    TrotterOperator,
)
```

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
| `.data()` → `(A1, A2)` | sparse/dense matrix pair | Return A₁=(A+Aᵀ)/2 and A₂=(A-Aᵀ)/(2i) using plain transpose; for real A these are Hermitian coefficients in A=A₁+iA₂ |

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
| `.data()` → `(A1, A2)` | Return (A+Aᵀ)/2 and (A-Aᵀ)/(2i); plain transpose is used, so this is not a general complex-matrix Hermitian decomposition |

---

#### `TrotterOperator(H1_list=None, H2_list=None, theta_list=None, target_list=None)`

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

#### `QPE(U, d, prepare_target=None, return_circuit=False, backend='torch', device='cpu', dtype=np.complex128)`

Estimates eigenphases of unitary `U` using a `d`-qubit phase register with grid spacing `1/2**d`. For an eigenstate `|ψ⟩`, `U|ψ⟩ = e^{2πiφ}|ψ⟩`; a superposition of eigenstates produces the corresponding phase distribution. Grid spacing is not a guaranteed estimation-error bound.

| Parameter | Type | Description |
|-----------|------|-------------|
| `U` | `Circuit` | Unitary circuit whose phase is estimated |
| `d` | int | Number of phase-register qubits; phase-grid spacing is `1/2**d` |
| `prepare_target` | `Circuit` \| None | Target-register preparation circuit; omitted preparation starts in the all-zero state, which need not be an eigenstate of `U` |
| `return_circuit` | bool | If `True`, return only the constructed `Circuit` without executing |
| `backend` | str | Execution backend when `return_circuit=False`; default `"torch"` |
| `device` | str | Execution device; default `"cpu"` |
| `dtype` | dtype | Complex dtype; default `np.complex128` |

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
| `decompositions` | list[tuple[Circuit, float]] | `(unitary_circuit, coefficient)` pairs with matching circuit widths; coefficients must be finite and non-negative |

**Returns** a `Circuit` acting on ancilla + system qubits that block-encodes `A / ‖α‖₁`.

Zero-coefficient terms are skipped. At least one positive coefficient is required.

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

#### `QSP_hamiltonian_simulation(U_H, n, alpha, m, t, epsilon, beta, flag)` → `tuple[Circuit, float, int, int, int]`

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
| `flag` | bool | Sign flag: `True` for `exp(−iHt)`, `False` for `exp(iHt)` |

**Returns** a 5-tuple `(circuit, factor, n_ancilla, n_qubits, degree)`, where `circuit` is a `Circuit` that is a `(2/beta, m+2, epsilon)`-block-encoding of the time-evolution operator, `factor` is the overall block-encoding normalization, `n_ancilla` is the ancilla qubit count, `n_qubits` is the total qubit count, and `degree` is the Chebyshev polynomial degree used.

---

### Quantum Singular Value Transformation (QSVT)

```python
from unitarylab.library import QSVT
```

#### `QSVT(H, function, target_error=1e-6, block_encoding_method='nagy', backend='torch', device='cpu', dtype=np.complex128)` → `QSVTResult`

Constructs an approximation of the scalar matrix function `f(H)` for Hermitian `H`. Complex-valued functions are handled through real and imaginary components. The target acts on eigenvalues of this Hermitian matrix; it is not an arbitrary singular-value transform for non-Hermitian inputs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `H` | `np.ndarray` | Hermitian matrix |
| `function` | `Callable` | Scalar spectral function `f(x)`, evaluated on real eigenvalue samples; it must accept the scalar/array inputs used by the builder |
| `target_error` | float | Desired approximation error for polynomial fitting |
| `block_encoding_method` | str | Block-encoding backend: `'nagy'` (default) or `'fable'` |
| `backend` | str | Backend used for generated-circuit evaluation; default `"torch"` |
| `device` | str | Execution device; default `"cpu"` |
| `dtype` | dtype | Complex dtype; default `np.complex128` |

**Returns** a `QSVTResult` with `.circuit`, `.evolution_result`, `.total_error`, and `.degree`.

---

### Hamiltonian Simulation (hamiltonian_simulation)

```python
from unitarylab.library import hamiltonian_simulation
```

#### `hamiltonian_simulation(H, t, method='trotter', target_error=1e-6, backend='torch', device='cpu', dtype=np.complex128, **kwargs)` → `HamiltonianSimulationResult`

Unified interface for approximating Hamiltonian time evolution `exp(−iHt)`. Depending on the method, `.circuit` is a direct evolution circuit or a block-encoding circuit with auxiliary qubits and a normalization factor. It is not universally an n-qubit implementation of the unscaled evolution operator.

| Parameter | Type | Description |
|-----------|------|-------------|
| `H` | `np.ndarray` | Square Hermitian matrix (dimension must be a power of 2, or padded automatically) |
| `t` | float | Evolution time |
| `method` | str | Simulation algorithm (see table below) |
| `target_error` | float | Requested tolerance passed to the selected method; its use depends on that method and does not certify the returned error |
| `backend` | str | Execution backend; default `"torch"` |
| `device` | str | Execution device; default `"cpu"` |
| `dtype` | dtype | Complex dtype; default `np.complex128` |
| `**kwargs` | — | Method-specific arguments |

**Available methods:**

| `method` | Algorithm | Method-specific `**kwargs` and defaults |
|----------|-----------|-----------------------------------------|
| `'trotter'` (default) | Trotter–Suzuki product formula | `order=1`, `steps=1000` |
| `'qdrift'` | Randomized Pauli product formula | `steps=5000` |
| `'taylor'` | Taylor-series truncation | `degree=5` |
| `'qsp'` or `'qsvt'` | QSP / QSVT block-encoding approach | `degree=15`, `beta=0.7`, `block_encoding_method='nagy'` |
| `'bessel-lcu'` | Bessel/Chebyshev LCU implementation | No method-specific keywords |
| `'cartan-lax'` | Cartan decomposition via Lax-pair | `evol_time=t`, `lr=1e-3`, `max_steps=100000`, `reps=5000` |
| `'cartan-optimization'` | Cartan decomposition via optimization | `evol_time=t`, `lr=1e-3`, `max_steps=100000`, `optimizer='SD'` |

Unsupported method-specific keywords raise `ValueError`.

Parameters such as `steps`, `degree`, and optimizer settings control the actual approximation. Some methods store `target_error` without adapting their circuit to meet it; inspect `.total_error` and the selected method's contract instead of treating a normal return as a tolerance certificate.

**Returns** a `HamiltonianSimulationResult` with `.circuit`, `.evolution_result`, `.total_error`, `.method`, and `.target_qubits`.

For Taylor/LCU and QSP-based constructions, initialize the required auxiliary register and follow the selected method's block extraction, normalization, or postselection contract. `.evolution_result` can be the recovered/scaled system block rather than the raw unitary matrix of `.circuit`. Do not append an auxiliary-width circuit to a system-only register or infer a reusable direct propagator from the result class alone.

---

### Linear System Solver (solve)

```python
from unitarylab.library import solve
```

#### `solve(A, b, method='hhl', backend='torch', device='cpu', dtype=np.complex128, precondition=None, **kwargs)` → `LinearSolverResult`

Quantum linear-system solver for `Ax = b`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `np.ndarray` | Coefficient matrix |
| `b` | `np.ndarray` | Right-hand-side vector |
| `method` | str | Solver algorithm (see table below) |
| `backend` | str | Execution backend; default `"torch"` |
| `device` | str | Execution device; default `"cpu"` |
| `dtype` | dtype | Complex dtype; default `np.complex128` |
| `precondition` | str \| None | `diagonal`/`jacobi`, `symmetric`/`ic`, `ilu`, or `None` |
| `**kwargs` | — | Method-specific arguments |

**Available methods:**

| `method` | Algorithm | Extra kwargs |
|----------|-----------|-------------|
| `'hhl'` (default) | Harrow–Hassidim–Lloyd; non-Hermitian square input is handled by the SDK's Hermitian dilation | `d` (phase qubits, default 6), `t` (evolution time, auto if `None`) |
| `'qsvt'` or `'qsvt_qlsa'` | QSVT-based quantum linear algebra | `epsilon` (target approximation accuracy, default `1e-3`) |
| `'schro'` | Schrödingerization; defaults to the classical implementation | `type='classical'` by default |
| `'schro_trotter'` | Schrödingerization; explicitly select the Trotter implementation | Pass `type='trotter'` |
| `'schro_classical'` | Schrödingerization via matrix exponentiation | `type='classical'` by default |
| `'aqc'` / `'discrete_adiabatic'` | Adiabatic solver | method-specific kwargs |
| `'vqls'` | Variational quantum linear solver | method-specific kwargs |
| `'cks'` | Chebyshev Krylov solver | `epsilon` and method-specific kwargs |

**Returns** a `LinearSolverResult` with `.solution`, `.matrix`, `.rhs`, `.circuit`, `.scaling_factor`, and optional preconditioning metadata.

The current dispatcher sends `schro`, `schro_trotter`, and `schro_classical` to `SCHROSolver` without deriving its `type` argument from the method name. Select the Trotter implementation explicitly:

```python
import numpy as np
from unitarylab.library import solve

A = np.array([[1.0, 0.0], [0.0, 2.0]])
b = np.array([1.0, 0.0])
result = solve(A, b, method="schro_trotter", type="trotter")
print(result.solution)
```

The SDK HHL interface's Hermitian dilation and `t` parameter are not the separate `HHLAlgorithm.run()` wrapper contract. Check the [HHL algorithm reference](algorithm-linear-systems-hhl.md) before generating a wrapper call. Returned vectors and scaling factors are method-specific; a solver name or requested tolerance is not a guarantee of convergence or solution accuracy.

---

### Schrödingerization Solvers (schrodingerization)

```python
from unitarylab.library.equation import (
    schro_classical,
    schro_trotter,
    initial_schro_fp,
)
```

#### `schro_classical(A, u0, T=1, na=5, R=4, order=2, point=1, b=None, scale_b=0.1)` → `ndarray`

Solve the Schrödingerization-lifted Schrödinger equation via matrix exponentiation. Returns `u(T)`.

Applicable to ODE/PDE: `du/dt = A u + b`

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | sparse matrix | System matrix |
| `u0` | ndarray | Initial condition vector |
| `T` | float | Final time |
| `na` | int | Auxiliary p-direction qubits (Nₐ = 2ⁿᵃ) |
| `R` | float | p-domain range [-πR, πR] |
| `order` | int | Smoothness order of the lifting function g(p); default `2` |
| `point` | int | Recovery point index (default 1) |
| `b` | ndarray \| None | Source term vector |
| `scale_b` | float | Source term scaling factor; default `0.1` |

---

#### `schro_trotter(u0, H1=None, H2=None, Nt=1, na=5, R=4, order=2, point=1, b=None, theta=None, device='cpu', backend='torch', dtype=np.complex128)` → `(ndarray, Circuit)`

Solve the Schrödingerization-lifted Schrödinger equation via a Trotter quantum circuit. Returns `(u(T), quantum circuit)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `u0` | ndarray | Initial condition vector |
| `H1` | callable \| None | Time-step-to-circuit factory for the H₁ contribution; the real-matrix operator helpers use H₁=(A+Aᵀ)/2 |
| `H2` | callable \| None | Time-step-to-circuit factory for the H₂ contribution; the real-matrix helpers use H₂=(A-Aᵀ)/(2i), a Hermitian coefficient rather than an anti-Hermitian matrix |
| `Nt` | int | Number of Trotter time steps |
| `na` | int | Auxiliary p-direction qubits |
| `R` | float | p-domain range [-πR, πR] |
| `order` | int | Smoothness order of the lifting function g(p) |
| `point` | int | Recovery point index (≥0); `-1` sums over p≥0 |
| `b` | ndarray \| None | Source term vector |
| `theta` | float \| None | Source term strength scale (per-step time step × scale factor) |
| `device` | str | Execution device; default `'cpu'` |
| `backend` | str | Execution backend; default `'torch'` |
| `dtype` | dtype | Complex dtype; default `np.complex128` |

---

#### `initial_schro_fp(p, order=1)` → `ndarray`

Construct the initial lifting function g(p) for the auxiliary p-direction in Schrödingerization.

| Parameter | Description |
|-----------|-------------|
| `p` | Discretized p-direction grid array |
| `order` | Smoothness order (higher → smoother) |

---

### Equation Parser (equation_parser)

```python
from unitarylab.library.equation import parse_equation, Equation

from unitarylab.library.equation.equation_parser import (
    BoundaryCondition,
    DiscreteFormat,
    InitialCondition,
    Preprocessing,
    SolutionMethod,
)
```

#### `parse_equation(json_data)` → `Equation`

Parse a JSON configuration dictionary and construct a complete `Equation` object, including boundary conditions, initial values, discrete format, and solver settings.

**Common `Equation` object interface (`eq`):**

| Attribute / Method | Description |
|--------------------|-------------|
| `eq.get_common_coefficients()` | Returns `(L, T, source, nx, na, R, point, porder, f0)` |
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

```python
from unitarylab.library import block_encode
from unitarylab.library.block_encoding import BlockEncodingResult, FABLE, Nagy
```

#### `block_encode(matrix, method='nagy', eps=1e-3, verbose=False)` → `BlockEncodingResult`

Block-encode a given matrix and return a result object containing the quantum circuit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `matrix` | ndarray \| list | Input matrix |
| `method` | str | Encoding method: `'nagy'` (default) or `'fable'` |
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
| `.matrix` | Original input matrix |
| `.get_encoded_matrix(device='cpu')` | Return the extracted block multiplied by `alpha`, approximating the original input matrix |
| `.get_unitary_matrix(device='cpu')` | Return the full unitary matrix |
| `.get_max_error()` | Return `(get_encoded_matrix() - matrix).max()` as currently implemented; raises `ValueError` when the original matrix is unavailable |

**Other exported classes:**

| Class | Description |
|-------|-------------|
| `FABLE` | Fast Approximate BLock-Encodings method |
| `Nagy` | Nagy block-encoding method |

---

### Pauli Operators

```python
from unitarylab.library.pauli_operator import (
    pauli_string_decomposition,
    pauli_string_to_matrix,
    pauli_string_circuit,
    pauli_string_evolution,
)
```

#### `pauli_string_decomposition(H, sets=None, partition_commuting=True, real_symmetric_hint=False)`

Decompose a matrix into Pauli-string terms. Use `sets` to restrict the candidate Pauli words and `partition_commuting=True` to group commuting terms.

| API | Returns | Purpose |
|---|---|---|
| `pauli_string_to_matrix(decomposition)` | `ndarray` | Convert a Pauli string, term, mapping, or term list back to a matrix. |
| `pauli_string_circuit(pauli_string)` | `Circuit` | Build a circuit implementing one Pauli word. |
| `pauli_string_evolution(pauli_string, theta)` | `Circuit` | Build a parameterized Pauli-evolution circuit. |

The subpackage also exports `pauli_string_multiply`, `pauli_string_product`, and `pauli_string_power` for explicit Pauli-expression algebra.

### PDE Algorithm Classes

PDE implementations are exported from `unitarylab.library.equation.examples`:

```python
from unitarylab.library.equation.examples import HeatEquationAlgorithm
```

| Category | Classes |
|---|---|
| Heat | `HeatEquationAlgorithm`, `Heat2dEquationAlgorithm`, `HeatVariableCoefficientEquationAlgorithm` |
| Advection and transport | `AdvectionEquationAlgorithm`, `MultiTransportEquationAlgorithm`, `TrafficFlowEquationAlgorithm` |
| Backward heat | `backHeatEquationAlgorithm`, `backHeat2dEquationAlgorithm` |
| Nonlinear | `BurgersEquationAlgorithm`, `Burgers2DEquationAlgorithm`, `HamiltonJacobiEquationAlgorithm` |
| Wave and field | `ElasticWaveEquationAlgorithm`, `ElasticWave2DEquationAlgorithm`, `HelmholtzEquationAlgorithm`, `MaxwellEquationAlgorithm` |
| Finance and stochastic | `BlackScholesEquationAlgorithm`, `OUProcessEquationAlgorithm` |
| Other | `SchrABCEquationAlgorithm`, `MultiEllipticEquationAlgorithm`, `GeneralLinearEquationAlgorithm` |

Inspect the selected SDK class or use the relevant algorithm reference before constructing its task-specific parameter dictionary; PDE classes do not share one universal parameter schema. A class exported by the SDK is not necessarily exported by `unitarylab_algorithms` under the same name.

## Package Boundary

`unitarylab.library` and its equation, Pauli, block-encoding, Hamiltonian, linear-solver,
[QML](simulator-unitarylab-references-qml.md), and [classical Ising/QUBO](simulator-unitarylab-references-qubocore.md) subpackages belong to the SDK.
`unitarylab_algorithms` is a separate package. Verify the requested SDK entry against the
installed SDK version before use.
