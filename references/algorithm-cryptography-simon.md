# Simon's Algorithm

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

Simon's problem supplies a function $f:\{0,1\}^n\to\{0,1\}^n$ with a nonzero secret $s$ and promise $f(x)=f(y)$ iff $x=y$ or $x\oplus y=s$. The standard quantum algorithm recovers $s$ with $O(n)$ queries and bounded error, whereas classical randomized query cost is $\Omega(2^{n/2})$.

The documented implementation constructs a CNOT oracle from the supplied nonzero secret `s`; it is not an arbitrary black-box oracle interface. The theoretical query separation does not imply a speedup for classical statevector simulation.

Source pointers: [algorithm-library usage example](../scripts/simon_algorithm.py) and [independent teaching implementation](../scripts/simon_implementation.py). Keep the teaching solver separate from `SimonAlgorithm`.

Algorithm-library source: `SimonAlgorithm` (`unitarylab_algorithms/cryptology/simon/algorithm.py`).

## Overview

Simon's algorithm:
1. Applies Hadamard gates to the $n$-qubit input register.
2. Queries the oracle $U_f$: entangles $f(x)$ into the output register.
3. Measures the output register (mid-circuit measurement), collapsing the input register to a superposition of $\{x_0, x_0 \oplus s\}$.
4. Applies Hadamard gates again to the input register.
5. Measures the input register to obtain a bit-string $y$ satisfying $y \cdot s \equiv 0 \pmod{2}$.
6. Repeats $O(n)$ times to collect linearly independent equations, then solves for $s$ via Gaussian elimination over $\mathbb{F}_2$.

## Prerequisites

- Hadamard gates and their effect on computational basis states.
- XOR (bitwise addition mod 2) and binary linear algebra.
- Python: `numpy`, `Circuit`, `Register`, `ClassicalRegister`.

## Reference Implementation Example

```python
from unitarylab_algorithms import SimonAlgorithm

algo = SimonAlgorithm()
result = algo.run(
    s='1010',      # Hidden string to find
    backend='torch'
)

print(result['Computed s'])      # Found hidden string (should match s)
print(result['status'])          # 'ok' if found == s
print(result['circuit_path'])    # SVG circuit diagram path
print(result['plot'])            # List of saved file dicts [{"format": ..., "filename": ...}]
```

## Core Parameters Explained

**`__init__(text_mode, algo_dir)` — Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Output text formatting mode. |
| `algo_dir` | `str` or `None` | `None` | Directory to save results; auto-derived from cwd if `None`. |

**`run(s, backend, device, dtype)` — Run parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `s` | `str` | `'1010'` | The hidden binary string $s$ to be found. Must have at least one `'1'`. |
| `backend` | `str` | `'torch'` | Forwarded to the SDK; the selected backend must support the circuit's mid-circuit measurement. `run()` does not enforce a Torch-only restriction. |
| `device` | `str` | `'cpu'` | Device for simulation (e.g. `'cpu'` or `'cuda'`). |
| `dtype` | dtype | `np.complex128` | Numerical dtype for simulation. |

**Common misunderstandings:**
- `s` cannot be all-zero (`'0000'`); this is the trivial case with no hidden structure.
- This implementation executes the circuit once, then reads all retained input-register marginal probabilities above `1e-5`. It does not collect a configurable number of input samples or implement the theoretical $O(n)$ repeated-query protocol.
- `s` must be a binary string containing a one. The source explicitly checks only for the presence of `'1'`; it does not fully validate the binary-string domain.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if `found_s == s`; `'failed'` otherwise. |
| `Computed s` | `str` | The binary string $s$ recovered by the classical post-processing step. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `plot` | `list` | List of saved file dicts, each with `'format'` and `'filename'` keys. |
| `circuit` | `Circuit` | The `Circuit` object constructed for the algorithm. |
| `Valid states` | `int` | Number of valid basis states measured from the simulation. |
| `computation time (s)` | `float` | Quantum simulation wall-clock time in seconds. |
| `Register size` | `int` | Register size $n$ (derived from `len(s)`). |
| `Equations` | `int` | Number of linearly independent $\mathbb{F}_2$ equations extracted. |

## Implementation Architecture

`SimonAlgorithm` in `algorithm.py` structures the algorithm into five ordered stages inside `run()`, with three classical helper methods for circuit oracle construction and linear algebra post-processing.

**`run(s, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Validation | Extracts `n = len(s)`; checks for all-zero string | Guards against trivial inputs |
| 2 — Circuit Construction | Creates `Circuit(rx, ry, cqr)` with two `n`-qubit registers + classical register; applies `qc.h(rx[:])`, calls `_build_simon_oracle(qc, s)`, measures `ry` to `cqr`, applies `qc.h(rx[:])` again | Full Simon protocol circuit in 4 lines |
| 3 — Simulation | `qc.execute(backend=backend, device=device, dtype=dtype)` → `result.marginal_probabilities(qubits=range(n), threshold=1e-5)` | Runs mid-circuit-measurement simulation; extracts x-register basis dictionary |
| 4 — Classical Post-Processing | `_get_basis_simple(state_basis_dict.keys(), n)` extracts $n-1$ linearly independent vectors; `_solve_simon_general(basis, n)` back-substitutes to recover `s` | $\mathbb{F}_2$ linear algebra |
| 5 — Export | `self.save_circuit(qc)` / `self.save_txt()` → `_build_return_dict(is_success, circuit_path, filename, qc)` | Saves SVG circuit + text results; packages return dict |

**Helper Methods:**

- **`_build_simon_oracle(qc, s)`** — Implements $U_f$ with CNOT gates. First, copies $x$ to $y$ register via `cx(i, i+n)` for each bit. Then, for each bit `i` where `s[i]='1'`, applies `cx(n-1-pivot_idx, n-1-i+n)` (where `pivot_idx = s.find('1')`) to create the $f(x) = f(x\oplus s)$ mapping.
- **`_get_basis_simple(state_list, n_qubits)`** — Greedy pivot selection: iterates bitstrings, keeps the first bitstring with each distinct leftmost-1 pivot position. Returns at most $n-1$ linearly independent vectors.
- **`_solve_simon_general(basis_list, n)`** — Back-substitution over $\mathbb{F}_2$: finds the free variable index (the non-pivot position), sets it to 1, then resolves all pivot variables in reverse order.
- **`_build_return_dict(success, circuit_path, filepath, circuit)`** — Packages the final return dict. Sets `status` to `'ok'` or `'failed'`, wraps file paths as `[{"format": ext, "filename": path}]`, and merges `self.output` fields (`Computed s`, `Valid states`, `computation time (s)`, `Register size`, `Equations`) into the result.

**Key design detail:** `qc.measure(ry[:], cqr[:])` occurs between oracle evaluation and the second input Hadamard layer. Backend support for this operation is an SDK constraint, not a class-level validation rule.

**Data flow:** `s` → `_build_simon_oracle(qc, s)` → `qc.execute()` → `result.marginal_probabilities(qubits=range(n), threshold=1e-5)` → `_get_basis_simple()` → `_solve_simon_general()` → `found_s` (`result['Computed s']`) → `_build_return_dict()`.

## Understanding the Key Quantum Components

The $n$-qubit input register $|x\rangle$ starts in $|0\rangle^n$. After Hadamard:
$$H^{\otimes n}|0\rangle^n = \frac{1}{\sqrt{2^n}}\sum_{x \in \{0,1\}^n}|x\rangle$$

### 2. Oracle ($U_f$)
The oracle implements $U_f|x\rangle|0\rangle \rightarrow |x\rangle|f(x)\rangle$. For the Simon oracle with hidden string $s$, the circuit maps pairs $(x, x \oplus s)$ to the same output. Implementation uses CNOT-based pattern matching on the bits of $s_{\text{target}}$.

### 3. Mid-Circuit Measurement on Output Register
Measuring $|f(x)\rangle$ collapses the input register to:
$$\frac{1}{\sqrt{2}}(|x_0\rangle + |x_0 \oplus s\rangle)$$
for some random $x_0$. This entangled state contains the hidden $s$ in its structure.

### 4. Final Hadamard and Interference
After the second Hadamard layer, only states $|y\rangle$ satisfying $y \cdot s \equiv 0 \pmod{2}$ have nonzero amplitude. This quantum interference eliminates all non-orthogonal components:
$$H^{\otimes n}\frac{|x_0\rangle + |x_0 \oplus s\rangle}{\sqrt2} = \frac{1}{\sqrt{2^{n-1}}}\sum_{y: y \cdot s = 0 \pmod 2}(-1)^{x_0\cdot y}|y\rangle$$

### 5. Classical Post-Processing
The measured bit-strings form a system of linear equations over $\mathbb{F}_2$. Gaussian elimination on $n-1$ linearly independent equations recovers $s$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Input register $|x\rangle$ | `rx = Register('x', n)`; `qc = Circuit(rx, ry, cqr)` |
| Output register $|y\rangle$ | `ry = Register('y', n)` |
| Initial superposition $H^{\otimes n}|0\rangle^n$ | `qc.h(rx[:])` before oracle |
| Oracle $U_f|x\rangle|0\rangle \to |x\rangle|f(x)\rangle$ | `_build_simon_oracle(qc, s)` — CNOT pattern |
| Mid-circuit collapse of output register | `qc.measure(ry[:], cqr[:])` between oracle and final Hadamard |
| Final Hadamard layer (interference) | `qc.h(rx[:])` after measurement |
| Linear equations $y \cdot s \equiv 0$ | Bitstrings from `result.marginal_probabilities(qubits=range(n), threshold=1e-5)` |
| Basis extraction (Gaussian elimination) | `_get_basis_simple(state_basis_dict.keys(), n)` — pivot selection |
| Back-substitution recovery of $s$ | `_solve_simon_general(basis, n)` — $\mathbb{F}_2$ algebra |
| Success condition | `found_s == s` in Stage 4 |

**Notes on encapsulation:** The oracle is a fixed CNOT construction parameterized by `s`. Post-processing retains rows with distinct leftmost-one pivots, then sets the first nonpivot variable to one and back-substitutes. It is not general row reduction or a rank-recovery retry loop.

## Mathematical Deep Dive

After state preparation and oracle:
$$|\psi_2\rangle = \frac{1}{\sqrt{2^n}}\sum_x |x\rangle|f(x)\rangle$$

After measuring the output register (obtaining result $f_0$) and second Hadamard on input:
$$|\text{output}\rangle = \frac{1}{\sqrt{2^{n-1}}}\sum_{y: y \cdot s = 0 \pmod 2} (-1)^{y \cdot x_0} |y\rangle$$

Each measured $y$ satisfies $y \cdot s = 0 \pmod 2$.

**Classical complexity:** Solving $n-1$ linear equations over $\mathbb{F}_2$ costs $O(n^3)$.

**Total quantum query complexity:** $O(n)$ oracle calls, vs. classical $\Omega(2^{n/2})$ — an exponential speedup.

## Hands-On Example

```python
from unitarylab_algorithms import SimonAlgorithm

# Test with a 6-bit hidden string
algo = SimonAlgorithm()
result = algo.run(s='101010', backend='torch')

print(f"Target s:  101010")
print(f"Found  s:  {result['Computed s']}")
print(f"Success:   {result['status'] == 'ok'}")
print(result['plot'])   # [{'format': 'txt', 'filename': '...'}]
```

## Minimal Manual Implementation

The following Python skeleton reconstructs the core components of Simon's algorithm at the oracle, circuit, and post-processing level.

### Step 1: Build the Simon oracle

```python
# Simplified reconstruction — mirrors SimonAlgorithm._build_simon_oracle()

def build_simon_oracle(qc, s: str, n: int):
    """Build U_f for Simon's problem with hidden string s.

    Structure: copy x → y via CX, then XOR the pivot column into all
    positions where s[i]='1'.
    """
    # Copy x to y
    for i in range(n):
        qc.cx(i, i + n)

    pivot_idx = s.find('1')    # leftmost '1' in s
    if pivot_idx < 0:
        return  # trivial case: s=0...0, nothing to do

    for i in range(n):
        if s[i] == '1':
            # XOR: qubit (n-1-pivot_idx) in x controls qubit (n-1-i+n) in y
            qc.cx(n - 1 - pivot_idx, n - 1 - i + n)
```

### Step 2: Build and run the full Simon circuit

```python
from unitarylab import Circuit, Register, ClassicalRegister

def simon_circuit(s_target: str, backend: str = 'torch'):
    n = len(s_target)
    rx = Register('x', n)
    ry = Register('y', n)
    cqr = ClassicalRegister('c', n)
    qc = Circuit(rx, ry, cqr)

    # H on input
    for i in range(n): qc.h(i)

    # Oracle
    build_simon_oracle(qc, s_target, n)

    # Mid-circuit measurement of output register
    qc.measure(ry[:], cqr[:])

    # H on input again
    for i in range(n): qc.h(i)

    return qc
```

### Step 3: Classical post-processing (Gaussian elimination over F₂)

```python
# Simplified reconstruction — mirrors _get_basis_simple() and _solve_simon_general()

def extract_basis(state_dict, n):
    """Greedy pivot selection: collect n-1 linearly independent vectors."""
    basis = {}
    for bits in state_dict:
        pivot = bits.find('1')
        if pivot >= 0 and pivot not in basis:
            basis[pivot] = bits
        if len(basis) >= n - 1:
            break
    return list(basis.values())

def solve_simon(basis, n):
    """Back-substitution over F₂ to find hidden string s."""
    pivot_positions = {bits.find('1'): bits for bits in basis}
    free_idx = next((i for i in range(n) if i not in pivot_positions), -1)
    if free_idx == -1:
        return "Error"
    s = ['0'] * n
    s[free_idx] = '1'  # match the source's first-free-variable rule
    for pivot_pos in sorted(pivot_positions.keys(), reverse=True):
        row = pivot_positions[pivot_pos]
        val = sum(int(row[j]) * int(s[j]) for j in range(n) if j != pivot_pos) % 2
        s[pivot_pos] = str(val)
    return ''.join(s)
```

## Debugging Tips

1. **`s` all zeros**: Will raise `ValueError`. Always include at least one `'1'` bit.
2. **Backend error**: Inspect the SDK's mid-circuit measurement support for the requested backend; the algorithm itself does not reject non-Torch names.
3. **`Computed s` differs from `s`**: Inspect retained marginal states, threshold, distinct pivot count, and back-substitution. There is no shots/measurement-count parameter to increase in this API.
4. **Odd register size vs. qubit count**: The circuit uses $2n$ qubits (input + output) plus a classical register of $n$ bits.

## Reference Implementation

Classiq can be used as a high-level reference implementation of Simon's algorithm.
It describes the algorithm with QMOD functions, automatic synthesis, and a declarative oracle definition.

In this version, the Simon circuit is expressed through `@qfunc`, `@qperm`,
`hadamard_transform`, `within_apply`, `create_model`, `synthesize`, and
`ExecutionSession`. The oracle is defined as a reversible permutation, and the
measured samples are post-processed over `GF(2)` to recover the hidden string.

### Example: Minimal Classiq Simon Run

```python
from classiq import *
from classiq.execution import ExecutionPreferences, ExecutionSession
import galois
import numpy as np

@qfunc
def simon_qfunc(
    f_qfunc: QCallable[QNum, Output[QNum]],
    x: QNum,
    res: Output[QNum],
):
    within_apply(
        lambda: hadamard_transform(x),
        lambda: f_qfunc(x, res),
    )

NUM_QUBITS = 5
S_SECRET = 6  # binary 00110

@qperm
def simon_oracle(s: CInt, x: Const[QNum], res: Output[QNum]):
    from classiq.qmod.symbolic import min
    res |= min(x, x ^ s)

@qfunc
def main(x: Output[QNum[NUM_QUBITS]], res: Output[QNum]):
    allocate(x)
    simon_qfunc(
        lambda x, res: simon_oracle(S_SECRET, x, res),
        x,
        res,
    )

qmod = create_model(
    main,
    constraints=Constraints(
        optimization_parameter=OptimizationParameter.WIDTH
    ),
)

qprog = synthesize(qmod)

prefs = ExecutionPreferences(num_shots=50 * NUM_QUBITS)

with ExecutionSession(qprog, execution_preferences=prefs) as es:
    result = es.sample()

bitstrings = [
    f"{x:0{NUM_QUBITS}b}"
    for x in result.dataframe["x"].tolist()
]

samples = [
    [int(b) for b in bs[::-1]]
    for bs in bitstrings
]

GF = galois.GF(2)

def is_independent_set(vectors):
    if not vectors:
        return True
    return np.linalg.matrix_rank(GF(vectors)) == len(vectors)

def get_independent_set(samples):
    independent = []
    for v in samples:
        if is_independent_set(independent + [v]):
            independent.append(v)
            if len(independent) == len(v) - 1:
                break
    return independent

def get_secret_integer(matrix):
    null_space = GF(matrix).T.left_null_space()
    return int(
        "".join(np.array(null_space)[0][::-1].astype(str)),
        2,
    )

matrix = get_independent_set(samples)

assert len(matrix) == NUM_QUBITS - 1, (
    "Insufficient independent samples; increase num_shots"
)

secret = get_secret_integer(matrix)

print("Expected secret:", S_SECRET)
print("Recovered secret:", secret)

assert secret == S_SECRET
```
