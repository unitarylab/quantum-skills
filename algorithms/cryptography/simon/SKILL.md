---
name: simon
description: "Use for implementing, explaining, running, or debugging Simon's algorithm in this repository, especially for oracle construction, measurement interpretation, GF(2) post-processing, simulator state extraction, and compatible reimplementation."
---

# Simon's Algorithm

## Purpose

Simon's algorithm solves Simon's problem: given a black-box function $f: \{0,1\}^n \rightarrow \{0,1\}^n$ with the promise $f(x) = f(y) \iff x \oplus y = s$ for some hidden string $s$, find $s$ using $O(n)$ quantum queries instead of the classical $O(2^{n/2})$.

Use this skill when you need to:
- Demonstrate an exponential quantum speedup over classical algorithms.
- Understand the hidden subgroup problem structure behind Shor's algorithm.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

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
- Python: `numpy`, `Circuit`, `Register`, `ClassicalRegister`, `State`.

## Using the Provided Implementation

```python
from unitarylab-algorithms import SimonAlgorithm

algo = SimonAlgorithm()
result = algo.run(
    s_target='1010',   # Hidden string to find
    backend='torch'
)

print(result['found_s'])      # Found hidden string (should match s_target)
print(result['status'])       # 'ok' if found_s == s_target
print(result['circuit_path']) # SVG circuit diagram path
print(result['plot'])         # ASCII result panel
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `s_target` | `str` | `'1010'` | The hidden binary string $s$ to be found. Must have at least one `'1'`. |
| `backend` | `str` | `'torch'` | Only `'torch'` is supported (requires mid-circuit measurement). |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- `s_target` cannot be all-zero (`'0000'`); this is the trivial case with no hidden structure.
- The algorithm is probabilistic; it runs the quantum circuit once and extracts equations from the measurement distribution. A single run may not find all $n-1$ linearly independent equations, but the simulation samples enough states.
- `backend` must be `'torch'` because the circuit includes mid-circuit measurements.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if `found_s == s_target`; `'failed'` otherwise. |
| `found_s` | `str` | The binary string $s$ recovered by the classical post-processing step. |
| `circuit_path` | `str` | Path to the saved SVG circuit diagram. |
| `message` | `str` | Human-readable summary. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`SimonAlgorithm` in `algorithm.py` structures the algorithm into five ordered stages inside `run()`, with three classical helper methods for circuit oracle construction and linear algebra post-processing.

**`run(s_target, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Validation | Extracts `n = len(s_target)`; checks for all-zero string | Guards against trivial inputs |
| 2 — Circuit Construction | Creates `Circuit(rx, ry, cqr)` with two `n`-qubit registers + classical register; applies `gs.h(rx[:])`, calls `_build_simon_oracle(gs, s_target)`, measures `ry` to `cqr`, applies `gs.h(rx[:])` again | Full Simon protocol circuit in 4 lines |
| 3 — Simulation | `gs.execute()` → `State(re_state)` → `state_obj.calculate_state(range(n))` | Runs mid-circuit-measurement simulation; extracts x-register basis dictionary |
| 4 — Classical Post-Processing | `_get_basis_simple(state_list, n)` extracts $n-1$ linearly independent vectors; `_solve_simon_general(basis, n)` back-substitutes to recover `s` | $\mathbb{F}_2$ linear algebra |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_build_simon_oracle(gs, s)`** — Implements $U_f$ with CNOT gates. First, copies $x$ to $y$ register via `cx(i, i+n)` for each bit. Then, for each bit `i` where `s[i]='1'`, applies `cx(n-1-pivot_idx, n-1-i+n)` (where `pivot_idx = s.find('1')`) to create the $f(x) = f(x\oplus s)$ mapping.
- **`_get_basis_simple(state_list, n_qubits)`** — Greedy pivot selection: iterates bitstrings, keeps the first bitstring with each distinct leftmost-1 pivot position. Returns at most $n-1$ linearly independent vectors.
- **`_solve_simon_general(basis_list, n)`** — Back-substitution over $\mathbb{F}_2$: finds the free variable index (the non-pivot position), sets it to 1, then resolves all pivot variables in reverse order.
- **`_update_last_result` / `_build_return`** — Store runtime fields and package result dict.

**Key design detail:** The circuit uses `Register`, `ClassicalRegister`, and `Circuit` with mixed quantum/classical registers. The mid-circuit measurement (`gs.measure(ry[:], cqr[:])`) collapses the output register during simulation, which is why only the `'torch'` backend is supported.

**Data flow:** `s_target` → oracle construction → `execute()` → `State.calculate_state()` → basis extraction → back-substitution → `found_s` → `_build_return()`.

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
$$H^{\otimes n}(|x_0\rangle + |x_0 \oplus s\rangle) = \sum_{y: y \cdot s = 0 \pmod 2} (\pm 1)|y\rangle$$

### 5. Classical Post-Processing
The measured bit-strings form a system of linear equations over $\mathbb{F}_2$. Gaussian elimination on $n-1$ linearly independent equations recovers $s$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Input register $|x\rangle$ | `rx = Register('x', n)`; `gs = Circuit(rx, ry, cqr)` |
| Output register $|y\rangle$ | `ry = Register('y', n)` |
| Initial superposition $H^{\otimes n}|0\rangle^n$ | `gs.h(rx[:])` before oracle |
| Oracle $U_f|x\rangle|0\rangle \to |x\rangle|f(x)\rangle$ | `_build_simon_oracle(gs, s_target)` — CNOT pattern |
| Mid-circuit collapse of output register | `gs.measure(ry[:], cqr[:])` between oracle and final Hadamard |
| Final Hadamard layer (interference) | `gs.h(rx[:])` after measurement |
| Linear equations $y \cdot s \equiv 0$ | Bitstrings from `state_obj.calculate_state(range(n))` |
| Basis extraction (Gaussian elimination) | `_get_basis_simple(state_list, n)` — pivot selection |
| Back-substitution recovery of $s$ | `_solve_simon_general(basis_list, n)` — $\mathbb{F}_2$ algebra |
| Success condition | `found_s == s_target` in Stage 4 |

**Notes on encapsulation:** The oracle construction uses a hardcoded CNOT-based pattern for the standard Simon oracle where $f(x) = f(x \oplus s)$ with $s = s_\text{target}$. The classical post-processing does not use a general Gaussian elimination library; instead it uses greedy pivot selection followed by $\mathbb{F}_2$ back-substitution implemented directly. The simulation must be run with `backend='torch'` because the mid-circuit measurement is required.

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
from unitarylab-algorithms import SimonAlgorithm

# Test with a 6-bit hidden string
algo = SimonAlgorithm()
result = algo.run(s_target='101010', backend='torch')

print(f"Target s:  101010")
print(f"Found  s:  {result['found_s']}")
print(f"Success:   {result['status'] == 'ok'}")
print(result['plot'])
```
## Reference Implementation (Classiq)

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
## Minimal Manual Implementation (UnitaryLab) 

The following Python skeleton reconstructs the core components of Simon's algorithm at the oracle, circuit, and post-processing level.

### Step 1: Build the Simon oracle

```python
# Simplified reconstruction — mirrors SimonAlgorithm._build_simon_oracle()

def build_simon_oracle(gs, s: str, n: int):
    """Build U_f for Simon's problem with hidden string s.

    Structure: copy x → y via CX, then XOR the pivot column into all
    positions where s[i]='1'.
    """
    # Copy x to y
    for i in range(n):
        gs.cx(i, i + n)

    pivot_idx = s.find('1')    # leftmost '1' in s
    if pivot_idx < 0:
        return  # trivial case: s=0...0, nothing to do

    for i in range(n):
        if s[i] == '1':
            # XOR: qubit (n-1-pivot_idx) in x controls qubit (n-1-i+n) in y
            gs.cx(n - 1 - pivot_idx, n - 1 - i + n)
```

### Step 2: Build and run the full Simon circuit

```python
# Exact usage example (uses actual API)
from unitarylab-algorithms import SimonAlgorithm
from unitarylab.core import Circuit, Register, State

def simon_circuit(s_target: str, backend: str = 'torch'):
    n = len(s_target)
    rx = Register('x', n)
    ry = Register('y', n)
    from unitarylab.core import ClassicalRegister
    cqr = ClassicalRegister('c', n)
    gs = Circuit(rx, ry, cqr, backend=backend)

    # H on input
    for i in range(n): gs.h(i)

    # Oracle
    build_simon_oracle(gs, s_target, n)

    # Mid-circuit measurement of output register
    gs.measure(list(range(n, 2*n)), list(range(n)))

    # H on input again
    for i in range(n): gs.h(i)

    return gs
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
    all_positions = set(range(n))
    free_vars = all_positions - set(pivot_positions.keys())
    s = ['0'] * n
    for pos in free_vars:
        s[pos] = '1'  # free variable set to 1
    for pivot_pos in sorted(pivot_positions.keys(), reverse=True):
        row = pivot_positions[pivot_pos]
        val = sum(int(row[j]) * int(s[j]) for j in range(n) if j != pivot_pos) % 2
        s[pivot_pos] = str(val)
    return ''.join(s)
```

## Debugging Tips

1. **`s_target` all zeros**: Will raise `ValueError`. Always include at least one `'1'` bit.
2. **`backend` not `'torch'`**: Will raise `ValueError`. Simon's algorithm requires mid-circuit measurement, supported only by `'torch'`.
3. **`found_s` differs from `s_target`**: The solver may find a different or trivial $s$ if too few linearly independent equations were collected. Re-run or increase measurements.
4. **Odd register size vs. qubit count**: The circuit uses $2n$ qubits (input + output) plus a classical register of $n$ bits.
