---
name: shor
description: Use this skill when the user asks for Shor integer factorization, quantum order-finding, period estimation, or implementing/running/debugging ShorAlgorithm in this repository (especially matrix/operator methods, IQFT-based phase post-processing, and continued-fraction period recovery). Keywords: shor, factor N, order finding, period finding, modular exponentiation, continued fraction, quantum factoring, ShorAlgorithm.
---
# Shor's Algorithm

## Purpose

Shor's algorithm factors an integer $N$ in polynomial time $O((\log N)^3)$, compared to the best known classical sub-exponential algorithms. It achieves this by reducing integer factorization to quantum period-finding via the Quantum Fourier Transform.

Use this skill when you need to:
- Factor a composite integer using a quantum simulator.
- Understand the quantum period-finding subroutine.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. **Classical pre-checks**: Even numbers and perfect powers are handled classically.
2. **Random base selection**: Choose random $a$ with $1 < a < N$ and $\gcd(a, N) = 1$.
3. **Quantum period-finding**: Build a circuit that implements $f(x) = a^x \bmod N$ and applies the inverse QFT to find the period $r$ of $f$.
4. **Classical post-processing**: Use the continued-fraction algorithm to extract $r$ from the QPE measurement, then compute $\gcd(a^{r/2} \pm 1, N)$ to find factors.
5. **Retry loop**: Automatically retries with a new $a$ if the current attempt fails (up to `max_retries` times).

Two circuit methods are supported:
- **`'matrix'`**: Uses a matrix-based modular exponentiation circuit.
- **`'operator'`**: Uses a modular multiplication operator circuit.

## Prerequisites

- Quantum Fourier Transform (QFT) and inverse QFT.
- Modular arithmetic and Euler's theorem.
- Continued fractions algorithm.
- Python: `numpy`, `Circuit`, `State`, `unitarylab.library.QFT`, `unitarylab.library.IQFT`.

## Using the Provided Implementation

```python
from unitarylab.algorithms import ShorAlgorithm

algo = ShorAlgorithm()
result = algo.run(
    N=15,              # Number to factor
    method='matrix',   # 'matrix' or 'operator'
    backend='torch',
    max_retries=15
)

print(result['factors'])       # List of factors, e.g. [3, 5]
print(result['period'])        # Found period r, or None for classical path
print(result['status'])        # 'ok' on success
print(result['circuit_path'])  # SVG circuit diagram path (if quantum path)
print(result['message'])       # Summary
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `N` | `int` | required | The composite integer to factor. |
| `method` | `str` | `'matrix'` | Circuit method: `'matrix'` or `'operator'`. |
| `backend` | `str` | `'torch'` | Simulation backend. Only `'torch'` supported. |
| `max_retries` | `int` | `15` | Maximum number of random base $a$ attempts. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- Not every attempt succeeds; the loop retries automatically. The returned factors may come from a classical shortcut (even `N`, shared GCD).
- `period` is `None` when a classical path (e.g., even `N`) is taken.
- Circuit qubit counts: `n_count = 2 * N.bit_length()` (counting register), plus `n_work` qubits for the work register.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` if max retries exhausted. |
| `factors` | `list` | List of prime factors found, e.g. `[3, 5]`. Empty on failure. |
| `period` | `int` or `None` | Period $r$ of $f(x) = a^x \bmod N$ if found via quantum path; `None` for classical shortcut. |
| `circuit_path` | `str` or `None` | Path to SVG circuit diagram; `None` for classical path. |
| `message` | `str` | How the factors were found. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`ShorAlgorithm` in `algorithm.py` uses a **retry loop** around the five-stage QPE framework. Classical shortcuts and random base selection are handled before the quantum circuit pipeline.

**`run(N, method, backend, max_retries, algo_dir)` — Outer Loop + Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| Pre-check | Handles `N % 2 == 0` and `gcd(a, N) > 1` cases without any quantum circuit | Classical short-circuit for trivial cases |
| 1 — Parameter Setup | Picks random `a`; computes `n_work`, `n_count = 2 * N.bit_length()`, `total_qubits` | Determines qubit counts for the chosen method |
| 2 — Circuit Construction | Creates `Circuit(total_qubits)`; applies H to counting register; sets work register to $|1\rangle$ via `gs.x(n_count)`; calls `_build_modular_matrix_circuit` or `_build_modular_operator_circuit`; appends `IQFT(n_count)` | Builds the QPE circuit for period finding |
| 3 — Simulation | `gs.execute()` → `State(result_vector)` → `state.measure(range(n_count), endian='little')` | Single-shot measurement of counting register |
| 4 — Classical Post-Processing | `phase = measure_int / 2^n_count`; `Fraction(phase).limit_denominator(N).denominator` gives `r`; checks `r % 2 == 0`; computes `gcd(a^(r/2)±1, N)` | Continued fractions + factor extraction |
| 5 — Export | `gs.draw(filename=..., title=...)` (only on success) | Saves SVG circuit diagram |

**Two Circuit Methods:**

- **`_build_modular_matrix_circuit(gs, n_count, n_work, a, N)`** — For each counting qubit `q`, computes `power_factor = a^(2^q) mod N`, builds a permutation matrix via `_get_modular_matrix(power_factor, N, n_work)`, and calls `gs.unitary(matrix, work_qubits, control=q)`. Simple and direct, but requires explicit matrix construction.

- **`_build_modular_operator_circuit(gs, n_count, n_work, n_work_actual, a, N, backend)`** — Uses an algebraic operator decomposition. Core sub-components:
  - `_multiple_mod(n_qubits, a, N, backend)` — Controlled modular multiplier; built from `_Add_constant_mod_opt` sub-circuits.
  - `_Add_constant_mod_opt(n_qubits, a, N, backend)` — Quantum modular adder using QFT-domain phase rotations (`_Ph`, `_Controlled_Ph`, `QFT`, `IQFT`).

**Helper Methods:**
- `_get_modular_matrix(a, N, n_qubits)` — Builds the permutation matrix for modular multiplication.
- `_Ph(n, a, gs)` / `_Controlled_Ph(n, a, gs, ctrl, data)` — Apply phase rotations in QFT domain.
- `_update_last_result` / `_build_return` — Store runtime fields and package result dict.

**Data flow:** `N` → random `a` → circuit method dispatch → `Circuit` → `execute()` → `state.measure()` → continued fractions → `gcd()` → factors → `_build_return()`.

## Understanding the Key Quantum Components
The $n_{\text{count}} = 2 \cdot \lfloor\log_2 N\rfloor$ counting qubits are placed in uniform superposition via Hadamard:
$$\frac{1}{\sqrt{2^{n_{\text{count}}}}}\sum_{x=0}^{2^{n_{\text{count}}}-1}|x\rangle|1\rangle_{\text{work}}$$

### 2. Modular Exponentiation Oracle
For each counting qubit $k$, a controlled operation applies $\times a^{2^k} \bmod N$ to the work register:
$$\sum_x |x\rangle |a^x \bmod N\rangle$$
The work register starts at $|1\rangle$ so that $a^x \bmod N$ is correctly accumulated.

### 3. Inverse QFT (Period Extraction)
After applying the modular exponentiation, the IQFT on the counting register produces peaks at multiples of $2^{n_{\text{count}}}/r$, where $r$ is the period. Measuring the counting register gives a value $\approx k \cdot 2^{n_{\text{count}}}/r$.

### 4. Continued Fractions (Classical)
The rational approximation $k/r \approx m/2^{n_{\text{count}}}$ is extracted using the continued fractions algorithm (`Fraction.limit_denominator(N)`), yielding the period $r$.

### 5. Factor Extraction
If $r$ is even and $a^{r/2} \not\equiv -1 \pmod{N}$:
$$\gcd(a^{r/2} - 1, N) \quad \text{and} \quad \gcd(a^{r/2} + 1, N)$$
yields non-trivial factors of $N$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| Counting register $H^{\otimes n}|0\rangle$ | `gs.h(range(n_count))` in Stage 2 |
| Work register initialized to $|1\rangle$ | `gs.x(n_count)` in Stage 2 |
| Controlled $a^{2^k} \bmod N$ (matrix method) | `_build_modular_matrix_circuit()` — `gs.unitary(matrix, work, control=q)` |
| Controlled $a^{2^k} \bmod N$ (operator method) | `_build_modular_operator_circuit()` via `_multiple_mod()` using QFT-domain adders |
| Inverse QFT (iQFT) | `gs.append(IQFT(n_count, backend), range(n_count))` from `unitarylab.library` |
| Measurement of counting register | `state.measure(range(n_count), endian='little')` → binary string → integer |
| Phase $\phi \approx k/r$ | `phase = measure_int / 2^n_count` |
| Continued fractions algorithm | `Fraction(phase).limit_denominator(N).denominator` → `r` |
| Factor extraction $\gcd(a^{r/2}\pm1, N)$ | `math.gcd(guess - 1, N)` and `math.gcd(guess + 1, N)` |
| Classical shortcut (even N or gcd>1) | Pre-loop checks before any quantum circuit construction |

**Notes on method differences:** The `'matrix'` method builds an explicit permutation matrix for each controlled power, making it circuit-efficient but memory-intensive for large $N$. The `'operator'` method uses QFT-domain phase additions and is more gate-intensive but architecturally general. Both append the same `IQFT` from `unitarylab.library`. The retry loop is necessary because the continued fractions step can fail for unlucky choices of `a` or measurement outcomes.

## Mathematical Deep Dive (the multiplicative order of $a$ modulo $N$), i.e., $a^r \equiv 1 \pmod{N}$.

**Factoring step:** Since $a^r - 1 = (a^{r/2}-1)(a^{r/2}+1) \equiv 0 \pmod{N}$, and $N$ divides the product but not (with high probability) either factor individually, $\gcd(a^{r/2} \pm 1, N)$ yields non-trivial factors.

**Success probability:** At least $50\%$ of bases $a$ produce a useful period. Multiple attempts are needed with probability exponentially small to fail.

**Complexity:** $O((\log N)^3)$ quantum gates — polynomial in $\log N$. Classical best: sub-exponential $e^{O((\log N)^{1/3}(\log\log N)^{2/3})}$.

## Hands-On Example

```python
from unitarylab.algorithm import ShorAlgorithm

algo = ShorAlgorithm()
result = algo.run(N=15, method='matrix', backend='torch', max_retries=20)
print(f"Factors of 15: {result['factors']}")    # [3, 5] or [5, 3]
print(f"Period r: {result['period']}")
print(result['plot'])
```


## Reference Implementation (Classiq)

Classiq can be used as a reference implementation for Shor-style period finding. It describes the modular multiplication based QPE workflow through high-level QMOD functions, automatic synthesis, and built-in modular arithmetic primitives.

### Example A: Minimal Classiq Shor Period-Finding Run

```python
from classiq import *


@qfunc
def period_finding(n: CInt, a: CInt, x: QNum, phase_var: QNum):
    x ^= 1
    qpe_flexible(
        lambda p: modular_multiply_constant_inplace(n, a**p, x),
        phase_var,
    )


modulo_num = 21
a_num = 11
x_len = modulo_num.bit_length()
phase_len = 2 * x_len


@qfunc
def main(phase_var: Output[QNum[phase_len, UNSIGNED, phase_len]]):
    x = QNum()
    allocate(x_len, x)
    allocate(phase_var)

    period_finding(modulo_num, a_num, x, phase_var)

    drop(x)


qprog = synthesize(
    main,
    preferences=Preferences(qasm3=True, optimization_level=1),
    constraints=Constraints(optimization_parameter="width"),
)

sample_result = execute(qprog).get_sample_result()
df = sample_result.dataframe
print(df)
```


## Minimal Manual Implementation (UnitaryLab) 

Below is a skeleton that reconstructs Shor's algorithm at the component level, matching the `ShorAlgorithm` structure in `algorithm.py`.

```python
# Simplified reconstruction — mirrors ShorAlgorithm.run() (matrix method)
import math, random
from fractions import Fraction
import numpy as np
from unitarylab.core import Circuit, State
from unitarylab.library import IQFT

def get_modular_matrix(mult: int, N: int, n_work: int) -> np.ndarray:
    """Build the 2^n_work x 2^n_work unitary for |x> -> |x*mult mod N>."""
    dim = 2**n_work
    U = np.zeros((dim, dim))
    for x in range(dim):
        y = (x * mult) % N if x < N else x
        U[y, x] = 1.0
    return U.astype(complex)

def build_shor_circuit(a: int, N: int, n_count: int, n_work: int,
                        backend: str = 'torch') -> Circuit:
    """Construct the quantum phase-estimation circuit for order finding."""
    total = n_count + n_work
    gs = Circuit(total, backend=backend)

    # 1. Superpose counting register; set work register to |1>
    gs.h(range(n_count))
    gs.x(n_count)

    # 2. Controlled-U^{2^j} gates (matrix method): |j>|x> -> |j>|x * a^{2^j} mod N>
    for j in range(n_count):
        mult = pow(a, 2**j, N)
        U = get_modular_matrix(mult, N, n_work)
        gs.unitary(U, range(n_count, total), j, '1')  # controlled on qubit j

    # 3. Inverse QFT on counting register
    gs.append(IQFT(n_count, backend=backend), range(n_count))
    return gs

def shor_factor(N: int, max_retries: int = 15, backend: str = 'torch'):
    """Full Shor factoring loop."""
    if N % 2 == 0:
        return [2, N // 2]

    n_work  = N.bit_length()
    n_count = 2 * n_work

    for _ in range(max_retries):
        a = random.randint(2, N - 1)
        g = math.gcd(a, N)
        if g > 1:
            return [g, N // g]   # classical shortcut

        gs = build_shor_circuit(a, N, n_count, n_work, backend)

        # Measure counting register
        sv   = gs.execute()
        from unitarylab.core import State
        meas = State(sv).measure(range(n_count), endian='little')
        phase_int = int(meas, 2)

        # Classical post-processing: continued fractions -> period r
        phase = phase_int / (2**n_count)
        r = Fraction(phase).limit_denominator(N).denominator

        if r % 2 != 0 or r == 0:
            continue
        guess = pow(a, r // 2, N)
        if guess in (1, N - 1):
            continue

        p = math.gcd(guess - 1, N)
        q = math.gcd(guess + 1, N)
        if p * q == N:
            return [p, q]
    return None  # failed
```

**Component roles**:
- `get_modular_matrix` — builds the $2^n \times 2^n$ permutation matrix for $|x\rangle \to |x \cdot \text{mult} \bmod N\rangle$. This is the `_build_modular_matrix_circuit` bypass used in the `'matrix'` method.
- `build_shor_circuit` — assembles $H^{\otimes n_{\text{count}}}$ + controlled-$U^{2^j}$ gates + IQFT, the three-stage quantum phase estimation structure.
- `shor_factor` — the outer retry loop: picks random $a$, builds + measures the circuit, applies continued fractions to extract period $r$, then computes $\gcd(a^{r/2}\pm 1, N)$.




## Debugging Tips

1. **Simulation is slow for large N**: Qubit count scales as $O(\log N)$, but the state vector grows as $2^{O(\log N)} = N^O$. Keep $N$ small (e.g., 15, 21, 35) for feasible simulation.
2. **`status='failed'` after `max_retries`**: Increase `max_retries`. Some choices of $N$ and $a$ lead to trivial periods more often.
3. **`period` is odd**: Odd periods are discarded; the loop retries. This is expected for some $a$.
4. **`factors = [1, N]`**: Trivial factors — the period-finding step gave an $a^{r/2} \equiv \pm 1 \pmod N$ case; retries automatically.
5. **`method='operator'` vs `'matrix'`**: `'matrix'` uses dense matrix multiplication; `'operator'` uses a modular multiplication operator circuit. Both are correct; `'matrix'` is simpler.
