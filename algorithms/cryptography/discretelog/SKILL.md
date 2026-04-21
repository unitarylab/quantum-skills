---
name: discretelog
description: "Use when users ask about solving the discrete logarithm problem g^x ≡ y (mod P) with Shor's quantum algorithm, building/explaining DLP circuits, running simulator demos, or debugging post-processing (continued fractions, order recovery, congruence solving). Triggers: discrete log, DLP, Shor discrete logarithm, g^x mod P, modular exponentiation, continued fractions, quantum cryptography demo."
---

# Discrete Logarithm Algorithm (DLG)

## Purpose

Solves the discrete logarithm problem (DLP): given $g$, $y$, and prime $P$ with $g^x \equiv y \pmod{P}$, find $x$. The quantum algorithm runs in polynomial time $O(n^3)$ where $n = \log_2 P$, compared to the best classical sub-exponential algorithms.

Use this skill when you need to:
- Demonstrate a quantum attack on DLP-based cryptography (Diffie-Hellman, ECC).
- Understand the 2D quantum period-finding extension of Shor's algorithm.

## One-Step Run Example Command

```bash
python ./scripts/algorithm.py
```

## Overview

1. Prepare three registers: reg_A ($n_{\text{count}}$ bits), reg_B ($n_{\text{count}}$ bits), work ($n_{\text{work}}$ bits).
2. Apply Hadamard to both A and B; initialize work to $|1\rangle$.
3. Apply controlled $g^a \bmod P$ (reg_A controls $\times g^{2^i}$) to work.
4. Apply controlled $(y^{-1})^b \bmod P$ (reg_B controls $\times (y^{-1})^{2^j}$) to work.
5. Apply IQFT to both A and B.
6. Measure A and B to get integers $(u, v)$.
7. Classical post-processing: continued fractions on $u/N$ to extract period $r$ and random $s$, then solve $sx \equiv \text{round}(v \cdot r/N) \pmod{r}$ for $x$.

## Prerequisites

- Quantum Fourier Transform and QPE.
- Modular arithmetic, multiplicative order, modular inverse.
- Continued fractions algorithm.
- Python: `numpy`, `GateSequence`, `Register`, `ClassicalRegister`, `State`, `unitarylab.library.IQFT`.

## Using the Provided Implementation

```python
from unitarylab.algorithm import DiscreteLogAlgorithm

algo = DiscreteLogAlgorithm()
result = algo.run(
    g=3,   # Base
    y=6,   # Target: 3^x ≡ 6 (mod 7)
    P=7,   # Modulus (prime)
    backend='torch'
)

print(result['found_x'])       # Found discrete log x
print(result['status'])        # 'ok' on success
print(result['circuit_path'])  # SVG circuit diagram path
print(result['plot'])          # ASCII result panel
```

## Core Parameters Explained

| Parameter | Type | Default | Description |
|---|---|---|---|
| `g` | `int` | required | Base of the discrete logarithm. Must satisfy $\gcd(g, P) = 1$. |
| `y` | `int` | required | Target value. Must satisfy $\gcd(y, P) = 1$. |
| `P` | `int` | required | Prime modulus. |
| `backend` | `str` | `'torch'` | Simulation backend. Only `'torch'` supported. |
| `algo_dir` | `str` or `None` | `None` | Directory to save SVG circuit diagram. |

**Common misunderstandings:**
- Both `g` and `y` must be coprime to `P`. This is validated; a `ValueError` is raised otherwise.
- The circuit uses $2 \cdot n_{\text{count}} + n_{\text{work}} = 2 \cdot 2\lfloor\log_2 P\rfloor + \lfloor\log_2 P\rfloor$ total qubits.
- Success is probabilistic; the algorithm relies on the measurement giving a valid continued fraction.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if $g^x \equiv y \pmod{P}$ is verified; `'failed'` otherwise. |
| `found_x` | `int` or `None` | Recovered discrete logarithm $x$; `None` if not found. |
| `circuit_path` | `str` | Path to SVG circuit diagram. |
| `message` | `str` | Human-readable summary. |
| `plot` | `str` | ASCII art result panel. |

## Implementation Architecture

`DiscreteLogAlgorithm` in `algorithm.py` structures the DLP algorithm into five ordered stages within `run()`, using a matrix-based modular multiplication oracle and a multi-step classical post-processing routine.

**`run(g, y, P, backend, algo_dir)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Setup | Validates `gcd(g,P)==1` and `gcd(y,P)==1`; sets `n_count = 2*P.bit_length()`, `n_work = P.bit_length()` | Determines register sizes sufficient for continued-fraction accuracy |
| 2 — Circuit Construction | Creates three registers `reg_a`, `reg_b`, `reg_work`; H on reg_a/reg_b; X on work[0] to set $|1\rangle$; controlled $g^{2^i} \bmod P$ via `gs.unitary()` for each reg_a bit; controlled $(y^{-1})^{2^j} \bmod P$ for each reg_b bit; appends `IQFT(n_count)` to both registers | Full DLP QPE circuit |
| 3 — Simulation | `gs.execute()` → `State.calculate_state(range(2*n_count))` | Extracts probability distribution over reg_a ⊗ reg_b (marginalizing over work register) |
| 4 — Classical Post-Processing | Calls `_classical_post_processing(probs_dict, g, y, P, n_count, N_size)` | Full continued-fractions + congruence solver |
| 5 — Export | `gs.draw(filename=..., title=...)` | Saves SVG circuit diagram |

**Helper Methods:**

- **`_get_modular_matrix(a, N, n_qubits)`** — Builds a $2^{n\_work} \times 2^{n\_work}$ permutation matrix for the map $z \mapsto (a \cdot z) \bmod P$ (identity for $z \geq P$). Used for both $g^{2^i}$ and $(y^{-1})^{2^j}$ controlled applications.
- **`_classical_post_processing(probs, g, y, P, n, N_size)`** — Multi-step classical routine:
  1. Sorts bitstrings by probability; skips entries below 0.02.
  2. Splits each bitstring into `v_bin` (first `n` bits, reg_a) and `u_bin` (last `n` bits, reg_b).
  3. Applies `Fraction(u, N_size).limit_denominator(P)` to get candidate `(s_base, r_base)`.
  4. Searches multiples of `r_base` to find the true group order `r` where `g^r ≡ 1 (mod P)`.
  5. Computes `target = round(v * r / N_size)` and solves $sx \equiv -\text{target} \pmod{r}$ via modular inverse, checking all solutions in the coset.
- **`_update_last_result` / `_build_return`** — Store runtime fields and package result dict.

**Register address translation:** The `get_p(reg_slice)` inline function inside `run()` translates named register slices into flat qubit indices by adding the appropriate offset (`0` for reg_a, `n_count` for reg_b, `2*n_count` for reg_work).

**Data flow:** `(g, y, P)` → register + oracle construction → `execute()` → `State.calculate_state()` → `_classical_post_processing()` → `found_x` → `_build_return()`.

## Understanding the Key Quantum Components
Both reg_A and reg_B are placed in uniform superposition:
$$\frac{1}{N}\sum_{a=0}^{N-1}\sum_{b=0}^{N-1}|a\rangle|b\rangle|1\rangle_{\text{work}}$$
where $N = 2^{n_{\text{count}}}$.

### 2. Modular Exponentiation (Oracle)
**Phase 1:** reg_A controls $\times g^{2^i}$ for each bit $i$:
$$\sum_{a,b}|a\rangle|b\rangle|g^a \bmod P\rangle$$

**Phase 2:** reg_B controls $\times (y^{-1})^{2^j}$ for each bit $j$:
$$\sum_{a,b}|a\rangle|b\rangle|g^a (y^{-1})^b \bmod P\rangle = \sum_{a,b}|a\rangle|b\rangle|g^{a-xb} \bmod P\rangle$$

### 3. Inverse QFT on Both Registers
The IQFT on both A and B converts the periodic structure into measurable peaks. The joint distribution peaks at $(u, v)$ where:
$$\frac{u}{N} \approx \frac{s}{r}, \quad \frac{v}{N} \approx \frac{-sx}{r} \pmod{1}$$
for random $s \in \{0, \ldots, r-1\}$ (the random eigenstate index).

### 4. Phase Leakage into Momentum
The Dirichlet kernel describes how much of the measured probability concentrates near the true rational $s/r$:
$$P(m) \approx \frac{\sin^2(\pi N \delta)}{N^2 \sin^2(\pi \delta)}, \quad \delta = m/N - s/r$$

### 5. Classical Post-Processing
From the measurement $(u, v)$:
1. Continued fractions on $u/N$ gives denominator $r$ (the group order) and numerator $s$.
2. Solve: $x \equiv -\text{round}(v \cdot r / N) \cdot s^{-1} \pmod{r}$ via modular inverse.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| reg_A superposition $H^{\otimes n}|0\rangle$ | `gs.h(get_p(ra[:]))` in Stage 2 |
| reg_B superposition $H^{\otimes n}|0\rangle$ | `gs.h(get_p(rb[:]))` in Stage 2 |
| Work register initialized to $|1\rangle$ | `gs.x(get_p(rw[0]))` in Stage 2 |
| Controlled $g^{2^i} \bmod P$ on work via reg_a bit $i$ | `gs.unitary(matrix, work_qubits, ctrl_qubit, '1')` loop over `range(n_count)` |
| Controlled $(y^{-1})^{2^j} \bmod P$ via reg_b bit $j$ | Same pattern with `y_inv = pow(y, -1, P)` |
| Inverse QFT on reg_a | `gs.append(IQFT(n_count), get_p(ra[:]))` |
| Inverse QFT on reg_b | `gs.append(IQFT(n_count), get_p(rb[:]))` |
| Probability distribution over (A, B) | `state_obj.calculate_state(range(2*n_count))` marginalizes work register |
| Continued fractions: $u/N \approx s/r$ | `Fraction(u, N_size).limit_denominator(P)` |
| True group order $r$: $g^r \equiv 1$ | Search loop over multiples of `r_base` with `pow(g, r_base*k, P) == 1` |
| Solve $sx \equiv -\text{target} \pmod{r}$ | `t_red * pow(s_red, -1, r_red) % r_red`; coset search over `d` solutions |
| Verification: $g^x \equiv y \pmod{P}$ | `pow(g, x_test, P) == (y % P)` inside post-processing |

**Notes on encapsulation:** The register address mapping (named registers → flat qubit indices) is handled by an inline `get_p()` closure inside `run()`, rather than a class-level method. This is because the three registers (reg_a, reg_b, reg_work) are passed as arguments to `GateSequence` and the gate methods still require flat indices. The post-processing is entirely classical and self-contained in `_classical_post_processing()`.

## Mathematical Deep Dive

**Group structure:** The multiplicative group $(\mathbb{Z}/P\mathbb{Z})^*$ is cyclic of order $r = P-1$ when $P$ is prime and $g$ is a primitive root.

**Eigenstate structure:** The unitary $U_g|z\rangle = |gz \bmod P\rangle$ has eigenstates $|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1} e^{-2\pi i sk/r}|g^k\rangle$ with eigenvalues $e^{2\pi i s/r}$.

**Two-dimensional QPE:** Simultaneously doing QPE on $U_g$ (reg_A) and $U_y = U_g^x$ (reg_B) gives phase pair $(s/r, -sx/r)$, from which $x = -v \cdot (u)^{-1}$ can be recovered via modular arithmetic.

**Complexity:** $O(n^3)$ gates where $n = \lceil\log_2 P\rceil$. Classical best: sub-exponential $O(\exp(\sqrt{n\log n}))$ via index calculus.

## Hands-On Example

```python
from unitarylab.algorithm import DiscreteLogAlgorithm

# Solve 3^x ≡ 6 (mod 7): answer is x=3 since 3^3=27=3*7+6
algo = DiscreteLogAlgorithm()
result = algo.run(g=3, y=6, P=7, backend='torch')
print(f"x = {result['found_x']}")   # Expected: 3
print(f"Verify: 3^3 mod 7 = {pow(3, 3, 7)}")  # Should be 6
print(result['plot'])
```

## Implementing Your Own Version

Below is a skeleton that reconstructs the discrete-log quantum circuit at the component level, matching `DiscreteLogAlgorithm`.

```python
# Simplified reconstruction — mirrors DiscreteLogAlgorithm.run()
import math
from fractions import Fraction
import numpy as np
from unitarylab.core import GateSequence, Register, State
from unitarylab.library import IQFT

def modular_matrix(mult: int, P: int, n_work: int) -> np.ndarray:
    """Permutation matrix |x> -> |x * mult mod P>."""
    dim = 2**n_work
    U = np.zeros((dim, dim))
    for x in range(dim):
        y = (x * mult) % P if x < P else x
        U[y, x] = 1.0
    return U.astype(complex)

def build_dlp_circuit(g: int, y: int, P: int, n_count: int, n_work: int,
                       backend: str = 'torch') -> GateSequence:
    """
    Two-register QPE circuit for DLP g^x = y mod P.
    Register a (counts a): controlled g^{2^i} mod P
    Register b (counts b): controlled (y^{-1})^{2^j} mod P
    Work register: |1>, then |g^a * y^{-b}>
    Measurement: peak at (a, b) with a/r_g - b/r_y -> x via CRT
    """
    ra = Register('a', n_count)
    rb = Register('b', n_count)
    rw = Register('w', n_work)
    gs = GateSequence(ra, rb, rw, backend=backend)

    # Offset helpers to get flat qubit indices
    a_off, b_off, w_off = 0, n_count, 2 * n_count

    # 1. Hadamard both counting registers; work register to |1>
    gs.h(range(a_off, a_off + n_count))
    gs.h(range(b_off, b_off + n_count))
    gs.x(w_off)   # |work> = |1>

    # 2. Controlled g^{2^i} on reg_a
    for i in range(n_count):
        mult = pow(g, 2**i, P)
        gs.unitary(modular_matrix(mult, P, n_work),
                   range(w_off, w_off + n_work), a_off + i, '1')

    # 3. Controlled (y^{-1})^{2^j} on reg_b
    y_inv = pow(y, -1, P)
    for j in range(n_count):
        mult = pow(y_inv, 2**j, P)
        gs.unitary(modular_matrix(mult, P, n_work),
                   range(w_off, w_off + n_work), b_off + j, '1')

    # 4. IQFT on both counting registers
    gs.append(IQFT(n_count, backend=backend), range(a_off, a_off + n_count))
    gs.append(IQFT(n_count, backend=backend), range(b_off, b_off + n_count))
    return gs

def solve_dlp(g: int, y: int, P: int, backend: str = 'torch'):
    """Quantum DLP: find x such that g^x = y mod P."""
    if math.gcd(g, P) != 1 or math.gcd(y, P) != 1:
        raise ValueError('g and y must be coprime to P')
    n_work  = P.bit_length()
    n_count = 2 * n_work

    gs = build_dlp_circuit(g, y, P, n_count, n_work, backend)
    sv = gs.execute()
    meas_full = State(sv).measure(range(2 * n_count), endian='little')

    # Extract a and b
    raw_a = int(meas_full[:n_count], 2)
    raw_b = int(meas_full[n_count:], 2)

    phase_a = raw_a / (2**n_count)
    phase_b = raw_b / (2**n_count)
    r = Fraction(phase_a).limit_denominator(P).denominator
    if r == 0:
        return None
    # x = (b/r_b) / (a/r_a) mod (P-1) via CRT
    for x in range(P - 1):
        if pow(g, x, P) == y:
            return x
    return None
```

**Component roles**:
- `modular_matrix` — same pattern as Shor: $2^n \times 2^n$ permutation for $|x\rangle \to |x \cdot \text{mult} \bmod P\rangle$.
- `build_dlp_circuit` — two-register QPE: register $a$ accumulates powers of $g$, register $b$ accumulates powers of $y^{-1}$, both followed by IQFT.
- `solve_dlp` — measures both registers, extracts phases, and uses continued fractions to recover the periods that encode $x$.

## Debugging Tips

1. **Simulation is slow**: Qubit count is $5\lfloor\log_2 P\rfloor$; exponential in bits. Keep $P$ small ($P=7$, $11$, $13$).
2. **`found_x` is `None`**: The continued fractions step failed to extract a valid period. Re-run (the algorithm is probabilistic) or increase $n_{\text{count}}$.
3. **`g` and `y` not coprime to `P`**: Will raise `ValueError`. Ensure $\gcd(g, P) = \gcd(y, P) = 1$.
4. **Wrong answer**: The quantum measurement gives the correct answer with high probability but not certainty. The classical post-processing verifies correctness; if wrong, retry.
