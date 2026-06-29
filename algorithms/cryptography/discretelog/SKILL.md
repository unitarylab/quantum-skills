---
name: discretelog
description: "Use when users ask about solving the discrete logarithm problem g^x ≡ y (mod P) with Shor-style two-register Fourier sampling, building/explaining DLP circuits, running simulator demos, or debugging post-processing (continued fractions plus two-dimensional Fourier-sample congruence solving). Triggers: discrete log, DLP, Shor discrete logarithm, g^x mod P, modular exponentiation, two-dimensional Fourier sampling, continued fractions, congruence solving, quantum cryptography demo."
---

# Discrete Logarithm Algorithm (DLG)

## Purpose

Solves the discrete logarithm problem (DLP): given $g$, $y$, and prime $P$ with $g^x \equiv y \pmod{P}$, find $x$. The quantum algorithm runs in polynomial time $O(n^3)$ where $n = \log_2 P$, compared to the best classical sub-exponential algorithms.

Use this skill when you need to:
- Demonstrate a quantum attack on DLP-based cryptography (Diffie-Hellman, ECC).
- Understand the two-register Fourier-sampling version of Shor's discrete logarithm algorithm.

## Overview

1. Prepare three registers: reg_A ($n_{\text{count}}$ bits), reg_B ($n_{\text{count}}$ bits), work ($n_{\text{work}}$ bits).
2. Apply Hadamard to both A and B; initialize work to $|1\rangle$.
3. Apply controlled $g^a \bmod P$ (reg_A controls $\times g^{2^i}$) to work.
4. Apply controlled $(y^{-1})^b \bmod P$ (reg_B controls $\times (y^{-1})^{2^j}$) to work.
5. Apply IQFT to both A and B.
6. Measure A and B to get integers $(u, v)$.
7. Classical post-processing: split the joint measurement into two Fourier samples `u` and `v`, use continued fractions on $u/N$ to estimate $s/r$, then combine `v` with the estimated period and solve the two-dimensional Fourier-sample congruence. The theoretical relation is $u x \equiv -v \pmod r$; in the source variables this is implemented as `real_s * x ≡ -target mod real_r`.

## Prerequisites

- Quantum Fourier Transform and two-register Fourier sampling.
- Modular arithmetic, multiplicative order, modular inverse.
- Continued fractions algorithm.
- Python: `numpy`, `Circuit`, `Register`, `unitarylab.library.IQFT`.

## Using the Provided Implementation

```python
from unitarylab_algorithms import DiscreteLogAlgorithm

algo = DiscreteLogAlgorithm()
result = algo.run(
    g=3,   # Base
    y=6,   # Target: 3^x ≡ 6 (mod 7)
    P=7,   # Modulus (prime)
    backend='torch'
)

print(result['Found x'])                  # Found discrete log x
print(result['status'])                   # 'ok' on success, 'failed' otherwise
print(result['circuit_path'])             # SVG circuit diagram path
print(result.get('plot', []))             # List of output file dicts [{"format": ..., "filename": ...}]
print(result.get('Detected period r'))    # Detected group order r
print(result.get('Computation time (s)')) # Simulation time in seconds
```

## Core Parameters Explained

**Constructor `DiscreteLogAlgorithm(text_mode, algo_dir)`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `'plain'` | Output text formatting mode. Use `'legacy'` for ASCII art panels. |
| `algo_dir` | `str` or `None` | `None` | Directory to save output files. Auto-derived from CWD if `None`. |

**`run(g, y, P, backend, device, dtype)` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `g` | `int` | required | Base of the discrete logarithm. Must satisfy $\gcd(g, P) = 1$. |
| `y` | `int` | required | Target value. Must satisfy $\gcd(y, P) = 1$. |
| `P` | `int` | required | Prime modulus. |
| `backend` | `str` | `'torch'` | Simulation backend. |
| `device` | `str` | `'cpu'` | Compute device (`'cpu'` or `'cuda'`). |
| `dtype` | | `np.complex128` | Numeric dtype for simulation. |

**Common misunderstandings:**
- Both `g` and `y` must be coprime to `P`. This is validated; a `ValueError` is raised otherwise.
- The circuit uses $2 \cdot n_{\text{count}} + n_{\text{work}} = 2 \cdot 2\lfloor\log_2 P\rfloor + \lfloor\log_2 P\rfloor$ total qubits.
- Success is probabilistic; the algorithm relies on a useful two-dimensional Fourier sample. Continued fractions are used to estimate the period component from `u/N`, but the discrete log is recovered only after combining the second sample coordinate `v` through a modular congruence.

## Return Fields

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` if $g^x \equiv y \pmod{P}$ is verified; `'failed'` otherwise. |
| `Found x` | `int` or `None` | Recovered discrete logarithm $x$; `None` if not found. |
| `Detected period r` | `int` or `None` | Detected multiplicative group order $r$; `None` if not found. |
| `Computation time (s)` | `float` | Wall-clock simulation time in seconds. |
| `circuit_path` | `str` | Path to saved SVG circuit diagram. |
| `plot` | `list` | List of output file dicts: `[{"format": "txt", "filename": "..."}]`. Use `.get('plot', [])`. |
| `circuit` | `Circuit` | The constructed `Circuit` object. |

## Implementation Architecture

`DiscreteLogAlgorithm` in `algorithm.py` structures the DLP algorithm into five ordered stages within `run()`, using a matrix-based modular multiplication oracle and a multi-step classical post-processing routine.

**`run(g, y, P, backend, device, dtype)` — Five Stages:**

| Stage | Code Action | Algorithmic Role |
|---|---|---|
| 1 — Parameter Setup | Validates `gcd(g,P)==1` and `gcd(y,P)==1`; sets `n_count = 2*P.bit_length()`, `n_work = P.bit_length()` | Determines register sizes sufficient for continued-fraction accuracy |
| 2 — Circuit Construction | Creates three registers `reg_a`, `reg_b`, `reg_work`; H on reg_a/reg_b; X on work[0] to set $|1\rangle$; controlled $g^{2^i} \bmod P$ via `qc.unitary()` for each reg_a bit; controlled $(y^{-1})^{2^j} \bmod P$ for each reg_b bit; appends `IQFT(n_count)` to both registers | Two-register DLP Fourier-sampling circuit |
| 3 — Simulation | `qc.execute(backend=backend, device=device, dtype=dtype)` → `res_vec.calculate_state(range(2*n_count))` | Extracts probability distribution over reg_a ⊗ reg_b (marginalizing over work register) |
| 4 — Classical Post-Processing | Calls `_classical_post_processing(probs_dict, g, y, P, n_count, N_size)` | Continued fractions plus two-dimensional Fourier-sample congruence solver |
| 5 — Export | `self.save_circuit(qc)` and `self.save_txt()` | Saves SVG circuit diagram and text result file |

**Helper Methods:**

- **`_get_modular_matrix(a, N, n_qubits)`** — Builds a $2^{n\_work} \times 2^{n\_work}$ permutation matrix for the map $z \mapsto (a \cdot z) \bmod P$ (identity for $z \geq P$). Used for both $g^{2^i}$ and $(y^{-1})^{2^j}$ controlled applications.
- **`_classical_post_processing(probs, g, y, P, n, N_size)`** — Multi-step classical routine. This is not a one-dimensional period-finding post-process; it uses the joint sample from both counting registers:
  1. Sorts bitstrings by probability; skips entries below 0.02.
  2. Splits each bitstring into `v_bin = bitstring[:n]` and `u_bin = bitstring[n:]`, then converts them to integers `u, v`.
  3. Applies `Fraction(u, N_size).limit_denominator(P)` to estimate the rational sample component `s/r`, producing candidate `(s_base, r_base)`.
  4. Searches multiples of `r_base` to find the true group order `r` where `g^r ≡ 1 (mod P)`.
  5. Computes `target = round(v * real_r / N_size)` from the second sample coordinate.
  6. Solves the source-level congruence `real_s * x ≡ -target mod real_r` via modular inverse and coset search. This corresponds to the RAG/theory relation $u x \equiv -v \pmod r$ for two-dimensional Fourier samples.
- **`_build_return_dict(is_success, circuit_path, filename, qc)`** — Packages the result dictionary returned by `run()`, including `status` (`'ok'`/`'failed'`), `circuit_path`, `plot` (list of file dicts), `circuit`, and the output fields `Found x`, `Detected period r`, and `Computation time (s)` merged from `self.output`.

**Register address translation:** The `get_p(reg_slice)` inline function inside `run()` translates named register slices into flat qubit indices by adding the appropriate offset (`0` for reg_a, `n_count` for reg_b, `2*n_count` for reg_work).

**Data flow:** `(g, y, P)` → register + oracle construction → `qc.execute()` → `res_vec.calculate_state()` → `_classical_post_processing()` → `found_x` → `_build_return_dict()`.

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
1. Split the measured bitstring exactly as the implementation does: `v_bin = bitstring[:n]`, `u_bin = bitstring[n:]`, then compute `u = int(u_bin, 2)` and `v = int(v_bin, 2)`.
2. Continued fractions on $u/N$ estimate a rational component $s/r$, giving `s_base` and `r_base`.
3. Search multiples of `r_base` until `pow(g, real_r, P) == 1`; scale the numerator at the same time to obtain `real_s`.
4. Use the second Fourier coordinate by computing `target = round(v * real_r / N)`.
5. Solve `real_s * x ≡ -target mod real_r`, including the gcd/coset case, and verify candidates with `pow(g, x, P) == y`.

The conceptual two-dimensional Fourier-sampling equation is
$$
u x \equiv -v \pmod r.
$$
The source expresses the same recovery step with estimated variables as
$$
\texttt{real\_s}\,x \equiv -\texttt{target} \pmod{\texttt{real\_r}}.
$$
Do not describe this as a purely one-dimensional continued-fraction period extraction. Continued fractions estimate the first coordinate's rational structure; the second coordinate is essential for recovering $x$.

## Theory-to-Code Mapping

| README / Theory Concept | Code Object or Location |
|---|---|
| reg_A superposition $H^{\otimes n}|0\rangle$ | `qc.h(get_p(ra[:]))` in Stage 2 |
| reg_B superposition $H^{\otimes n}|0\rangle$ | `qc.h(get_p(rb[:]))` in Stage 2 |
| Work register initialized to $|1\rangle$ | `qc.x(get_p(rw[0]))` in Stage 2 |
| Controlled $g^{2^i} \bmod P$ on work via reg_a bit $i$ | `qc.unitary(matrix, work_qubits, ctrl_qubit, '1')` loop over `range(n_count)` |
| Controlled $(y^{-1})^{2^j} \bmod P$ via reg_b bit $j$ | Same pattern with `y_inv = pow(y, -1, P)` |
| Inverse QFT on reg_a | `qc.append(IQFT(n_count), get_p(ra[:]))` |
| Inverse QFT on reg_b | `qc.append(IQFT(n_count), get_p(rb[:]))` |
| Probability distribution over (A, B) | `state_obj.calculate_state(range(2*n_count))` marginalizes work register |
| Split joint two-register sample | `v_bin = bitstring[:n]`, `u_bin = bitstring[n:]`; then `u, v = int(u_bin, 2), int(v_bin, 2)` |
| Continued fractions: $u/N \approx s/r$ | `Fraction(u, N_size).limit_denominator(P)` |
| True group order $r$: $g^r \equiv 1$ | Search loop over multiples of `r_base` with `pow(g, r_base*k, P) == 1` |
| Second coordinate target | `target = int(round((v * real_r) / N_size))` |
| Solve `real_s * x ≡ -target mod real_r` | `t_red * pow(s_red, -1, r_red) % r_red`; coset search over `d` solutions |
| Verification: $g^x \equiv y \pmod{P}$ | `pow(g, x_test, P) == (y % P)` inside post-processing |

**Notes on encapsulation:** The register address mapping (named registers → flat qubit indices) is handled by an inline `get_p()` closure inside `run()`, rather than a class-level method. This is because the three registers (reg_a, reg_b, reg_work) are passed as arguments to `Circuit` and the gate methods still require flat indices. The post-processing is entirely classical and self-contained in `_classical_post_processing()`.

## Mathematical Deep Dive

**Group structure:** The multiplicative group $(\mathbb{Z}/P\mathbb{Z})^*$ is cyclic of order $r = P-1$ when $P$ is prime and $g$ is a primitive root.

**Eigenstate structure:** The unitary $U_g|z\rangle = |gz \bmod P\rangle$ has eigenstates $|u_s\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1} e^{-2\pi i sk/r}|g^k\rangle$ with eigenvalues $e^{2\pi i s/r}$.

**Two-dimensional Fourier sampling:** The two counting registers sample a correlated pair. In ideal notation, the pair satisfies $u x \equiv -v \pmod r$. In the finite-$N$ implementation, `u/N` is processed with continued fractions to estimate `real_s/real_r`, while `v` is rounded into `target`; the final recovery solves `real_s * x ≡ -target mod real_r`.

**Complexity:** $O(n^3)$ gates where $n = \lceil\log_2 P\rceil$. Classical best: sub-exponential $O(\exp(\sqrt{n\log n}))$ via index calculus.

## Hands-On Example (UnitaryLab)

```python
from unitarylab_algorithms import DiscreteLogAlgorithm

# Solve 3^x ≡ 6 (mod 7): answer is x=3 since 3^3=27=3*7+6
algo = DiscreteLogAlgorithm()
result = algo.run(g=3, y=6, P=7, backend='torch')
print(f"x = {result['Found x']}")                    # Expected: 3
print(f"Status: {result['status']}")                  # 'ok' on success, 'failed' on failure
print(f"Period r: {result['Detected period r']}")     # Detected group order
print(f"Time: {result['Computation time (s)']} s")    # Simulation time
print(f"Verify: 3^3 mod 7 = {pow(3, 3, 7)}")        # Should be 6
print(result.get('plot', []))  # [{"format": "txt", "filename": "..."}]
```

## Implementing Your Own Version

Below is a skeleton that reconstructs the discrete-log quantum circuit at the component level, matching `DiscreteLogAlgorithm`.

```python
# Simplified reconstruction — mirrors DiscreteLogAlgorithm.run()
import math
from fractions import Fraction
import numpy as np
from unitarylab import Circuit, Register
from unitarylab.library import IQFT

def modular_matrix(mult: int, P: int, n_work: int) -> np.ndarray:
    """Permutation matrix |x> -> |x * mult mod P>."""
    dim = 2**n_work
    U = np.zeros((dim, dim))
    for x in range(dim):
        y = (x * mult) % P if x < P else x
        U[y, x] = 1.0
    return U.astype(complex)

def build_dlp_circuit(g: int, y: int, P: int, n_count: int, n_work: int) -> Circuit:
    """
    Two-register Fourier-sampling circuit for DLP g^x = y mod P.
    Register a (counts a): controlled g^{2^i} mod P
    Register b (counts b): controlled (y^{-1})^{2^j} mod P
    Work register: |1>, then |g^a * y^{-b}>
    Measurement: joint Fourier sample whose coordinates are post-processed via
    continued fractions plus the congruence s*x = -target (mod r).
    """
    ra = Register('a', n_count)
    rb = Register('b', n_count)
    rw = Register('w', n_work)
    qc = Circuit(ra, rb, rw, name='DLP_circuit')

    # Offset helpers to get flat qubit indices
    a_off, b_off, w_off = 0, n_count, 2 * n_count

    # 1. Hadamard both counting registers; work register to |1>
    qc.h(range(a_off, a_off + n_count))
    qc.h(range(b_off, b_off + n_count))
    qc.x(w_off)   # |work> = |1>

    # 2. Controlled g^{2^i} on reg_a
    for i in range(n_count):
        mult = pow(g, 2**i, P)
        qc.unitary(modular_matrix(mult, P, n_work),
                   range(w_off, w_off + n_work), a_off + i, '1')

    # 3. Controlled (y^{-1})^{2^j} on reg_b
    y_inv = pow(y, -1, P)
    for j in range(n_count):
        mult = pow(y_inv, 2**j, P)
        qc.unitary(modular_matrix(mult, P, n_work),
                   range(w_off, w_off + n_work), b_off + j, '1')

    # 4. IQFT on both counting registers
    qc.append(IQFT(n_count), range(a_off, a_off + n_count))
    qc.append(IQFT(n_count), range(b_off, b_off + n_count))
    return qc

def solve_dlp(g: int, y: int, P: int, backend: str = 'torch', device: str = 'cpu'):
    """Quantum DLP: find x such that g^x = y mod P."""
    if math.gcd(g, P) != 1 or math.gcd(y, P) != 1:
        raise ValueError('g and y must be coprime to P')
    n_work  = P.bit_length()
    n_count = 2 * n_work

    qc = build_dlp_circuit(g, y, P, n_count, n_work)
    res_vec = qc.execute(backend=backend, device=device, dtype=np.complex128)
    probs_dict = res_vec.calculate_state(range(2 * n_count))

    # Sort by probability and post-process the two-dimensional Fourier sample.
    sorted_probs = sorted(probs_dict.items(), key=lambda item: item[1]['prob'], reverse=True)
    for bitstring, data in sorted_probs:
        if data['prob'] < 0.02:
            continue
        v_bin, u_bin = bitstring[:n_count], bitstring[n_count:]
        u, v = int(u_bin, 2), int(v_bin, 2)
        if u == 0:
            continue
        frac = Fraction(u, 2**n_count).limit_denominator(P)
        r_base, s_base = frac.denominator, frac.numerator
        real_r = real_s = None
        for k in range(1, 10):
            if pow(g, r_base * k, P) == 1:
                real_r, real_s = r_base * k, s_base * k
                break
        if real_r is None:
            continue
        target = int(round((v * real_r) / (2**n_count)))
        d = math.gcd(real_s, real_r)
        if (-target) % d:
            continue
        s_red, r_red, t_red = real_s // d, real_r // d, (-target) // d
        try:
            x0 = (t_red * pow(s_red, -1, r_red)) % r_red
        except ValueError:
            continue
        for i in range(d):
            x_test = (x0 + i * r_red) % real_r
            if pow(g, x_test, P) == y % P:
                return x_test
    return None
```

**Component roles**:
- `modular_matrix` — same pattern as Shor: $2^n \times 2^n$ permutation for $|x\rangle \to |x \cdot \text{mult} \bmod P\rangle$.
- `build_dlp_circuit` — two-register Fourier sampling: register $a$ accumulates powers of $g$, register $b$ accumulates powers of $y^{-1}$, both followed by IQFT.
- `solve_dlp` — measures both registers, uses continued fractions on `u/N` to estimate `s/r`, combines `v` through `target = round(v*r/N)`, and solves the congruence `s*x ≡ -target (mod r)`.



## Reference Implementation (Classiq)

Classiq provides a high-level QMOD implementation of the discrete logarithm algorithm. It uses `@qfunc` to define modular exponentiation, the discrete-log oracle, inverse `qft`, model creation, synthesis, and execution.

This reference is useful for understanding how the Shor-style discrete logarithm workflow can be expressed through automatic circuit synthesis.

### Example A: Minimal Classiq Discrete Logarithm Model

```python
from classiq import *
from classiq.qmod.symbolic import ceiling, log


@qfunc
def modular_exponentiation(N: CInt, a: CInt, x: QArray, pw: QArray):
    repeat(
        count=pw.len,
        iteration=lambda index: control(
            pw[index],
            lambda: modular_multiply_constant_inplace(
                N,
                a ** (2**index),
                x,
            ),
        ),
    )


@qfunc
def discrete_log_oracle(
    g_generator: CInt,
    x_element: CInt,
    N_modulus: CInt,
    alpha: QArray,
    beta: QArray,
    func_res: Output[QNum],
) -> None:
    allocate(ceiling(log(N_modulus, 2)), func_res)
    func_res ^= 1

    # Apply x^alpha mod N
    modular_exponentiation(N_modulus, x_element, func_res, alpha)

    # Apply g^beta mod N
    modular_exponentiation(N_modulus, g_generator, func_res, beta)


@qfunc
def discrete_log(
    g: CInt,
    x: CInt,
    N: CInt,
    order: CInt,
    alpha: Output[QArray],
    beta: Output[QArray],
    func_res: Output[QArray],
) -> None:
    reg_len = ceiling(log(order, 2))

    allocate(reg_len, alpha)
    allocate(reg_len, beta)

    hadamard_transform(alpha)
    hadamard_transform(beta)

    discrete_log_oracle(g, x, N, alpha, beta, func_res)

    invert(lambda: qft(alpha))
    invert(lambda: qft(beta))


MODULU_NUM = 5
G_GENERATOR = 3
X_LOG_ARG = 2
ORDER = MODULU_NUM - 1


@qfunc
def main(
    alpha: Output[QNum],
    beta: Output[QNum],
    func_res: Output[QNum],
) -> None:
    discrete_log(
        G_GENERATOR,
        X_LOG_ARG,
        MODULU_NUM,
        ORDER,
        alpha,
        beta,
        func_res,
    )


qmod = create_model(
    main,
    constraints=Constraints(max_width=13),
    preferences=Preferences(optimization_level=1),
    execution_preferences=ExecutionPreferences(num_shots=4000),
)

qprog = synthesize(qmod)
result = execute(qprog).result_value()
print(result)

```

## Debugging Tips

1. **Simulation is slow**: Qubit count is $5\lfloor\log_2 P\rfloor$; exponential in bits. Keep $P$ small ($P=7$, $11$, $13$).
2. **`Found x` is `None`**: The joint sample may not yield a usable continued-fraction estimate or solvable congruence. Re-run (the algorithm is probabilistic) or increase $n_{\text{count}}$.
3. **`g` and `y` not coprime to `P`**: Will raise `ValueError`. Ensure $\gcd(g, P) = \gcd(y, P) = 1$.
4. **Wrong answer**: The quantum measurement gives the correct answer with high probability but not certainty. The classical post-processing verifies correctness; if wrong, retry.
