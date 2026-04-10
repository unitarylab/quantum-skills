---
name: shor
description: Shor integer factorization algorithm guide for using the quantum simulator implementation and for writing a compatible implementation from scratch
---

# Shor Integer Factorization Guide


## What It Does

This module implements **Shor's algorithm** for integer factorization.

Given a composite integer $N$, it tries to find non-trivial factors of $N$ by reducing factorization to **order finding** for:

$$f(x) = a^x \bmod N$$

The implementation combines:
- fast classical exits for easy cases,
- a quantum period-finding circuit built with `GateSequence`,
- classical continued-fraction post-processing,
- automatic retry logic over random bases `a`.

This is important because classical integer factorization is hard for large inputs, while Shor's algorithm can solve the order-finding subproblem in polynomial time on a quantum computer.

---

## How This Implementation Works

The public entry point is the `ShorAlgorithm` class in `algorithm.py`.

### Main Flow

The `run()` method follows this structure:

1. Check whether `N` is even.
2. Randomly choose a base `a` with `2 <= a <= N-1`.
3. If `gcd(a, N) > 1`, return factors classically.
4. Build a quantum circuit for modular exponentiation and inverse QFT.
5. Simulate the circuit with the selected backend.
6. Measure the counting register and estimate the phase.
7. Use continued fractions to infer the period `r`.
8. Convert `r` into candidate factors.
9. Retry with a new `a` if the attempt is invalid.

### Supported Circuit Construction Modes

The implementation supports two methods:

- `matrix`: builds controlled modular multiplication from an explicit permutation matrix.
- `operator`: builds modular multiplication from arithmetic subcircuits.

The `matrix` method is easier to understand and is the best starting point for users trying the algorithm for small `N`.

### Register Sizing

For an input integer `N`:

```python
n_work = N.bit_length()
n_count = 2 * n_work
```

Then:

- `matrix` mode uses `n_work_actual = n_work`
- `operator` mode uses `n_work_actual = n_work * 2 + 2`

Total qubits:

```python
total_qubits = n_count + n_work_actual
```

---

## Using the Existing Implementation

### Import Path

You can import Shor from either of these paths:

```python
from engine.algorithms import ShorAlgorithm
```

### Quickstart

```python
from engine.algorithms import ShorAlgorithm

shor = ShorAlgorithm()

result = shor.run(
    N=15,
    method='matrix',
    backend='torch',
    max_retries=15,
    algo_dir='./shor_results'
)

print(result['status'])
print(result['factors'])
print(result['period'])
print(result['circuit_path'])
print(result['message'])
print(result['plot'])
```

### Recommended First Example

Use `N = 15` first. It is small, standard for demos, and aligns with the existing module default example.

```python
from engine.algorithms import ShorAlgorithm

shor = ShorAlgorithm()
result = shor.run(15, method='matrix', backend='torch')

if result['status'] == 'ok':
    p, q = result['factors']
    print(f"Success: {p} * {q} = {p*q}")
else:
    print(f"Failed: {result['message']}")
```

### Return Value

The `run()` method returns a dictionary with this structure:

```python
{
    "status": "ok" or "failed",
    "factors": [p, q],
    "period": r,
    "circuit_path": "./shor_results/...svg" or None,
    "message": "runtime summary",
    "plot": "formatted ASCII report"
}
```

### Output Behavior

On success, the module also:

- caches the full result in `self.last_result`,
- saves a circuit diagram to `algo_dir`,
- renders an ASCII summary through `format_result_ascii()`.

You can inspect the cached result after a run:

```python
shor = ShorAlgorithm()
shor.run(15)

last = shor.last_result
print(last['N'])
print(last['a'])
print(last['period'])
print(last['computation_time'])
```

---

## Writing Your Own Version

If you want to implement your own compatible Shor module, keep the same high-level structure.

### Minimal Class Skeleton

```python
import math
import random
from fractions import Fraction

from engine import GateSequence, State, IQFT

class ShorAlgorithm:
    def __init__(self):
        self.last_result = None

    def run(self, N, method='matrix', backend='torch', max_retries=15, algo_dir='./shor_results'):
        if N % 2 == 0:
            return {"status": "ok", "factors": [2, N // 2]}

        for _ in range(max_retries):
            a = random.randint(2, N - 1)
            gcd_val = math.gcd(a, N)
            if gcd_val > 1:
                return {"status": "ok", "factors": [gcd_val, N // gcd_val]}

            n_work = N.bit_length()
            n_count = 2 * n_work
            total_qubits = n_count + n_work

            gs = GateSequence(total_qubits, backend=backend)
            gs.h(range(n_count))
            gs.x(n_count)

            self._build_modular_matrix_circuit(gs, n_count, n_work, a, N)
            gs.append(IQFT(n_count, backend=backend), range(n_count))

            result_vector = gs.execute()
            measure_bin = State(result_vector).measure(range(n_count), endian='little')
            measure_int = int(measure_bin, 2)

            phase = measure_int / (2 ** n_count)
            r = Fraction(phase).limit_denominator(N).denominator

            if r % 2 == 0:
                guess = pow(a, r // 2, N)
                if guess not in (1, N - 1):
                    p = math.gcd(guess - 1, N)
                    if p > 1 and N % p == 0:
                        return {"status": "ok", "factors": [p, N // p], "period": r}

        return {"status": "failed", "factors": []}
```

### Required Pieces

Your own implementation needs these parts:

- a public `run()` method that orchestrates retry, circuit construction, simulation, and post-processing,
- a modular multiplication implementation,
- inverse QFT on the counting register,
- measurement of the counting register,
- continued-fraction reconstruction of the period,
- factor extraction from the recovered period.

### Core Mathematical Condition

After estimating the order `r`, factor extraction depends on:

1. `r` must be even.
2. `a^(r/2) mod N` must not be `1`.
3. `a^(r/2) mod N` must not be `N - 1`.

Then candidate factors come from:

```python
import math

p = math.gcd(pow(a, r // 2, N) - 1, N)
q = math.gcd(pow(a, r // 2, N) + 1, N)
```

---

## Key Code Blocks

### 1. Quantum Circuit Setup

This is the core pattern used in the existing implementation:

```python
from engine import GateSequence, IQFT

n_work = N.bit_length()
n_count = 2 * n_work
total_qubits = n_count + n_work

gs = GateSequence(total_qubits, name=f'Shor_N{N}_a{a}_matrix', backend=backend)
gs.h(range(n_count))
gs.x(n_count)

self._build_modular_matrix_circuit(gs, n_count, n_work, a, N)
gs.append(IQFT(n_count, backend=backend), range(n_count))
```

Meaning:

- `gs.h(range(n_count))` creates a superposition over counting states.
- `gs.x(n_count)` initializes the work register to `|1⟩`.
- modular multiplication writes phase information into the counting register.
- `IQFT` converts phase information into a measurable frequency pattern.

### 2. Matrix-Based Controlled Modular Multiplication

This is the easiest part to reuse when building a small demonstration version:

```python
import numpy as np

def _get_modular_matrix(self, a, N, n_qubits):
    dim = 2 ** n_qubits
    matrix = np.zeros((dim, dim))
    for y in range(dim):
        if y < N:
            target = (a * y) % N
        else:
            target = y
        matrix[target, y] = 1.0
    return matrix


def _build_modular_matrix_circuit(self, gs, n_count, n_work, a, N):
    total_qubits = n_count + n_work
    for q in range(n_count):
        power_factor = pow(a, 2 ** q, N)
        matrix = self._get_modular_matrix(power_factor, N, n_work)
        gs.unitary(matrix, range(n_count, total_qubits), q, '1')
```

Why it works:

- each counting qubit controls multiplication by $a^{2^q} \bmod N$,
- this is the standard repeated-squaring structure used in modular exponentiation,
- the control pattern lets phase estimation recover the order of `a mod N`.

### 3. Post-Processing the Measured Phase

```python
from fractions import Fraction

from engine import State

result_vector = gs.execute()
result_state = State(result_vector)
measure_bin = result_state.measure(range(n_count), endian='little')
measure_int = int(measure_bin, 2)

phase = measure_int / (2 ** n_count)
r = Fraction(phase).limit_denominator(N).denominator
```

This is the bridge from quantum output to a classical period estimate.

### 4. Factor Recovery

```python
import math

if r % 2 == 0 and r > 0:
    guess = pow(a, r // 2, N)
    if guess != N - 1 and guess != 1:
        p = math.gcd(guess - 1, N)
        q = math.gcd(guess + 1, N)
```

This is the exact classical decision point where a valid order becomes a factorization.

---

## Practical Notes

### Which Method Should You Try First?

- Choose `matrix` if you want a compact, readable implementation for small examples.
- Choose `operator` if you want to study arithmetic circuit construction in more detail.

### When Runs Fail

Failures are normal in Shor-style demonstrations. Common reasons:

- the random base `a` does not yield a useful period,
- the inferred period `r` is odd,
- the value $a^{r/2} \bmod N$ is trivial,
- the simulator cost grows quickly with larger `N`.

This is why the module retries automatically with new random bases.

### Practical Limits

This implementation is best treated as a **quantum simulator demonstration** and an **educational reference**, not a large-scale factoring tool.

As `N` grows:

- qubit counts increase,
- matrix dimensions grow exponentially,
- simulation becomes expensive very quickly.

### Suggested Learning Path

1. Run `ShorAlgorithm().run(15, method='matrix')`.
2. Read `_get_modular_matrix()` and `_build_modular_matrix_circuit()` first.
3. Inspect how `IQFT` is appended.
4. Trace the continued-fraction recovery of `r`.
5. Only then move to the `operator` arithmetic subcircuit path.

### Environment Expectation

The repository README indicates the simulator depends on `unitarylab` and uses the `torch` backend in examples. Make sure the project dependencies and simulator backend are installed before running the algorithm.

---

## Summary

Use this module when you want to:

- demonstrate Shor's order-finding workflow on a simulator,
- inspect a complete classical-plus-quantum factoring pipeline,
- reuse a working matrix-based period-finding template,
- extend the implementation with your own modular arithmetic circuit design.

For first-time users, start with `N = 15` and `method='matrix'`. For contributors, treat the matrix path as the reference implementation and the operator path as the deeper circuit-engineering version.