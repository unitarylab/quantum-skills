---
name: simon
description: "Use for implementing, explaining, running, or debugging Simon's algorithm in this repository, especially for oracle construction, measurement interpretation, GF(2) post-processing, simulator state extraction, and compatible reimplementation."
---

## One Step to Run Simon Example
```bash
python ./scripts/algorithm.py
```

# Simon Hidden-String Guide

## What It Does

This module implements Simon's algorithm for recovering a hidden nonzero bit string $s$ from a black-box function with the promise:

$$
f(x) = f(y) \iff x \oplus y = s
$$

Given a target secret string such as `1010`, the implementation:

- builds a Simon circuit with two quantum registers,
- runs it on the simulator,
- extracts measurement constraints on the hidden string,
- solves the resulting linear system over $\mathrm{GF}(2)$,
- saves the circuit diagram and formats a readable result summary.

This is useful both as a simulator demo of Simon's exponential query advantage and as a reference for writing a compatible implementation in this codebase.

## Trigger Phrases (User Commands)

This skill should be triggered when the user asks for Simon-algorithm tasks such as implementation, explanation, debugging, or adaptation to this repository.

Typical trigger prompts include:

- "Implement the Simon algorithm"
- "Write a compatible SimonAlgorithm in this repository"
- "Explain the run flow of the Simon hidden string algorithm"
- "Help me debug why the Simon algorithm cannot find 's'"
- "Build a Simon oracle using GateSequence"
- "How to find Simon's 's' from measurement results using GF(2)"
- "Run the Simon algorithm with secret 1010 on the torch backend"
- "Create a SimonAlgorithm implementation compatible with engine.algorithms"
- "Why does my Simon circuit fail to recover the hidden string?"

To improve activation reliability, user requests should mention one or more of these keywords: `Simon`, `hidden string`, `oracle`, `GF(2)`, `SimonAlgorithm`, `GateSequence`, `torch backend`.

---

## How This Implementation Works

The public entry point is the `SimonAlgorithm` class in `algorithm.py`.

### Main Flow

The `run()` method follows this structure:

1. Validate the input hidden string `s_target`.
2. Reject the all-zero string.
3. Enforce the `torch` backend.
4. Create two `n`-qubit quantum registers `x` and `y`, plus an `n`-bit classical register.
5. Apply Hadamards to the input register `x`.
6. Build the Simon oracle `U_f` determined by `s_target`.
7. Measure the `y` register into a classical register.
8. Apply Hadamards again to the `x` register.
9. Execute the circuit on the simulator.
10. Extract basis states from the `x` register and solve for `s` classically.
11. Save the circuit diagram and return a structured result dictionary.

### Important Implementation Detail

This implementation does not repeatedly execute the circuit to collect one measured bit string at a time.

Instead, after a single simulator execution it calls:

```python
state_basis_dict = state_obj.calculate_state(range(n))
```

and uses the resulting support on the `x` register as the source of candidate equations. It then selects linearly independent vectors and solves for `s` with a simple back-substitution routine.

That is a simulator-friendly shortcut specific to this implementation, and it is the main behavior to preserve if you want your own version to be compatible with the current module.

### Register Layout

For a secret string of length `n`:

```python
rx = Register('x', n)
ry = Register('y', n)
cqr = ClassicalRegister('cr', n)
```

So the circuit uses:

- `n` qubits in the input register `x`,
- `n` qubits in the output register `y`,
- `n` classical bits for the mid-circuit measurement.

Total quantum qubits:

```python
total_qubits = 2 * n
```

---

## Using the Existing Implementation

### Import Path

You can import Simon from:

```python
from engine.algorithms import SimonAlgorithm
```

### Quickstart

```python
from engine.algorithms import SimonAlgorithm

simon = SimonAlgorithm()

result = simon.run(
    s_target='1010',
    backend='torch',
    algo_dir='./simon_results'
)

print(result['status'])
print(result['found_s'])
print(result['circuit_path'])
print(result['message'])
print(result['plot'])
```

### Recommended First Example

Start with a short nonzero secret string such as `1010` or `1101`.

```python
from engine.algorithms import SimonAlgorithm

simon = SimonAlgorithm()
result = simon.run(s_target='1101', backend='torch')

if result['status'] == 'ok':
    print(f"Recovered secret: {result['found_s']}")
else:
    print(f"Run failed: {result['message']}")
```

### Return Value

The `run()` method returns a dictionary with this structure:

```python
{
    "status": "ok" or "failed",
    "found_s": "1101",
    "circuit_path": "./simon_results/...svg",
    "message": "runtime summary",
    "plot": "formatted ASCII report"
}
```

### Cached Result State

On completion, the module also stores the full execution record in `self.last_result`.

```python
simon = SimonAlgorithm()
simon.run('1010')

last = simon.last_result
print(last['s_target'])
print(last['found_s'])
print(last['valid_states'])
print(last['equations'])
print(last['comp_time'])
```

### Input Constraints

The current implementation assumes:

- `s_target` is a binary string,
- `s_target` is not all zeros,
- `backend` must be `'torch'`.

If `s_target` contains no `'1'`, the code raises:

```python
ValueError("秘密字符串 s 不能为全 0 串！")
```

If the backend is not `torch`, the code raises:

```python
ValueError("Simon 算法目前仅支持 'torch' 后端进行含测量模拟...")
```

---

## Writing Your Own Version

If you want to implement your own compatible Simon module, keep the same high-level structure and return format.

### Minimal Class Skeleton

```python
import os
import time

from engine import GateSequence, Register, ClassicalRegister, State


class SimonAlgorithm:
    def __init__(self):
        self.log_prefix = "INFO: "
        self.last_result = None

    def run(self, s_target='1010', backend='torch', algo_dir='./simon_results'):
        n = len(s_target)
        if s_target.find('1') == -1:
            raise ValueError("secret string must not be all zeros")
        if backend != 'torch':
            raise ValueError("only torch backend is supported")

        rx = Register('x', n)
        ry = Register('y', n)
        cqr = ClassicalRegister('cr', n)
        gs = GateSequence(rx, ry, cqr, name=f'Simon_{s_target}', backend=backend)

        gs.h(rx[:])
        self._build_simon_oracle(gs, s_target)
        gs.measure(ry[:], cqr[:])
        gs.h(rx[:])

        result_state = State(gs.execute())
        basis_dict = result_state.calculate_state(range(n))
        basis = self._get_basis_simple(list(basis_dict.keys()), n)
        found_s = self._solve_simon_general(basis, n)

        os.makedirs(algo_dir, exist_ok=True)
        circuit_path = os.path.join(algo_dir, f"Simon_s_{s_target}_circuit.svg")
        gs.draw(filename=circuit_path, title=f"Simon Algorithm (s={s_target})")

        return {
            "status": "ok" if found_s == s_target else "failed",
            "found_s": found_s,
            "circuit_path": circuit_path,
        }
```

### Required Pieces

Your own implementation needs these parts:

- a public `run()` method,
- a Simon oracle builder,
- Hadamards before and after the oracle,
- a mid-circuit measurement of the `y` register,
- extraction of basis vectors from the `x` register,
- a classical solver over $\mathrm{GF}(2)$,
- the same result packaging pattern.

### Classical Recovery Condition

Each sampled or extracted bit string `y` must satisfy:

$$
y \cdot s \equiv 0 \pmod 2
$$

The classical post-processing stage recovers `s` from enough linearly independent equations of that form.

---

## Key Code Blocks

### 1. Circuit Construction Pattern

This is the core setup used in the existing implementation:

```python
from engine import Register, ClassicalRegister, GateSequence
n = len(s_target)

rx = Register('x', n)
ry = Register('y', n)
cqr = ClassicalRegister('cr', n)
gs = GateSequence(rx, ry, cqr, name=f'Simon_{s_target}', backend=backend)

gs.h(rx[:])
self._build_simon_oracle(gs, s_target)
gs.measure(ry[:], cqr[:])
gs.h(rx[:])
```

Meaning:

- the first Hadamard layer creates a uniform superposition over inputs,
- the oracle encodes the Simon promise,
- the `y` register is measured mid-circuit,
- the second Hadamard layer turns the hidden-period structure into linear constraints on `s`.

### 2. Oracle Construction

This module uses the following oracle builder:

```python
def _build_simon_oracle(self, gs, s):
    n = len(s)
    for i in range(n - 1, -1, -1):
        gs.cx(i, i + n)

    pivot_idx = s.find('1')
    for i in range(n):
        if s[i] == '1':
            gs.cx(n - 1 - pivot_idx, n - 1 - i + n)
```

Why it matters:

- the first loop copies input structure from `x` into `y`,
- the second loop uses the first `1` in `s` as a pivot,
- every `1` in `s` adds the corresponding controlled-X relation needed to enforce the Simon promise.

If you are writing your own version in this repository, this is the most important block to mirror.

### 3. Extracting Equations from the Final State

After execution, the current implementation derives equation candidates like this:

```python
from engine import State, GateSequence

re_state = gs.execute()
state_obj = State(re_state)
state_basis_dict = state_obj.calculate_state(range(n))
```

Then it chooses a small linearly independent subset:

```python
basis = self._get_basis_simple(list(state_basis_dict.keys()), n)
```

The selection routine is:

```python
def _get_basis_simple(self, state_list, n_qubits):
    basis_list, seen_pivots = [], set()
    for bin_str in state_list:
        pivot = bin_str.find('1')
        if pivot != -1 and pivot not in seen_pivots:
            seen_pivots.add(pivot)
            basis_list.append(bin_str)
        if len(basis_list) == n_qubits - 1:
            break
    return basis_list
```

This is a lightweight independence heuristic rather than a full Gaussian elimination pass.

### 4. Solving for the Hidden String

The hidden string is reconstructed by back-substitution:

```python
def _solve_simon_general(self, basis_list, n):
    s_vec = [0] * n
    pivot_map = {y.find('1'): y for y in basis_list if y.find('1') != -1}
    free_idx = next((i for i in range(n) if i not in pivot_map), -1)

    if free_idx == -1:
        return "Error"
    s_vec[free_idx] = 1

    for p in sorted(pivot_map.keys(), reverse=True):
        y_vec = pivot_map[p]
        dot_sum = 0
        for j in range(p + 1, n):
            if y_vec[j] == '1':
                dot_sum ^= s_vec[j]
        s_vec[p] = dot_sum
    return "".join(map(str, s_vec))
```

This is the classical decision point that turns the orthogonality constraints into a candidate secret string.

---

## Practical Notes

### Backend Limitation

The implementation explicitly rejects non-`torch` backends, so use:

```python
backend='torch'
```

### Which Secrets To Try First

Use short nonzero binary strings first, for example:

- `11`
- `101`
- `1010`
- `1101`

These keep the circuit small and make it easier to inspect the generated SVG diagram.

### Why the Algorithm Can Fail

Failures are possible in this implementation when:

- the extracted basis vectors are insufficient,
- the pivot-based heuristic does not recover enough independent equations,
- the reconstructed `found_s` does not match `s_target`.

Unlike the textbook presentation, this module does not loop over many separate circuit executions, so its recovery strategy is tightly coupled to simulator state inspection.

### Practical Limits

This module is best treated as a simulator demonstration and educational reference. As `n` grows, state-vector simulation becomes expensive because the total quantum register width is `2n`.

### Environment Expectation

The repository README indicates the simulator depends on `unitarylab`, and the examples use the `torch` backend. Ensure the backend dependency is installed before running the algorithm.

---

## Summary

Use this module when you want to:

- demonstrate Simon's hidden-string workflow on the project simulator,
- study a complete quantum-plus-classical implementation,
- inspect how a Simon oracle is encoded with `GateSequence`,
- build your own compatible version using the same circuit and return conventions.

For first-time use, start with `SimonAlgorithm().run('1010', backend='torch')`. For contributors, treat `_build_simon_oracle()`, `_get_basis_simple()`, and `_solve_simon_general()` as the key reference points.