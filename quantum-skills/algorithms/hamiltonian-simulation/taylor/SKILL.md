---
name: taylor
description: Taylor-series Hamiltonian simulation via Linear Combination of Unitaries (LCU), approximating e^{-iHt} with a truncated polynomial expansion of the evolution operator.
---

## One Step to Run Taylor Example
```bash
python ./scripts/algorithm.py
```

# Taylor-Series Hamiltonian Simulation Skill Guide

## Overview

Taylor simulation approximates

$$
U(t)=e^{-iHt}
$$

using a truncated series, then compiles the resulting operator as a linear combination of unitaries (LCU).

### Key Insight

Instead of composing deterministic short products (as in Trotter), this method builds a polynomial approximation of the evolution operator itself. Complex series coefficients are split into magnitude and phase, where phase is embedded in unitary blocks and magnitude becomes LCU weight.

### Why Taylor-Series Matters:

1. It offers a systematic approximation route via truncation order.
2. It pairs naturally with LCU machinery.
3. It can be very accurate when order and slicing are chosen well.
4. It creates a useful bridge between analytic expansions and circuit synthesis.

### Real Applications:

1. Matrix-function based simulation prototypes.
2. Controlled approximation studies in small-to-medium systems.
3. Cross-checking product-formula and QSP implementations.
4. Educational demonstrations of LCU from complex coefficients.

## Learning Objectives

After using this skill, you should be able to:
1. Explain truncated Taylor expansion for $e^{-iHt}$.
2. Understand why time slicing is used before truncation.
3. Follow dynamic programming accumulation of Pauli coefficients.
4. Convert complex coefficients into unitary-phase and nonnegative weights.
5. Interpret ancilla block extraction in final matrix reconstruction.

## Prerequisites

### Essential knowledge:

1. Taylor expansion of exponential operators.
2. Pauli-string decomposition and multiplication.
3. Linear Combination of Unitaries concept.

### Mathematical comfort:

1. Complex numbers, coefficient phase and magnitude.
2. Norm-based error interpretation.
3. Truncation versus slicing tradeoffs.

### Recommended:

1. High-level familiarity with `LCU` helper from `core.linear_combination_unitary`.
2. Awareness of helper functions: `pauli_string_multiply`, `pauli_string_power`, `pauli_string_circuit`.
3. Start with low-dimensional matrices before large expansions.

## Using the Provided Implementation

### Quick Start Example

```python
import numpy as np
from engine.algorithms.taylor import Taylor

H = np.array([
    [1.0, 0.4],
    [0.4, -0.5]
], dtype=complex)

sim = Taylor(
    H=H,
    t=0.8,
    target_error=1e-4,
    degree=10,
)

print("method:", sim.method)
print("degree:", sim.degree)
print("lambda=||H||*t:", sim.lam)
print("error:", sim.total_error)
```

### Core Parameters Explained

```python
class Taylor(HamiltonianSimulationResult):
    def __init__(
        self,
        H: np.ndarray,
        t: float,
        target_error: float,
        degree: int = 10
    ):
        ...
```

Parameter meaning:
1. `H`: Hermitian Hamiltonian.
2. `t`: total evolution time.
3. `target_error`: desired approximation level used in degree heuristic.
4. `degree`: initial truncation-order hint (internally adjusted and capped at 15).

Important constructor logic:

```python
self.alpha = np.linalg.norm(H, 2)
self.lam = self.alpha * self.t
self.time_split = 0.5
self.degree = min(max(degree, int(np.ceil(self.lam * 1.5 + np.log(1/self.target_error) * 1.5))), 15)
```

Return dictionary contains:

This constructor returns an object. A practical logging dictionary is:

```python
result = {
    "method": sim.method,
    "target_qubits": sim.target_qubits,
    "degree": sim.degree,
    "lambda": sim.lam,
    "time_split": sim.time_split,
    "circuit": sim.circuit,
    "evolution_result": sim.evolution_result,
    "total_error": sim.total_error,
}
```

## Understanding the Core Components

### 1) Time slicing and per-slice decomposition

From `_run` in `algorithms/taylor/taylor.py`:

```python
r = int(self.lam / self.time_split) + 1
ans_decomposition = self._pauli_decompose(self.H, self.t / r)
L = len(ans_decomposition)
```

Interpretation:
1. Total time is split into `r` slices.
2. One-slice Hamiltonian scale is `t / r`.
3. Truncation is applied at slice level, then lifted to total evolution.

### 2) Dynamic-programming series accumulation

Code excerpt:

```python
ans_term_map = dict()
for k in range(self.degree + 1):
    ans_term_map[k] = defaultdict(complex)
ans_term_map[0]["I" * self.target_qubits] = 1.0

for k in range(1, self.degree + 1):
    for str_prev in ans_term_map[k-1]:
        for i in range(L):
            ans_str, ans_val = pauli_string_multiply(str_prev, ans_decomposition[i][0])
            ans_term_map[k][ans_str] += (
                ans_term_map[k-1][str_prev]
                * ans_val
                * ans_decomposition[i][1]
                * -1j / k
            )
```

Interpretation:
1. `ans_term_map[k]` stores all Pauli-string coefficients at order `k`.
2. Recurrence applies product-rule accumulation with `-i/k` factor.
3. This avoids naive full symbolic expansion from scratch each order.

### 3) Slice powering and LCU conversion

Code excerpt:

```python
ans_term_list = [(key, val) for key, val in ans_term_list.items()]
term_list = pauli_string_power(ans_term_list, r)

LCU_terms = list()
for key, coef in term_list:
    magnitude = abs(coef)
    phase = cmath.phase(coef)
    U_rotation = self._make_U_rotation(pauli_string_circuit(key), phase, self.target_qubits)
    LCU_terms.append((U_rotation, magnitude))

circuit = self._circuit = LCU(LCU_terms)
```

Interpretation:
1. One-slice approximation is raised to power `r`.
2. Complex coefficient phase is encoded into unitary block.
3. Magnitude becomes nonnegative LCU coefficient.

### 4) Matrix block extraction and renormalization

Code excerpt:

```python
m = len(LCU_terms)
lcu_matrix = circuit.get_matrix()
U_approx = np.zeros_like(self.H, dtype=complex)
for i in range(len(U_approx)):
    for j in range(len(U_approx)):
        U_approx[i, j] = lcu_matrix[i*m, j*m]
s = sum(alpha for _, alpha in LCU_terms)
U_approx = U_approx * s
```

Interpretation:
1. The code extracts the ancilla-|0> block by indexed strides.
2. It multiplies back by coefficient sum to undo LCU normalization.

### 5) External package function notes (brief)

1. `LCU` builds a combined gate sequence from weighted unitary blocks.
2. `pauli_string_power` composes a Pauli-sum representation by repeated multiplication.
3. `pauli_string_circuit` converts Pauli strings to executable circuits.

## Hands-On Example: Hamiltonian Simulation

Compare truncation settings under fixed time.

```python
import numpy as np
from engine.algorithms.taylor import Taylor

H = np.array([
    [0.9, 0.2, 0.0, 0.0],
    [0.2, -0.9, 0.2, 0.0],
    [0.0, 0.2, 0.6, 0.1],
    [0.0, 0.0, 0.1, -0.6],
], dtype=complex)

for deg in [6, 8, 10, 12]:
    sim = Taylor(H=H, t=0.8, target_error=1e-4, degree=deg)
    print(f"degree_hint={deg}, effective_degree={sim.degree}, error={sim.total_error:.3e}")
```

What to look for:
1. Effective degree may be greater than input hint due to heuristic floor.
2. Higher effective degree generally improves accuracy but can increase cost sharply.

## Mathematical Deep Dive

Taylor truncation of one-slice evolution with $\Delta t=t/r$:

$$
e^{-iH\Delta t} \approx \sum_{k=0}^{K}\frac{(-iH\Delta t)^k}{k!}
$$

If

$$
H=\sum_{\ell=1}^{L}\alpha_\ell H_\ell,
$$

then each order expands as

$$
(-iH\Delta t)^k = (-i\Delta t)^k
\sum_{l_1,\dots,l_k=1}^{L}
\alpha_{l_1}\cdots\alpha_{l_k}
H_{l_1}\cdots H_{l_k}.
$$

Thus one-slice truncated evolution can be written in LCU form

$$
\widetilde{U}_r = \sum_{j=0}^{m-1}\beta_j V_j.
$$

LCU preparation and selection are:

$$
B|0\rangle = \frac{1}{\sqrt{s}}\sum_j \sqrt{\beta_j}|j\rangle,
\quad
s=\sum_j\beta_j
$$

$$
\mathrm{select}(V)|j\rangle|\psi\rangle = |j\rangle V_j|\psi\rangle
$$

$$
W=(B^\dagger\otimes I)\,\mathrm{select}(V)\,(B\otimes I).
$$

Acting on $|0\rangle|\psi\rangle$:

$$
W|0\rangle|\psi\rangle
=
\frac{1}{s}|0\rangle\widetilde{U}_r|\psi\rangle
+
\sqrt{1-\frac{1}{s^2}}\,|\Phi\rangle.
$$

So the post-selection success probability is $1/s^2$.

The implementation computes this in Pauli-string coefficient space using dynamic programming, then raises the slice approximation to the $r$-th power.

For each resulting complex coefficient $a_j$ with unitary block $U_j$:

$$
a_j U_j = |a_j| \cdot \left(e^{i\arg(a_j)}U_j\right)
$$

so LCU receives nonnegative weights $|a_j|$ and phase-adjusted unitaries.

In the literature for $T=(\sum_\ell \alpha_\ell)t$, representative complexity expressions are

$$
O\left(T\,\frac{\log(T/\epsilon)}{\log\log(T/\epsilon)}\right)
$$

for oracle queries and

$$
O\left(nT\,\frac{\log^2(T/\epsilon)}{\log\log(T/\epsilon)}\right)
$$

for additional two-qubit-gate overhead.

Practical implications:
1. Degree controls truncation error.
2. Slice count controls local approximation regime.
3. LCU term growth can dominate runtime and memory.

Implementation-consistent notes:
1. The full construction in code matches the sequence: time slicing, truncated series accumulation in Pauli space, then LCU realization with phase/magnitude split.
2. The phase-magnitude decomposition is explicit in implementation: each complex coefficient is converted to magnitude weight plus global phase on the corresponding unitary block.
3. The current implementation does not explicitly include a separate oblivious amplitude amplification stage; it focuses on direct truncated-series plus LCU construction.
4. Literature-level query complexity formulas are theoretical references. In this concrete implementation, explicit Pauli expansion and powering costs dominate practical scaling, and effective degree is capped at 15.

## Summary Checklist

1. [ ] Validate Hermitian `H` and manageable matrix size.
2. [ ] Record `lam`, `time_split`, and effective `degree`.
3. [ ] Track LCU term growth and runtime.
4. [ ] Verify results against exact evolution on small systems.
5. [ ] Sweep degree/time to find stable operating region.

## Real-World Applications

1. High-accuracy simulation studies with explicit approximation control.
2. LCU-based algorithm prototyping and verification.
3. Matrix-function experiments in quantum linear algebra contexts.
4. Controlled comparisons with Trotter and QSP methods.
