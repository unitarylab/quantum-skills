---
name: qsp
description: "Use when: you need block-encoding based Hamiltonian simulation with Quantum Signal Processing phases, parity-aware polynomial construction, and implementation-level guidance for this repository."
---

# Quantum Signal Processing (QSP) Hamiltonian Simulation Skill Guide

## Overview

QSP simulates

$$
U(t)=e^{-iHt}
$$

by transforming a block-encoding of $H$ through carefully designed polynomial phase sequences.

### Key Insight

Instead of approximating $e^{-iHt}$ directly at matrix level, QSP approximates scalar target functions (cosine and sine components) over the normalized spectrum and then lifts them to matrix functions through block-encoding and phase-factor circuits.

### Why QSP Matters:

1. It is one of the most powerful polynomial-transformation frameworks in quantum algorithms.
2. It gives high-accuracy approximations with principled phase synthesis.
3. It integrates naturally with block-encoding-based algorithm stacks.
4. It is foundational for modern fault-tolerant simulation strategies.

### Real Applications:

1. High-precision Hamiltonian simulation for quantum chemistry.
2. Polynomial matrix-function transforms in linear algebra workflows.
3. Building blocks for singular-value transformation pipelines.
4. Advanced algorithm prototyping in block-encoding ecosystems.

## Learning Objectives

After using this skill, you should be able to:
1. Explain how block-encoding feeds into QSP.
2. Understand parity-separated construction for cosine and sine parts.
3. Use `QSP(H, t, target_error, degree, beta)` safely.
4. Interpret the final LCU-style combination used in this implementation.
5. Extract and compare matrix-level approximation error.

## Prerequisites

### Essential knowledge:

1. Block-encoding concept for Hermitian matrices.
2. Controlled operations and ancilla management.
3. Chebyshev-polynomial approximation basics.

### Mathematical comfort:

1. Spectral mapping from matrix eigenvalues to scalar polynomial approximation.
2. Bessel-function based coefficient construction.
3. Complex phase manipulation in circuit synthesis.

### Recommended:

1. Understand `block_encode` and `qsp_solver` interfaces at a high level.
2. Start with low-dimensional Hermitian matrices for debugging.
3. Keep `beta` in a moderate range and inspect numerical stability.

## Using the Provided Implementation

### Quick Start Example

```python
import numpy as np
from engine.algorithms.qsp import QSP

H = np.array([
    [0.8, 0.2],
    [0.2, -0.8]
], dtype=complex)

sim = QSP(
    H=H,
    t=1.2,
    target_error=1e-4,
    degree=15,
    beta=0.7,
)

print("method:", sim.method)
print("degree:", sim.degree)
print("alpha:", sim.alpha)
print("beta:", sim.beta)
print("error:", sim.total_error)
```

### Core Parameters Explained

```python
class QSP(HamiltonianSimulationResult):
    def __init__(
        self,
        H: np.ndarray,
        t: float,
        target_error: float,
        degree: int = 15,
        beta: float = 0.7
    ) -> None:
        ...
```

Parameter meaning:
1. `H`: Hermitian Hamiltonian.
2. `t`: evolution time.
3. `target_error`: target approximation level used in degree estimation.
4. `degree`: initial degree hint (implementation currently caps final degree at 15).
5. `beta`: scaling factor for bounded target functions, must satisfy `0 < beta < 1`.

Relevant constructor logic from `algorithms/qsp/qsp.py`:

```python
encoded_H = block_encode(H)
self.UH = encoded_H.circuit
self.alpha = encoded_H.alpha
self.beta = beta
self.m = encoded_H.total_qubits
self.degree = min(max(degree, int(np.ceil(self.alpha * t * 1.5 + np.log(1/self.target_error) * 1.5))), 15)
```

Return dictionary contains:

This constructor returns an object. A practical logging dictionary is:

```python
result = {
    "method": sim.method,
    "target_qubits": sim.target_qubits,
    "degree": sim.degree,
    "alpha": sim.alpha,
    "beta": sim.beta,
    "m": sim.m,
    "circuit": sim.circuit,
    "evolution_result": sim.evolution_result,
    "total_error": sim.total_error,
}
```

## Understanding the Core Components

### 1) Block-encoding initialization and safety checks

Code excerpt:

```python
encoded_H = block_encode(H)
self.UH = encoded_H.circuit
self.alpha = encoded_H.alpha
self.beta = beta
self.m = encoded_H.total_qubits
...
if beta <= 0 or beta >= 1:
    raise ValueError("beta must be between 0 and 1, strictly speaking (0, 1)!")
```

Interpretation:
1. `block_encode` produces a circuit encoding scaled Hamiltonian.
2. `alpha` controls normalization from matrix space to polynomial domain.
3. `beta` enforces bounded target magnitude for stable phase solving.

### 2) Parity-aware QSP polynomial block encoding

Code excerpt from `_qsp_block_encoding`:

```python
if parity == 0:
    coef[1::2] = [0] * (len(coef[1::2]))
else:
    coef[::2] = [0] * (len(coef[::2]))

if is_coef_cheby:
    coef_cheby = coef
else:
    coef_cheby = poly2cheb(coef)

coef_cheby = coef_cheby[parity::2]
phi_proc, out = qsp_solver(coef_cheby, parity, opts)
```

Interpretation:
1. Even and odd target components are handled separately.
2. Coefficients are converted to Chebyshev basis when needed.
3. `qsp_solver` provides phase factors used to compile the circuit.

### 3) Phase sequence circuit construction

Code excerpt from `_u_phi`:

```python
gs = GateSequence(n + m + 1)
gs.h(n + m)

if parity == 0:
    for i in range(int(d / 2), 0, -1):
        gs.append(u, list(range(n + m)))
        gs.append(self._gate_z_pi(m, phi[2 * i - 1]), list(range(n, n + m + 1)))
        gs.append(u.dagger(), list(range(n + m)))
        gs.append(self._gate_z_pi(m, phi[2 * i - 2]), list(range(n, n + m + 1)))
else:
    ...

gs.h(n + m)
```

Interpretation:
1. Alternation of `U` and `U^dagger` with phase gadgets realizes polynomial transformation.
2. Parity determines sequence shape.
3. A dedicated phase qubit is used with initial/final Hadamard.

### 4) Cos/Sin split and final combination

Code excerpt from `_run`:

```python
parity = 0
coef = np.zeros(d + 1)
coef[0] = jn(0, t) * beta
...
cos_Ht = self._qsp_block_encoding(True, coef, parity, opts, self.UH, n, m)

parity = 1
coef = np.zeros(d + 1)
...
sin_Ht = self._qsp_block_encoding(True, coef, parity, opts, self.UH, n, m)

qc = GateSequence(n + m + 2)
qc.h(n + m + 1)
qc.s(n + m + 1)
qc.z(n + m + 1)
qc.append(cos_Ht, list(range(n + m + 1)), [n + m + 1], [0])
qc.append(sin_Ht, list(range(n + m + 1)), [n + m + 1], [1])
qc.h(n + m + 1)

U_approx = self._circuit.get_matrix()[:len(self.H), :len(self.H)] * factor
```

Interpretation:
1. The implementation separately approximates cosine and sine polynomial blocks.
2. It then combines them through a selection qubit pattern.
3. Final matrix block is rescaled by `factor = 2 / beta`.

### 5) External package function notes (brief)

1. `block_encode` returns circuit and scaling metadata.
2. `qsp_solver` computes feasible phase factors for target polynomial.
3. `jn` from SciPy is used to build coefficient sequences.

## Hands-On Example: Hamiltonian Simulation

Sweep beta and target error to inspect behavior.

```python
import numpy as np
from engine.algorithms.qsp import QSP

H = np.array([
    [1.0, 0.15],
    [0.15, -1.0]
], dtype=complex)

betas = [0.6, 0.7, 0.8]
eps_list = [1e-2, 1e-3, 1e-4]

for beta in betas:
    for eps in eps_list:
        sim = QSP(H=H, t=1.0, target_error=eps, degree=15, beta=beta)
        print(f"beta={beta:.1f}, eps={eps:.0e}, degree={sim.degree}, err={sim.total_error:.3e}")
```

What to look for:
1. Stability and error trends under different beta choices.
2. Interaction between target error and effective polynomial requirements.

## Mathematical Deep Dive

Suppose block-encoding provides

$$
\langle 0^m|\,U_H\,|0^m\rangle = H/\alpha
$$

For scalar $x\in[-1,1]$, QSP uses a phase sequence

$$
\Phi=(\phi_0,\phi_1,\dots,\phi_d)
$$

to construct

$$
U_\Phi(x)=e^{i\phi_0 Z}\prod_{k=1}^{d}W(x)e^{i\phi_k Z}
=
\begin{pmatrix}
P(x) & i\sqrt{1-x^2}Q(x)\\
i\sqrt{1-x^2}Q^*(x) & P^*(x)
\end{pmatrix}.
$$

with

$$
W(x)=e^{i\arccos(x)X}=xI+i\sqrt{1-x^2}X
$$

and

$$
R_z(\phi)=e^{i\phi Z}=\cos(\phi)I+i\sin(\phi)Z.
$$

The bounded polynomial relation is

$$
|P(x)|^2 + (1-x^2)|Q(x)|^2 = 1.
$$

Then with $x \in [-1,1]$ representing normalized spectrum, QSP builds polynomial approximations to:

$$
\beta \cos(\alpha t x), \quad \beta \sin(\alpha t x)
$$

with parity separation:
1. Cosine uses even terms.
2. Sine uses odd terms.

Implementation then combines both blocks to approximate:

$$
e^{-iHt} = \cos(Ht) - i\sin(Ht)
$$

and rescales by the factor used in the code.

A standard asymptotic query bound in the QSP Hamiltonian-simulation literature is

$$
O\left(t + \frac{\log(1/\epsilon)}{\log\log(1/\epsilon)}\right),
$$

which motivates the polynomial-approximation framework.

Practical implications:
1. Good block-encoding quality and phase solving are both critical.
2. Parameter choices (`beta`, target error) strongly affect numerical robustness.

Implementation-consistent notes:
1. The implementation follows the bounded-polynomial QSP route and parity split (even cosine / odd sine), with explicit safety constraint `0 < beta < 1`.
2. The practical pipeline here is block-encoding plus numerical phase synthesis (`block_encode` + `qsp_solver`) and circuit assembly, rather than direct eigendecomposition of $H$.
3. Literature-level asymptotic complexity guarantees are theoretical references. This concrete implementation uses fixed solver options and a degree cap (`<= 15`), so performance/accuracy should be validated empirically through `total_error` sweeps.

## Summary Checklist

1. [ ] Confirm `H` is Hermitian and dimensions are correct.
2. [ ] Choose `beta` strictly inside `(0,1)`.
3. [ ] Record `alpha`, `degree`, and final error for every run.
4. [ ] Validate against exact evolution on small systems.
5. [ ] Keep a fallback parameter set for stable phase-solving behavior.

## Real-World Applications

1. Precision-oriented simulation workflows.
2. Block-encoding based quantum linear algebra primitives.
3. Long-term pathway toward fault-tolerant simulation stacks.
4. Research prototypes using QSP and related polynomial frameworks.
