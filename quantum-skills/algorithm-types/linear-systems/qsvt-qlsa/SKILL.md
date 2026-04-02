---
name: qsvt-qlsa
description: Quantum Singular Value Transformation (QSVT) for Quantum Linear System Algorithm (QLSA) - a high-performance quantum method to solve linear systems using singular value polynomial transformation and block-encoding. Includes workflow, setup, and example usage.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# QSVT-QLSA (Quantum Linear System via Quantum Singular Value Transformation)

## Algorithm Overview

Quantum Singular Value Transformation (QSVT) is a versatile framework for applying polynomial functions to the singular values of matrices represented by block-encoded unitary operators. QSVT-QLSA leverages this to implement efficient quantum linear solvers, improving upon HHL by reducing condition number dependence and using polynomial spectral filtering.

### Core ideas

- Encode matrix $A$ as a block of a unitary operator $U_A$ (block-encoding).
- Prepare input state $|b\rangle$ in a register.
- Use QSVT to apply polynomial $P(x)$ approximating $1/x$ on the singular values of $A$.
- Post-select and uncompute to obtain output state approximating $|x\rangle = A^{-1}|b\rangle$.
- The polynomial degree is set by desired precision and condition number $\kappa$.

### Advantages

- Optimizes polynomial approximations to achieve near-optimal query complexity.
- Can reduce dependence to $O(\kappa\text{ polylog}(\kappa/\epsilon))$ in some methods.
- Avoids explicit QPE; better suited for NISQ-compatible or block-encoding based architectures.

---

## Mathematical Foundation

### Block-encoding of A

A matrix $A$ with $\|A\|\le 1$ has a unitary block-encoding $U_A$ such that:

$$U_A = \begin{pmatrix}A & * \\ * & *\end{pmatrix}$$

on $a+n$ qubits, with $a$ ancillas. When $|0^a\rangle$ is ancilla state:

$$\langle0^a| U_A |0^a\rangle = A$$

### Singular value transformation

Given SVD $A=\sum_j \sigma_j |u_j\rangle\langle v_j|$, QSVT implements:

$$P(A) = \sum_j P(\sigma_j) |u_j\rangle\langle v_j|$$

For linear systems, choose polynomial $P(x) \approx 1/x$ over domain $[1/\kappa,1]$.

### State evolution

Apply to input

$$|b\rangle = \sum_j \beta_j |v_j\rangle$$

and obtain:

$$P(A)|b\rangle \approx \sum_j \beta_j \frac{1}{\sigma_j} |u_j\rangle$$

Normalized output approximates $|x\rangle = A^{-1}|b\rangle / \|A^{-1}|b\rangle\|$.

---

## Features

| Feature | Description |
|---------|-------------|
| Block-encoding | Encodes A in a unitary operator on ancilla + data qubits |
| Polynomial transform | Applies target polynomial to singular values |
| Precision control | Variable polynomial degree d for accuracy \epsilon |
| Condition number | Handles well-conditioned matrices with efficient scaling |
| Integration | Works with QPE, QAE, HHL, quantum optimization workflows |

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.linear_algebra.qsvt_qlsa import QSVTQLSA
from engine.core import GateSequence
import numpy as np
```

### Step 1: Prepare block-encoding for A

```python
# Example matrix A with ||A|| <=1 (sparse / normalized)
A = np.array([[0.5, 0.2], [0.2, 0.5]])

# Build block-encoding circuit or use helper method in framework
ua = GateSequence(n_qubits=3, backend='torch')
# placeholder: ua = build_block_encoding(A, ...)
```

### Step 2: Prepare input state b

```python
psi = GateSequence(n_qubits=2, backend='torch')
psi.h(0); psi.h(1)  # |++> example
```

### Step 3: Instantiate QSVT-QLSA

```python
qsvt = QSVTQLSA()
```

### Step 4: Define polynomial and parameters

```python
# For 1/x approximation on [1/kappa,1]
kappa = 5.0
epsilon = 1e-3

# The algorithm may provide helper to compute polynomial coefficients:
# coeffs = qsvt.compute_inverse_poly(kappa, epsilon)
```

### Step 5: Run algorithm

```python
result = qsvt.run(
    block_encoding=ua,
    state_b=psi,
    kappa=kappa,
    epsilon=epsilon,
    d=80,  # polynomial degree
    backend='torch',
    algo_dir='./qsvt_qlsa_results'
)

print('Output solution state:', result['output_state'])
print('Success probability:', result['success_prob'])
print('Polynomial degree:', result['degree'])
```

### Step 6: Extract observables

```python
# for example expectation of Pauli Z on first qubit
print('Z expectation:', result['observable']['Z0'])
```

---

## Practical Examples

### Example 1: small matrix solve

```python
A = np.array([[0.5, 0.2], [0.2, 0.5]])
psi = GateSequence(n_qubits=2, backend='torch')
psi.h(0); psi.h(1)

qsvt = QSVTQLSA()
res = qsvt.run(block_encoding=ua, state_b=psi, kappa=5.0, epsilon=1e-2, d=30, backend='torch')
print(res)
```

### Example 2: analyzing condition number

```python
for kappa in [2.0, 5.0, 10.0]:
    res = qsvt.run(block_encoding=ua, state_b=psi, kappa=kappa, epsilon=1e-3, d=50, backend='torch')
    print('kappa', kappa, 'success', res['success_prob'])
```

---

## Performance

- Complexity: polynomial degree $d = O(\kappa \, \text{polylog}(\kappa/\epsilon))$ for optimal schemes.
- Space: $O(n + a)$ qubits with $a$ ancillas for block encoding.
- Quantum runtime dominated by block-encoding calls and controlled rotations in QSVT sequence.
- Post-selection and amplitude estimation determine final success probability.

---

## Integration Notes

- Use with QAE and quantum preconditioning modules.
- QSVT can implement HHL-like inversion and arbitrary spectral filtering.
- Works as a building block in quantum machine learning and differential equation solvers.

---

## References

- Gilyén, A., Su, Y., Low, G. H., & Wiebe, N. (2019). "Quantum singular value transformation and beyond." 
- Childs, A. M., Kothari, R., & Somma, R. D. (2017). "Quantum linear systems algorithm with exponentially improved dependence on precision." 
- Harrow, A. W., Hassidim, A., & Lloyd, S. (2009). "Quantum algorithm for linear systems of equations." 

---

## License

MIT License - Authored by Yuanchun He