---
name: qsp
description: Quantum Signal Processing (QSP) - a powerful framework for implementing polynomial transformations of quantum singular values via controlled phase rotations and block-encoded operators. Supports eigenvalue filtering, matrix function evaluation, and integration with QSVT/HHL.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Quantum Signal Processing (QSP)

## Algorithm Overview

Quantum Signal Processing (QSP) is a key component in modern quantum algorithms for applying polynomial functions to singular values or eigenvalues of operators using a sequence of phase-encoded unitaries. It forms the basis for Quantum Singular Value Transformation (QSVT), Hamiltonian simulation, and quantum linear system solvers.

### Key concept

QSP realizes a target polynomial $P(x)$ on singular values via a product of alternating operators:

$$U_{\phi}(x) = e^{i\phi_0 Z} e^{i \theta(x) X} e^{i\phi_1 Z} ... e^{i \theta(x) X} e^{i\phi_d Z}$$

where $\phi_j$ are phase angles and $\theta(x)$ encodes input parameter (e.g. through controlled operator application). Output is a transformed matrix function that can approximate $f(A)$ for block-encoded matrix $A$.

### Why QSP is useful

- Implements arbitrary real polynomials on an operator spectrum
- Enables QSVT, QAE, QLSA, and phase estimation with near-optimal complexity
- Avoids deep controlled-U exponentiation overhead when optimized
- Supports robust approximation with low-degree polynomials

---

## Mathematical Foundation

Given a base block-encoded unitary $U$ with singular values $\sigma \in [0,1]$ and associated parameter $x=\sigma$, QSP finds phase sequence $\vec{\phi}$ such that:

$$P(x) \approx \Re\{\langle0|\Phi(x)|0\rangle\}$$

with

$$\Phi(x) = e^{i \phi_0 Z}W(x)e^{i\phi_1 Z}W(x)...e^{i\phi_d Z}$$

and $W(x)=e^{i\arccos(x) X}$. The resulting polynomial is even or odd depending on parity.

### Real polynomial synthesis

- Target polynomial must satisfy parity constraint for implementability.
- Phase angles computed numerically (e.g. via quantum signal processing solvers).
- Degree $d$ determines precision $\epsilon=O(1/2^d)$ in approximating $f(x)$.

---

## Features

| Feature | Description |
|---------|-------------|
| Polynomial transform | Realize $P(x)$ for singular values or eigenvalues |
| Low-degree implementation | Achieve high accuracy with relatively small $d$ |
| Block-encoding integration | Directly combine with QSVT and LCU primitives |
| Efficient phase control | Uses single-qubit phase gates + controlled block-encoding |
| Applicability | Hamiltonian simulation, QLSA, eigenvalue filtering |

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.linear_algebra.qsp import QSPAlgorithm
from engine.core import GateSequence
import numpy as np
```

### Step 1: Define block-encoded operator and input state

```python
# A simple block-encoding placeholder for demonstration
A = np.array([[0.5, 0.2], [0.2, 0.5]])
U = GateSequence(n_qubits=2, backend='torch')
# ... construct block encoding of A in U ...

# input state |b>
psi = GateSequence(n_qubits=2, backend='torch')
psi.h(0); psi.h(1)
```

### Step 2: Set target polynomial function

```python
# For example f(x)=1/x approximated over x∈[1/κ,1]
kappa = 5.0
epsilon = 1e-3

target_fn = lambda x: 1.0/x
```

### Step 3: Instantiate QSP algorithm

```python
qsp = QSPAlgorithm()
```

### Step 4: Compute QSP phases and polynomial degree

```python
# The API may provide helper for phase extraction and coefficient solver
phases, degree = qsp.compute_phases(target_fn, kappa, epsilon)
print('Computed phases', phases)
```

### Step 5: Run QSP circuit

```python
results = qsp.run(
    block_encoding=U,
    input_state=psi,
    phases=phases,
    degree=degree,
    backend='torch',
    algo_dir='./qsp_results'
)

print('Output state', results['output_state'])
print('Success probability', results['success_prob'])
```

---

## Practical Examples

### Example 1: Polynomial approximation of inverse

```python
# Reuse A, psi from above
phases, degree = qsp.compute_phases(lambda x: 1/x, kappa=4.0, epsilon=1e-3)
res = qsp.run(block_encoding=U, input_state=psi, phases=phases, degree=degree, backend='torch')
print('Approx inverse output', res['output_state'])
```

### Example 2: Bounded filter function

```python
phases, degree = qsp.compute_phases(lambda x: x**2, kappa=5.0, epsilon=1e-4)
res = qsp.run(block_encoding=U, input_state=psi, phases=phases, degree=degree)
print('Filter output', res['output_state'])
```

---

## Performance

- Complexity depends on polynomial degree $d$ and number of block-encoding calls.
- Degree scaling: $d = O(\kappa \log(1/\epsilon))$ for efficient QSVT-based inversion.
- Space complexity: $O(n+a)$ qubits with ancilla for block encoding.
- Accurate phase computations are critical; numerical methods may be used.

---

## Integration notes

- QSP is the low-level mechanism for QSVT; combine with QSVTQLSA, HHL, and amplitude amplification modules.
- Use with spectral gap/condition number preconditioning for stable behavior.
- Supports both sparse and dense operator pre-processing.

---

## References

- Gilyén, A., Su, Y., Low, G. H., & Wiebe, N. (2019). "Quantum singular value transformation and beyond." 
- Low, G. H., & Chuang, I. L. (2017). "Optimal Hamiltonian simulation by quantum signal processing."
- Childs, A. M., Kothari, R., & Somma, R. D. (2017). "Quantum linear systems algorithm with exponentially improved dependence on precision."

---

## License

MIT License - Authored by Yuanchun He