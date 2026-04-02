---
name: lcu
description: Linear Combination of Unitaries (LCU) - A powerful quantum algorithmic technique for implementing linear transformations by expressing them as a sum of unitary operations. Supports efficient state preparation, controlled unitary execution, and measurement-based post-selection for various applications in quantum algorithms.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# LCU (Linear Combination of Unitaries)

## Algorithm Overview

Linear Combination of Unitaries (LCU) is a key quantum subroutine that allows you to realize a target operator
$A$ through a linear combination of easily implementable unitary operations:

$$A = \sum_{j} \alpha_j U_j$$

where $U_j$ are unitary and $\alpha_j$ are coefficients. LCU forms the core of quantum simulation methods, quantum linear solvers (HHL/QSVT), and matrix functions via block-encoding and quantum signal processing.

### Core steps

1. Prepare an ancilla index register in a superposition that encodes the coefficients $\alpha_j$.
2. Apply a controlled unitary operation $\sum_j |j\rangle\langle j|\otimes U_j$.
3. Uncompute ancilla and perform post-selection on the ancilla state to obtain the resulting target state.
4. Optionally use oblivious amplitude amplification to boost success probability.

### Why use LCU

- Enables implementation of non-unitary operators via unitary resources
- Facilitates quantum simulation of sparse Hamiltonians
- Supports quantum linear algebra primitives such as $A^{-1}$ and $\,e^{-iAt}$
- Combines cleanly with QPE/QSVT/HHL architectures

---

## Mathematical Foundation

Given $A=\sum_j \alpha_j U_j$ with $\sum_j |\alpha_j| \le 1$ (normalized), define state preparation unitary $V$ such that:

$$V|0\rangle = \frac{1}{\sqrt{s}}\sum_j \sqrt{\alpha_j}|j\rangle,\quad s=\sum_j |\alpha_j|$$

Then LCU circuit:

$$W = (V^{\dagger}\otimes I)(\sum_j |j\rangle\langle j|\otimes U_j)(V\otimes I)$$

Applies $A/s$ to target state conditioned on ancilla $|0\rangle$ post-selection.

### Success probability

Post-selecting ancilla in $|0\rangle$ gives amplitude $A/s$. Success probability is $\|A|\psi\rangle\|^2/s^2$. Use amplitude amplification to raise to $O(1)$.

---

## Features

| Feature | Description |
|---------|-------------|
| Block-encoding via ancilla | Encodes matrix by ancilla-prepared superposition |
| Controlled unitary multiplexing | Implements $\sum_j |j\rangle\langle j|\otimes U_j$ |
| Post-selection | Uses ancilla measurement to realize non-unitary operation |
| Amplitude amplification | Optional for success boost |
| Universality | Works for Hamiltonian simulation, inverses, and matrix functions |

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.linear_algebra.lcu import LCUAlgorithm
from engine.core import GateSequence
import numpy as np
```

### Step 1: Define unitaries and coefficients

```python
# Example simple operator A = 0.6 I + 0.4 X
alpha = [0.6, 0.4]
U0 = GateSequence(n_qubits=1, backend='torch')
U0.i(0)
U1 = GateSequence(n_qubits=1, backend='torch')
U1.x(0)
```

### Step 2: Prepare ancilla state for coefficients

```python
# ancilla form of V (prepares superposition of |0>, |1>)
V = GateSequence(n_qubits=1, backend='torch')
# This is a placeholder; actual V depends on sqrt(alpha_j)
V.ry(0, 2*np.arcsin(np.sqrt(alpha[1]/sum(alpha))))
```

### Step 3: Instantiate LCU algorithm

```python
lcu = LCUAlgorithm()
```

### Step 4: Run LCU transform

```python
result = lcu.run(
    state_b=psi,              # input state |psi>
    unitaries=[U0, U1],
    coefficients=alpha,
    prep_unitary=V,
    backend='torch',
    algo_dir='./lcu_results'
)

print('Output state:', result['output_state'])
print('Success probability:', result['success_prob'])
```

### Step 5: Optional amplitude amplification

```python
res2 = lcu.amplitude_amplify(result)
print('Amplified success:', res2['success_prob'])
```

---

## Practical Examples

### Example 1: Hamiltonian simulation with LCU

```python
# Hamiltonian H = 0.5 X + 0.5 Z
dt = 0.2
alpha = [0.5j*dt, 0.5j*dt]  # using taylor expansion coefficients for e^{-iHt}
U0 = GateSequence(n_qubits=1, backend='torch')
U0.x(0)
U1 = GateSequence(n_qubits=1, backend='torch')
U1.z(0)

psi = GateSequence(n_qubits=1, backend='torch')
psi.h(0)

lcu = LCUAlgorithm()
res = lcu.run(state_b=psi, unitaries=[U0, U1], coefficients=alpha, prep_unitary=V, backend='torch')
print(res)
```

### Example 2: Linear system prelude for QSVT/HHL

```python
# A = 0.75 I + 0.25 X, building A^{-1} through LCU then QSVT
res = lcu.run(state_b=psi, unitaries=[U0, U1], coefficients=[0.75,0.25], prep_unitary=V, backend='torch')
print('lcu output', res['output_state'])
```

---

## Performance

- Time complexity: dominated by controlled-$U_j$ resources and state-prep depth.
- Space complexity: $O(n+a)$ qubits, where $a$ is ancilla register size.
- Success probability depends on $s=\sum_j |\alpha_j|$; amplitude amplification can mitigate this.

---

## Integration Notes

- LCU is a foundational building block for Hamiltonian simulation, QSVT, and QPE-based algorithms.
- Use with efficient coefficient extraction modules, sparse matrix utilities, and preconditioning.
- Supports both exact and approximate polynomial series expansion for matrix functions.

---

## References

- Childs, A. M., Kothari, R., & Somma, R. D. (2017). "Quantum linear systems algorithm with exponentially improved dependence on precision."
- Berry, D. W., Childs, A. M., & Kothari, R. (2015). "Hamiltonian simulation with nearly optimal dependence on all parameters."
- Low, G. H., & Chuang, I. L. (2017). "Hamiltonian simulation by qubitization."

---

## License

MIT License - Authored by Yuanchun He