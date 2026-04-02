---
name: hadamard-transform
description: Hadamard Transform - A fundamental quantum algorithm for creating uniform superposition states. Supports multi-qubit Hadamard layers and efficient state manipulation for quantum algorithm applications.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Hadamard Transform

## Algorithm Overview

The Hadamard Transform (HT) is a basic yet vital quantum operation that converts computational basis states into uniform superposition states using single-qubit Hadamard gates. It forms the basis of many quantum algorithms, including Quantum Fourier Transform (QFT), Grover's algorithm, and various quantum sampling and optimization protocols.

### Key properties

- Converts $|0...0\rangle$ to uniform superposition $\frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$
- Applies Hadamard gate $H$ to every qubit in an $n$-qubit register
- Reversible and unitary with $H^2 = I$
- Supports parallel implementation for high-performance simulation

### Primary use cases

- Initial state preparation in quantum algorithms
- Phase estimation and QPE initialization
- Grover’s diffusion operator construction
- Quantum machine learning feature maps

---

## Mathematical Description

For a single qubit, Hadamard gate is:

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1 \\ 1 & -1\end{pmatrix}$$

On an $n$-qubit register, Hadamard transform is:

$$H^{\otimes n} |x\rangle = \frac{1}{\sqrt{2^n}} \sum_{y=0}^{2^n-1} (-1)^{x \cdot y} |y\rangle$$

where $x \cdot y$ denotes bitwise dot product modulo 2.

### In action

- $H^{\otimes n} |0...0\rangle = \frac{1}{\sqrt{2^n}}\sum_{x=0}^{2^n-1}|x\rangle$
- $H^{\otimes n} |x\rangle$: creates equal-amplitude state with phase pattern from $(-1)^{x\cdot y}$

---

## Features

| Feature | Description |
|--------|-------------|
| Uniform superposition | Prepare $2^n$ basis states with equal amplitude |
| Reversible | $H^{\otimes n}$ is unitary, inverse is itself | 
| Simplest quantum layer | Single-qubit gates, perfect for hardware or simulators |
| Supports controlled operations | Can be used to build larger controlled-H structures |
| Integration | Foundational module for Grover, QFT, etc. |

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.fundamental_algorithm.hadamard_transform import HadamardTransform
from engine.core import GateSequence
import numpy as np
```

### Step 1: Instantiate HadamardTransform

```python
ht = HadamardTransform()
```

### Step 2: Create a gateway circuit

```python
n_qubits = 3
circuit = GateSequence(n_qubits=n_qubits, backend='torch')
```

### Step 3: Apply Hadamard transform

```python
circuit = ht.apply(circuit, target_qubits=list(range(n_qubits)))
```

if the module provides direct function:

```python
circuit.hadamard_all()  # if available
```

### Step 4: Execute and inspect state

```python
result = circuit.execute()
print('Final state vector:', result)
```

### Step 5: Validate uniform amplitudes

```python
state = result.reshape(-1)
amplitudes = np.abs(state) ** 2
print('Probability uniformity:', np.allclose(amplitudes, 1/2**n_qubits))
```

---

## Practical Examples

### Example 1: Basic 2-qubit Hadamard

```python
from engine.algorithms.fundamental_algorithm.hadamard_transform import HadamardTransform
from engine.core import GateSequence

ht = HadamardTransform()
qs = GateSequence(n_qubits=2, backend='torch')
qs = ht.apply(qs, target_qubits=[0, 1])
state = qs.execute()
print(state)  # should be [0.5, 0.5, 0.5, 0.5] in magnitude squares
```

### Example 2: Hadamard + Phase Oracle (Grover initialization)

```python
ht = HadamardTransform()
qs = GateSequence(n_qubits=3, backend='torch')
qs = ht.apply(qs, target_qubits=[0,1,2])
# apply oracle with controlled phase on |111>
qs.phase_oracle([0,1,2], True) # pseudo-code
qs = ht.apply(qs, target_qubits=[0,1,2])
```

### Example 3: Repeated Hadamard (self-inverse)

```python
qs = GateSequence(n_qubits=3, backend='torch')
qs = ht.apply(qs, [0,1,2])
qs = ht.apply(qs, [0,1,2])
final_state = qs.execute()
# should return |000>
```

---

## Performance

- Time complexity: $O(n 2^n)$ for full state vector simulation (for generating amplitudes), but local gate application cost is $O(n)$ per step.
- Space complexity: $O(2^n)$ for state vector.
- Perfectly parallelizable via vectorized matrix transformations.

---

## Integration Notes

- Often used as first layer in quantum circuits.
- Combine with controlled phase oracles for Grover's diffusion.
- Can be reused in amplitude amplification and QPE pipelines.

---

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*.
- Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search".

---

## License

MIT License - Authored by Yuanchun He