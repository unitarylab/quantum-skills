---
name: hadamard-test
description: Hadamard Test - A fundamental quantum algorithm for estimating the expectation value of a unitary operator. Supports flexible control qubit configurations and efficient state manipulation for quantum algorithm applications.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Hadamard Test

## Algorithm Overview

The Hadamard Test is a key quantum algorithm for measuring the real and imaginary parts of the expectation value
of a unitary operator $U$ on a given quantum state $|\psi\rangle$. This is essential for many algorithms, such as
quantum phase estimation, variational quantum circuits, and quantum machine learning.

### Core idea

1. Add a control (ancilla) qubit initialized in $|0\rangle$.
2. Apply Hadamard to the control qubit.
3. Apply a controlled-$U$ gate with the control qubit controlling the application of $U$ on the target state.
4. Apply Hadamard to the control qubit again.
5. Measure the control qubit.

The measurement probability gives:

- with direct Hadamard: $\operatorname{Re}(\langle\psi | U | \psi\rangle)$
- with phase $S$ gate before final Hadamard: $\operatorname{Im}(\langle\psi | U | \psi\rangle)$

## Why this is useful

- Estimating expectation values in VQE and QAOA
- Implementing controlled-unitary-based subroutines
- Validating operator properties (Hermitian, unitary)
- Constructing stochastic circuits with fewer measurement shots

---

## Algorithm details

### Hadamard test circuit for real part

```
control:  H ---●--- H --- Measure
              |
target: |psi> - U -
```

Probability of measuring control as $|0\rangle$:

$$P(0) = \frac{1}{2}[1 + \operatorname{Re}(\langle\psi|U|\psi\rangle)]$$

Thus:

$$\operatorname{Re}(\langle\psi|U|\psi\rangle) = 2P(0) - 1$$

### Hadamard test circuit for imaginary part

Insert an $S$ gate (phase gate) on control before final Hadamard:

```
control:  H ---●--- S --- H --- Measure
              |
target: |psi> - U -
```

Probability of measuring control as $|0\rangle$:

$$P(0) = \frac{1}{2}[1 - \operatorname{Im}(\langle\psi|U|\psi\rangle)]$$

Thus:

$$\operatorname{Im}(\langle\psi|U|\psi\rangle) = 1 - 2P(0)$$

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.fundamental_algorithm.hadamard_test import HadamardTest
from engine.core import GateSequence
import numpy as np
```

### Step 1: Prepare an input state $|\psi\rangle$

```python
# Example: single qubit |0> state after Hadamard -> |+>
psi = GateSequence(n_qubits=1, backend='torch')
psi.h(0)
```

### Step 2: Define the unitary operator $U$

```python
U = GateSequence(n_qubits=1, backend='torch')
U.rz(0, np.pi/4)  # Example unitary
```

### Step 3: Instantiate HadamardTest

```python
ht = HadamardTest()
```

### Step 4: Estimate real part

```python
result_real = ht.run(
    state_prep=psi,
    unitary=U,
    target_qubits=[0],
    control_qubit=1,
    mode='real',
    shots=1000,
    backend='torch'
)

print('Real part estimate:', result_real['expectation'])
```

### Step 5: Estimate imaginary part

```python
result_imag = ht.run(
    state_prep=psi,
    unitary=U,
    target_qubits=[0],
    control_qubit=1,
    mode='imag',
    shots=1000,
    backend='torch'
)

print('Imag part estimate:', result_imag['expectation'])
```

### Step 6: Compare against exact value

```python
# Analytical expectation value for this example
exact = np.exp(1j * np.pi/4) * 0.5 + np.exp(-1j * np.pi/4) * 0.5
print('Exact:', complex(exact))
```

---

## Practical examples

### Example 1: identity unitary

```python
U_id = GateSequence(n_qubits=1, backend='torch')
# identity gate is no-op

res_real = ht.run(state_prep=psi, unitary=U_id, target_qubits=[0], control_qubit=1, mode='real')
res_imag = ht.run(state_prep=psi, unitary=U_id, target_qubits=[0], control_qubit=1, mode='imag')

print('Identity real:', res_real['expectation'])  # ~1.0
print('Identity imag:', res_imag['expectation'])  # ~0.0
```

### Example 2: Pauli-Z operator

```python
U_z = GateSequence(n_qubits=1, backend='torch')
U_z.z(0)

res_real = ht.run(state_prep=psi, unitary=U_z, target_qubits=[0], control_qubit=1, mode='real')
print('Z real (|+> basis):', res_real['expectation'])  # ~0.0
```

### Example 3: complex gate

```python
U_complex = GateSequence(n_qubits=1, backend='torch')
U_complex.rz(0, np.pi/3)

res_real = ht.run(state_prep=psi, unitary=U_complex, target_qubits=[0], control_qubit=1, mode='real')
res_imag = ht.run(state_prep=psi, unitary=U_complex, target_qubits=[0], control_qubit=1, mode='imag')
print('complex real:', res_real['expectation'])
print('complex imag:', res_imag['expectation'])
```

---

## Performance

- Measurement counts (`shots`) affects statistical error.
- A standard error scales as $1/\sqrt{N_{shots}}$.
- Mini-batch and averaging are effective for noisy simulators.

---

## References

- A. Kitaev, "Quantum Measurements and the Abelian Stabilizer Problem", 1995.
- M. Nielsen & I. Chuang, "Quantum Computation and Quantum Information", 2000.

---

## License

MIT License - Authored by Yuanchun He