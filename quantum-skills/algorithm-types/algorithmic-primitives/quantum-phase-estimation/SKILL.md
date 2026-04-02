---
name: quantum-phase-estimation
description: Quantum Phase Estimation (QPE) - A fundamental quantum algorithm to estimate eigenvalues of unitary operators with high precision. Supports controlled-unitary synthesis, inverse QFT, and phase-to-amplitude estimation workflows.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Quantum Phase Estimation (QPE)

## Algorithm Overview

Quantum Phase Estimation is a central algorithm in quantum computing for estimating the eigenvalues of a unitary operator $U$. It is used in many quantum algorithms such as Shor's factoring, quantum simulation, and eigenvalue problems (HHL, QAE).

QPE maps an eigenstate $|\psi\rangle$ with eigenvalue $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$ into a register that encodes $\phi$ as a binary fraction.

### Key benefits

- Extracts eigenphases with precision $2^{-d}$ using $d$ phase qubits.
- Offers a way to transform phase estimation into amplitude estimation.
- Provides exact results when $\phi$ is representable in $d$ bits.
- Works with any unitary operator and its controlled powers.

### High-level circuit

1. Prepare a phase register of $d$ qubits in $|0\rangle^{\otimes d}$.
2. Prepare a target register in eigenstate $|\psi\rangle$.
3. Apply Hadamard to all phase qubits.
4. Apply controlled-$U^{2^k}$ gates from each phase qubit.
5. Apply inverse QFT on phase register.
6. Measure phase register to obtain binary estimate of $\phi$.

---

## Mathematical foundation

Given $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$ (0≤φ<1), QPE:

- Creates state: $\frac{1}{2^{d/2}}\sum_{k=0}^{2^{d}-1} |k\rangle U^{k}|\psi\rangle$
- On eigenstate reduces to: $\frac{1}{2^{d/2}}\sum_{k=0}^{2^{d}-1} e^{2\pi i k\phi}|k\rangle|\psi\rangle$
- After $\mathrm{iQFT}$ on phase register, measurement yields $\tilde{\phi}=m/2^{d}$.

Precision: if φ has exact d-bit expansion, result is exact; otherwise the nearest integer rounding error ≤$1/2^{d+1}$.

---

## Features

| Feature | Description |
|---------|-------------|
| Controlled unitary powers | $U^{2^{k}}$ for k=0..d-1 |
| Inverse QFT | Efficient phase-to-bit conversion |
| Parameterized precision | Tune d for required accuracy |
| Eigenvalue estimation | Essential in Shor, HHL, QAE |
| Integration | Basis for many high-level algorithms |

---

## Usage guide

### Prerequisites

```python
from engine.algorithms.fundamental_algorithm.qpe import QPEAlgorithm
from engine.core import GateSequence
import numpy as np
```

### Step 1: Define target unitary U and eigenstate |ψ⟩

```python
# Example: U = Rz(2πθ) on one qubit, eigenstate |+>
theta = 0.125  # desired phase: 0.125
U = GateSequence(n_qubits=1, backend='torch')
U.rz(0, 2*np.pi*theta)

psi = GateSequence(n_qubits=1, backend='torch')
psi.h(0)  # |+> is eigenstate of Rz
```

### Step 2: Instantiate QPE algorithm

```python
qpe = QPEAlgorithm()
```

### Step 3: Run with precision d

```python
result = qpe.run(
    U=U,
    eigenstate=psi,
    d=6,
    backend='torch',
    algo_dir='./qpe_results'
)

print('Estimated phase:', result['phase'])
print('Estimated value:', float(result['phase'])*1.0)
```

### Step 4: Analyze results

```python
exact = theta
estimate = result['phase']
error = abs(exact - estimate)
print(f'Exact φ={exact}, estimate={estimate}, error={error:.6f}')
```

---

## Practical examples

### Example 1: single-qubit rotation eigenphase

```python
# phase 1/8, d=5 gives exact 0.125

theta = 1/8
U = GateSequence(n_qubits=1, backend='torch')
U.rz(0, 2*np.pi*theta)

psi = GateSequence(n_qubits=1, backend='torch')
psi.h(0)

qpe = QPEAlgorithm()
res = qpe.run(U=U, eigenstate=psi, d=5, backend='torch')
print(res)
```

### Example 2: controlled-OP with unknown eigenstate

```python
# If psi is not exact eigenstate, result is distribution over possible values.
psi_state = GateSequence(n_qubits=1, backend='torch')
psi_state.x(0)  # |1>

# U as phase rotation for |+>
U = GateSequence(n_qubits=1, backend='torch')
U.rz(0, 2*np.pi*0.2)

res = qpe.run(U=U, eigenstate=psi_state, d=6, backend='torch')
print(res)
```

---

## Performance

- Time complexity: $O(d \cdot \text{cost}(U)) + O(d^2)$ for iQFT
- Space complexity: $O(2^{d+n})$ state-vector in simulator
- Accuracy controlled by d; doubling d halves phase error

---

## Integration notes

- Use with Amplitude Estimation (QAE) for probability estimation.
- Use estimated eigenphase in Hamiltonian simulation and energy spectra.
- Combine with QFT / iQFT modules from `hadamard_transform` or existing implementation.

---

## References

- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*.
- Shor, P. (1994). "Algorithms for quantum computation: discrete logarithms and factoring." 

---

## License

MIT License - Authored by Yuanchun He: quantum-phase-estimation