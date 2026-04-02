---
name: hhl
description: HHL (Harrow-Hassidim-Lloyd) algorithm for solving linear systems using quantum computing. Includes eigenvalue estimation, controlled rotation, and output state reconstruction with application notes and usage workflow.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# HHL Algorithm (Quantum Linear System Solver)

## Algorithm Overview

The HHL algorithm is a pioneering quantum algorithm for solving linear systems of equations of the form:

$$A \mathbf{x} = \mathbf{b}$$

where $A$ is a Hermitian, sparse, and well-conditioned matrix, and $\mathbf{b}$ is a known vector. HHL returns a quantum state proportional to $|x\rangle$ that encodes the solution vector $\mathbf{x} = A^{-1}\mathbf{b}$, achieving exponential speedup for certain problem classes compared to classical solvers.

### Core steps

1. Prepare input state $|b\rangle$.
2. Perform Hamiltonian simulation to implement $e^{iAt}$.
3. Use Quantum Phase Estimation (QPE) to estimate eigenvalues $\lambda$ of $A$.
4. Apply conditional rotation mapping $|\lambda\rangle\to|1/\lambda\rangle$.
5. Uncompute QPE register and post-select on ancilla measurement.
6. Output state approximates $|x\rangle \propto A^{-1}|b\rangle$.

### Requirements and assumptions

- $A$ must be Hermitian or embedded in a Hermitian block, i.e., $A = A^{\dagger}$.
- $A$ should be sparse and efficiently simulatable.
- Condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ should be manageable.
- Input vector $|b\rangle$ must be prepareable in polylog time.
- Output is a quantum state. Extracting full $\mathbf{x}$ requires tomography or application-specific observables.

---

## Mathematical Foundation

Given eigen-decomposition $A|u_j\rangle = \lambda_j|u_j\rangle$, express $|b\rangle = \sum_j \beta_j |u_j\rangle$.

After phase estimation, state becomes:

$$\sum_j \beta_j |u_j\rangle |\tilde\lambda_j\rangle$$

Conditional rotation on ancilla adds amplitude proportional to $1/\lambda_j$:

$$\sum_j \beta_j |u_j\rangle |\tilde\lambda_j\rangle \left(\sqrt{1-\frac{C^2}{\tilde\lambda_j^2}}|0\rangle + \frac{C}{\tilde\lambda_j}|1\rangle\right)$$

Uncompute eigenvalue register and measure ancilla in $|1\rangle$ yields:

$$|x\rangle \approx \frac{1}{\|A^{-1}|b\rangle\|} \sum_j \frac{\beta_j}{\lambda_j} |u_j\rangle$$

Success probability is $\Omega(1/\kappa^2)$; repeated runs or amplitude amplification can boost it.

---

## Features

| Feature | Description |
|---------|-------------|
| Quantum linear system solver | Solves $A x = b$ in quantum state form |
| Phase estimation | QPE extracts eigenvalues of matrix $A$ |
| Controlled rotation | Implements $1/\lambda$ transform in ancilla amplitude |
| Post-selection | Conditioned on ancilla $|1\rangle$ for inversion outcome |
| Integration | Works with HHL-based algorithms, QAE, and quantum ML |

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.linear_algebra.hhl import HHLAlgorithm
from engine.core import GateSequence
import numpy as np
```

### Step 1: Define matrix $A$ and prepare $|b\rangle$

```python
# Example: 2x2 Hermitian matrix
A = np.array([[1.0, 0.5], [0.5, 1.0]])

# Prepare |b> = |0> + |1> normalized
psi = GateSequence(n_qubits=1, backend='torch')
psi.h(0)  # |+>
```

### Step 2: Instantiate HHL algorithm

```python
hhl = HHLAlgorithm()
```

### Step 3: Configure and run HHL

```python
result = hhl.run(
    A=A,
    state_b=psi,
    t=1.0,            # time for Hamiltonian simulation
    d=4,              # QPE precision bits
    c=1.0,            # scaling constant for controlled rotation
    backend='torch',
    algo_dir='./hhl_results'
)

print('Output state vector:', result['output_state'])
print('Success probability:', result['success_prob'])
```

### Step 4: Extract solution-related observables

```python
# For example, expectation of Pauli-Z in solution state
print('Expected Z:', result['observable']['Z'])
```

---

## Practical Examples

### Example 1: Simple system with A=[[1,0.5],[0.5,1]]

```python
A = np.array([[1.0, 0.5], [0.5, 1.0]])
psi = GateSequence(n_qubits=1, backend='torch')
psi.h(0)

hhl = HHLAlgorithm()
res = hhl.run(A=A, state_b=psi, t=1.0, d=6, c=1.0, backend='torch')

print('x-state approx:', res['output_state'])
print('success prob:', res['success_prob'])
```

### Example 2: Handling condition number and precision

```python
for d in [3, 5, 7]:
    res = hhl.run(A=A, state_b=psi, t=1.0, d=d, c=1.0, backend='torch')
    print('d=', d, 'phase error:', abs(res['estimated_phase'] - 0.25))
```

---

## Performance Considerations

- Complexity depends on Hamiltonian simulation cost and QPE precision $d$.
- QPE and controlled rotation induce additional depth and gate overhead.
- Post-selection may require repeated runs; use amplitude amplification for quadratic improvement.
- Full classical vector extraction from $|x\rangle$ is expensive; prefer observable estimation.

---

## Integration Notes

- Combine HHL with QAE to estimate expectation value of $x$ efficiently.
- Use with linear algebra utilities in the engine for matrix loading / sparsity patterns.
- Keep $\kappa$ low (via preconditioning) to maintain realistic runtime.

---

## References

- A. W. Harrow, A. Hassidim, S. Lloyd, "Quantum algorithm for linear systems of equations," Phys. Rev. Lett. 103, 150502 (2009).
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*.

---

## License

MIT License - Authored by Yuanchun He
description: HHL Algorithm - A quantum algorithm for solving linear systems of equations. Supports efficient state preparation, Hamiltonian simulation, and quantum phase estimation for exponential speedup in certain cases.
license: MIT