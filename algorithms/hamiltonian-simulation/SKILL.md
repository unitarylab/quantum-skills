---
name: hamiltonian-simulation
description: Quantum Hamiltonian simulation methods for approximating time evolution e^{-iHt}. Currently includes Trotter-Suzuki decomposition and QDrift randomized sampling.
---

# Hamiltonian Simulation

Approximate the time-evolution operator $e^{-iHt}$ for a given Hermitian matrix $H$.

## Available Methods

### 1. Trotter-Suzuki Decomposition
See reference: `./trotter/SKILL.md`

Deterministic product formula. Decomposes $e^{-iHt}$ into a sequence of simpler exponentials.
- Key parameters: `order` (Suzuki order), `steps` (number of Trotter steps).

### 2. QDrift (Randomized)
See reference: `./qdrift/SKILL.md`

Stochastic channel simulation via $\lambda$-weighted random sampling of Pauli terms.
- Key parameter: `steps` (number of random samples).

## Unified Entry Point

All methods are accessible through `hamiltonian_simulation(H, t, method, target_error, **kwargs)`:

```python
from hamiltonian_simulation import hamiltonian_simulation

result = hamiltonian_simulation(H, t, method='trotter', target_error=1e-4, order=2, steps=80)
print(result.target_qubits, result.total_error)
```

| Method | `method` string | Extra kwargs |
|--------|----------------|--------------|
| Trotter | `'trotter'` | `order`, `steps` |
| QDrift | `'qdrift'` | `steps` |

The returned object exposes lazy properties: `circuit`, `evolution_result`, `total_error`.

## Prerequisites

1. Basic linear algebra (Hermitian matrices, matrix exponential).
2. High-level understanding of Hamiltonian simulation goals.
3. NumPy for matrix construction.
