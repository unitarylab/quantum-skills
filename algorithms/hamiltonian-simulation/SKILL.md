---
name: hamiltonian-simulation
description: Quantum Hamiltonian simulation methods for approximating time evolution e^{-iHt}. Includes Trotter-Suzuki decomposition, QDrift randomized sampling, Cartan decomposition, QSP polynomial spectral transformation, and Taylor series LCU expansion.
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

### 3. Cartan Decomposition
See reference: `./cartan/SKILL.md`

Structural decomposition via Lie algebra splitting $\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}$. Iterates a Lax flow to build a circuit of the form $K \cdot e^{-i\eta} \cdot K^\dagger$.
- Key parameters: `lr` (learning rate), `max_steps` (hard cap on Lax update steps), `reps` (number of repetitions).

### 4. QSP (Quantum Signal Processing)
See reference: `./qsp/SKILL.md`

Block-encodes the Hamiltonian and applies polynomial spectral transformations via interleaved signal-processing rotations. Approximates $\cos(tH)$ and $\sin(tH)$ as Chebyshev series and merges them via LCU.
- Key parameters: `degree` (upper bound on polynomial degree per time slice), `beta` (block-encoding scaling factor).

### 5. Taylor Series (LCU)
See reference: `./taylor/SKILL.md`

Truncates the Taylor series of $e^{-iHt}$ and implements the result as a Linear Combination of Unitaries over Pauli string products. Applies adaptive time-slicing to keep the per-slice spectral weight small.
- Key parameters: `degree` (Taylor truncation order, capped at 15).

## Method Selection Guidance

This top-level skill is an index and routing guide. For executable examples, open the corresponding method-specific `SKILL.md`.

Conceptual dispatch logic:

```text
if method == "trotter":
    use ./trotter/SKILL.md
elif method == "qdrift":
    use ./qdrift/SKILL.md
elif method == "cartan-lax":
    use ./cartan/SKILL.md
elif method == "qsp":
    use ./qsp/SKILL.md
elif method == "taylor":
    use ./taylor/SKILL.md
```

| Method | `method` string | Extra kwargs |
|--------|----------------|--------------|
| Trotter | `'trotter'` | `order`, `steps` |
| QDrift | `'qdrift'` | `steps` |
| Cartan | `'cartan-lax'` | `lr`, `max_steps`, `reps` |
| QSP | `'qsp'` | `degree`, `beta` |
| Taylor | `'taylor'` | `degree` |

The returned object exposes lazy properties: `circuit`, `evolution_result`, `total_error`.

## Prerequisites

1. Basic linear algebra (Hermitian matrices, matrix exponential).
2. High-level understanding of Hamiltonian simulation goals.
3. NumPy for matrix construction.
