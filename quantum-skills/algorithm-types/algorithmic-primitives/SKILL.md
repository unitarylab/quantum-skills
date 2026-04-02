---
name: algorithmic-primitives
description: A comprehensive collection of fundamental quantum algorithmic primitives that serve as building blocks for more complex quantum algorithms. Includes quantum phase estimation, amplitude estimation, hadamard testing, and linear combinations of unitaries.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Algorithmic Primitives

## Overview

Algorithmic Primitives are fundamental quantum subroutines that form the foundation of modern quantum algorithms. These core building blocks encapsulate key quantum mechanical operations and measurements, enabling the construction of sophisticated quantum algorithms for optimization, simulation, machine learning, and cryptography.

This skill collection contains essential primitives that transform quantum information in ways impossible for classical computers, providing exponential or polynomial speedups for specific computational tasks. Each primitive is mathematically rigorous, efficiently implementable on quantum hardware, and proven effective in advancing quantum algorithm development.

## Purpose & Role

Algorithmic primitives serve multiple critical functions:

- **Foundation for Complex Algorithms**: Combine primitives to construct algorithms for specific applications
- **Quantum Speedup Mechanism**: Leverage quantum mechanical properties (superposition, entanglement, interference) to achieve computational advantages
- **Bridging Classical-Quantum Gap**: Enable quantum computers to solve problems intractable on classical systems
- **Modular Design**: Enable code reuse and algorithm composition
- **Proof of Concept**: Demonstrate quantum advantage in controlled computational scenarios

## Contained Algorithms

### 1. Quantum Phase Estimation (QPE)

**Type**: Eigenvalue Estimation | **Complexity**: $O(1/\epsilon)$ gates for $\epsilon$ precision

Quantum Phase Estimation is a central algorithm for computing the eigenvalues of unitary operators with exponential precision. It maps an eigenstate $|\psi\rangle$ with eigenvalue $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$ into a register encoding $\phi$ as a binary fraction.

**Key Features**:
- Extracts eigenphases with precision $2^{-d}$ using $d$ phase register qubits
- Provides exact results when phase is representable in $d$ bits
- Works with any unitary operator and its controlled powers
- Foundation for many quantum algorithms (Shor's, HHL, eigenvalue solvers)

**Applications**:
- Factoring and discrete logarithm problems (Shor's algorithm)
- Quantum simulation with phase kickback
- Eigenvalue problems in quantum chemistry
- Building block for amplitude estimation

**Related Skills**: `quantum-phase-estimation/`

---

### 2. Quantum Amplitude Estimation (QAE)

**Type**: Probability Estimation | **Speedup**: Quadratic ($O(1/\epsilon)$ vs classical $O(1/\epsilon^2)$)

Quantum Amplitude Estimation is a powerful algorithm providing high-precision estimation of success probabilities through converting amplitude information into phase information. By combining Grover iteration with Quantum Phase Estimation, QAE achieves quadratic speedup compared to classical Monte Carlo sampling.

**Key Features**:
- Quadratic speedup: Query complexity $O(1/\epsilon)$ vs classical $O(1/\epsilon^2)$
- Adjustable precision through configurable phase register bits
- Universal: applicable to arbitrary initial states and target conditions
- Direct phase-to-amplitude conversion for robust estimation

**Applications**:
- Derivative pricing and option valuation in finance
- Portfolio risk assessment and Value-at-Risk (VaR) computation
- Probability estimation in quantum machine learning
- Monte Carlo acceleration for scientific computing
- Counting solutions in combinatorial optimization

**Mathematical Basis**: Converts amplitude $a$ to phase $\theta$ where $\sqrt{a} = \sin\theta$, then uses QPE to estimate $\theta$

**Related Skills**: `amplitude-estimation/`

---

### 3. Hadamard Test

**Type**: Expectation Value Estimation | **Utility**: Core subroutine for measurement

The Hadamard Test is a fundamental quantum measurement technique for estimating the real and imaginary parts of expectation values of unitary operators. It uses a control (ancilla) qubit to efficiently extract operator expectation values $\langle\psi | U | \psi\rangle$ with high precision.

**Key Features**:
- Measures real part: $\operatorname{Re}(\langle\psi | U | \psi\rangle)$ with standard Hadamard
- Measures imaginary part: $\operatorname{Im}(\langle\psi | U | \psi\rangle)$ with phase gate variant
- Minimal qubit overhead: requires only one ancilla qubit
- Efficient classical post-processing from binary measurement data

**Circuit Structure**:
```
control:  H ---●--- H --- Measure
              |
target:  |ψ⟩ - U -
```

**Applications**:
- Variational Quantum Eigensolver (VQE) energy measurements
- QAOA expectation value evaluation
- Quantum machine learning observable measurements
- Operator property validation (Hermitian, unitary verification)
- Constructing efficient measurement protocols for quantum algorithms

**Related Skills**: `hadamard-test/`

---

### 4. Linear Combination of Unitaries (LCU)

**Type**: Operator Synthesis | **Application**: Non-unitary operator implementation

Linear Combination of Unitaries is a powerful quantum technique for implementing linear transformations through expressing them as sums of unitary operations: $A = \sum_j \alpha_j U_j$. LCU enables realization of non-unitary operators using unitary resources, forming the foundation of quantum linear algebra algorithms.

**Key Features**:
- Encodes coefficients $\alpha_j$ in ancilla superposition states
- Combines controlled unitaries for weighted operator application
- Supports oblivious amplitude amplification for success probability boosting
- Success probability amplifiable from exponentially small to $O(1)$

**Core Mechanism**:
1. Prepare ancilla index register encoding coefficients
2. Apply controlled-unitary operation: $\sum_j |j\rangle\langle j| \otimes U_j$
3. Conditionally apply inverse state-preparation to ancilla
4. Post-select ancilla in target state

**Applications**:
- Quantum simulation of sparse Hamiltonians (Hamiltonian simulation)
- Quantum linear solvers (HHL algorithm): implementing $A^{-1}$
- Matrix functions via block-encoding and quantum signal processing
- Quantum machine learning with matrix operations
- Quantum linear regression and quantum data fitting

**Mathematical Foundation**: Maps state preparation unitary $V$ such that LCU circuit $W = (V^\dagger \otimes I)(\sum_j |j\rangle\langle j| \otimes U_j)(V \otimes I)$ applies $A/s$ to target state conditioned on ancilla measurement.

**Related Skills**: `lcu/`

---

## Hierarchy & Relationships

### Build-up Structure
```
Quantum Phase Estimation (QPE)
    ↓
    ├─→ Quantum Amplitude Estimation (QAE)
    ├─→ HHL Algorithm (Quantum Linear Systems)
    └─→ Eigenvalue Solvers

Hadamard Test
    ↓
    ├─→ VQE (Variational Quantum Eigensolver)
    ├─→ QAOA (Quantum Approximate Optimization)
    └─→ Measurement Protocols

Linear Combination of Unitaries (LCU)
    ↓
    ├─→ Quantum Simulation
    ├─→ Block-Encoded Operators
    └─→ QSVT (Quantum Signal Processing)
```

## Common Use Patterns

| Scenario | Primary Primitive | Secondary Support |
|----------|---------|---------|
| Eigenvalue computation | QPE | Hadamard Test |
| Probability estimation | QAE | QPE |
| Expectation measurements | Hadamard Test | LCU |
| Hamiltonian simulation | LCU | QPE |
| Quantum optimization | QAE + Hadamard Test | LCU |

## Technical Characteristics

| Primitive | Gate Depth | Qubit Overhead | Success Probability | Primary Cost |
|-----------|-----------|----------------|-------------------|------------|
| QPE | $O(1/\epsilon)$ | Moderate | 1.0 (deterministic) | Controlled-$U^{2^k}$ gates |
| QAE | $O(1/\epsilon)$ | Moderate | 1.0 | QPE calls + Grover iterations |
| Hadamard Test | $O(1)$ | 1 ancilla | 1.0 | Controlled-$U$ gates |
| LCU | $O(log N)$ | $log N$ indices | $O(1/s^2)$ | State preparation + amplitude amplification |

## Implementation Considerations

### Choice of Primitive

- **For eigenvalue problems**: Use QPE directly
- **For probability/counting**: Use QAE (superior to classical methods)
- **For operator expectations**: Use Hadamard Test (measurement subroutine)
- **For non-unitary operators**: Use LCU (with amplitude amplification for efficiency)

### Precision & Accuracy

- **Precision Requirements**: Define tolerance $\epsilon$ from problem constraints
- **Gate Budget**: Inverse relationship between precision and gate count
- **Error Mitigation**: Consider noise channels and error correction techniques
- **Classical Post-Processing**: Leverage classical computation for result refinement

## Integration with Quantum Skills Framework

These primitives integrate with the broader quantum skills hierarchy:

- **Upper Level**: Forms algorithms under `algorithm-types/` parent
- **Lower Level**: Implemented using gates from `programming-languages/`
- **Foundation**: Based on quantum gates and basic unitary operations
- **Applications**: Drive implementations in domain-specific skills (optimization, machine learning, simulation)

## Related Quantum Skills

- Parent Skill: `algorithm-types/` - Algorithm categorization and strategy
- Gate Implementation: `programming-languages/` - Quantum gate implementations
- Learning Path: `learning-paths/` - Structured progression through quantum concepts
- Machine Learning: `algorithm-types/quantum-machine-learning/` - Applications of these primitives

## References & Further Reading

- **Quantum Phase Estimation**: Cleve, Ekert, Macchiavello, Mosca (1998)
- **Quantum Amplitude Estimation**: Brassard, Hoyer, Mosca, Tapp (2000)
- **Hadamard Test**: Aharonov, Ben-Or (2008)
- **Linear Combination of Unitaries**: Low, Chuang (2019)
- **Applications in Quantum Chemistry**: Cao et al. (2019)
- **Quantum Algorithm Zoo**: https://quantumalgorithmzoo.org/

---