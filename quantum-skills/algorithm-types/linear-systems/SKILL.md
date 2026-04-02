---
name: linear-systems
description: A collection of quantum algorithms for solving linear systems of equations, including HHL, QSVT-based linear solvers, and related techniques. Provides efficient state preparation, block-encoding, and polynomial transformations for various applications in quantum linear algebra and optimization.
license: MIT
metadata:
    skill-author: UnitaryLab Team
---

# Quantum Linear Systems Solvers

## Overview

Quantum Linear Systems algorithms represent one of the most impactful applications of quantum computing, offering exponential speedup for solving sparse, well-conditioned linear systems of equations. These algorithms address the fundamental problem:

$$A \mathbf{x} = \mathbf{b}$$

where $A$ is an $n \times n$ matrix and $\mathbf{b}$ is an $n$-dimensional vector. Classical computers require $O(n^3)$ or $O(n^2 \log(1/\epsilon))$ time for exact or approximate solutions. Quantum algorithms can achieve $O(\text{polylog}(n) \cdot \kappa \cdot f(A))$ complexity under favorable conditions, providing exponential speedup for certain matrix classes.

## Purpose & Role

Quantum linear system solvers are fundamental to:

- **Scientific Computing**: Accelerating computational simulations (physics, chemistry, materials science)
- **Machine Learning**: Fast training and inference in quantum ML algorithms
- **Optimization**: Solving linear constraints and dual problems
- **Quantum Simulation**: Implementing time evolution and spectral methods
- **Finance**: Portfolio optimization and risk assessment
- **Combinatorial Problems**: Converting problems to linear systems

## Contained Algorithms

### 1. HHL Algorithm (Harrow-Hassidim-Lloyd)

**Type**: Quantum Linear System Solver | **Speedup**: Exponential for sparse matrices | **Complexity**: $O(\kappa^2 \text{ polylog}(n) / \epsilon)$

The HHL algorithm is the pioneering quantum algorithm for solving linear systems of the form $A \mathbf{x} = \mathbf{b}$. It returns a quantum state proportional to $|\mathbf{x}\rangle$ that encodes the solution vector $\mathbf{x} = A^{-1}\mathbf{b}$, achieving exponential speedup compared to classical solvers for specific problem instances.

**Algorithm Steps**:
1. **State Preparation**: Encode input vector $|\mathbf{b}\rangle$ into quantum register
2. **Hamiltonian Simulation**: Implement $e^{iAt}$ for controlled time steps, enabling eigenvalue extraction
3. **Phase Estimation**: Use Quantum Phase Estimation (QPE) to estimate eigenvalues $\lambda_j$ of matrix $A$
4. **Conditional Rotation**: Apply controlled rotation mapping $|\lambda_j\rangle \to |1/\lambda_j\rangle$ in ancilla amplitude
5. **Uncomputation**: Reverse QPE register preparation
6. **Post-selection**: Condition on ancilla measurement in $|1\rangle$ state to extract solution $|x\rangle$

**Key Features**:
- **Exponential Speedup**: $O(\kappa^2 \text{ polylog}(n))$ vs classical $O(n^3)$
- **Matrix Requirements**: Requires Hermitian, sparse, well-conditioned matrices
- **Condition Number**: Success probability depends on condition number $\kappa = \lambda_{\max}/\lambda_{\min}$
- **Output Form**: Quantum state encoding; requires tomography or observable measurement for classical output
- **Foundational**: Forms basis for many advanced quantum algorithms

**Requirements & Assumptions**:
- Matrix $A$ must be Hermitian ($A = A^{\dagger}$)
- Matrix should be sparse with efficient quantum simulation
- Condition number $\kappa$ should be manageable (not exponentially large)
- Input vector $|\mathbf{b}\rangle$ must be efficiently state-preparable
- Output interpretation requires problem-specific observables

**Mathematical Foundation**:
Using eigendecomposition $A|u_j\rangle = \lambda_j|u_j\rangle$, express $|\mathbf{b}\rangle = \sum_j \beta_j |u_j\rangle$. After phase estimation and controlled rotation:

$$|\psi'\rangle = \sum_j \beta_j |u_j\rangle |\tilde\lambda_j\rangle \left(\sqrt{1-\frac{C^2}{\tilde\lambda_j^2}}|0\rangle + \frac{C}{\tilde\lambda_j}|1\rangle\right)$$

Uncomputing eigenvalue register and post-selecting ancilla in $|1\rangle$ yields the desired solution state.

**Applications**:
- Quantum chemistry simulations (molecular scattering)
- Quantum machine learning (QML) algorithms
- Quantum optimization (constraint satisfaction)
- Scientific computing acceleration
- Differential equation solving

**Success Probability**: $\Omega(1/\kappa^2)$ - success probability scales inversely with condition number squared; can be boosted via amplitude amplification

**Related Skills**: `hhl/` - Detailed implementation and variants

---

### 2. QSVT-QLSA (Quantum Singular Value Transformation - Quantum Linear System Algorithm)

**Type**: Advanced Linear System Solver | **Improvement**: Condition-dependent complexity reduction | **Complexity**: $O(\kappa \text{ polylog}(\kappa/\epsilon))$

Quantum Singular Value Transformation (QSVT) is a versatile framework for applying polynomial functions to the singular values of block-encoded matrix operators. QSVT-QLSA leverages this powerful technique to implement improved quantum linear solvers, offering better complexity scaling and enhanced robustness compared to classical HHL.

**Core Concepts**:

1. **Block-Encoding**: Represent matrix $A$ as a block within a unitary operator:
$$U_A = \begin{pmatrix}A & * \\ * & *\end{pmatrix}$$

2. **Polynomial Transformation**: Apply polynomial $P(x)$ approximating $1/x$ to matrix singular values
3. **Spectral Filtering**: Use polynomial of carefully chosen degree to optimize accuracy-complexity trade-off
4. **State Transformation**: Transform input state $|\mathbf{b}\rangle$ to output approximating $A^{-1}|\mathbf{b}\rangle$

**Key Features**:
- **Optimized Complexity**: Reduces dependence to $O(\kappa \text{ polylog}(\kappa/\epsilon))$ - better than HHL's quadratic condition number dependence
- **Polynomial-Based**: Avoids explicit QPE; uses polynomial approximations instead
- **Block-Encoding Compatible**: Natural fit for block-encoded matrix representations
- **Flexible Precision**: Tunable polynomial degree for accuracy-circuit depth trade-off
- **NISQ-Friendly**: Potentially more suitable for near-term quantum devices

**Algorithm Steps**:
1. **Block-Encoding Preparation**: Construct unitary block-encoding $U_A$ for matrix $A$
2. **Input State Preparation**: Encode vector $|\mathbf{b}\rangle$ in target register
3. **QSVT Application**: Apply sequence of controlled-$U_A$ and single-qubit rotations implementing polynomial $P$ on singular values
4. **Ancilla Manipulation**: Handle ancilla qubits carrying polynomial transformation information
5. **Measurement & Post-selection**: Extract solution state with appropriate measurement basis

**Mathematical Foundation**:

For Singular Value Decomposition $A = \sum_j \sigma_j |u_j\rangle\langle v_j|$, QSVT implements:

$$P(A) = \sum_j P(\sigma_j) |u_j\rangle\langle v_j|$$

For linear system solving, choose polynomial $P(x) \approx 1/x$ over the singular value domain $[1/\kappa, 1]$. Applying to input state:

$$P(A)|\mathbf{b}\rangle = \sum_j P(\sigma_j) \beta_j |u_j\rangle \approx \sum_j \frac{\beta_j}{\sigma_j} |u_j\rangle$$

This approximates the desired solution $A^{-1}|\mathbf{b}\rangle$ up to normalization and approximation error.

**Requirements & Assumptions**:
- Matrix $A$ must have $\|A\| \leq 1$ (normalized or block-scaled)
- Should be well-conditioned (manageable $\kappa$)
- Efficient block-encoding must exist
- Polynomial approximation error must be compatible with problem tolerance

**Advanced Capabilities**:
- **Variable Condition Number Handling**: Polynomial degree scales efficiently with $\kappa$
- **Precision Control**: Direct control over approximation error via polynomial degree
- **Error Bounds**: Rigorous error analysis and confidence intervals
- **Integration with QAE**: Combines with Quantum Amplitude Estimation for probability estimation

**Applications**:
- High-precision quantum machine learning
- Scientific computing with demanding accuracy requirements
- Optimization with complex constraint structures
- Quantum chemistry simulations with stringent convergence criteria
- Portfolio analysis with risk constraints

**Advantages over HHL**:
- Better asymptotic complexity in condition number: $O(\kappa)$ vs $O(\kappa^2)$
- More flexible error control through polynomial degree tuning
- Natural integration with other QSVT-based algorithms
- Potential for improved numerical stability

**Related Skills**: `qsvt-qlsa/` - Detailed implementation and variants

---

## Algorithm Comparison

| Aspect | HHL | QSVT-QLSA |
|--------|-----|-----------|
| **Condition Number Dependence** | $O(\kappa^2)$ | $O(\kappa)$ |
| **Overall Complexity** | $O(\kappa^2 \text{ polylog}(n))$ | $O(\kappa \text{ polylog}(\kappa/\epsilon))$ |
| **Key Subroutine** | Quantum Phase Estimation (QPE) | Polynomial transformation |
| **Matrix Representation** | Direct simulation | Block-encoding |
| **Success Probability** | $\Omega(1/\kappa^2)$ | $\Omega(1/\kappa^2)$ (with optimized polynomial) |
| **Error Control** | Phase estimation precision | Polynomial degree |
| **Implementation Complexity** | Moderate | Higher (polynomial design) |
| **NISQ Suitability** | Requires precise QPE | Better suited for near-term devices |
| **Scalability** | Limited by QPE depth | More flexible scaling |

## Use Case Selection Guide

### Use HHL When:
- Seeking foundational understanding of quantum linear solvers
- Matrix has specific structure enabling efficient simulation
- Condition number is small (< 100)
- QPE implementation is available
- Historical comparison or baseline needed

### Use QSVT-QLSA When:
- Optimized performance is critical
- Condition number is moderate to large (100-10,000)
- Block-encoding representation is natural
- Polynomial approximation flexibility needed
- Near-term quantum device compatibility desired

## Integration with Broader Quantum Framework

### Dependencies
- **Prerequisite Skills**: `algorithmic-primitives/` - QPE, Hadamard Test, LCU
- **Required Components**: Quantum phase estimation, state preparation, amplitude amplification

### Applications & Extensions
- **Quantum Machine Learning**: `algorithm-types/quantum-machine-learning/` - QML algorithms using HHL/QSVT
- **Optimization**: `algorithm-types/quantum-optimization/` - Linear constraints in optimization
- **Simulation**: `algorithm-types/quantum-simulation/` - Hamiltonian evolution and dynamics

### Related Skills
- **Algorithmic Primitives**: Core subroutines (QPE, QAE, Hadamard Test, LCU)
- **Programming Languages**: Gate implementations and quantum circuit compilation
- **Learning Paths**: Step-by-step progression through linear algebra concepts

## Technical Considerations

### Implementation Challenges

1. **State Preparation**: Efficiently preparing $|\mathbf{b}\rangle$ is non-trivial; often as expensive as the algorithm itself
2. **Condition Number Dependency**: Performance degrades significantly for ill-conditioned matrices
3. **Output Interpretation**: Solution is a quantum state; extracting classical bits requires tomography
4. **Error Accumulation**: Errors in state preparation, phase estimation, and rotation compound

### Practical Guidelines

- **Problem Preprocessing**: Normalize and precondition matrix $A$ to improve $\kappa$
- **Hybrid Approaches**: Combine quantum solver with classical preprocessing
- **Error Mitigation**: Employ error correction or mitigation techniques for noisy hardware
- **Benchmarking**: Compare with classical solvers on small instances before scaling

## Performance Metrics

| Metric | HHL | QSVT-QLSA | Notes |
|--------|-----|-----------|-------|
| **Gate Depth** | $O(\kappa^2 \log(n/\epsilon))$ | $O(\kappa \log(1/\epsilon))$ | Lower is better for near-term devices |
| **Qubit Count** | $O(\log n + a)$ | $O(\log n + a)$ | $a$ ancillas for precision |
| **Circuit Constant** | Higher | Moderate | Affects practical performance |
| **Noise Resilience** | Moderate | Good | QPE vs polynomial-based |

## References & Further Reading

- **HHL Original**: Harrow, Hassidim, Lloyd (2008) - "Quantum algorithm for solving linear systems of equations"
- **QSVT Framework**: Low, Chuang (2019) - "Hamiltonian Simulation by Qubitization"
- **QSVT-QLSA**: Chakraborty, Gilyén, Jeffery (2020)
- **Block-Encoding**: Gilyén, Su, Low, Wiebe (2019)
- **Quantum ML Applications**: Biamonte et al. (2017)

---