---
name: qsvt-qlsa
description: A quantum algorithm for solving linear systems of equations using Quantum Singular Value Transformation (QSVT), providing exponential speedup over classical methods for certain types of problems. This skill includes efficient implementations and educational resources for understanding and utilizing QSVT-based quantum linear systems algorithms in various applications.
---

# One Step to Run QSVT Example
```bash
python ./scripts/algorithm.py
```

# Quantum Singular Value Transformation Linear Systems Algorithm (QSVT-QLSA) Skill Guide

## Overview

**Quantum Singular Value Transformation (QSVT)** is a cutting-edge quantum algorithm that solves linear systems of equations $Ax = b$ by applying polynomial functions directly to singular values. Given an $N \times N$ matrix $A$ with condition number $\kappa$, QSVT:

1. **Approximates the inverse function** $f(x) = 1/x$ using Chebyshev polynomials
2. **Constructs polynomial block-encodings** via Quantum Signal Processing (QSP)
3. **Applies QSVT procedure** to efficiently implement the polynomial
4. **Post-processes result** to extract the normalized solution state

### Key Innovation

Instead of using phase estimation (like HHL) or linear combinations (like LCU/HHL), QSVT directly transforms the **singular values** of $A$ using:

$$\text{QSVT: } A = U\Sigma V^\dagger \rightarrow \text{Poly}^{(SV)}(A) = U \cdot \text{Poly}(\Sigma) \cdot V^\dagger$$

This enables:
- **Better scaling**: $O(\kappa^2 \alpha \sqrt{\log(\kappa^3/\epsilon)})$ gate complexity (vs. $O(\kappa/\epsilon)$ for HHL)
- **Polynomial functions**: Apply any polynomial transformation to eigenvalues
- **Non-symmetric matrices**: Works directly with full SVD

### Why QSVT-QLSA Matters

1. **Improved Complexity**: Polynomial dependence on error tolerance $\epsilon$
2. **Handles Non-Hermitian Systems**: Works for arbitrary (non-symmetric) matrices
3. **Foundation for QSP**: Quantum Signal Processing is basis for modern quantum algorithms
4. **Practical Speedup**: Proven speedup for many linear algebra problems

### Real Applications

- Solving differential equations (PDEs, ODEs)
- Quantum machine learning (kernel methods, regression)
- Linear regression and least-squares problems
- Engineering simulations and scientific computing

---

## Learning Objectives

After mastering this skill, you will be able to:

1. Understand singular value transformation vs. eigenvalue-based methods
2. Grasp Chebyshev polynomial approximation of $f(x) = 1/x$
3. Understand Quantum Signal Processing (QSP) phase sequences
4. Explain block-encoding and polynomial scaling
5. Use `QSVTLinearSolverAlgorithm` class effectively
6. Calculate required polynomial degree and approximation error
7. Implement core components (Chebyshev coefficients, QSP phases)
8. Analyze complexity and compare with HHL algorithm
9. Apply QSVT to non-symmetric matrices
10. Understand QSP theorem and its quantum circuit realization

---

## Prerequisites

- **Essential knowledge**:
  - Linear algebra: SVD, matrix norms, condition numbers
  - Quantum block-encodings and their properties
  - Quantum signal processing basics
  - Chebyshev polynomials and approximation theory
  - Basic quantum gates and measurements
- **Recommended background**: [HHL Algorithm](../hhl/SKILL.md) for comparison
- **Mathematical comfort**: Complex polynomials, Fourier analysis, optimization

---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms.linear_algebra import QSVTLinearSolverAlgorithm
from engine import GateSequence
import numpy as np

# Step 1: Create block-encoding of matrix A
# Example: Diagonal matrix with eigenvalues 0.8, 0.4
def block_encoding_diag(a0=0.8, a1=0.4):
    theta0 = 2 * np.arccos(a0)
    theta1 = 2 * np.arccos(a1)
    gs = GateSequence(2)
    anc, sys = 1, 0
    gs.cry(theta0, sys, anc, [0])
    gs.cry(theta1, sys, anc, [1])
    return gs

# Convert to non-diagonal via Hadamard
def block_encoding_nondiag_via_hadamard(a0=0.8, a1=0.4):
    U = block_encoding_diag(a0, a1)
    gs = GateSequence(2)
    sys = 0
    gs.h(sys)
    gs.append(U, [0, 1])
    gs.h(sys)
    return gs

# Step 2: Create block-encoding of A
be_A = block_encoding_nondiag_via_hadamard(a0=0.8, a1=0.4)

# Step 3: Define RHS vector b
b_vec = np.array([1.0, 0.0])

# Step 4: Run QSVT solver
algo = QSVTLinearSolverAlgorithm()
result = algo.run(
    gs_be=be_A,           # Block-encoding of A
    b=b_vec,              # Right-hand side vector
    kappa=5.0,            # Condition number
    alpha=1.0,            # Block-encoding scaling
    epsilon=0.001,        # Target accuracy
    n_sys=1,              # Number of system qubits
    n_anc_be=1,           # Number of auxiliary qubits
    backend='torch'
)

# Step 5: Inspect results
print(result['plot'])
print(f"Solution vector: {result['solution_vector']}")
print(f"Circuit path: {result['circuit_path']}")
```

### Core Parameters Explained

```python
algo.run(
    gs_be: GateSequence,          # Block-encoding unitary U_A representing matrix A
    b: np.ndarray,                # Right-hand side vector (in ℝ^(2^n_sys))
    kappa: float,                 # Condition number κ = σ_max / σ_min
    alpha: float,                 # Block-encoding scaling factor (typical: 1.0)
    epsilon: float,               # Target approximation accuracy
    n_sys: int,                   # Number of system (data) qubits
    n_anc_be: int,                # Number of ancilla qubits for block-encoding
    backend: str = 'torch',       # Simulation backend
    algo_dir: str = './qsvt_results'  # Output directory
)
```

**Return dictionary contains:**
- `solution_vector`: Final solution state $|x\rangle \approx A^{-1}|b\rangle / (κ||b||)$
- `full_circuit`: Complete QSVT quantum circuit
- `circuit_path`: Location of circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

---

## Understanding the Core Components

### 1. The Five-Step QSVT Pipeline

```python
def run(self, gs_be, b, kappa, alpha, epsilon, n_sys, n_anc_be):
    """
    Complete QSVT pipeline for solving Ax = b:
    
    Step 1: Classical preprocessing (Chebyshev coefficients)
    Step 2: Build QSVT quantum circuit
    Step 3: Execute quantum simulation
    Step 4: Classical post-processing (extract solution)
    Step 5: Export circuit diagram
    """
    
    # ========== Stage 1: Chebyshev Coefficients ==========
    # Approximate f(x) = 1/x using Chebyshev polynomials
    # f(x) = (1 - (1 - x²)^b) / x
    
    # Calculate parameter b based on condition number and accuracy
    b_param = ceil((κ·α)² · log(κ·α/ε))
    
    # Calculate truncation point j₀
    j_0 = ceil(√(b·log(4b/ε)))
    
    # Compute Chebyshev expansion coefficients
    c_cheby = _chebyshev_factor(kappa, alpha, epsilon)
    # Result: list of coefficients for odd-order Chebyshev polynomials
    
    # ========== Stage 2: Build QSVT Circuit ==========
    # Total qubits = n_sys + n_anc_be + 1 (extra control qubit for QSVT)
    
    circuit = GateSequence(n_sys + n_anc_be + 1)
    
    # Initialize system register with normalized |b⟩
    circuit.initialize(b / ||b||, system_qubits)
    
    # Apply QSP block-encoding via Chebyshev coefficients
    # This implements polynomial transformation of singular values
    qsp_circuit = QSP_block_encoding(
        is_coef_cheby=True,
        coef=c_cheby,              # Chebyshev coefficients
        parity=1,                  # Odd function
        opts=optimization_options,
        u=gs_be.dagger(),          # U_A† for polynomial application
        n=n_sys,
        m=n_anc_be
    )
    circuit.append(qsp_circuit, all_qubits)
    
    # ========== Stage 3: Execute Simulation ==========
    final_state = circuit.execute()
    state_array = np.asarray(final_state, dtype=complex).reshape(-1)
    
    # ========== Stage 4: Classical Post-Processing ==========
    # Extract system register (first 2^n_sys amplitudes)
    raw_solution = state_array[0 : 2**n_sys]
    
    # Apply scaling factor to recover physical solution
    # scaling = ||b|| · κ · α · 2 (factor of 2 from function normalization)
    scaling_factor = ||b|| * kappa * alpha * 2
    solution = raw_solution * scaling_factor
    
    # ========== Stage 5: Export Circuit ==========
    circuit.draw(filename=circuit_path, title=f"QSVT Solver (κ={kappa})")
    
    return solution
```

### 2. Chebyshev Polynomial Approximation

The QSVT uses Chebyshev polynomials to approximate $f(x) = 1/x$:

```python
def _chebyshev_factor(self, kappa: float, alpha: float, 
                      epsilon: float) -> np.ndarray:
    """
    Compute Chebyshev expansion coefficients for:
    f(x) = (1 - (1 - x²)^b) / x
    
    This function ε-approximates 1/x on domain D_α = [1/(κα), 1/α]
    
    Mathematical Foundation:
    For integer b ≥ (κα)² log(κα/ε), the truncated expansion:
    g(x) = 4 Σⱼ₌₀^j₀ (-1)^j · [central_binom_ratio] · T_{2j+1}(x)
    
    is ε-close to f(x), where j₀ = √(b log(4b/ε))
    """
    
    # Step 1: Calculate polynomial order from condition number and accuracy
    b = ceil((kappa * alpha) ** 2 * log(kappa * alpha / epsilon))
    
    # Step 2: Calculate truncation index
    j = ceil(sqrt(b * log(4 * b / epsilon)))
    
    # Step 3: Use high-precision arithmetic for central binomial ratio
    # This is C(2b, b) / 4^b = central_binom_ratio
    mid_v = _central_binom_ratio(b, dps=50)
    value = (1 - mid_v) / 2
    
    # Step 4: Build coefficient array iteratively
    j = int(j) + 1
    coefficient_list = np.zeros(j, dtype=float)
    
    for k in range(j):
        # Each coefficient includes sign, value, and normalization
        coefficient_list[k] = 4 * (-1) ** k * value
        
        # Update central binomial ratio recursively
        mid_v = mid_v * (b - k) / (b + k + 1)
        value -= mid_v
    
    # Normalize by 2·κ·α to ensure proper scaling
    return coefficient_list / (2 * kappa * alpha)
```

**Example**: For $\kappa = 5, \alpha = 1, \epsilon = 0.001$:
- $b = \text{ceil}(25 \cdot \log(5000)) \approx 210$
- $j_0 = \text{ceil}(\sqrt{210 \cdot \log(840)}) \approx 40$
- Returns 40 Chebyshev coefficients for odd-order polynomials

### 3. Quantum Signal Processing (QSP)

QSP converts polynomial coefficients into phase sequences:

```python
def apply_QSP_to_polynomial(coefficients, unitary_U, system_qubits, 
                            ancilla_qubits):
    """
    QSP Theorem: For polynomial P(x) and unitary U representing A,
    there exist phases Φ = (φ₀, φ₁, ..., φd) such that:
    
    U_Φ(x) = exp(i·φ₀·Z) · ∏ⱼ₌₁^d [O(x)·exp(i·φⱼ·Z)]
    
    produces a block-encoding of P(A) where:
    
    O(x) = [[x, -√(1-x²)], [√(1-x²), x]] = U_A(x)·Z
    
    with U_A(x) being the block-encoding unitary of A.
    
    Mathematical Conditions for Valid (P,Q):
    1. deg(P) ≤ d, deg(Q) ≤ d-1
    2. P has parity d mod 2, Q has parity (d-1) mod 2
    3. |P(x)|² + (1-x²)|Q(x)|² = 1 for all x ∈ [-1,1]
    """
    
    circuit = GateSequence(len(system_qubits) + len(ancilla_qubits) + 1)
    
    # Initial phase
    circuit.rz(2 * phases[0], ancilla_qubits[0])
    
    # Apply alternating unitary and phase gates
    for j in range(1, len(phases)):
        # Apply O(x) - the block-encoding structure
        circuit.append(unitary_U, system_qubits + ancilla_qubits)
        
        # Apply relative phase
        circuit.rz(2 * phases[j], ancilla_qubits[0])
    
    return circuit
```

### 4. Block-Encoding Structure

The key ingredient is the block-encoding of matrix $A$:

```python
def block_encoding_for_diagonal_matrix(eigenvalues: List[float]) -> GateSequence:
    """
    Create block-encoding of diagonal matrix with given eigenvalues.
    
    For eigenvalues λ = [λ₀, λ₁, ..., λₙ], construct:
    U_A = [[λ₀, √(1-λ₀²)], [√(1-λ₀²), -λ₀] ] ⊗ I
        ⊕ [[λ₁, √(1-λ₁²)], [√(1-λ₁²), -λ₁] ] ⊗ I
        ...
    
    This unitary encodes A as (1,m)-block encoding:
    U_A = [[A, ·], [·, ·]]  in block form
    """
    n = len(eigenvalues)
    n_anc = int(np.ceil(np.log2(n)))
    
    circuit = GateSequence(n_anc + 1)
    
    # For each eigenvalue, create a controlled ancilla-dependent rotation
    for j, lam in enumerate(eigenvalues):
        # Compute rotation angle: 2·arccos(λⱼ)
        theta = 2 * np.arccos(lam)
        
        # Apply controlled RY based on ancilla state
        circuit.cry(theta, 0, 1, control_value=format(j, f'0{n_anc}b'))
    
    return circuit
```

### 5. Post-Processing and Solution Extraction

```python
def extract_solution_from_qsvt(state_array: np.ndarray, n_sys: int,
                               b: np.ndarray, kappa: float, alpha: float) -> np.ndarray:
    """
    Extract solution from QSVT output state.
    
    QSVT produces state proportional to:
    |ψ_out⟩ ∝ A⁻¹|b⟩ / (κ·||b||)
    
    The amplitudes in the first 2^n_sys positions encode the solution.
    We need to:
    1. Extract system register amplitudes
    2. Apply classical scaling factors
    3. Normalize result
    """
    
    # Extract system-register amplitudes (first 2^n_sys entries)
    raw_state = state_array[0 : 2**n_sys]
    
    # Apply scaling factor
    # The QSVT output includes factors: 1/κ (from matrix inversion)
    #                                   1/α (from block-encoding)
    #                                   1/2 (from Chebyshev normalization)
    # We need to multiply by ||b|| · κ · α · 2 to recover physical solution
    
    norm_b = np.linalg.norm(b)
    scaling = norm_b * kappa * alpha * 2
    
    solution = raw_state * scaling
    
    return solution
```

---

## Hands-On Example: 2×2 System

Let's solve a simple $2 \times 2$ system:

```python
from engine.algorithms.linear_algebra import QSVTLinearSolverAlgorithm
from engine import GateSequence
import numpy as np

# Define the linear system: [[0.8, 0.2], [0.2, 0.6]] · x = [1, 0]
# First eigenvalue: 0.8, second: 0.6

def simple_2x2_block_encoding():
    """Block-encode a diagonal matrix with eigenvalues [0.8, 0.6]"""
    gs = GateSequence(2)
    
    # Eigenvalue 0 (0.8): rotation angle = 2·arccos(0.8)
    theta0 = 2 * np.arccos(0.8)
    # Eigenvalue 1 (0.6): rotation angle = 2·arccos(0.6)
    theta1 = 2 * np.arccos(0.6)
    
    # Apply ancilla-controlled rotations
    # If ancilla = |0⟩, apply first rotation
    gs.cry(theta0, 0, 1, [0])  # cry(angle, control, target, control_value)
    # If ancilla = |1⟩, apply second rotation
    gs.cry(theta1, 0, 1, [1])
    
    return gs

# Add Hadamard to make non-diagonal
def make_nondiagonal(diagonal_be):
    gs = GateSequence(2)
    gs.h(0)  # Hadamard on system qubit
    gs.append(diagonal_be, [0, 1])
    gs.h(0)  # Hadamard again
    return gs

# Run the algorithm
algo = QSVTLinearSolverAlgorithm()

be_A = make_nondiagonal(simple_2x2_block_encoding())
b = np.array([1.0, 0.0])

result = algo.run(
    gs_be=be_A,
    b=b,
    kappa=4.0,              # κ = 0.8 / 0.2 ≈ 4
    alpha=1.0,
    epsilon=0.001,
    n_sys=1,                # 1 qubit for 2-dimensional system
    n_anc_be=1              # 1 ancilla qubit for 2 eigenvalues
)

print("Solution from QSVT:")
print(result['solution_vector'])

print("\nCircuit diagram saved to:")
print(result['circuit_path'])

# Verification (classical)
A = np.array([[0.8, 0.2], [0.2, 0.6]])
x_classical = np.linalg.solve(A, b)
print(f"\nClassical solution: {x_classical}")
```

**Expected Output:**
- QSVT approximates the solution with error bounded by $O(\epsilon)$
- For $\epsilon = 0.001$, expect relative error $< 0.1\%$

---

## Implementing Core Components from Scratch

### Complete QSVT Implementation Template

```python
import numpy as np
from typing import List, Dict, Any, Tuple
from engine import GateSequence, State
import mpmath as mp

class MyQSVTSolver:
    """Educational implementation of QSVT for linear systems."""
    
    def solve_linear_system(self, gs_be: GateSequence, b: np.ndarray,
                           kappa: float, epsilon: float,
                           n_sys: int, n_anc_be: int) -> Tuple[np.ndarray, float]:
        """
        Solve Ax = b using QSVT.
        
        Args:
            gs_be: Block-encoding unitary of A
            b: Right-hand side vector
            kappa: Condition number
            epsilon: Target accuracy
            n_sys: System qubits
            n_anc_be: Ancilla qubits for block-encoding
        
        Returns:
            (solution_state, computation_time)
        """
        
        # STEP 1: Calculate Chebyshev approximation order
        b_param = self._calculate_polynomial_order(kappa, epsilon)
        j_0 = self._calculate_truncation_index(b_param, epsilon)
        
        print(f"Polynomial degree: {2*j_0}  (Chebyshev order)")
        print(f"Block-Encoding calls: O({int(np.sqrt(b_param * np.log(b_param)))})")
        
        # STEP 2: Generate Chebyshev coefficients
        c_cheby = self._compute_chebyshev_coefficients(kappa, epsilon, b_param)
        print(f"Generated {len(c_cheby)} Chebyshev coefficients")
        
        # STEP 3: Convert to QSP phase sequence
        phases = self._chebyshev_to_QSP_phases(c_cheby)
        print(f"Generated {len(phases)} QSP phases")
        
        # STEP 4: Build and execute QSVT circuit
        total_qubits = n_sys + n_anc_be + 1
        circuit = GateSequence(total_qubits)
        
        # Initialize |b⟩ state
        b_normalized = b / np.linalg.norm(b)
        circuit.initialize(b_normalized, list(range(n_sys)))
        
        # Apply QSVT using phases and block-encoding
        circuit.append(
            self._build_QSVT_circuit(gs_be, phases, n_sys, n_anc_be),
            list(range(total_qubits))
        )
        
        # STEP 5: Execute simulation
        final_state = circuit.execute()
        state_array = np.asarray(final_state, dtype=complex).reshape(-1)
        
        # STEP 6: Extract and scale solution
        solution = self._extract_solution(state_array, n_sys, b, kappa)
        
        return solution
    
    def _calculate_polynomial_order(self, kappa: float, epsilon: float) -> float:
        """
        Calculate polynomial order b ≥ (κα)² log(κα/ε)
        Assumes α = 1.0 for simplicity.
        """
        return np.ceil((kappa ** 2) * np.log(kappa / epsilon))
    
    def _calculate_truncation_index(self, b: float, epsilon: float) -> float:
        """
        Calculate truncation index:
        j₀ = √(b·log(4b/ε))
        """
        return np.ceil(np.sqrt(b * np.log(4 * b / epsilon)))
    
    def _compute_chebyshev_coefficients(self, kappa: float, epsilon: float,
                                       b: float) -> np.ndarray:
        """
        Compute Chebyshev expansion coefficients using high precision.
        
        For f(x) = (1 - (1-x²)^b) / x, we expand:
        f(x) = 4 Σⱼ₌₀^j₀ (-1)^j [central_binom] T_{2j+1}(x)
        """
        
        # Set high precision for binomial calculation
        mp.mp.dps = 50
        
        # Calculate central binomial coefficient: C(2b,b) / 4^b
        b_int = int(b)
        
        # Use logarithmic form to avoid overflow
        log_central = (mp.loggamma(2*b_int + 1) - 
                      2*mp.loggamma(b_int + 1) - 
                      b_int * mp.log(4))
        central_binom = float(mp.e ** log_central)
        
        # Initialize value
        value = (1.0 - central_binom) / 2.0
        j_0 = int(self._calculate_truncation_index(b, epsilon)) + 1
        
        # Build coefficient list
        coeffs = np.zeros(j_0, dtype=float)
        mid_v = central_binom
        
        for k in range(j_0):
            coeffs[k] = 4 * ((-1) ** k) * value
            
            # Update using recurrence
            mid_v = mid_v * (b - k) / (b + k + 1)
            value -= mid_v
        
        # Normalize
        return coeffs / (2.0 * kappa)
    
    def _chebyshev_to_QSP_phases(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Convert Chebyshev coefficients to QSP phase sequence.
        
        This requires solving optimization problem with loss function:
        L(Φ_W) = (1/d̃) Σⱼ [Re[⟨0|U_Φ(xⱼ)|0⟩] - f(xⱼ)]²
        
        For educational purposes, we provide a simplified version.
        """
        
        # In production, this uses iterative optimization
        # For now, return placeholder phases
        # Real implementation uses QSP_block_encoding module
        
        return np.concatenate([
            [coefficients[0] + np.pi/4],
            coefficients[1:-1] + np.pi/2,
            [coefficients[-1] + np.pi/4]
        ])
    
    def _build_QSVT_circuit(self, gs_be: GateSequence, phases: List[float],
                           n_sys: int, n_anc_be: int) -> GateSequence:
        """
        Build complete QSVT circuit.
        
        Structure:
        U_Φ = exp(i·φ₀·Z) · ∏ⱼ₌₁^d [O(x)·exp(i·φⱼ·Z)]
        
        where O(x) involves the block-encoding unitary.
        """
        
        total_qubits = n_sys + n_anc_be + 1
        circuit = GateSequence(total_qubits)
        
        # Apply phases and block-encoding gates
        # Note: Detailed implementation requires QSP module
        
        return circuit
    
    def _extract_solution(self, state: np.ndarray, n_sys: int,
                         b: np.ndarray, kappa: float) -> np.ndarray:
        """
        Extract solution from QSVT output state.
        """
        
        # Extract system register (first 2^n_sys entries)
        sys_state = state[0 : 2**n_sys]
        
        # Apply scaling: multiply by ||b|| · κ · 2
        scaling = np.linalg.norm(b) * kappa * 2
        
        solution = sys_state * scaling
        
        return solution
```

---

## Mathematical Deep Dive

### Singular Value Transformation

For matrix $A$ with SVD $A = U\Sigma V^\dagger$, QSVT applies:

$$\text{QSVT: Poly}^{(SV)}(A) = U \cdot \text{Poly}(\Sigma) \cdot V^\dagger$$

This differs from eigenvalue-based methods which apply to $A^\dagger A$ instead of $A$ directly.

### Chebyshev Approximation

The key polynomial is:
$$f(x) = \frac{1 - (1-x^2)^b}{x}$$

For integer $b \geq (\kappa\alpha)^2 \log(\kappa\alpha/\epsilon)$, this is $\epsilon$-close to $1/x$ on the domain $D_\alpha = [1/(\kappa\alpha), 1/\alpha]$.

**Chebyshev Expansion** (truncated at $j_0 = \sqrt{b\log(4b/\epsilon)}$):

$$f(x) \approx 4\sum_{j=0}^{j_0} (-1)^j \left[\frac{\sum_{i=j+1}^{b}\binom{2b}{b+i}}{2^{2b}}\right] T_{2j+1}(x)$$

where $T_n(x)$ are Chebyshev polynomials of the first kind.

### Quantum Signal Processing Theorem

**QSP Theorem**: For polynomial $P(x)$ and unitary $U$ representing $A$, there exist phases $\Phi = (\phi_0, \phi_1, \ldots, \phi_d)$ such that:

$$U_\Phi(x) = e^{i\phi_0 Z} \prod_{j=1}^{d} [O(x) e^{i\phi_j Z}] = \begin{pmatrix} P(x) & -Q(x)\sqrt{1-x^2} \\ Q^*(x)\sqrt{1-x^2} & P^*(x) \end{pmatrix}$$

### Complexity Analysis

**Gate Complexity:**
$$O\left(\kappa^2 \alpha \sqrt{\log\left(\frac{\kappa^3\|b\|\alpha}{\epsilon}\right)}\right)$$

- Polynomial degree: $O(\sqrt{b\log(b/\epsilon)})$ where $b = \Theta((\kappa\alpha)^2 \log(\kappa\alpha/\epsilon))$
- Gate count for block-encoding: Typically $O(s\log N)$ for sparse matrices
- Amplitude amplification overhead: $O(\kappa)$ factor

**Success Probability:**
$$P_{\text{success}} = \frac{1}{\kappa^2}$$

Can be improved to $\Omega(1)$ using amplitude amplification.

---

## Debugging Tips

1. **Chebyshev coefficients diverging?**
   - Check polynomial order: b ≥ (κα)² log(κα/ε)
   - Verify high-precision arithmetic (mp.dps = 50)
   - Ensure κ and α are positive

2. **Solution vector has wrong scale?**
   - Verify scaling factor: ||b|| · κ · α · 2
   - Check block-encoding amplitude is exactly α
   - Ensure initial state normalization

3. **QSP phase generation failing?**
   - Verify Chebyshev coefficients first
   - Check convergence of optimization
   - Compare with reference implementation

4. **Matrix condition number too large?**
   - QSVT complexity scales with κ²
   - Preconditioning can reduce κ
   - Consider iterative refinement

5. **Non-Hermitian matrix issues?**
   - Use Hadamard sandwich for SVD-based algorithm
   - Verify block-encoding of A†A doesn't lose information
   - Check if A is actually diagonalizable
