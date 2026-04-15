---
name: quantum-signal-processing
description: A quantum algorithm for signal processing tasks, leveraging quantum phase estimation and amplitude amplification techniques to achieve efficient signal analysis and transformation. This skill includes implementations and educational resources for understanding and utilizing quantum signal processing algorithms in various applications.
---

## One Step to Run QSP Example
```bash
python ./scripts/algorithm.py
```

# Quantum Signal Processing (QSP) Skill Guide

## Overview

**Quantum Signal Processing (QSP)** applies polynomial functions to quantum signals via phase sequences and signal operators. For target $P(x)$, QSP:

1. **Encodes the signal** in a unitary operator $W(x)$ (e.g., via rotation angles)
2. **Applies polynomial transformation** using alternating phase gates and signal operators
3. **Implements the polynomial** with depth linear in degree $d$: $O(d)$ gates

### Key Innovation

QSP works directly at circuit level and enables polynomial transformations of encoded eigenvalues/singular values.

### The QSP Framework

The fundamental QSP theorem states that for any polynomial $P(x)$ satisfying certain constraints, there exists a phase sequence $\Phi = (\phi_0, \phi_1, \ldots, \phi_d)$ such that:

$$U_\Phi = e^{i\phi_0 Z} \prod_{k=1}^{d} [W(x) e^{i\phi_k Z}] = \begin{pmatrix} P(x) & -Q(x)\sqrt{1-x^2} \\ Q^*(x)\sqrt{1-x^2} & P^*(x) \end{pmatrix}$$

where:
- $P(x), Q(x)$ are Chebyshev polynomials
- $W(x) = \begin{pmatrix} x & i\sqrt{1-x^2} \\ i\sqrt{1-x^2} & x \end{pmatrix}$ is the signal operator
- The $(0,0)$ block encodes polynomial evaluation: $P(x) = \langle 0|U_\Phi|0\rangle$

### Why QSP Matters

1. Foundation for QSVT and many modern algorithms
2. Direct route to polynomial function evaluation $P(A)$
3. High-precision Hamiltonian simulation primitives

### Real Applications

- Simulating quantum chemistry (molecular dynamics)
- Quantum machine learning (kernel methods, feature maps)
- Linear systems solving (via polynomial inversion)
- Digital quantum simulation of physical systems

---

## Learning Objectives

After mastering this skill, you will be able to:

1. Explain the QSP theorem and the role of $W(x)$
2. Use phase sequences to implement target polynomials
3. Use `QSPAlgorithm` for simulation-oriented tasks
4. Analyze degree/error trade-offs and convergence
5. Apply QSP as a building block for broader algorithms

---

## Prerequisites

- **Essential knowledge**:
    - Quantum gates and circuit construction
    - Complex numbers and Chebyshev polynomials
    - Matrix algebra and eigenvalues
    - Polynomial approximation theory
    - Basic optimization (for phase fitting)

---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms.linear_algebra import QSPAlgorithm
import numpy as np

# Step 1: Create QSP solver instance
algo = QSPAlgorithm(seed=42)

# Step 2: Run QSP for Hamiltonian simulation
# Target: Approximate exp(-i·τ·x) where τ is evolution time
# x: eigenvalue between -1 and 1

result = algo.run(
    target_tau=1.2,        # Evolution parameter τ
    degree=14,             # Polynomial degree d
    x_value=0.7,           # Test eigenvalue x ∈ [-1,1]
    backend='torch',       # Simulation backend
    algo_dir='./qsp_results'
)

# Step 3: Inspect results
print(result['plot'])
print(f"Approximation error: {result['error']:.6e}")
print(f"Circuit path: {result['circuit_path']}")

# Step 4: Access detailed information
print(f"Simulated value: {result_old['qsp_val']}")
print(f"Ideal value: {result['ideal_val']}")
```

### Core Parameters Explained

```python
algo.run(
    target_tau: float,              # Evolution parameter τ (controls function: e^{-iτx})
    degree: int,                    # Polynomial approximation degree d
    x_value: float = 0.5,           # Test point (eigenvalue in [-1,1])
    backend: str = 'torch',         # Simulation backend
    algo_dir: str = './qsp_results' # Output directory
)
```

**Return dictionary contains:**
- `error`: Approximation error at test point $|P(x) - e^{-i\tau x}|^2$
- `circuit`: Complete QSP quantum circuit
- `circuit_path`: Location of circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

### Accuracy and Convergence

For approximating $e^{-i\tau x}$ using degree-$d$ polynomial:
- **Error scaling**: $O(\exp(-d))$ (exponential convergence!)
- **Required degree**: $d \approx \tau$ for $\epsilon$-accuracy
- **Gate complexity**: $O(\tau)$ gates

---

## Understanding the Core Components

### 1. The Five-Step QSP Pipeline

```python
def run(self, target_tau, degree, x_value=0.5):
    """
    Complete QSP algorithm pipeline:
    
    Step 1: Optimize phase sequence Φ for target polynomial
    Step 2: Construct quantum circuit from phases
    Step 3: Execute quantum simulation
    Step 4: Verify approximation error
    Step 5: Export circuit diagram
    """
    
    # ========== Stage 1: Phase Optimization ==========
    # Goal: Find phases Φ = (φ₀, φ₁, ..., φ_d) such that
    # Φ-controlled circuit approximates exp(-i·τ·x)
    
    x_samples = np.linspace(-1, 1, 2*d+1)  # Chebyshev sample points
    
    def target_function(x):
        """Target polynomial: exp(-i·τ·x)"""
        return np.exp(-1j * target_tau * x)
    
    def loss_function(phases):
        """Minimize approximation error over sample points"""
        error = 0.0
        for x in x_samples:
            simulated = simulate_circuit(phases, x)
            ideal = target_function(x)
            error += |simulated - ideal|²
        return error / len(x_samples)
    
    # Optimize phases using L-BFGS-B
    optimized_phases = minimize(loss_function, 
                                initial_guess=random_phases,
                                method='L-BFGS-B')
    phases = optimized_phases.x
    
    # ========== Stage 2: Circuit Construction ==========
    # Build QSP circuit: Rz(2φ₀) → [W(x) → Rz(2φ₁)] → ...
    
    circuit = GateSequence(1)
    
    # Initial phase rotation
    circuit.rz(2 * phases[0], qubit=0)
    
    # Alternating signal operator and phase gates
    for k in range(1, degree + 1):
        # Signal operator: rotation by 2·arccos(x)
        theta = arccos(clip(x_value, -1, 1))
        circuit.rx(2 * theta, qubit=0)  # RX approximates W(x) operation
        
        # Phase rotation
        circuit.rz(2 * phases[k], qubit=0)
    
    # ========== Stage 3: Execution ==========
    final_state = circuit.execute()
    
    # ========== Stage 4: Error Analysis ==========
    simulated_value = final_state[0]
    ideal_value = np.exp(-1j * target_tau * x_value)
    error = |simulated_value - ideal_value|²
    
    # ========== Stage 5: Export ==========
    circuit.draw(filename=circuit_path)
    
    return error, circuit
```

### 2. The Signal Operator W(x)

The signal operator encodes the input value $x$ as a unitary:

```python
def signal_operator(self, x: float) -> np.ndarray:
    """
    Quantum signal operator: W(x)
    
    For x ∈ [-1,1]:
    W(x) = [[x, i√(1-x²)],
            [i√(1-x²), x]]
    
    Satisfies: W(x)† = W(x), det(W(x)) = 1 (unitary)
    
    Eigenvalues: λ₀ = 1, λ₁ = 1 (both equal to 1):
    W(x) encodes x in angle: θ = arccos(x)
    """
    
    # Clip x to valid range
    x_clipped = np.clip(x, -1.0, 1.0)
    
    # Calculate off-diagonal element
    sqrt_term = np.sqrt(1 - x_clipped**2)
    
    # Construct 2×2 matrix
    W = np.array([
        [x_clipped, 1j * sqrt_term],
        [1j * sqrt_term, x_clipped]
    ], dtype=complex)
    
    return W

def circuit_representation_of_W(self, x: float, qubit: int) -> GateSequence:
    """
    Implement W(x) on quantum circuit using RX rotation.
    
    Key insight: W(x) can be decomposed as:
    W(x) = Rz(φ) · Rx(2·arccos(x)) · Rz(-φ)  for suitable φ
    
    In our implementation, we use RX(2·arccos(x)) directly:
    RX(θ) = [[cos(θ/2), -i·sin(θ/2)],
             [-i·sin(θ/2), cos(θ/2)]]
    """
    
    circuit = GateSequence(1)
    theta = np.arccos(np.clip(x, -1, 1))
    circuit.rx(2 * theta, qubit)  # Apply RX gate
    
    return circuit
```

### 3. Phase Optimization via Loss Minimization

```python
def optimize_phases(self, target_tau: float, degree: int) -> np.ndarray:
    """
    Optimize phase sequence Φ = (φ₀, φ₁, ..., φ_d) to minimize
    approximation error for target function exp(-i·τ·x).
    
    Key Points:
    - Sample at Chebyshev points: x_k = cos((2k-1)π/(4d+2))
    - Use high-precision optimization (L-BFGS-B)
    - Initialize with small random values to break symmetry
    """
    
    # Chebyshev sampling points (optimal for polynomial approximation)
    num_samples = 2 * degree + 1
    x_samples = np.linspace(-1, 1, num_samples)
    
    def matrix_of_phases(phases, x):
        """
        Compute matrix product for QSP at point x with given phases.
        
        U_Φ(x) = Rz(φ₀) · [W(x) · Rz(φ₁)] · ...
        """
        
        # Signal operator
        W_x = self.signal_operator(x)
        
        # Initial phase
        Rz_init = np.array([
            [np.exp(1j * phases[0]), 0],
            [0, np.exp(-1j * phases[0])]
        ], dtype=complex)
        
        U = Rz_init
        
        # Alternating applications
        for k in range(1, len(phases)):
            U = U @ W_x @ np.array([
                [np.exp(1j * phases[k]), 0],
                [0, np.exp(-1j * phases[k])]
            ], dtype=complex)
        
        return U
    
    def get_p_value(phases, x):
        """Extract P(x) from (0,0) element of U_Φ(x)"""
        U = matrix_of_phases(phases, x)
        return U[0, 0]
    
    def loss_function(phases):
        """Mean squared error over all sample points"""
        total_error = 0.0
        for x in x_samples:
            simulated = get_p_value(phases, x)
            target = np.exp(-1j * target_tau * x)
            total_error += np.abs(simulated - target) ** 2
        
        return total_error / len(x_samples)
    
    # Optimize using L-BFGS-B
    initial_guess = np.random.randn(degree + 1) * 0.1
    result = minimize(
        loss_function,
        initial_guess,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    
    return result.x
```

### 4. Quantum Circuit Construction

```python
def build_qsp_circuit(self, phases: np.ndarray, x_value: float, 
                      degree: int, backend: str = 'torch') -> GateSequence:
    """
    Construct complete QSP quantum circuit.
    
    Circuit structure:
    |ψ⟩ → Rz(2φ₀) → [Rx(2θ) → Rz(2φ₁)] → ... → [Rx(2θ) → Rz(2φ_d)]
    
    where θ = arccos(x_value)
    """
    
    circuit = GateSequence(1, backend=backend)
    
    # Calculate rotation angle from x value
    theta = np.arccos(np.clip(x_value, -1.0, 1.0))
    
    # Initial phase gate
    circuit.rz(2 * phases[0], qubit=0)
    
    # Alternating signal operators (RX) and phase gates (RZ)
    for k in range(1, len(phases)):
        # Signal operator approximation: Rx(2θ)
        circuit.rx(2 * theta, qubit=0)
        
        # Phase gate
        circuit.rz(2 * phases[k], qubit=0)
    
    return circuit
```

---

## Hands-On Example: Approximating exp(-iτx)

Let's implement QSP for Hamiltonian simulation:

```python
from algorithm import QSPAlgorithm
import numpy as np

# Create QSP solver
algo = QSPAlgorithm(seed=42)

# Example 1: Short-time evolution (small τ)
print("=" * 60)
print("Example 1: Short-time evolution (τ = 0.5)")
print("=" * 60)

result_short = algo.run(
    target_tau=0.5,        # Small evolution time
    degree=6,              # Low degree sufficient
    x_value=0.3,           # Low eigenvalue
    backend='torch'
)

print(result_short['plot'])
print(f"\nError for short evolution: {result_short['error']:.8e}")

# Example 2: Long-time evolution (large τ)
print("\n" + "=" * 60)
print("Example 2: Long-time evolution (τ = 2.0)")
print("=" * 60)

result_long = algo.run(
    target_tau=2.0,        # Large evolution time
    degree=20,             # Higher degree needed
    x_value=0.8,           # Higher eigenvalue
    backend='torch'
)

print(result_long['plot'])
print(f"\nError for long evolution: {result_long['error']:.8e}")

# Example 3: Varying degree
print("\n" + "=" * 60)
print("Example 3: Error vs. approximation degree")
print("=" * 60)

degrees = [4, 8, 12, 16, 20]
errors = []

for d in degrees:
    result = algo.run(
        target_tau=1.5,
        degree=d,
        x_value=0.5,
        backend='torch'
    )
    errors.append(result['error'])
    print(f"Degree {d}: Error = {result['error']:.6e}")

# Analyze convergence rate
print("\nConvergence analysis:")
for i in range(1, len(errors)):
    ratio = errors[i-1] / errors[i]
    print(f"  Error({degrees[i-1]}/{degrees[i]}) = {ratio:.2f}")
```

---

## Implementing QSP From Scratch

### Complete Implementation Template

```python
import numpy as np
from typing import List, Tuple, Callable
from scipy.optimize import minimize
from engine.core import GateSequence

class MyQSP:
    """Educational QSP implementation."""
    
    def approximate_function(self, target_func: Callable[[float], complex],
                            degree: int, 
                            x_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Approximate target_func using QSP of given degree.
        
        Args:
            target_func: Function to approximate, e.g., lambda x: np.exp(-1j*tau*x)
            degree: Polynomial approximation degree
            x_samples: Sample points for optimization (typically Chebyshev points)
        
        Returns:
            (optimized_phases, final_error)
        """
        
        # STEP 1: Define loss function
        def loss(phases):
            """Mean squared error over all sample points"""
            total_error = 0.0
            
            for x in x_samples:
                # Simulate QSP at this point
                simulated = self._qsp_evaluation(phases, x)
                
                # Compare with target
                target = target_func(x)
                
                # Accumulate squared error
                total_error += np.abs(simulated - target) ** 2
            
            return total_error / len(x_samples)
        
        # STEP 2: Optimize phases
        initial_phases = np.random.randn(degree + 1) * 0.1
        
        result = minimize(
            loss,
            initial_phases,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        optimized_phases = result.x
        final_error = loss(optimized_phases)
        
        return optimized_phases, final_error
    
    def _qsp_evaluation(self, phases: np.ndarray, x: float) -> complex:
        """
        Evaluate QSP at point x with given phase sequence.
        
        Returns the (0,0) element of:
        U_Φ = Rz(φ₀) · [W(x) · Rz(φ₁)] · [W(x) · Rz(φ₂)] · ...
        """
        
        # Signal operator
        W = self._signal_operator(x)
        
        # Initialize with Rz(φ₀)
        U = np.diag([np.exp(1j * phases[0]), np.exp(-1j * phases[0])])
        
        # Alternating W and Rz
        for k in range(1, len(phases)):
            Rz = np.diag([np.exp(1j * phases[k]), np.exp(-1j * phases[k])])
            U = U @ W @ Rz
        
        # Return (0,0) element (this is P(x))
        return U[0, 0]
    
    def _signal_operator(self, x: float) -> np.ndarray:
        """
        Signal operator W(x) for x ∈ [-1,1].
        
        W(x) = [[x, i√(1-x²)],
                [i√(1-x²), x]]
        """
        
        x = np.clip(x, -1.0, 1.0)
        sqrt_term = np.sqrt(1.0 - x**2)
        
        return np.array([
            [x, 1j * sqrt_term],
            [1j * sqrt_term, x]
        ], dtype=complex)
    
    def construct_circuit(self, phases: np.ndarray, x: float) -> GateSequence:
        """
        Build quantum circuit for QSP.
        """
        
        circuit = GateSequence(1)
        
        # Initial phase
        circuit.rz(2 * phases[0], 0)
        
        # Alternating gates
        theta = np.arccos(np.clip(x, -1.0, 1.0))
        
        for k in range(1, len(phases)):
            circuit.rx(2 * theta, 0)
            circuit.rz(2 * phases[k], 0)
        
        return circuit
    
    def analyze_convergence(self, target_func: Callable[[float], complex]):
        """
        Analyze how error decreases with polynomial degree.
        """
        
        degrees = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        errors = []
        
        for d in degrees:
            # Chebyshev sample points
            x_samples = np.linspace(-1, 1, 2*d+1)
            
            phases, error = self.approximate_function(target_func, d, x_samples)
            errors.append(error)
            
            print(f"Degree {d:2d}: Error = {error:.6e}")
        
        return degrees, errors
```

---

## Mathematical Deep Dive

### QSP Theorem

For any polynomial $P(x)$ with $\|P\|_\infty \leq 1$ on $[-1,1]$, there exists a phase sequence $\Phi = (\phi_0, \phi_1, \ldots, \phi_d)$ such that:

$$U_\Phi(x) = e^{i\phi_0 Z} \prod_{j=1}^{d} [W(x) e^{i\phi_j Z}]$$

produces the block form:
$$U_\Phi = \begin{pmatrix} P(x) & -Q(x)\sqrt{1-x^2} \\ Q^*(x)\sqrt{1-x^2} & P^*(x) \end{pmatrix}$$

where $Q(x)$ is another polynomial determined by $P(x)$.

### Signal Operator Properties

The signal operator $W(x) = \begin{pmatrix} x & i\sqrt{1-x^2} \\ i\sqrt{1-x^2} & x \end{pmatrix}$:

1. **Unitarity**: $W(x)^\dagger W(x) = I$ for all $x \in [-1,1]$
2. **Eigenvalues**: Both eigenvalues equal 1
3. **Diagonal form**: Can be diagonalized with unitary $V$:
   $$W(x) = V \begin{pmatrix} e^{i\theta} & 0 \\ 0 & e^{-i\theta} \end{pmatrix} V^\dagger$$
   where $\theta = \arccos(x)$

### Approximation Error

For approximating $e^{-i\tau x}$ using degree-$d$ polynomial:

**Error bound:**
$$\max_{x \in [-1,1]} |P_d(x) - e^{-i\tau x}| \leq C \cdot e^{-d/D}$$

where $C, D$ are constants depending on $\tau$.

**Convergence rate:** Exponential in degree $d$ (much better than polynomial!)

### Chebyshev Points

Optimal sampling for polynomial approximation:
$$x_k = \cos\left(\frac{(2k-1)\pi}{2n}\right), \quad k = 1, 2, \ldots, n$$

These minimize Runge phenomena and provide optimal interpolation.
