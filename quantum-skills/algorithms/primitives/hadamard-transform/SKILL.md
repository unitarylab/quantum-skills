---
name: hadamard-transform
description: A quantum algorithm for performing the Hadamard transform, which is a fundamental operation in quantum computing that creates superposition states. This skill includes efficient implementations and educational resources for understanding and utilizing the Hadamard transform in various quantum algorithms and applications.
--- 

## One Step to Run Hadamard Transform Example
```bash
python ./scripts/algorithm.py
```
# Hadamard Transform Skill Guide

## Overview

**The Hadamard Transform** is one of the most fundamental and widely-used quantum gates. It:

1. **Creates equal superposition**: Transforms $|0\rangle \rightarrow \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$
2. **Is self-inverse**: $H^2 = I$ (applying it twice returns to original state)
3. **Basis changes**: Maps computational basis $\{|0\rangle, |1\rangle\}$ to $\{|+\rangle, |-\rangle\}$
4. **Enables quantum parallelism**: $H^{\otimes n}|0\rangle^{\otimes n}$ creates superposition of all $2^n$ basis states

### The Single-Qubit Hadamard Matrix

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Key Properties:**
- $H|0\rangle = |+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
- $H|1\rangle = |-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$
- $H^2 = I$ (self-adjoint and involutory)
- $H^\dagger = H$ (Hermitian)

### The Multi-Qubit Hadamard Transform

For $n$ qubits, the global Hadamard transform is:
$$H^{\otimes n} = \underbrace{H \otimes H \otimes \cdots \otimes H}_{n \text{ times}}$$

Applied to the initial state $|0\rangle^{\otimes n}$:
$$H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle$$

This creates a **uniform superposition** of all possible $n$-qubit basis states with equal amplitude $\frac{1}{\sqrt{2^n}}$.

### Why Hadamard Matters

1. **Foundation for quantum algorithms**: Grover's algorithm, quantum Fourier transform, most quantum algorithms
2. **Superposition creation**: Essential for quantum advantage
3. **Efficient**: Only $O(n)$ gates for $n$ qubits
4. **Quantum parallelism**: Encodes information in amplitude relationships

### Real Applications

- **Grover's Algorithm**: Initial superposition creation
- **Quantum Fourier Transform**: Key component in period-finding
- **Phase Estimation**: State preparation
- **Quantum Machine Learning**: Feature map creation
- **Random number generation**: Creates uniform probability distribution

---

## Learning Objectives

After mastering this skill, you will be able to:

1. Understand single-qubit Hadamard gate and its matrix form
2. Explain superposition creation: $|+\rangle, |-\rangle$ states
3. Understand multi-qubit Hadamard transform $H^{\otimes n}$
4. Grasp the self-inverse property: $H^2 = I$
5. Create uniform superposition of all basis states
6. Use the provided `HadamardTransformAlgorithm` class
7. Verify property-based testing (reflexive property)
8. Implement Hadamard from scratch
9. Understand amplitude relationships and phase patterns
10. Apply Hadamard in quantum algorithm design

---

## Prerequisites

- **Essential knowledge**:
  - Basic quantum mechanics (kets, bras, superposition)
  - Matrix operations and linear algebra
  - Complex numbers
  - Basic quantum gates (Pauli gates)
- **Recommended**: Familiarity with quantum computing basics (qubits, measurement)
- **Mathematical comfort**: Linear algebra, matrix multiplication

---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms import HadamardTransformAlgorithm
import numpy as np

# Step 1: Create Hadamard Transform solver
algo = HadamardTransformAlgorithm()

# Step 2: Generate uniform superposition of 3 qubits
result = algo.run(
    n_qubits=3,           # Number of qubits
    mode='superposition', # Create uniform superposition
    backend='torch'       # Simulation backend
)

# Step 3: Inspect results
print(result['plot'])
print(f"Status: {result['status']}")
print(f"Probabilities: {result['probabilities']}")

# Step 4: Verify self-inverse property
result_reflexive = algo.run(
    n_qubits=2,
    mode='reflexive_test', # Test that H² = I
    backend='torch'
)

print(result_reflexive['plot'])
```

### Core Parameters Explained

```python
algo.run(
    n_qubits: int = 3,              # Number of qubits (default 3)
    mode: str = "superposition",    # "superposition" or "reflexive_test"
    backend: str = 'torch',         # Simulation backend
    algo_dir: str = './hadamard_results'  # Output directory
)
```

**Parameters:**
- `n_qubits`: How many qubits to transform (dimension: $2^n$)
- `mode`: 
  - `"superposition"`: Create $\frac{1}{\sqrt{2^n}}\sum_{x} |x\rangle$
  - `"reflexive_test"`: Verify $H^2 = I$ on random state
- `backend`: 'torch' for PyTorch-based simulation

**Return dictionary contains:**
- `state_vector`: Final quantum state amplitudes
- `probabilities`: Probability distribution over basis states
- `circuit_path`: Location of circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

---

## Understanding the Core Components

### 1. Single-Qubit Hadamard Gate

```python
def single_qubit_hadamard():
    """
    The fundamental single-qubit Hadamard gate.
    
    Matrix form:
    H = (1/√2) * [[1,  1],
                   [1, -1]]
    
    Properties:
    - Hermitian: H† = H
    - Unitary: H†H = I
    - Involutory: H² = I (self-inverse)
    - Determinant: det(H) = -1
    """
    
    H = (1/np.sqrt(2)) * np.array([
        [1, 1],
        [1, -1]
    ], dtype=complex)
    
    # Verify properties
    assert np.allclose(H @ H, np.eye(2)), "H² should equal I"
    assert np.allclose(H.conj().T, H), "H should be Hermitian"
    
    return H

# Apply to basis states
H = single_qubit_hadamard()
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

plus_state = H @ ket0  # |+⟩ = (|0⟩ + |1⟩)/√2
minus_state = H @ ket1  # |-⟩ = (|0⟩ - |1⟩)/√2

print(f"|+⟩ = {plus_state}")   # [0.707, 0.707]
print(f"|-⟩ = {minus_state}")  # [0.707, -0.707]
```

### 2. Multi-Qubit Hadamard Transform

```python
def multi_qubit_hadamard(n_qubits: int) -> np.ndarray:
    """
    Construct the full n-qubit Hadamard transform.
    
    H⊗n = H ⊗ H ⊗ ... ⊗ H  (n times)
    
    Key insight: For computational efficiency, we apply
    Hadamard gates individually rather than constructing
    the full 2^n × 2^n matrix.
    
    Applied to |0⟩^⊗n:
    H⊗n |0...0⟩ = (1/√(2^n)) Σ_{x=0}^{2^n-1} |x⟩
    
    This creates equal superposition of all basis states.
    """
    
    # Single qubit Hadamard
    H1 = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    
    # Build tensor product iteratively
    H_n = H1.copy()
    for _ in range(n_qubits - 1):
        H_n = np.kron(H_n, H1)
    
    return H_n / np.sqrt(2 ** (n_qubits - 1))

# Create superposition
n = 3
H3 = multi_qubit_hadamard(n)
initial_state = np.zeros(2**n)
initial_state[0] = 1.0  # |0⟩^⊗3

superposition = H3 @ initial_state

print(f"Superposition state amplitudes:")
print(superposition)  # All equal to 1/√8 ≈ 0.3536

print(f"Probabilities (all equal):")
probs = np.abs(superposition) ** 2
print(probs)  # All equal to 0.125 (1/8)
```

### 3. Applying Hadamard on Quantum Circuit

```python
def apply_hadamard_circuit(n_qubits: int, 
                           initial_state: Optional[np.ndarray] = None) -> 'GateSequence':
    """
    Build quantum circuit with Hadamard gates.
    
    Steps:
    1. Initialize qubits (default to |0⟩^⊗n)
    2. Apply H gate to each qubit individually
    3. Execute and measure
    """
    
    circuit = GateSequence(n_qubits, name=f'Hadamard_n{n_qubits}')
    
    # Initialize with custom state if provided
    if initial_state is not None:
        circuit.initialize(initial_state, target=list(range(n_qubits)))
    
    # Apply Hadamard to each qubit
    for qubit in range(n_qubits):
        circuit.h(qubit)  # Single-qubit Hadamard
    
    return circuit
```

### 4. Superposition Creation Example

```python
def create_superposition(n_qubits: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Create uniform superposition of all 2^n basis states.
    
    Expected result:
    |ψ⟩ = (1/√(2^n)) Σ_{x=0}^{2^n-1} |x⟩
    
    Probability of measuring any basis state: 1/2^n
    """
    
    circuit = GateSequence(n_qubits)
    
    # Apply Hadamard to all qubits
    for q in range(n_qubits):
        circuit.h(q)
    
    # Execute simulation
    state = np.asarray(circuit.execute(), dtype=complex)
    
    # Calculate probability distribution
    probs = np.abs(state) ** 2
    prob_dict = {}
    
    for idx in range(2**n_qubits):
        if probs[idx] > 1e-12:
            binary_str = format(idx, f'0{n_qubits}b')
            prob_dict[binary_str] = float(probs[idx])
    
    return state, prob_dict

# Example: 2-qubit superposition
state_2, probs_2 = create_superposition(2)
print("2-qubit superposition:")
print("State:", np.round(state_2, 3))  # [0.5, 0.5, 0.5, 0.5]
print("Probs:", probs_2)  # {00: 0.25, 01: 0.25, 10: 0.25, 11: 0.25}
```

### 5. Verifying Self-Inverse Property

```python
def verify_hadamard_involution(n_qubits: int, 
                               test_state: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Verify that H² = I by applying Hadamard twice.
    
    For any state |ψ⟩:
    (H⊗n)² |ψ⟩ = |ψ⟩
    
    This property is fundamental and used in many algorithms.
    """
    
    # Generate random test state if not provided
    if test_state is None:
        rng = np.random.default_rng()
        test_state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
        test_state = test_state / np.linalg.norm(test_state)
    
    # Build circuit: initialize → H → H
    circuit = GateSequence(n_qubits)
    circuit.initialize(test_state, target=list(range(n_qubits)))
    
    # Apply Hadamard twice
    for _ in range(2):
        for q in range(n_qubits):
            circuit.h(q)
    
    # Execute and retrieve final state
    final_state = np.asarray(circuit.execute(), dtype=complex)
    
    # Check if returned to original
    is_recovered = np.allclose(final_state, test_state, atol=1e-10)
    max_deviation = np.max(np.abs(final_state - test_state))
    
    return is_recovered, max_deviation
```

---

## Hands-On Examples

### Example 1: Creating Single-Qubit Superposition

```python
from algorithm import HadamardTransformAlgorithm
import numpy as np

algo = HadamardTransformAlgorithm()

# Create |+⟩ = (|0⟩ + |1⟩)/√2
result = algo.run(n_qubits=1, mode='superposition')

print("=" * 60)
print("Example 1: Single-Qubit Superposition")
print("=" * 60)
print(result['plot'])

expected_prob = 0.5
actual_probs = result['probabilities']

print(f"\nTheory: P(0) = P(1) = {expected_prob}")
print(f"Measured: P(0) = {actual_probs.get('0', 0):.4f}, P(1) = {actual_probs.get('1', 0):.4f}")
```

### Example 2: Multi-Qubit Uniform Superposition

```python
# Create uniform superposition of 4 basis states
result_2 = algo.run(n_qubits=2, mode='superposition')

print("\n" + "=" * 60)
print("Example 2: Two-Qubit Uniform Superposition")
print("=" * 60)
print(result_2['plot'])

expected_prob = 0.25
print(f"\nAll 4 basis states should have probability {expected_prob}")
for state, prob in result_2['probabilities'].items():
    print(f"  |{state}⟩: {prob:.4f}")
```

### Example 3: Verifying H² = I

```python
# Test on random 3-qubit state
result_3 = algo.run(n_qubits=3, mode='reflexive_test')

print("\n" + "=" * 60)
print("Example 3: Reflexive Property H² = I")
print("=" * 60)
print(result_3['plot'])

print("\nExplanation:")
print("- A random 3-qubit state was initialized")
print("- Hadamard was applied twice: H² ")
print("- The final state should equal the initial state")
print("- Deviation shows numerical precision of simulation")
```

### Example 4: Batch Testing with Different Sizes

```python
print("\n" + "=" * 60)
print("Example 4: Varying Qubit Count")
print("=" * 60)

for n in [1, 2, 3, 4]:
    result = algo.run(n_qubits=n, mode='superposition')
    expected_prob = 1 / (2**n)
    
    print(f"\nn = {n} qubits:")
    print(f"  Expected probability per state: {expected_prob:.6f}")
    print(f"  Total states created: {len(result['probabilities'])}")
    
    # Verify uniformity
    probs = list(result['probabilities'].values())
    is_uniform = all(np.isclose(p, expected_prob, atol=1e-5) for p in probs)
    print(f"  Uniformity check: {'✓ PASS' if is_uniform else '✗ FAIL'}")
```

---

## Implementing Hadamard From Scratch

### Complete Implementation Template

```python
import numpy as np
from typing import Dict, Tuple, List, Optional
from engine.core import GateSequence

class MyHadamardTransform:
    """Educational Hadamard Transform implementation."""
    
    def create_hadamard_matrix(self) -> np.ndarray:
        """
        Construct single-qubit Hadamard matrix.
        
        H = (1/√2) * [[1,  1],
                       [1, -1]]
        """
        return (1 / np.sqrt(2)) * np.array([
            [1.0, 1.0],
            [1.0, -1.0]
        ], dtype=complex)
    
    def create_n_qubit_hadamard(self, n: int) -> np.ndarray:
        """
        Construct n-qubit Hadamard transform: H⊗n
        
        Uses tensor product (Kronecker product).
        """
        H1 = self.create_hadamard_matrix()
        
        # Start with single-qubit
        H_n = H1.copy()
        
        # Iteratively tensor with more Hadamards
        for _ in range(n - 1):
            H_n = np.kron(H_n, H1)
        
        return H_n
    
    def apply_hadamard_to_state(self, state: np.ndarray, 
                                n_qubits: int) -> np.ndarray:
        """
        Apply n-qubit Hadamard to quantum state.
        
        Args:
            state: State vector of dimension 2^n_qubits
            n_qubits: Number of qubits
        
        Returns:
            Transformed state vector
        """
        H_n = self.create_n_qubit_hadamard(n_qubits)
        return H_n @ state
    
    def create_superposition(self, n_qubits: int) -> np.ndarray:
        """
        Create uniform superposition of all basis states.
        
        H^⊗n |0...0⟩ = (1/√(2^n)) Σ_{x=0}^{2^n-1} |x⟩
        """
        # Initial state: |0...0⟩
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply Hadamard
        return self.apply_hadamard_to_state(state, n_qubits)
    
    def build_circuit(self, n_qubits: int, mode: str = 'superposition',
                     initial_state: Optional[np.ndarray] = None) -> GateSequence:
        """
        Build Hadamard circuit.
        
        Args:
            n_qubits: Number of qubits
            mode: 'superposition' or 'double' (for reflexive test)
            initial_state: Optional custom initial state
        
        Returns:
            Constructed GateSequence
        """
        circuit = GateSequence(n_qubits, name=f'Hadamard_{mode}')
        
        # Initialize with custom state if provided
        if initial_state is not None:
            circuit.initialize(initial_state, target=list(range(n_qubits)))
        
        # Apply Hadamard gates
        for q in range(n_qubits):
            circuit.h(q)
        
        # For reflexive test, apply again
        if mode == 'double':
            for q in range(n_qubits):
                circuit.h(q)
        
        return circuit
    
    def verify_involution(self, n_qubits: int) -> Tuple[bool, float]:
        """
        Verify H² = I on a random state.
        """
        # Generate random state
        rng = np.random.default_rng(42)
        psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
        psi = psi / np.linalg.norm(psi)
        
        # Apply H twice
        circuit = self.build_circuit(n_qubits, mode='double', initial_state=psi)
        final_state = np.asarray(circuit.execute(), dtype=complex)
        
        # Check recovery
        is_recovered = np.allclose(final_state, psi, atol=1e-10)
        deviation = np.max(np.abs(final_state - psi))
        
        return is_recovered, deviation
    
    def calculate_probabilities(self, state_vector: np.ndarray,
                               n_qubits: int) -> Dict[str, float]:
        """
        Calculate measurement probabilities from state vector.
        
        Returns dictionary mapping binary strings to probabilities.
        """
        probs = np.abs(state_vector) ** 2
        prob_dict = {}
        
        for idx in range(2**n_qubits):
            if probs[idx] > 1e-12:
                binary_str = format(idx, f'0{n_qubits}b')
                prob_dict[binary_str] = float(probs[idx])
        
        return dict(sorted(prob_dict.items()))
    
    def run(self, n_qubits: int, mode: str = 'superposition') -> Dict:
        """
        Execute Hadamard transform and return results.
        """
        
        if mode == 'superposition':
            state = self.create_superposition(n_qubits)
            probs = self.calculate_probabilities(state, n_qubits)
            
            expected_prob = 1 / (2**n_qubits)
            is_uniform = all(np.isclose(p, expected_prob, atol=1e-5) for p in probs.values())
            
            return {
                'state': state,
                'probabilities': probs,
                'is_success': is_uniform,
                'message': 'Uniform superposition created'
            }
        
        elif mode == 'reflexive_test':
            circuit = self.build_circuit(n_qubits, mode='double')
            state = np.asarray(circuit.execute(), dtype=complex)
            
            # Generate test state for comparison
            rng = np.random.default_rng()
            test_state = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
            test_state = test_state / np.linalg.norm(test_state)
            
            is_recovered, deviation = self.verify_involution(n_qubits)
            
            return {
                'state': state,
                'is_success': is_recovered,
                'deviation': deviation,
                'message': f'H² = I verified (deviation: {deviation:.2e})'
            }
```

---

## Mathematical Properties

### The Hadamard Matrix Properties

1. **Unitarity**: $H^\dagger H = HH^\dagger = I$
2. **Hermitian**: $H^\dagger = H$
3. **Involution**: $H^2 = I$ (applying twice gives identity)
4. **Eigenvalues**: $\lambda_1 = 1, \lambda_2 = -1$
5. **Trace**: $\text{Tr}(H) = 0$
6. **Determinant**: $\det(H) = -1$

### State Transformations

**Input-Output pairs**:
- $H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$
- $H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle$
- $H|+\rangle = |0\rangle$
- $H|-\rangle = |1\rangle$

### Tensor Product Formula

$$H^{\otimes n} = \bigotimes_{i=1}^{n} H = \underbrace{H \otimes H \otimes \cdots \otimes H}_{n \text{ times}}$$

**Dimension**: $2^n \times 2^n$

**Sparsity**: Only 2 non-zero entries per row (highly structured!)

---

## Debugging Tips

1. **Superposition not uniform?**
   - Check all qubits get Hadamard applied
   - Verify qubit indexing (might be off by one)
   - Ensure initial state is exactly |0...0⟩
   - Check for floating-point errors (use atol in comparison)

2. **H² = I test failing?**
   - Verify circuit applies Hadamard exactly twice
   - Check state normalization before and after
   - Use high-precision comparison (atol=1e-10)
   - Ensure two operations are independent (not collapsed)

3. **Wrong amplitude values?**
   - Should see ±1/√(2^n) for all amplitudes in superposition
   - Check sign patterns (related to parity)
   - Verify phase is 0 or π (real or negative real)

4. **Circuit construction issues?**
   - Ensure qubits are indexed correctly
   - Verify gate application order (shouldn't matter for H)
   - Check backend is set correctly
   - Compare with manual matrix multiplication

5. **Numerical precision problems?**
   - Use `np.allclose()` with appropriate tolerance
   - For state vector: atol=1e-10
   - For probabilities: atol=1e-5
   - Use 64-bit floats (default in NumPy)

---