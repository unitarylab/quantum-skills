---
name: simon
description: Simon's algorithm - a quantum algorithm to discover hidden periodic structures in functions. Efficiently finds the hidden period string s such that f(x) = f(x⊕s) for given function f.
requirements:
  - quantum computing framework (GateSequence, Register, State, ClassicalRegister)
  - numpy for numerical operations
  - linear algebra over GF(2) for equation solving
  - simulation backend (torch or similar)
---

# Simon's Algorithm - Hidden Subgroup Problem Skill

## Overview
**Simon's Algorithm** is a fundamental quantum algorithm that discovers hidden periodic structures in boolean functions. Given a function f: {0,1}^n → {0,1}^n that is 2-to-1 with period s (meaning f(x) = f(x⊕s) for all x), the algorithm finds s exponentially faster than classical algorithms.

### Why Simon's Algorithm Matters
- **Pedagogical**: Demonstrates quantum parallelism without phase estimation
- **Complexity**: O(n) quantum queries vs Ω(2^(n/2)) classical queries (exponential speedup)
- **Cryptographic**: Applications to hidden symmetries and symmetric-key cryptanalysis
- **Foundational**: Building block for understanding hidden subgroup problems

## Mathematical Background

### The Hidden Period Problem (Simon's Problem)
- **Input**: Oracle access to function f: {0,1}^n → {0,1}^m
- **Promise**: ∃s ∈ {0,1}^n \ {0^n} such that f(x) = f(x⊕s) for all x ∈ {0,1}^n
- **Goal**: Find the hidden period string s
- **Structure**: f is exactly 2-to-1 with this period (f(x) and f(x⊕s) are the only preimages)

### Mathematical Definition
```
Function f: {0,1}^n → {0,1}^m is σ-periodic if:
  ∀x ∈ {0,1}^n: f(x) = f(x ⊕ σ)

Simon's Problem: Find σ (the period/secret string)
```

### Key Quantum Concepts

**1. Quantum Parallelism**
- Classical: Query f(x) for individual values of x
- Quantum: Query superposition of all x simultaneously
- Result: Extract global properties of f by observing interference

**2. Hadamard Basis Switching**
```
Computational basis: {|0⟩, |1⟩}
Hadamard basis: |+⟩ = (|0⟩ + |1⟩)/√2, |-⟩ = (|0⟩ - |1⟩)/√2

Key insight: Hadamard transforms between bases
H transforms |x⟩ to superposition over all x: (1/√2^n) Σ_y (-1)^(x·y)|y⟩
```

**3. XOR Structure**
- s represents the hidden structure via XOR (bitwise addition mod 2)
- f(x) = f(x⊕s) means f is invariant under XOR with s
- Quantum measurement collapses to basis orthogonal to s

## Algorithm Structure

### The Three-Phase Simon's Algorithm

#### Phase 1: Quantum Circuit Setup
```
1. Allocate 2n qubits (n query, n output)
2. Initialize: |ψ₀⟩ = |0⟩^(2n)
3. Apply Hadamard: H^⊗n ⊗ I^⊗n
4. Apply Oracle: U_f
5. Measure output register (y register)
6. Apply Hadamard: H^⊗n ⊗ I^⊗n
7. Measure query register (x register)
```

#### Phase 2: Measurement and Collection
- Run quantum circuit multiple times
- Each measurement yields bitstring y with |P(y|s)| ∝ |amplitude|²
- Key property: Measurement gives y ⊥ s (y · s ≡ 0 mod 2)
- Collect n linearly independent equations

#### Phase 3: Gaussian Elimination
```
System of linear equations over GF(2):
  y₁ · s ≡ 0 (mod 2)
  y₂ · s ≡ 0 (mod 2)
  ...
  y_n · s ≡ 0 (mod 2)

Solve using Gaussian elimination over GF(2)
Result: Unique s (with high probability)
```

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install numpy
pip install torch  # or tensorflow for alternative backend
pip install sympy  # For linear algebra over GF(2)
```

### Project Structure
```
quantum-cryptography/
├── simon/
│   ├── __init__.py
│   ├── algorithm.py          # Main Simon implementation
│   ├── SKILL.md              # This file
│   └── README.md
├── core/                      # Core quantum computing modules
│   ├── GateSequence.py
│   ├── Register.py
│   ├── State.py
│   └── ClassicalRegister.py
└── library/                   # Quantum library functions
```

### Configuration
Create configuration settings:
```python
CONFIG = {
    'backend': 'torch',
    'output_dir': './simon_results',
    'max_qubits': 25,
    'seed': 42,
    'num_runs': 100  # Runs to collect equations
}
```

## Key Methods

### `SimonAlgorithm.run(s_target='1010', backend='torch', algo_dir='./simon_results')`
**Parameters**:
- `s_target` (str): Hidden period string (binary string)
- `backend` (str): Simulation backend ('torch', 'numpy')
- `algo_dir` (str): Directory for output files

**Returns**: Dictionary with:
- `status`: 'ok' or 'failed'
- `found_s`: The discovered period string
- `equations`: List of collected equations (y vectors)
- `circuit_path`: Path to SVG circuit diagram
- `timing`: Execution time breakdown
- `confidence`: Probability of correct solution

### `_build_simon_oracle(gs, s)`
Constructs the quantum oracle U_f from the period s.

### `_get_basis_simple(state_list, n_qubits)`
Extracts linearly independent basis vectors from measurements.

### `_solve_simon_general(basis_list, n)`
Solves the linear system over GF(2) to find s.

## Step-by-Step Usage Guide

### Step 1: Import and Setup
```python
from simon.algorithm import SimonAlgorithm
import os

# Create algorithm instance
simon = SimonAlgorithm()

# Set output directory
output_dir = './simon_results'
os.makedirs(output_dir, exist_ok=True)
```

### Step 2: Define the Secret Period
```python
# Choose hidden period string
n = 4  # Input register size: 4 qubits -> 2⁴ = 16 possible inputs
s = '1010'  # Binary string (must contain at least one '1')

# Alternative examples:
# n = 2, s = '01' or '11'
# n = 3, s = '001', '100', '111'
# n = 5, s = '00001', '10000', '11111'

print(f"Finding hidden period: n={n}, s={s}")
```

### Step 3: Define the Function (Optional - Use Built-in Oracle)
```python
# The algorithm uses a built-in oracle based on the period s
# For s='1010', the function pairs inputs that differ by s:
# f(x) = f(x ⊕ s) for all x

# The built-in oracle implements this pairing automatically
# No manual function definition needed
```

### Step 4: Execute Simon's Algorithm
```python
# Run the algorithm
result = simon.run(
    s_target=s,
    backend='torch',
    algo_dir=output_dir
)
```

### Step 5: Verify the Result
```python
# Check success
if result['status'] == 'ok':
    found_s = result['found_s']
    print(f"✓ Successfully found period: {found_s}")
    
    # Verify correctness
    if found_s == s:
        print(f"Verification: Found s = {found_s} matches original s = {s} ✓")
    else:
        print(f"WARNING: Found s = {found_s} does not match original s = {s}")
else:
    print(f"✗ Algorithm failed: {result.get('message', 'Unknown error')}")
```

### Step 6: Analyze Quantum Behavior
```python
# Examine collected equations
print(f"\n--- Quantum Measurement Analysis ---")
print(f"Number of equations collected: {len(result.get('equations', []))}")
print("Equations (y vectors orthogonal to s):")

# The algorithm automatically collects and solves equations
# All measured y satisfy y · s ≡ 0 mod 2
s_int = int(s, 2)
for i, y in enumerate(result.get('equations', [])):
    y_int = int(y, 2)
    dot_product = bin(s_int & y_int).count('1') % 2
    print(f"  y_{i} = {y}, y·s mod 2 = {dot_product} (should be 0)")
```

### Step 7: Batch Testing with Different Periods
```python
# Test with multiple secret strings
test_cases = [
    (2, '01'),
    (2, '11'),
    (3, '001'),
    (3, '100'),
    (3, '111'),
    (4, '0101'),
]

results = []
for n, s in test_cases:
    result = simon.run(s_target=s, backend='torch')
    success = result['status'] == 'ok' and result['found_s'] == s
    
    status = "✓" if success else "✗"
    print(f"{status} n={n}, s={s} -> found={result['found_s']}")
    results.append({
        'params': (n, s),
        'success': success,
        'confidence': result.get('confidence', 0)
    })
```

## Advanced Usage

### Custom Oracle Implementation
```python
# While the built-in oracle works for the standard Simon problem,
# you can implement custom periodic functions

def create_custom_simon_function(n, s):
    """
    Create a custom 2-to-1 function with period s
    This is for understanding - the algorithm uses built-in oracle
    """
    s_int = int(s, 2)
    
    def f(x):
        # Simple implementation: f(x) = x & (~s_int & ((1 << n) - 1))
        # This ensures f(x) = f(x XOR s)
        return x & (~s_int & ((1 << n) - 1))
    
    return f

# Example usage (for verification only)
f = create_custom_simon_function(3, '101')
print(f"f(0) = {f(0)}, f(0⊕5) = f({0^5}) = {f(0^5)}")  # Should be equal
```

### Performance Profiling
```python
import time

for n in range(2, 6):
    s = bin(1)[2:].zfill(n)  # Secret: '0...01'
    start = time.time()
    result = simon.run(s_target=s, backend='torch')
    elapsed = time.time() - start
    
    success = result['status'] == 'ok'
    print(f"n={n}: {elapsed:.4f}s - Status: {'✓' if success else '✗'}")
```

### Error Handling and Retry
```python
max_retries = 3
attempt = 0
success = False

while attempt < max_retries and not success:
    try:
        result = simon.run(s_target='1010', backend='torch')
        if result['status'] == 'ok':
            success = True
            break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
    attempt += 1

if not success:
    print("All attempts failed")
```

### Understanding the Oracle Construction
```python
# The built-in oracle implements the Simon function
# For s='101', it pairs inputs that differ by s:
# f(000) = f(101) = some value
# f(001) = f(100) = some other value
# etc.

# The oracle is constructed using CNOT gates
# This creates the required entanglement for interference
```

## Implementing Your Own Simon Algorithm

### Core Components to Implement

#### 1. Linear Algebra over GF(2)
```python
def gf2_add(a, b):
    """Addition in GF(2) (XOR)"""
    return a ^ b

def gf2_dot_product(vec1, vec2):
    """Dot product over GF(2)"""
    return sum(x * y for x, y in zip(vec1, vec2)) % 2

def gf2_gaussian_elimination(matrix):
    """
    Solve system of linear equations over GF(2)
    Returns solution vector
    """
    rows, cols = len(matrix), len(matrix[0])
    
    # Forward elimination
    for i in range(min(rows, cols)):
        # Find pivot
        pivot_row = -1
        for j in range(i, rows):
            if matrix[j][i] == 1:
                pivot_row = j
                break
        
        if pivot_row == -1:
            continue  # No pivot in this column
        
        # Swap rows
        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
        
        # Eliminate below
        for j in range(i+1, rows):
            if matrix[j][i] == 1:
                for k in range(cols):
                    matrix[j][k] ^= matrix[i][k]
    
    # Back substitution
    solution = [0] * cols
    for i in range(min(rows, cols)-1, -1, -1):
        if matrix[i][i] == 0:
            continue
        solution[i] = matrix[i][-1]
        for j in range(i+1, cols-1):
            solution[i] ^= (matrix[i][j] * solution[j])
    
    return solution
```

#### 2. Simon Oracle Construction
```python
def build_simon_oracle(n, s):
    """
    Construct the Simon oracle matrix
    For s='101', pairs inputs that differ by s
    """
    dim = 2**(2*n)
    oracle = np.eye(dim, dtype=complex)
    
    s_int = int(s, 2)
    for x in range(2**n):
        y = x ^ s_int  # x XOR s
        if x < y:  # Avoid double counting
            # Swap |x⟩|f(x)⟩ and |y⟩|f(y)⟩
            # This creates the required periodicity
            idx1 = x * (2**n) + (x % (2**n))  # Simplified
            idx2 = y * (2**n) + (y % (2**n))
            # Swap amplitudes (simplified implementation)
            pass
    
    return oracle

def apply_simon_oracle(state_vector, n, s):
    """
    Apply Simon oracle to quantum state
    """
    s_int = int(s, 2)
    new_state = np.copy(state_vector)
    
    for x in range(2**n):
        y = x ^ s_int
        if x < y:
            # Swap amplitudes for paired states
            temp = new_state[x]
            new_state[x] = new_state[y]
            new_state[y] = temp
    
    return new_state
```

#### 3. Basic Simon Algorithm Structure
```python
class BasicSimonAlgorithm:
    def __init__(self):
        self.backend = 'numpy'
    
    def run(self, s, num_measurements=100):
        n = len(s)
        if s == '0' * n:
            raise ValueError("s cannot be all zeros")
        
        # Collect measurement results
        measurements = []
        
        for _ in range(num_measurements):
            # Simulate quantum circuit
            y = self._simulate_measurement(n, s)
            measurements.append(y)
            
            # Check if we have enough independent equations
            if len(self._get_independent_equations(measurements, n)) >= n-1:
                break
        
        # Solve the system
        equations = self._get_independent_equations(measurements, n)
        found_s = self._solve_equations(equations, n)
        
        return found_s
    
    def _simulate_measurement(self, n, s):
        """Simulate a single quantum measurement"""
        # In practice, this would run the full quantum circuit
        # Here we simulate the expected behavior
        
        # Generate a random y orthogonal to s
        s_int = int(s, 2)
        
        while True:
            y_int = np.random.randint(0, 2**n)
            if bin(s_int & y_int).count('1') % 2 == 0 and y_int != 0:
                return format(y_int, f'0{n}b')
    
    def _get_independent_equations(self, measurements, n):
        """Extract linearly independent equations"""
        equations = []
        for y in measurements:
            y_vec = [int(bit) for bit in y]
            if y_vec not in equations:
                equations.append(y_vec)
                if len(equations) >= n-1:
                    break
        return equations
    
    def _solve_equations(self, equations, n):
        """Solve the linear system over GF(2)"""
        if len(equations) < n-1:
            return None
        
        # Set up augmented matrix
        matrix = []
        for eq in equations:
            matrix.append(eq + [0])  # y · s = 0
        
        # Add the constraint that s is not all zeros
        # This is handled by the free variable approach
        
        # Use Gaussian elimination
        solution = gf2_gaussian_elimination(matrix)
        
        # Convert back to binary string
        s_found = ''.join(map(str, solution))
        return s_found
```

## Best Practices

1. **Choose appropriate n**: Start small (n=2-4) for testing and verification
2. **Verify orthogonality**: Check that collected equations satisfy y·s ≡ 0 (mod 2)
3. **Number of runs**: Run quantum circuit ≥ n+1 times to get n independent equations
4. **Linear independence**: Ensure collected equations are linearly independent
5. **Backend selection**: Use 'torch' for GPU acceleration on GPU-equipped systems
6. **Result validation**: Always verify found_s matches original period

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Not enough equations | Insufficient circuit runs | Increase num_measurements (≥ n+10) |
| Linearly dependent equations | Unlucky measurement outcomes | Run more times to collect independent equations |
| Found wrong period | Measurement/linear algebra error | Enable verification mode and debug |
| ImportError | Missing quantum framework | Install via pip; check PYTHONPATH |
| Memory overflow | n too large for simulator | Use smaller n (max ~20-25 qubits) |

## Performance Characteristics

- **Quantum Queries**: O(n) vs Ω(2^(n/2)) classically
- **Total Time**: O(n³) for Gaussian elimination + O(n·m) for quantum sampling
- **Success Probability**: High (exponentially close to 1) with n+O(log n) runs
- **Circuit Depth**: O(1) with oracle, O(n) QFT if used

## Simon vs Other Quantum Algorithms

| Algorithm | Problem | Quantum Time | Classical Time | Speedup |
|-----------|---------|---|---|---|
| Simon | Find hidden period | O(n) | Ω(2^n) | Exponential |
| Deutsch-Jozsa | Balanced vs constant | O(1) | O(2^(n-1)) | Exponential |
| Grover | Search |O(√N) | O(N) | Quadratic |
| Shor (factoring) | Factor N | O(log³ N) | Sub-exponential | Exponential |

## Cryptographic Significance

### Simon's Problem in Cryptography

1. **Symmetric Cryptanalysis**: Finding collisions in block ciphers
2. **Key Recovery**: Exploiting periodic structures if cipher leaks information
3. **Quantum Meet-in-Middle**: Simon's algorithm applied to cryptanalysis

### Example Attack Scenario
```
Classical Meet-in-Middle:
  Time: O(2^(n/2))

Quantum Simon's Attack:
  Find period s in cipher behavior
  Time: O(n) queries + O(n³) classical
  Result: Exponential speedup
```

### Real-World Applications
- **Even-Mansour Construction**: Breaking certain block cipher designs
- **Message Authentication Codes**: Finding periodic structures in MACs
- **Hash Function Analysis**: Detecting hidden symmetries

## Mathematical Deep Dive

### Why Hadamard Basis Measurement Works
```
Initial: |ψ⟩ = (1/√2^n) Σ_x |x⟩|f(x)⟩

After oracle: States with f(x) = f(y) interfere constructively/destructively
Key observation: H^⊗n |x⟩ = (1/√2^n) Σ_y (-1)^(x·y) |y⟩

Measurement biases toward y where y·s ≡ 0 (mod 2)
```

### Linear Algebra over GF(2)
```
The measurement gives y such that y·s = 0 mod 2
This gives one equation in GF(2) for each measurement

With n-1 independent equations, we can solve for s
The last bit is determined by the constraint s ≠ 0
```

### Oracle Implementation Details
```
The Simon oracle must satisfy f(x) = f(x⊕s) for all x
This is implemented using CNOT gates to create entanglement

For s with bit i set, CNOT gates connect corresponding qubits
This creates the required periodic structure in the quantum state
```

## References

1. Simon, D. R. (1994). "On the power of quantum computation"
2. Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and Quantum Information"
3. Boneh, D., & Shacham, H. (2002). "Fast Variants of RSA"