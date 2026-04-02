---
name: simon
description: Simon's algorithm - a quantum algorithm to discover hidden periodic structures in functions. Efficiently finds the hidden period string s such that f(x) = f(x⊕s) for given function f.
requirements:
  - quantum computing framework (GateSequence, Register, State)
  - numpy for numerical operations
  - QAOA or quantum phase estimation framework
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
1. Allocate n+m qubits (n query, m output)
2. Initialize: |ψ₀⟩ = |0⟩^(n+m)
3. Apply Hadamard: H^⊗n ⊗ I^⊗m
4. Apply Oracle: U_f
5. Apply Hadamard: H^⊗n ⊗ I^⊗m
6. Measure first n qubits
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
cryptology/
├── simon/
│   ├── __init__.py
│   ├── algorithm.py          # Main Simon implementation
│   ├── SKILL.md              # This file
│   └── README.md
├── skills/
│   └── skill.md              # General skills reference
└── core/                      # Core quantum computing modules
    ├── GateSequence.py
    ├── Register.py
    ├── State.py
    └── ...
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

### `SimonAlgorithm.run(n, s, backend='torch', algo_dir='./simon_results')`
**Parameters**:
- `n` (int): Number of qubits (input size)
- `s` (str or int): Hidden period string (binary string or integer)
- `backend` (str): Simulation backend ('torch', 'numpy')
- `algo_dir` (str): Directory for output files

**Returns**: Dictionary with:
- `status`: 'ok' or 'failed'
- `found_s`: The discovered period string
- `equations`: List of collected equations (y vectors)
- `circuit_path`: Path to SVG circuit diagram
- `timing`: Execution time breakdown
- `confidence`: Probability of correct solution

### `_build_oracle(f, n)`
Constructs the quantum oracle U_f from classical function f.

### `_quantum_sampling()`
Executes quantum circuit and collects measurement results.

### `_gaussian_elimination_gf2(equations)`
Solves linear system over GF(2) to find s.

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
n = 3  # Input register size: 3 qubits -> 2³ = 8 possible inputs
s = '011'  # Binary string (or s = 0b011 as integer)

# Alternative examples:
# n = 2, s = '01' or '11'
# n = 4, s = '1010'
# n = 5, s = '00001'

print(f"Finding hidden period: n={n}, s={s}")
```

### Step 3: Define the Function (Optional - Use Default Oracle)
```python
def create_simon_function(n, s):
    """Create a 2-to-1 function with period s"""
    s_int = int(s, 2)
    
    def f(x):
        # Simple implementation: f(x) = x AND (NOT s)
        # This ensures f(x) = f(x XOR s)
        return x & (~s_int & ((1 << n) - 1))
    
    return f

f = create_simon_function(n, s)
```

### Step 4: Execute Simon's Algorithm
```python
# Run the algorithm
result = simon.run(
    n=n,
    s=s,
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
print(f"Number of equations collected: {len(result['equations'])}")
print(f"Equations (y vectors orthogonal to s):")

for i, y in enumerate(result['equations']):
    # Verify: Each y should be orthogonal to s
    s_int = int(s, 2)
    y_int = int(y, 2)
    dot_product = bin(s_int & y_int).count('1') % 2
    print(f"  y_{i} = {y}, y·s mod 2 = {dot_product} (should be 0)")

print(f"Execution time: {result['timing'].get('total', 0):.4f}s")
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
    result = simon.run(n=n, s=s, backend='torch')
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
def create_custom_oracle(n, secret_pattern):
    """
    Create custom periodic function
    f(x) = MSB (first bit) of (x AND pattern)
    """
    def f(x):
        return (x & secret_pattern) >> (n - 1)
    return f

# Use in algorithm
result = simon.run(
    n=4,
    s='0001',
    backend='torch'
)
```

### Performance Profiling
```python
import time

for n in range(2, 8):
    start = time.time()
    s = bin(1)[2:].zfill(n)  # Secret: '0...01'
    result = simon.run(n=n, s=s, backend='torch')
    elapsed = time.time() - start
    
    print(f"n={n}: {elapsed:.4f}s - Status: {result['status']}")
```

### Error Handling and Retry
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        result = simon.run(n=3, s='101', backend='torch')
        if result['status'] == 'ok':
            break
    except Exception as e:
        print(f"Attempt {attempt+1} failed: {e}")
        if attempt == max_retries - 1:
            print("All attempts failed")
```

### Quantum Superposition Visualization
```python
# Examine superposition before measurement
result = simon.run(
    n=2,
    s='11',
    backend='torch',
    visualize=True
)

# Access amplitudes
if 'amplitudes' in result:
    for state, amplitude in result['amplitudes'].items():
        print(f"|{state}⟩: {amplitude:.4f}")
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
| Not enough equations | Insufficient circuit runs | Increase num_runs (≥ n+10) |
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

## Mathematical Deep Dive

### Why Hadamard Basis Measurement Works
```
Initial: |ψ⟩ = (1/√2^n) Σ_x |x⟩|f(x)⟩

After oracle: Hadamard transforms to basis orthogonal to s
Key observation: H^⊗n |x⟩ = (1/√2^n) Σ_y (-1)^(x·y) |y⟩

Measurement biases measurement toward y where y·s ≡ 0 (mod 2)
```

### Gaussian Elimination over GF(2)
```
Linear system: A·s = 0 (mod 2)
where A has rows y₁, y₂, ..., y_n

Solution methods:
  1. Row reduction to row echelon form
  2. Back substitution to find s
  3. Kernel of matrix A over GF(2)
```

## Extension: Hidden Subgroup Problem
Simon's algorithm is a special case of the **Hidden Subgroup Problem (HSP)**:
- **General HSP**: Find subgroup H ≤ G where f(g·h) = f(g) for all g ∈ G, h ∈ H
- **Simon's case**: G = (Z/2Z)^n, want period/subgroup of size 2

Other HSP variations:
- **Abelian HSP**: Shor's algorithm (exponential improvement)
- **Non-Abelian HSP**: Open problem (related to graph isomorphism)

## References
- Simon, D.R. (1994). "On the Power of Quantum Computation over Z_2"
- Nielsen & Chuang (2010). "Quantum Computation and Quantum Information"
- Kuperberg, G. (2005). "A Subexponential-Time Quantum Algorithm for the Dihedral HSP"
- Boneh & Lipton (1995). "Quantum Cryptanalysis of Hidden Linear Functions"
- Recent surveys on quantum algorithm applications
