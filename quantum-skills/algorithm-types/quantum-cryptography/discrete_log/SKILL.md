---
name: discretelog
description: Implements the quantum Discrete Logarithm Problem (DLP) algorithm to solve equations of the form g^x ≡ y (mod P).
requirements:
  - quantum computing framework (GateSequence, Register, State, IQFT)
  - numpy for matrix operations
  - fractions module for continued fraction extraction
  - simulation backend (torch or similar)
---

# Discrete Logarithm Problem (DLP) Quantum Algorithm Skill

## Overview
This skill implements a quantum algorithm to solve the Discrete Logarithm Problem: given g, y, and P, find x such that g^x ≡ y (mod P). The solution combines quantum computing with classical post-processing using continued fractions and modular arithmetic.

## Mathematical Background

### The Discrete Logarithm Problem (DLP)
The DLP is a fundamental hardness assumption in cryptography:
- **Definition**: Given a cyclic group G of order r with generator g, and an element y ∈ G, find the exponent x (where 0 ≤ x < r) such that g^x = y.
- **Classical Hardness**: No known polynomial-time classical algorithm exists for generic groups
- **Quantum Advantage**: Shor's algorithm solves DLP in O(log³ P) time using quantum computers

### Mathematical Formulation
```
Find x such that: g^x ≡ y (mod P)
where:
  - g is a primitive root modulo P (or at least has high order)
  - y is the target residue
  - P is typically a prime modulus
  - r = order(g) is the multiplicative order of g modulo P
```

### Key Number Theory Concepts
1. **Modular Order**: r is the smallest positive integer where g^r ≡ 1 (mod P)
2. **Fermat's Little Theorem**: If P is prime, then g^(P-1) ≡ 1 (mod P) for gcd(g,P)=1
3. **Continued Fractions**: Used to extract r from quantum measurement results
4. **Modular Inverse**: Essential for solving the congruence equation sx ≡ -target (mod r)

## Algorithm Phases

### Phase 1: Parameter Preparation
- **Validation**: Ensure gcd(g, P) = 1 and gcd(y, P) = 1
- **Qubit Allocation**:
  - Counting registers: n_count = 2 × bit_length(P)
  - Work registers: n_work = bit_length(P)
  - Total qubits: 2 × n_count + n_work
- **Precision**: Ensures 2^n_count ≥ r² where r is the order

### Phase 2: Quantum Circuit Construction
- **Superposition Initialization**: Apply Hadamard gates to counting registers
- **Controlled Modular Multiplication**:
  - Phase A: Apply controlled g^(2^i) for i = 0 to n_count-1
  - Phase B: Apply controlled (y^(-1))^(2^j) for j = 0 to n_count-1
- **Inverse QFT**: Apply IQFT to extract phase information

### Phase 3: Quantum Simulation
- Execute circuit on specified backend (torch/numpy)
- Measure first 2×n_count qubits
- Extract probability distribution over measurement outcomes

### Phase 4: Classical Post-Processing
**Continued Fraction Extraction**:
1. Convert measured bitstring to integers u, v
2. Compute Fraction(u / N_size).limit_denominator(P)
3. Find actual order r by checking multiples

**Solving Congruence**:
1. Extract target = round(v × r / N)
2. Solve sx ≡ -target (mod r) using modular inverse
3. Verify solution: g^x ≡ y (mod P)

### Phase 5: Result Export
- Save circuit diagram as SVG
- Provide detailed execution report

## Key Methods

### `run(g, y, P, backend='torch', algo_dir='./dlg_results')`
**Parameters**:
- `g` (int): Base value; must satisfy gcd(g, P) = 1
- `y` (int): Target value; must satisfy gcd(y, P) = 1
- `P` (int): Modulus (typically prime)
- `backend` (str): Simulation backend ('torch', 'numpy')
- `algo_dir` (str): Directory for output files

**Returns**: Dictionary with keys:
- `status`: 'ok' or 'failed'
- `found_x`: The discrete logarithm x (or None)
- `circuit_path`: Path to SVG circuit diagram
- `message`: Execution status message
- `plot`: Formatted result report

### `_get_modular_matrix(a, N, n_qubits)`
Constructs unitary matrix for modular multiplication by a mod N.

### `_classical_post_processing(probs, g, y, P, n, N_size)`
Processes measured probabilities to extract order and solve for x.

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install numpy
pip install torch  # or tensorflow for alternative backend
```

### Project Structure
```
cryptology/
├── discrete_log/
│   ├── __init__.py
│   ├── algorithm.py          # Main DLP implementation
│   ├── SKILL.md              # This file
│   └── README.md
├── skills/
│   └── skill.md              # General skills reference
└── core/                       # Core quantum computing modules
    ├── GateSequence.py
    ├── Register.py
    ├── State.py
    └── ...
```

### Configuration
Create a configuration dict or environment variables:
```python
CONFIG = {
    'backend': 'torch',        # 'torch' or 'numpy'
    'output_dir': './dlg_results',
    'max_qubits': 30,          # Maximum qubits for simulation
    'seed': 42                 # For reproducible results
}
```
```python
from discrete_log.algorithm import DiscreteLogAlgorithm

dlg = DiscreteLogAlgorithm()
result = dlg.run(g=3, y=6, P=7, backend='torch')
print(result['plot'])
# Expected: x = 3 (since 3^3 ≡ 6 mod 7)
```

## Step-by-Step Usage Guide

### Step 1: Import and Initialize
```python
from discrete_log.algorithm import DiscreteLogAlgorithm
import os

# Create algorithm instance
dlg = DiscreteLogAlgorithm()

# Optional: Set output directory
output_dir = './dlg_results'
os.makedirs(output_dir, exist_ok=True)
```

### Step 2: Prepare Parameters
```python
# Define the DLP parameters
g = 3      # Base (generator)
y = 6      # Target value
P = 7      # Prime modulus

# Verify preconditions
import math
assert math.gcd(g, P) == 1, "g must be coprime to P"
assert math.gcd(y, P) == 1, "y must be coprime to P"
print(f"Parameters valid: {g}^x ≡ {y} (mod {P})")
```

### Step 3: Execute Algorithm
```python
# Run the DLP solver
result = dlg.run(
    g=g,
    y=y,
    P=P,
    backend='torch',           # Choose backend
    algo_dir=output_dir
)
```

### Step 4: Verify Results
```python
# Check success status
if result['status'] == 'ok':
    found_x = result['found_x']
    print(f"✓ Successfully found x = {found_x}")
    
    # Manual verification
    verified = pow(g, found_x, P) == y
    print(f"Verification: {g}^{found_x} ≡ {y} (mod {P}): {verified}")
else:
    print(f"✗ Algorithm failed: {result['message']}")
```

### Step 5: Analyze Results
```python
# Access detailed information
print(f"Circuit path: {result['circuit_path']}")
print(f"Message: {result['message']}")

# Display formatted report
print(result['plot'])

# Access last result object for detailed metrics
last_result = dlg.last_result
print(f"Computation time: {last_result['comp_time']:.4f}s")
print(f"Detected order r: {last_result['r']}")
```

### Step 6: Batch Processing
```python
# Multiple test instances
test_cases = [
    (3, 6, 7),      # Expected: x = 3
    (2, 3, 5),      # Expected: x = 4
    (2, 8, 11),     # Expected: x = 3
]

results = []
for g, y, P in test_cases:
    result = dlg.run(g=g, y=y, P=P, backend='torch')
    results.append({
        'params': (g, y, P),
        'found_x': result['found_x'],
        'status': result['status']
    })
    print(f"{g}^x ≡ {y} (mod {P}) → x={result['found_x']}")
```

## Advanced Usage

### Customizing Precision
```python
# Inspect algorithm parameters
P = 1000000007  # Large prime
n_count = 2 * P.bit_length()
print(f"Recommended n_count: {n_count}")
print(f"Total qubits: {2*n_count + P.bit_length()}")
```

### Error Handling
```python
try:
    result = dlg.run(g=1, y=1, P=7)
except ValueError as e:
    print(f"Input validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Backend Comparison
```python
# Performance testing across backends
import time
for backend in ['torch', 'numpy']:
    start = time.time()
    result = dlg.run(g=3, y=6, P=7, backend=backend)
    elapsed = time.time() - start
    print(f"{backend}: {elapsed:.4f}s - Status: {result['status']}")
```

## Best Practices
1. **Choose appropriate n_count**: Larger values increase accuracy but require more computation
2. **Verify input**: Ensure g and y are coprime to P
3. **Backend selection**: Use 'torch' for GPU acceleration if available
4. **Result validation**: Always check that g^x ≡ y (mod P)
5. **Order detection**: May require checking multiple continued fraction candidates

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No solution found | Insufficient qubit precision | Increase n_count (2 × bit_length(P)) |
| Order not detected | Weak probability peaks | Run multiple trials or adjust thresholds |
| Invalid inputs | g or y not coprime to P | Verify gcd(g, P) = 1 and gcd(y, P) = 1 |
| Computation timeout | Large P value | Use smaller test cases or reduce n_count |

## Performance Characteristics
- **Time Complexity**: O(log³ P) quantum + O(log P) classical
- **Space Complexity**: O(log P) qubits
- **Success Probability**: Increases with n_count; typically 80%+ for reasonable parameters

## API Reference

### DiscreteLogAlgorithm Class

**Constructor:**
```python
dlg = DiscreteLogAlgorithm()
```

**Main Methods:**

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `run()` | g, y, P, backend, algo_dir | dict | Execute DLP solver |
| `_get_modular_matrix()` | a, N, n_qubits | ndarray | Generate unitary matrix |
| `_classical_post_processing()` | probs, g, y, P, n, N_size | tuple | Extract order and solve |
| `format_result_ascii()` | - | str | Format results as ASCII table |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `log_prefix` | str | Logging prefix (default: "INFO: ") |
| `last_result` | dict | Results from last execution |

## Algorithm Comparison

| Algorithm | Problem | Classical Time | Quantum Time | Output |
|-----------|---------|---|---|---|
| DLP Solver | g^x ≡ y (mod P) | O(√P) or worse | O(log³ P) | x value |
| Shor (Factoring) | N = p×q | O(log³ N) | O(log³ N) | p, q factors |
| Shor (DLP) | g^x ≡ y (mod P) | Subexponential | O(log³ P) | x value |
| Period Finding | Period of f(x) | Variable | O(log N) | r (period) |

## Cryptographic Applications

### 1. Elliptic Curve Cryptography (ECC)
- DLP hardness is basis for ECDSA, ECDH
- Quantum algorithm breaks ECC security

### 2. Diffie-Hellman Key Exchange
```
Setup: g, P known publicly
Alice: Choose a, send A = g^a mod P
Bob: Choose b, send B = g^b mod P
Shared Secret: g^(ab) mod P

Attack: Use DLP to find a or b from A or B
```

### 3. El Gamal Encryption
- Public key: y = g^x mod P
- Ciphertext depends on DLP hardness
- Quantum algorithm breaks confidentiality

## Extension Points

### Custom Backends
To add new simulation backend:
```python
class CustomBackend:
    def __init__(self):
        pass
    
    def execute(self, circuit):
        """Execute circuit and return state vector"""
        pass
```

### Modified Qubit Allocation
Adjust `_Phase 1` in `algorithm.py`:
```python
n_count = 3 * P.bit_length()  # Higher precision
n_work = 2 * P.bit_length()   # Larger work register
```

## Troubleshooting Guide

### Issue: ImportError for quantum modules
```
Solution: Ensure quantum framework is installed in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/quantum/core
```

### Issue: Out of Memory
```
Solution 1: Reduce n_count (may affect accuracy)
Solution 2: Use GPU-based backend 'torch'
Solution 3: Solve smaller instances first
```

### Issue: Inconsistent Results
```
Solution 1: Increase n_count for better phase resolution
Solution 2: Run multiple trials and compare
Solution 3: Set random seed for reproducibility
```

## References
- Shor's Algorithm for Discrete Logarithm Problem
- Quantum Phase Estimation (QPE)
- Continued Fractions for Period Finding
- Post-Quantum Cryptography Recommendations (NIST)
- Elementary Number Theory (modular arithmetic foundations)