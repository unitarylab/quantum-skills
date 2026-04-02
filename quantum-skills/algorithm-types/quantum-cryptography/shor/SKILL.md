---
name: shor
description: Implements Shor's algorithm for integer factorization, which efficiently factors large integers using quantum computing. Combines quantum period finding with classical reduction techniques.
requirements:
  - quantum computing framework (GateSequence, Register, State, IQFT)
  - numpy for numerical operations
  - fractions module for continued fraction analysis
  - simulation backend (torch or similar)
---

# Shor's Algorithm for Integer Factorization Skill

## Overview
This skill implements **Shor's Algorithm**, one of the most significant quantum algorithms that efficiently factors large integers in polynomial time O(log³ N). It combines quantum period finding with classical number theory to break RSA encryption and other factorization-based cryptosystems.

## Mathematical Background

### The Factorization Problem
- **Definition**: Given a composite number N = p × q where p, q are prime, find p and q.
- **Classical Hardness**: Best known classical algorithms require sub-exponential time
- **Quantum Advantage**: Shor's algorithm factors N in O(log³ N) quantum time
- **Cryptographic Impact**: Breaks RSA, Diffie-Hellman, and most public-key cryptography

### Mathematical Formulation
```
Input: N = p × q (composite, odd number)
Goal: Find factors p and q
Method: Reduce factorization to order-finding problem
  1. Pick random a: gcd(a, N) = 1
  2. Find r = order(a mod N): a^r ≡ 1 (mod N)
  3. If r is even: Compute gcd(a^(r/2) ± 1, N)
  4. Result: Non-trivial factor of N
```

### Key Number Theory Concepts

**1. Order Problem**
- order(a, N) = smallest positive r where a^r ≡ 1 (mod N)
- Classical: No known efficient algorithm (within polynomial time)
- Quantum: Solvable via Quantum Phase Estimation (QPE)

**2. Factorization Reduction**
```
If a^r ≡ 1 (mod N), then:
  a^r - 1 ≡ 0 (mod N)
  (a^(r/2) - 1)(a^(r/2) + 1) ≡ 0 (mod N)

If r is even and a^(r/2) ≢ ±1 (mod N):
  gcd(a^(r/2) - 1, N) = p or q (non-trivial factor)
```

**3. Quantum Phase Estimation (QPE)**
- Estimates eigenvalue phase with precision O(1/2^n)
- Extracts order r via continued fractions
- Core quantum speedup mechanism

## Algorithm Structure

### The Five-Phase Shor's Algorithm

#### Phase 1: Classical Preprocessing
- Input validation: Ensure N is odd composite
- Trial division: Check small primes (2, 3, 5, ...)
- GCD check: Verify gcd(a, N) = 1 for random a
- Early termination: If r is odd, pick new a

#### Phase 2: Quantum Order Finding
```
1. Allocate qubits: n_count + n_work qubits
2. Initialize superposition: |0⟩ → H⊗n |0⟩
3. Phase kickback: Apply controlled a^(2^k) mod N
4. Inverse QFT: Extract phase information
5. Measure: Get phase estimate ϕ ≈ s/r
```

#### Phase 3: Continued Fraction Extraction
- Convert phase ϕ to fraction s/r via continued fractions
- Search multiples of denominator r for true order
- Verify: Check if a^r ≡ 1 (mod N)

#### Phase 4: Factor Extraction
```
If r is even:
  x = a^(r/2) mod N
  
If x ≢ ±1 (mod N):
  p = gcd(x - 1, N)
  q = gcd(x + 1, N)
  Return factors (p, q)
Else:
  Retry with new random a
```

#### Phase 5: Result Verification
- Confirm: N = p × q
- Export: Circuit diagram and metrics
- Statistics: Success probability, timing data

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
├── shor/
│   ├── __init__.py
│   ├── algorithm.py          # Main Shor implementation
│   ├── parameters.json       # Algorithm configuration
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
Create a configuration dict in `parameters.json`:
```json
{
  "backend": "torch",
  "output_dir": "./shor_results",
  "max_qubits": 30,
  "seed": 42,
  "trial_limit": 10
}
```

## Key Methods

### `ShorAlgorithm.run(N, backend='torch', algo_dir='./shor_results')`
**Parameters**:
- `N` (int): Number to factor (composite, odd, > 1)
- `backend` (str): Simulation backend ('torch', 'numpy')
- `algo_dir` (str): Directory for output files

**Returns**: Dictionary with:
- `status`: 'ok' or 'failed'
- `factors`: Tuple (p, q) or None
- `circuit_path`: Path to SVG circuit diagram
- `trials`: Number of attempts needed
- `timing`: Execution time breakdown

### `_classical_preprocessing(N)`
Validates input and performs trial division for small factors.

### `_quantum_order_finding(a, N, n_count)`
Executes quantum phase estimation to find order r.

### `_continued_fraction_extraction(phase, N)`
Converts quantum phase to order estimate via continued fractions.

### `_factor_extraction(a, r, N)`
Derives factors from order using GCD operations.

## Step-by-Step Usage Guide

### Step 1: Import and Initialize
```python
from shor.algorithm import ShorAlgorithm
import json
import os

# Create algorithm instance
shor = ShorAlgorithm()

# Load configuration
with open('shor/parameters.json', 'r') as f:
    config = json.load(f)

output_dir = config.get('output_dir', './shor_results')
os.makedirs(output_dir, exist_ok=True)
```

### Step 2: Prepare Input
```python
# Choose a number to factor
N = 15  # Simple: 15 = 3 × 5
# N = 77  # Classic: 77 = 7 × 11
# N = 91  # Challenge: 91 = 7 × 13
# N = 143  # Harder: 143 = 11 × 13

# Verify it's suitable for factorization
if N % 2 == 0:
    print(f"N is even, trivial factor: 2")
    N = N // 2

print(f"Factoring N = {N}")
```

### Step 3: Execute Algorithm
```python
# Run Shor's algorithm
result = shor.run(
    N=N,
    backend=config.get('backend', 'torch'),
    algo_dir=output_dir
)
```

### Step 4: Verify Factors
```python
# Check success status
if result['status'] == 'ok':
    p, q = result['factors']
    print(f"✓ Successfully factored: {N} = {p} × {q}")
    
    # Verify result
    product = p * q
    if product == N:
        print(f"Verification: {p} × {q} = {product} ✓")
    else:
        print(f"ERROR: {p} × {q} = {product} ≠ {N}")
else:
    print(f"✗ Factorization failed: {result.get('message', 'Unknown error')}")
```

### Step 5: Analyze Performance
```python
# Access detailed metrics
print(f"\n--- Performance Statistics ---")
print(f"Trials attempted: {result['trials']}")
print(f"Circuit path: {result['circuit_path']}")

if 'timing' in result:
    timing = result['timing']
    print(f"Classical preprocessing: {timing.get('preprocess', 0):.4f}s")
    print(f"Quantum simulation: {timing.get('quantum', 0):.4f}s")
    print(f"Classical post-processing: {timing.get('postprocess', 0):.4f}s")
```

### Step 6: Batch Factorization
```python
# Factor multiple numbers
test_numbers = [15, 21, 35, 77, 91, 143, 221]

results = []
for N in test_numbers:
    if N % 2 == 0:
        continue
    
    result = shor.run(N=N, backend='torch')
    if result['status'] == 'ok':
        p, q = result['factors']
        results.append({
            'N': N,
            'factors': (p, q),
            'trials': result['trials']
        })
        print(f"{N} = {p} × {q} (trials: {result['trials']})")
    else:
        print(f"{N}: Failed after {result['trials']} trials")
```

## Advanced Usage

### Custom Random Seed
```python
import random
random.seed(42)
result = shor.run(N=91, backend='torch')
```

### Multiple Backend Comparison
```python
import time

for backend in ['torch', 'numpy']:
    start = time.time()
    result = shor.run(N=77, backend=backend)
    elapsed = time.time() - start
    
    status = "✓" if result['status'] == 'ok' else "✗"
    print(f"{backend:8s}: {status} {elapsed:.4f}s - Trials: {result['trials']}")
```

### Large Number Factorization
```python
# For larger N, may need more qubits
N = 10007 * 10009  # ~100 million

# Note: Simulation backend limits practical N size
# In real quantum computer: exponentially harder classically
# In simulator: limited by classical simulation resources

result = shor.run(N=N, backend='torch')
```

### Error Handling
```python
try:
    result = shor.run(N=2, backend='torch')  # Should fail: even number
except ValueError as e:
    print(f"Input validation error: {e}")
except Exception as e:
    print(f"Algorithm error: {e}")
```

## Best Practices
1. **Input validation**: Ensure N is odd composite number
2. **Backend selection**: Use 'torch' for faster GPU computation
3. **Random seed**: Set seed for reproducible testing
4. **Trial limits**: Monitor number of attempts to find usable 'a'
5. **Verification**: Always check N = p × q after factorization
6. **Error handling**: Prepare for cases where algorithm needs retries

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| ImportError | Missing quantum framework | Install via pip and add to PYTHONPATH |
| "N is even" | Input is 2-smooth number | Divide by 2 and factor remainder |
| Order not found | Phase measurement imprecise | Increase n_count for better precision |
| No factors found | Unlucky random 'a' choice | Algorithm automatically retries |
| Computation timeout | N too large for simulator | Use real quantum computer or smaller N |
| gcd(a,N)≠1 | Bad random selection | Algorithm retries with new a automatically |

## Performance Characteristics
- **Time Complexity**: O(log³ N) quantum + O(log N) classical per trial
- **Space Complexity**: O(log N) logical qubits
- **Success Probability**: ~(4/π²) ≈ 40% per trial for random even order
- **Average Trials**: ~π²/4 ≈ 2.5 attempts needed

## Algorithm Comparison

| Algorithm | Problem | Time | Space | Quantum? |
|-----------|---------|------|-------|----------|
| Trial Division | Factor N | O(√N) | O(1) | No |
| Pollard's rho | Factor N | O(N^(1/4)) | O(1) | No |
| Quadratic Sieve | Factor N | O(exp(1.9√(log N))) | O(log N) | No |
| Shor's Algorithm | Factor N | O(log³ N) | O(log N) | Yes ⚡ |

## Cryptographic Impact

### RSA Encryption Break
```
RSA Security = Difficulty of factoring N
Shor's Algorithm = Efficient quantum factorization
Conclusion = RSA broken by quantum computers
```

### Timeline of Security
```
2024: Quantum computers: ~100-1000 qubits (not yet threatening)
2030: Estimated: NIST becomes vulnerable? (speculative)
2040+: Large-scale quantum computers likely break RSA (estimate)
```

### Post-Quantum Cryptography
- **Lattice-based**: Learning With Errors (LWE)
- **Hash-based**: Merkle signatures
- **Code-based**: McEliece cryptosystem
- **Multivariate**: Polynomial equations
- These are resistant to Shor's algorithm!

## Mathematical Deep Dive

### Phase Kickback Operator
```
Define: U_a|y⟩ = |a·y mod N⟩

Eigenstate analysis:
  U_a|ψ_s⟩ = e^(2πi·s/r)|ψ_s⟩  where a^r ≡ 1 (mod N)
  
Phase estimation extracts: s/r
Continued fractions recover: r (the order)
```

### Why Continued Fractions Work
```
If phase ϕ ≈ s/r with s < r:

Fraction(ϕ).limit_denominator() finds (s, r)
because continued fractions give best rational approximations
```

## References
- Shor, P.W. (1994). "Polynomial-Time Algorithms for Prime Factorization"
- Nielsen & Chuang (2010). "Quantum Computation and Quantum Information"
- de Wolf, R. (2019). "Quantum Computing: Lecture Notes"
- NIST Post-Quantum Cryptography Standardization
- Quantum Supremacy: From Theory to Practice (recent surveys)
