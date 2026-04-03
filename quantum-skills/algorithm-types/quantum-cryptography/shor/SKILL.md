---
name: shor
description: Shor's algorithm - a quantum algorithm to factor large integers in polynomial time. Efficiently finds prime factors exponentially faster than classical algorithms.
requirements:
  - quantum computing framework (GateSequence, Register, State)
  - numpy for matrix operations
  - fractions for continued fraction expansion
  - QFT/IQFT (Quantum/Inverse Quantum Fourier Transform) modules
  - simulation backend (torch or similar)
---

# Shor's Integer Factoring Algorithm - Period Finding Problem Skill

## Overview
**Shor's Algorithm** is a quantum algorithm that factors large integers in polynomial time, exponentially faster than any known classical algorithm. Given a composite integer N, the algorithm finds its prime factors by exploiting quantum parallelism to find the period of a modular exponential function.

### Why Shor's Algorithm Matters
- **Cryptographic Impact**: Breaks RSA encryption and threatens internet security
- **Complexity**: O((log N)³) quantum time vs sub-exponential classical time (exponential speedup)
- **Foundational**: First quantum algorithm to demonstrate exponential advantage
- **Real-world Threat**: Endangers current public-key cryptography infrastructure

## Mathematical Background

### The Integer Factoring Problem
- **Input**: Composite integer N to be factored
- **Promise**: N is not prime and not a prime power
- **Goal**: Find non-trivial factors p, q such that N = p × q
- **Hardness**: Classical algorithms require sub-exponential time O(exp(c(log N)^(1/3)(log log N)^(2/3)))

### Period Finding and Factoring
The quantum algorithm reduces factoring to finding the period of the function:

```
f(x) = a^x mod N
```

where a is randomly chosen such that gcd(a, N) = 1.

If f has period r (smallest r such that a^r ≡ 1 mod N), and r is even with a^(r/2) ≢ -1 mod N, then:

```
p = gcd(a^(r/2) - 1, N)
q = gcd(a^(r/2) + 1, N)
```

give non-trivial factors of N.

### Key Quantum Concepts

**1. Quantum Phase Estimation**
- Classical: Finds eigenvalues of unitary operators
- Quantum: Estimates phase e^(2πiθ) with high precision
- Application: Extracts periodic information from quantum superpositions

**2. Modular Exponentiation**
```
U_a |y⟩ = |a·y mod N⟩
```
- Creates controlled modular multiplication
- Enables quantum evaluation of a^x mod N for all x simultaneously

**3. Continued Fractions**
- Classical tool for Diophantine approximation
- Quantum measurement gives s/2^n ≈ k/r
- Extracts period r from measurement outcomes

## Algorithm Structure

### The Five-Phase Shor's Algorithm

#### Phase 1: Classical Pre-Processing
```
1. Check if N is even: if N % 2 == 0, return [2, N//2]
2. Choose random a ∈ [2, N-1]
3. Compute g = gcd(a, N): if g > 1, return [g, N//g]
4. Verify gcd(a, N) = 1, otherwise retry
```

#### Phase 2: Quantum Circuit Construction
```
1. Set register sizes: n_count = 2·⌈log₂N⌉, n_work = ⌈log₂N⌉
2. Initialize: |0⟩^(n_count) ⊗ |1⟩^(n_work)
3. Create superposition: H^⊗(n_count) on counting register
4. Apply controlled modular exponentiation oracle
5. Apply inverse QFT: IQFT on counting register
```

#### Phase 3: Quantum Execution
```
1. Execute quantum circuit
2. Measure counting register to get integer s
3. Extract phase information: θ = s / 2^(n_count)
```

#### Phase 4: Classical Post-Processing
```
1. Apply continued fraction expansion to θ
2. Find best rational approximation k/r
3. Verify r is valid period: a^r ≡ 1 mod N
4. Check conditions: r even and a^(r/2) ≢ -1 mod N
```

#### Phase 5: Factor Extraction
```
1. Compute a^(r/2) mod N
2. Calculate p = gcd(a^(r/2) - 1, N)
3. Calculate q = gcd(a^(r/2) + 1, N)
4. Verify p·q = N and p,q are non-trivial factors
```

## Installation & Setup

### Prerequisites
```bash
# Required Python packages
pip install numpy
pip install torch  # or tensorflow for alternative backend
pip install fractions  # Built-in Python module
```
### Configuration
Create configuration settings:
```python
CONFIG = {
    'backend': 'torch',
    'output_dir': './shor_results',
    'max_qubits': 30,
    'max_retries': 15,
    'seed': 42
}
```

## Key Methods

### `ShorAlgorithm.run(N, method='matrix', backend='torch', max_retries=15, algo_dir='./shor_results')`
**Parameters**:
- `N` (int): The composite number to factor
- `method` (str): Implementation method ('matrix' or 'operator')
- `backend` (str): Simulation backend ('torch', 'numpy')
- `max_retries` (int): Maximum attempts with different random a values
- `algo_dir` (str): Directory for output files

**Returns**: Dictionary with:
- `status`: 'ok' or 'failed'
- `factors`: List of prime factors found
- `period`: The discovered period r (or None)
- `circuit_path`: Path to SVG circuit diagram (or None)
- `message`: Status message
- `plot`: ASCII-formatted result summary

### `_get_modular_matrix(a, N, n_qubits)`
Creates unitary matrix for modular multiplication by a modulo N.

### `_build_modular_matrix_circuit(gs, n_count, n_work, a, N)`
Builds controlled modular exponentiation using matrix method.

### `_build_modular_operator_circuit(gs, n_count, n_work, n_work_actual, a, N, backend)`
Builds controlled modular exponentiation using operator method.

## Step-by-Step Usage Guide

### Step 1: Import and Setup
```python
from shor.algorithm import ShorAlgorithm
import os

# Create algorithm instance
shor = ShorAlgorithm()

# Set output directory
output_dir = './shor_results'
os.makedirs(output_dir, exist_ok=True)
```

### Step 2: Choose Number to Factor
```python
# Example: Factor N = 15 = 3 × 5
N = 15

# Alternative examples:
# N = 21 = 3 × 7
# N = 35 = 5 × 7
# N = 51 = 3 × 17
# N = 91 = 7 × 13

print(f"Factoring N = {N}")
```

### Step 3: Execute Shor's Algorithm
```python
# Run with matrix method (recommended for beginners)
result = shor.run(
    N=N,
    method='matrix',
    backend='torch',
    max_retries=15,
    algo_dir=output_dir
)
```

### Step 4: Analyze the Results
```python
# Check success and display results
print(result['plot'])  # ASCII-formatted summary

if result['status'] == 'ok':
    factors = result['factors']
    period = result['period']
    print(f"✓ Successfully factored {N} = {factors[0]} × {factors[1]}")
    print(f"Found period r = {period}")
    
    # Manual verification
    if len(factors) == 2:
        product = factors[0] * factors[1]
        if product == N:
            print(f"✓ Verification: {factors[0]} × {factors[1]} = {product}")
        else:
            print(f"✗ Verification failed: {product} ≠ {N}")
else:
    print(f"✗ Algorithm failed: {result.get('message', 'Unknown error')}")
```

### Step 5: Examine Quantum Circuit
```python
# The circuit diagram is automatically saved
circuit_path = result.get('circuit_path')
if circuit_path:
    print(f"Circuit diagram saved to: {circuit_path}")
    
    # Open in browser or image viewer
    import webbrowser
    webbrowser.open(circuit_path)
```

### Step 6: Batch Testing with Different Numbers
```python
# Test factoring multiple numbers
test_cases = [15, 21, 35, 51, 91]

results = []
for N in test_cases:
    result = shor.run(N=N, method='matrix', backend='torch')
    success = result['status'] == 'ok'
    
    status = "✓" if success else "✗"
    factors = result.get('factors', [])
    factors_str = f"{factors[0]}×{factors[1]}" if len(factors) == 2 else "N/A"
    print(f"{status} N={N} -> {factors_str}")
    
    results.append({
        'N': N,
        'success': success,
        'factors': factors
    })
```

## Advanced Usage

### Comparing Matrix vs Operator Methods
```python
# Test both implementation methods
methods = ['matrix', 'operator']

for method in methods:
    result = shor.run(N=15, method=method, backend='torch')
    success = result['status'] == 'ok'
    factors = result.get('factors', [])
    
    print(f"Method {method}: {'✓' if success else '✗'} -> {factors}")
```

### Custom Backend Selection
```python
# Use different simulation backends
backends = ['torch', 'numpy']

for backend in backends:
    result = shor.run(N=15, method='matrix', backend=backend)
    print(f"Backend {backend}: status={result['status']}")
```

### Performance Benchmarking
```python
import time

# Test with different composite numbers
test_numbers = [15, 21, 35, 51, 91]

for N in test_numbers:
    start = time.time()
    result = shor.run(N=N, method='matrix', backend='torch')
    elapsed = time.time() - start
    
    success = result['status'] == 'ok'
    factors = result.get('factors', [])
    print(f"N={N}: {elapsed:.4f}s - Status: {'✓' if success else '✗'} - Factors: {factors}")
```

### Analyzing Quantum Measurements
```python
# Run with detailed output
result = shor.run(N=15, method='matrix', backend='torch')

# Access measurement details (if available in result)
if 'measure_int' in result:
    measure_int = result['measure_int']
    measure_bin = result['measure_bin']
    phase = result['phase']
    
    print("Quantum measurement details:")
    print(f"  Measured integer: {measure_int}")
    print(f"  Binary representation: {measure_bin}")
    print(f"  Phase: {phase:.6f}")
    print(f"  Estimated period: {result.get('period')}")
```

## Implementing Your Own Shor Algorithm

### Core Components to Implement

#### 1. Classical Helper Functions
```python
def mod_pow(base, exp, mod):
    """Modular exponentiation using fast exponentiation"""
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def extended_gcd(a, b):
    """Extended Euclidean algorithm for gcd and coefficients"""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def mod_inverse(a, m):
    """Compute modular inverse using extended gcd"""
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse doesn't exist for {a} mod {m}")
    return x % m
```

#### 2. Continued Fraction Expansion
```python
def continued_fractions(a, b, max_terms=20):
    """Compute continued fraction expansion of a/b"""
    cf = []
    for _ in range(max_terms):
        cf.append(a // b)
        a, b = b, a % b
        if b == 0:
            break
    return cf

def convergents(cf):
    """Compute convergents from continued fraction"""
    h, k = [0, 1], [1, 0]
    for q in cf:
        h.append(q * h[-1] + h[-2])
        k.append(q * k[-1] + k[-2])
    return list(zip(h[1:], k[1:]))

def find_period_from_phase(phase, N, max_denominator=None):
    """Find period r from phase measurement"""
    if max_denominator is None:
        max_denominator = N
    
    frac = Fraction(phase).limit_denominator(max_denominator)
    return frac.denominator, frac.numerator
```

#### 3. Modular Arithmetic Oracle
```python
def create_modular_exponentiation_oracle(a, N, n_qubits):
    """
    Create controlled modular exponentiation oracle
    U |x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩ where f(x) = a^x mod N
    """
    dim = 2**n_qubits
    # This is a simplified version - full implementation requires
    # quantum circuit construction
    pass

def create_qft_oracle(n_qubits):
    """Create quantum Fourier transform oracle"""
    dim = 2**n_qubits
    omega = np.exp(2j * np.pi / dim)
    
    qft_matrix = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            qft_matrix[i, j] = omega**(i * j)
    
    qft_matrix /= np.sqrt(dim)
    return qft_matrix
```

#### 4. Basic Shor Algorithm Structure
```python
class BasicShorAlgorithm:
    def __init__(self):
        self.backend = 'numpy'
    
    def run(self, N, max_retries=10):
        # Phase 1: Classical pre-processing
        if N % 2 == 0:
            return [2, N // 2]
        
        for attempt in range(max_retries):
            a = random.randint(2, N - 1)
            if math.gcd(a, N) > 1:
                return [math.gcd(a, N), N // math.gcd(a, N)]
            
            # Phase 2: Quantum period finding (simplified)
            r = self._quantum_period_finding(a, N)
            
            if r is None or r % 2 == 1:
                continue
                
            # Phase 3: Factor extraction
            factors = self._extract_factors(a, r, N)
            if factors:
                return factors
        
        return None  # Failed
    
    def _quantum_period_finding(self, a, N):
        """Simplified quantum period finding simulation"""
        # In practice, this would run the full quantum circuit
        # Here we simulate the expected behavior
        
        # Simulate measurement outcome
        n_count = 2 * N.bit_length()
        s = random.randint(0, 2**n_count - 1)
        phase = s / (2**n_count)
        
        # Extract period using continued fractions
        r, _ = find_period_from_phase(phase, N)
        
        # Verify it's a valid period
        if mod_pow(a, r, N) == 1:
            return r
        return None
    
    def _extract_factors(self, a, r, N):
        """Extract factors from period"""
        if r % 2 == 1:
            return None
            
        ar2 = mod_pow(a, r // 2, N)
        if ar2 == N - 1:  # Trivial case
            return None
            
        p = math.gcd(ar2 - 1, N)
        q = math.gcd(ar2 + 1, N)
        
        if p > 1 and q > 1 and p * q == N:
            return [p, q]
        return None
```

## Best Practices

1. **Choose appropriate N**: Start with small composites (N<100) for testing and verification
2. **Use matrix method**: 'matrix' method is simpler and more reliable for beginners
3. **Set reasonable retries**: max_retries=10-20 is usually sufficient
4. **Verify results**: Always check that p×q = N for found factors
5. **Backend selection**: Use 'torch' for GPU acceleration when available
6. **Circuit visualization**: Examine generated circuit diagrams for debugging

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Algorithm fails | Bad random a choice | Increase max_retries |
| Wrong factors | Invalid period found | Verify period: a^r ≡ 1 mod N |
| Memory overflow | N too large | Use smaller N for testing |
| ImportError | Missing dependencies | Install numpy, torch |
| No factors found | Trivial period cases | Algorithm handles automatically |

## Performance Characteristics

- **Quantum Queries**: O(log N) for period finding
- **Total Time**: O((log N)³) polynomial time
- **Success Probability**: High with multiple random a trials
- **Circuit Depth**: O(log N) with optimized modular arithmetic

## Shor vs Other Quantum Algorithms

| Algorithm | Problem | Quantum Time | Classical Time | Impact |
|-----------|---------|---|---|---|
| Shor (Factoring) | Factor N | O((log N)³) | Sub-exp | Breaks RSA |
| Shor (DL) | Find x: g^x=y mod P | O((log P)³) | Sub-exp | Breaks D-H, ECC |
| Grover | Search | O(√N) | O(N) | Quadratic speedup |
| Deutsch-Jozsa | Balanced vs constant | O(1) | O(2^(n-1)) | Exponential |

## Cryptographic Significance

### Breaking Modern Cryptography

**1. RSA Cryptosystem**
```
Classical: Security based on factoring hardness
Quantum: Private keys recoverable in polynomial time
Impact: All RSA-based systems vulnerable
```

**2. Digital Signatures**
```
RSA signatures: Based on factoring
Quantum: Signatures forgeable, private keys recoverable
Impact: Certificate authorities, code signing compromised
```

**3. Key Exchange Protocols**
```
Diffie-Hellman: Based on discrete log (broken by Shor)
Quantum: Session keys recoverable
Impact: TLS/SSL, IPsec, VPNs at risk
```

### Post-Quantum Cryptography
Shor's algorithm motivates development of:
- **Lattice-based cryptography** (hard even for quantum computers)
- **Hash-based signatures** (XMSS, LMS)
- **Multivariate cryptography**
- **Supersingular isogeny key exchange** (SIKE)

## Mathematical Deep Dive

### Why Quantum Fourier Transform Works
```
The QFT transforms periodic states into peaked distributions:
For periodic function f(x) = f(x+r), the QFT concentrates
measurement probability around multiples of 2^n/r

This allows extracting the period r from measurement statistics.
```

### Continued Fractions and Diophantine Approximation
```
Quantum measurement gives s/2^n ≈ k/r + ε
Continued fraction expansion finds the best rational approximation:

The convergents h_m/k_m satisfy |s/2^n - h_m/k_m| < 1/(k_m·k_{m+1})

For periodic functions, this recovers the exact period r.
```

### Modular Arithmetic in Quantum Circuits
```
Controlled modular multiplication U_a |x⟩|y⟩ = |x⟩|a·y mod N⟩
can be implemented using quantum adders and comparators.

The circuit depth is O(log N) with carry-lookahead techniques.
```

## References

1. Shor, P. W. (1994). "Algorithms for quantum computation: discrete logarithms and factoring"
2. Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and Quantum Information"
3. Bernstein, D. J. (2009). "Introduction to post-quantum cryptography"
4. Mosca, M. (2018). "Cybersecurity in an era with quantum computers"