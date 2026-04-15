---
name: discretelog
description: "Use when users ask about solving the discrete logarithm problem g^x ≡ y (mod P) with Shor's quantum algorithm, building/explaining DLP circuits, running simulator demos, or debugging post-processing (continued fractions, order recovery, congruence solving). Triggers: discrete log, DLP, Shor discrete logarithm, g^x mod P, modular exponentiation, continued fractions, quantum cryptography demo."
---
## One Step to Run Discrete Log Example
```bash
python ./scripts/algorithm.py
```
# 🔐 Discrete Logarithm Quantum Algorithm (Shor's DLG) Guide

---

## What It Solves

### The Mathematical Problem
Find **x** such that:
$$g^x \equiv y \pmod{P}$$

Where:
- **g** = base (generator)
- **y** = target value
- **P** = modulus, usually prime
- **x** = discrete logarithm

### Why It Matters
This problem underpins:
- Diffie-Hellman key exchange
- ElGamal encryption
- ECC
- ECDSA digital signatures

Classical computers solve this in **sub-exponential time** → Quantum computers solve it in **polynomial time** (O(n³)).

### Trigger Prompts (What Users Can Type)

Use or trigger this skill when users input prompts like:

- "Calculate the discrete logarithm using a quantum algorithm: g^x ≡ y (mod P)"
- "Implement Shor's Discrete Logarithm Algorithm (DLP)"
- "Write a discrete logarithm quantum circuit for me and explain each step"
- "Why can't I find x in my continued fractions post-processing?"
- "Give me a quantum simulation example of discrete logarithms (including IQFT)"
- "How to recover r and x from measured u and v?"
- "Please debug the issue of failing to recover r in DLP"
- "Quantum cryptography demo: Breaking g^x ≡ y (mod P)"

---

## Theory & Math

### Algorithm Overview (5-Stage Process)

1. **Parameter preparation**: validate inputs and choose register sizes from the bit length of $P$.
2. **Quantum circuit construction**: initialize registers A, B, and Work; apply Hadamards, controlled modular multiplications, and inverse QFT.
3. **Quantum execution**: run the circuit and obtain the measurement probability distribution.
4. **Classical post-processing**: use continued fractions to infer $r$ and $s$, then solve $sx \equiv -\text{target} \pmod r$.
5. **Results export**: format the output and save the circuit visualization.

### Key Quantum Operations

| Operation | Purpose | Implementation |
|-----------|---------|-----------------|
| **|+⟩ superposition** | Create coherent combination of all possible (a,b) pairs | Hadamard gates on registers A & B |
| **Controlled g^a** | Apply modular exponentiation controlled by register A | Unitary matrices with conditional control qubits |
| **Controlled y^-b** | Apply inverse target exponentiation controlled by register B | Modular inverse preprocessing |
| **Inverse QFT** | Extract periodicity information | Map time-domain to frequency-domain |
| **Measurement** | Collapse to computational basis | Get two integers u, v |

### Mathematical Relationships After Measurement

After measuring $(u, v)$:

1. **u/N ≈ s/r** → Extract group order r and multiplier s
2. **v/N ≈ -sx/r (mod 1)** → Contains information about x
3. **sx ≡ -⌊v·r/N⌉ (mod r)** → Linear congruence to solve classically

---

## Using the Implementation

### Basic Usage

```python
from engine.algorithms import DiscreteLogAlgorithm

dlg = DiscreteLogAlgorithm()
result = dlg.run(g=3, y=6, P=7, backend='torch')

print(f"Found x = {result['found_x']}")
print(f"Status: {result['status']}")
```

### Understanding the Return Value

```python
result = {
    "status": "ok",                       # or "failed"
    "found_x": 3,
    "circuit_path": "./dlg_results/...",
    "message": "成功",
    "plot": "...ASCII art..."
}
```

### Common Parameters

```python
result = dlg.run(
    g=2,
    y=13,
    P=29,
    backend='torch',
    algo_dir='./my_results'
)
```

### Accessing Last Result

The instance caches the most recent run:

```python
last = dlg.last_result
print(f"Computation time: {last['comp_time']:.4f} sec")
print(f"Order found: r = {last['r']}")
print(f"Success: {last['success']}")
```

---

## Building Your Own

### Architecture Overview

Core structure:

```
DiscreteLogAlgorithm
├── run(g, y, P, backend, algo_dir)
├── _get_modular_matrix()
├── _classical_post_processing()
├── _update_last_result()
├── _build_return()
└── format_result_ascii()
```

### Minimal Implementation Notes

```python
def run(self, g: int, y: int, P: int, backend: str = 'torch', algo_dir: str = './dlg_results'):
    if math.gcd(g, P) != 1 or math.gcd(y, P) != 1:
        raise ValueError("g and y must be coprime to P")

    n_count = 2 * P.bit_length()  # counting register (for QFT)
    n_work = P.bit_length()       # work register
    N_size = 2 ** n_count
```

Larger $N$ usually gives better phase resolution.

#### Circuit Skeleton

```python
from engine import Register, GateSequence,IQFT

ra = Register("reg_a", n_count)
rb = Register("reg_b", n_count)
rw = Register("reg_work", n_work)
gs = GateSequence(ra, rb, rw, name=f'DLP_{g}^x_{y}', backend=backend)

def get_p(reg_slice):
    reg, idxs = reg_slice[0]
    if reg.name == "reg_a": offset = 0
    elif reg.name == "reg_b": offset = n_count
    else: offset = 2 * n_count
    return [i + offset for i in idxs]

gs.h(get_p(ra[:]))
gs.h(get_p(rb[:]))
gs.x(get_p(rw[0]))

for i in range(n_count):
    gs.unitary(self._get_modular_matrix(pow(g, 2**i, P), P, n_work), get_p(rw[:]), get_p(ra[i])[0], '1')

y_inv = pow(y, -1, P)
for j in range(n_count):
    gs.unitary(self._get_modular_matrix(pow(y_inv, 2**j, P), P, n_work), get_p(rw[:]), get_p(rb[j])[0], '1')

gs.append(IQFT(n_count, backend=backend), get_p(ra[:]))
gs.append(IQFT(n_count, backend=backend), get_p(rb[:]))
```

The circuit prepares:
$$|\psi\rangle = \frac{1}{N} \sum_{a,b=0}^{N-1} |a\rangle |b\rangle |g^a y^{-b} \bmod P\rangle$$

After IQFT, phase information encodes the hidden periodicity.

#### Execution and Post-Processing

```python
res_vec = gs.execute()
state_obj = State(res_vec)
probs_dict = state_obj.calculate_state(range(2 * n_count))
```

Classical post-processing should:
- sort outcomes by probability
- recover an order candidate from $u/N$
- solve the congruence for $x$
- verify with $g^x \equiv y \pmod P$

```python
def _classical_post_processing(self, probs, g, y, P, n, N_size):
    # Sort by probability (most likely first)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1]['prob'], reverse=True)
    
    for bitstring, data in sorted_probs:
        # Skip low-probability measurements
        if data['prob'] < 0.02: 
            continue
        
        # Parse bit string into u (from first n bits) and v (from next n bits)
        v_bin, u_bin = bitstring[:n], bitstring[n:]
        u, v = int(u_bin, 2), int(v_bin, 2)
        
        if u == 0: 
            continue  # u=0 would lead to division by zero
        
        frac = Fraction(u, N_size).limit_denominator(P)
        r_base, s_base = frac.denominator, frac.numerator

        real_r = None
        for k in range(1, 10):
            if pow(g, r_base * k, P) == 1:
                real_r = r_base * k
                real_s = s_base * k
                break

        if real_r is None:
            continue

        target = int(round((v * real_r) / N_size))
        d = math.gcd(real_s, real_r)

        if (-target) % d == 0:
            s_red = real_s // d
            r_red = real_r // d
            t_red = (-target) // d

            try:
                x0 = (t_red * pow(s_red, -1, r_red)) % r_red

                for i in range(d):
                    x_test = (x0 + i * r_red) % real_r
                    if pow(g, x_test, P) == (y % P):
                        return x_test % real_r, real_r, "成功"
            except ValueError:
                continue

    return None, None, "未找到解"
```

#### Result Formatting

```python
def _build_return(self, x, success, path, msg):
    return {
        "status": "ok" if success else "failed",
        "found_x": x,
        "circuit_path": path,
        "message": msg,
        "plot": self.format_result_ascii()
    }

def format_result_ascii(self) -> str:
    """Create pretty-printed output"""
    if self.last_result is None:
        return "No execution yet"
    
    res = self.last_result
    status_emoji = 'PASS' if res['success'] else 'FAIL'

    return f"""
{'='*70}
DISCRETE LOG (DLP) ALGORITHM EXECUTION RESULTS
{'='*70}

Status: {status_emoji} {'Successfully found discrete logarithm' if res['success'] else 'Failed to find solution'}

─────────────────────────────────────────────────────────────
Input Parameters
─────────────────────────────────────────────────────────────
  Equation: {res['g']}^x ≡ {res['y']} (mod {res['P']})
  Register size n: {res['n']}

─────────────────────────────────────────────────────────────
Quantum Computation Results
─────────────────────────────────────────────────────────────
  Computation time: {res['comp_time']:.4f} second(s)
  Detected order r: {res['r']}

─────────────────────────────────────────────────────────────
Classical Post-Processing Results
─────────────────────────────────────────────────────────────
  Final solution x: {res['x'] if res['success'] else 'N/A'}
  Verification: {res['g']}^{res['x'] if res['success'] else '?'} ≡ {res['y']} (mod {res['P']}) {'✓' if res['success'] else '✗'}

─────────────────────────────────────────────────────────────
Output file: {res['path']}
Remarks: {res['msg']}
{'='*70}
"""
```

---

## Example

```python
from engine.algorithms import DiscreteLogAlgorithm

dlg = DiscreteLogAlgorithm()
result = dlg.run(g=3, y=5, P=11, backend='torch')

if result['status'] == 'ok':
    print(f"x = {result['found_x']}")
else:
    print(result['message'])
```

---

## Troubleshooting & Tips

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **"g and y must be coprime to P"** | Input validation failed | Ensure P is prime and gcd(g,P)=gcd(y,P)=1 |
| **"未找到解" (No solution)** | Classical post-processing failed | Increase n_count; try smaller P for testing |
| **High failure rate** | Continued fractions missing true order | Expand search range in `for k in range(1, 10)` |
| **OutOfMemory error** | Register sizes too large | Use smaller P values; optimize matrix storage |
| **Slow execution** | Large modular exponentiation | Pre-compute powers of g and y^-1 |

### Best Practices

1. Start with small primes such as $P < 50$.
2. Always verify with `pow(g, x, P) == y % P`.
3. Use `n_count = 2 * P.bit_length()` as the default precision setting.
4. If post-processing is unstable, log $(u, v)$ and the fraction estimate.

### Tips for Implementing Your Own

1. Test `_get_modular_matrix()` first and confirm it is unitary.
2. IQFT works best when the period is not too close to $N$.
3. Continued fractions are the fragile step; tune `limit_denominator()` if needed.
4. Ensure `pow(y, -1, P)` exists before circuit construction.

---

## Summary Table

| Aspect | Detail |
|--------|--------|
| **Problem** | Solve g^x ≡ y (mod P) for x |
| **Classical Complexity** | O(exp(c·∛n)) — sub-exponential (very slow) |
| **Quantum Complexity** | O(n³) — polynomial (very fast) |
| **Required Qubits** | ~5·log₂(P) qubits |
| **Main Quantum Operations** | Hadamard, Controlled-U(mult), IQFT |
| **Key Classical Step** | Continued fractions + linear congruence |
| **Success Rate** | ~50% per single run (rerun if needed) |
| **Best For** | Teaching quantum algorithms, proof-of-concept DLP attacks |

---



