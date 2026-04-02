---
name: amplitude-estimation
description: Quantum Amplitude Estimation - A powerful quantum algorithm to estimate the amplitude of target states with quadratic speedup. Supports flexible state preparation and configurable precision.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Quantum Amplitude Estimation (QAE)

## Algorithm Overview

Quantum Amplitude Estimation (QAE) is a fundamental quantum algorithm that provides **high-precision estimation** of success probabilities in quantum states by cleverly converting **amplitude information into phase information**. Through the combination of Grover iteration and Quantum Phase Estimation (QPE), QAE achieves quadratic speedup: where classical Monte Carlo sampling requires $O(1/\epsilon^2)$ queries, quantum amplitude estimation needs only $O(1/\epsilon)$ queries—achieving **quadratic acceleration**.

### Key Advantages

- **Quadratic Speedup**: Query complexity reduced from $O(1/\epsilon^2)$ to $O(1/\epsilon)$
- **Flexible Precision**: Adjust estimation accuracy by tuning phase register bits $d$
- **Universality**: Applicable to arbitrary initial states and target conditions
- **Theoretical Foundation**: Built on proven Grover iteration and quantum phase estimation mathematics

### Applications

- **Finance**: Derivative pricing, portfolio risk assessment, option pricing
- **Machine Learning**: Probability estimation in quantum generative models
- **Optimization**: Fast probability estimation for feasible solutions
- **Scientific Computing**: Quantum acceleration of Monte Carlo methods

---

## Mathematical Foundation

### Problem Formulation

Consider a quantum circuit $U$ that prepares a target state:

$$U|0\rangle = \sqrt{a}|\text{good}\rangle + \sqrt{1-a}|\text{bad}\rangle$$

Where:
- $a \in (0,1)$ is the **success probability** to be estimated
- $|\text{good}\rangle$ represents target states (qubits all in |0⟩)
- $|\text{bad}\rangle$ represents background states
- **Goal**: Estimate the value of $a$ through quantum measurement

### Angle Parameterization

Define parametrization angle $\theta$ such that:

$$\sqrt{a} = \sin\theta, \quad \sqrt{1-a} = \cos\theta$$

Therefore $a = \sin^2\theta$ where $\theta \in (0, \pi/2)$

### Grover Iteration Operator

Define two reflection operations:

**Phase Reversal Oracle** (marks good states):
$$S_f = I - 2|\text{good}\rangle\langle\text{good}|$$

**Diffusion Operator** (reflection about initial state):
$$S_\psi = 2|\psi\rangle\langle\psi| - I, \quad \text{where } |\psi\rangle = U|0\rangle$$

**Grover Iteration Operator**:
$$Q = S_\psi S_f$$

### Eigenvalue Analysis

In the 2D subspace spanned by $\{\text{good}, \text{bad}\}$, operator $Q$ has eigenvalues:

$$Q|\psi_\pm\rangle = e^{\pm i2\theta}|\psi_\pm\rangle$$

With eigenstates:
$$|\psi_\pm\rangle = \frac{1}{\sqrt{2}}(|\text{good}\rangle \pm i|\text{bad}\rangle)$$

**Key Insight**: The eigenphases $\pm 2\theta$ contain the target probability information!

### Amplitude-to-Phase Conversion

Apply Quantum Phase Estimation (QPE) to operator $Q$:

1. Measure phase $\phi \approx \frac{2\theta}{2\pi}$
2. Recover target angle $\theta \approx \pi \phi$
3. Final estimate $\hat{a} = \sin^2(\pi \phi)$

### Precision and Complexity

- **Resolution**: Phase register with $d$ bits provides resolution $\delta\phi \approx 1/2^d$
- **Amplitude Error**: Worst-case error $\delta a \approx \pi/2^d$
- **Query Complexity**: $O(2^d) \approx O(1/\epsilon)$
- **Peak Success Probability**: ~$4/\pi^2 \approx 0.405$ (improvable with larger $d$)

---

## Core Components

### 1. Grover Iteration

```
Prepare state U|0⟩
    ↓
Apply phase oracle S_f
    ↓
Apply diffusion S_ψ
    ↓
Result: Q|ψ⟩ = e^(±i2θ)|ψ⟩
```

### 2. Quantum Phase Estimation (QPE)

- Employs $d$ phase register qubits to extract eigenphases
- Implements controlled $Q^{2^k}$ operations through controlled Grover iteration
- Applies inverse QFT to recover phase information

### 3. Classical Post-Processing

- Select highest-probability basis state from measurement results
- Recover corresponding phase $\phi$
- Calculate estimated success probability $\hat{a}$

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.fundamental_algorithm.amplitude_estimation import AmplitudeEstimationAlgorithm
from engine.core import GateSequence
import numpy as np
```

### Step 1: Define State Preparation Circuit

Create a `GateSequence` to prepare the superposition containing the target state:

```python
# Example: Prepare uniform superposition state
U = GateSequence(n_qubits=3, backend='torch')

# Add quantum gates to prepare desired state
U.h(0)      # Hadamard on qubit 0
U.h(1)      # Hadamard on qubit 1
U.h(2)      # Hadamard on qubit 2
# U now prepares uniform superposition from |000>
```

### Step 2: Instantiate Algorithm

```python
qae = AmplitudeEstimationAlgorithm()
```

### Step 3: Configure Precision and Target Conditions

Key parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `U` | `GateSequence` | — | State preparation circuit (without ancilla qubits) |
| `good_zero_qubits` | `List[int]` | — | Indices defining "good" states (must be \|0\rangle) |
| `d` | `int` | 6 | Phase register bits, controls precision $\delta a \approx \pi/2^d$ |
| `backend` | `str` | 'torch' | Quantum simulation backend |
| `algo_dir` | `str` | './qae_results' | Result output directory |

Precision vs $d$ relationship:

| $d$ | Resolution $1/2^d$ | Worst Error $\pi/2^d$ |
|-----|-------------------|----------------------|
| 3 | 0.125 | 0.393 |
| 4 | 0.0625 | 0.196 |
| 5 | 0.0313 | 0.098 |
| 6 | 0.0156 | 0.049 |
| 8 | 0.00391 | 0.0123 |

### Step 4: Execute Algorithm

```python
# Method 1: Default precision (d=6)
results = qae.run(
    U=U,
    good_zero_qubits=[0, 1, 2],     # All three must be |0> for success
    backend='torch',
    algo_dir='./amplitude_estimation_results'
)

# Method 2: Specify higher precision
results = qae.run(
    U=U,
    good_zero_qubits=[2],            # Only qubit 2 must be |0>
    d=8,                             # 8-bit precision register
    backend='torch'
)
```

### Step 5: Parse Results

```python
# Extract key results
estimated_amplitude = results['estimated_amplitude']      # Estimated a value
estimated_phase = results['estimated_phase']              # Extracted phase φ
histogram = results['phase_histogram']                    # Phase histogram
circuit_path = results.get('circuit_path', None)          # Circuit diagram path

print(f"Estimated success probability: {estimated_amplitude:.6f}")
print(f"Extracted phase: {estimated_phase:.6f}")
print(f"Top 5 measurement outcomes:")
for bits, prob in list(histogram.items())[:5]:
    print(f"  {bits}: {prob:.4f}")
```

---

## Practical Examples

### Example 1: Basic Uniform Search

Estimate probability of finding a target among 8 candidates (theoretical value $a = 1/8 = 0.125$):

```python
from engine.algorithms.fundamental_algorithm.amplitude_estimation import AmplitudeEstimationAlgorithm
from engine.core import GateSequence

# Prepare 3-qubit uniform superposition
U = GateSequence(n_qubits=3, backend='torch')
U.h(0)
U.h(1)
U.h(2)

# Define target: all qubits in |0>
qae = AmplitudeEstimationAlgorithm()
results = qae.run(
    U=U,
    good_zero_qubits=[0, 1, 2],
    d=6,                             # 6-bit precision
    backend='torch'
)

print(f"Theoretical probability: 0.125")
print(f"Estimated probability:  {results['estimated_amplitude']:.6f}")
print(f"Estimation error:       {abs(results['estimated_amplitude'] - 0.125):.6f}")
```

### Example 2: High-Precision Estimation

Use more phase bits for superior precision:

```python
results = qae.run(
    U=U,
    good_zero_qubits=[1],
    d=10,                            # 10-bit precision (error ≈ π/1024 ≈ 0.003)
    backend='torch',
    algo_dir='./high_precision_results'
)

estimated = results['estimated_amplitude']
print(f"High-precision estimate: {estimated:.8f}")
```

### Example 3: Complex State Preparation

Prepare states with more sophisticated circuits:

```python
# Complex target state preparation
U = GateSequence(n_qubits=4, backend='torch')
U.h(0)
U.h(1)
U.rz(0, np.pi/4)                    # Add phase
U.cnot(0, 2)                        # Create entanglement
U.h(3)

# Estimate target probability
results = qae.run(
    U=U,
    good_zero_qubits=[2, 3],        # Complex success conditions
    d=7,
    backend='torch'
)

hist = results['phase_histogram']
print(f"Top 3 measurement outcomes:")
for i, (bits, prob) in enumerate(list(hist.items())[:3]):
    print(f"  {i+1}. {bits}: {prob:.4f}")
```

### Example 4: Error Analysis Across Precisions

Compare results at different precision levels:

```python
# Test different d values to find optimal balance
theoretical_prob = 0.125
for d_val in [4, 6, 8, 10]:
    results = qae.run(
        U=U,
        good_zero_qubits=[0],
        d=d_val,
        backend='torch'
    )
    estimate = results['estimated_amplitude']
    error = abs(estimate - theoretical_prob)
    print(f"d={d_val}: estimate={estimate:.6f}, error={error:.6f}")
```

---

## Algorithm Flowchart

```
┌──────────────────────────────────────────────┐
│ Input: U (prep), good_zero_qubits, d (prec) │
└────────────────┬─────────────────────────────┘
                 │
         ┌───────▼────────────┐
         │ Phase 1: Setup     │
         │ Compute total bits │
         └───────┬────────────┘
                 │
    ┌────────────▼──────────────────┐
    │ Phase 2: Build Grover + QPE   │
    │ · Prepare initial state        │
    │ · Construct Grover operator   │
    │ · Build d-bit QPE circuit      │
    └────────┬───────────────────────┘
             │
  ┌──────────▼─────────────────┐
  │ Phase 3: Quantum Simulate  │
  │ · Execute full circuit      │
  │ · Extract phase register   │
  │ · Build phase histogram    │
  └──────────┬──────────────────┘
             │
   ┌─────────▼──────────────────┐
   │ Phase 4: Classical Process │
   │ · Find most probable phase │
   │ · Recover angle θ = πφ    │
   │ · Compute a = sin²(πφ)    │
   └─────────┬──────────────────┘
             │
  ┌──────────▼─────────────────┐
  │ Phase 5: Export & Archive  │
  │ · Save circuit diagram      │
  │ · Return result dictionary  │
  └──────────┬──────────────────┘
             │
  ┌──────────▼─────────────────┐
  │ Output: Estimated prob â    │
  │         Phase φ             │
  │         Histogram           │
  │         Circuit path        │
  └─────────────────────────────┘
```

---

## Performance Analysis

### Complexity Comparison

| Metric | Complexity | Description |
|--------|-----------|-------------|
| **Query Calls** | $O(2^d)$ | $d$ = phase register bits |
| **Equivalent Precision** | $O(1/\epsilon)$ | Precision requirement $\epsilon = \pi/2^d$ |
| **Gate Count** | $O(\text{poly}(d))$ | Polynomial in phase bits |
| **Space** | $O(2^n)$ | $n$ = total qubit count |

### Classical vs Quantum

| Method | Samples | Precision | Advantage |
|--------|---------|-----------|-----------|
| Classical Monte Carlo | $O(1/\epsilon^2)$ | $\pm\epsilon$ | Simple implementation |
| **Quantum Amplitude Estimation** | **$O(1/\epsilon)$** | **$\pm\epsilon$** | **Quadratic speedup** |

### Peak Success Probability

- Single-shot success probability ≈ $4/\pi^2 \approx 40.5\%$
- Larger $d$ concentrates distribution but doesn't increase peak height
- Run multiple times and average for better reliability

---

## Relationship to Other Algorithms

### Grover vs QAE

- **Grover's Search**: Goal is to **find** index of target state
- **QAE**: Goal is to **estimate** probability of target state
- **Relationship**: QAE uses Grover iteration as eigenstate operation

### Amplitude Amplification vs QAE

- **AA (Amplitude Amplification)**: Prepare more target instances ($p$ → ~1)
- **QAE (Amplitude Estimation)**: Precisely measure original success probability $p$
- **Complementary**: AA for enhancement, QAE for measurement

### Quantum Phase Estimation (QPE)

- **General Purpose**: Extract eigenphases from any unitary operator
- **Role in QAE**: Extract phase from Grover operator encoding amplitude information

---

## Best Practices

### Precision Selection

1. **Rough Estimation**: $d = 4$ (error ~20%)
2. **Standard Precision**: $d = 6$ (error ~5%)
3. **High Precision**: $d = 8$ (error ~1%)
4. **Very High Precision**: $d = 10$ (error ~0.3%)

### Result Interpretation

- Always check main peak height in histogram
- If main peak < 30%, consider increasing $d$
- For low-probability events ($a < 0.01$), use larger $d$
- Multiple runs help average out noise

### Circuit Optimization

```python
# Minimize U circuit depth to reduce noise
# Use efficient QFT implementations
# Merge adjacent controlled operations
```

### Hyperparameter Tuning

```python
# Test different d values to find sweet spot
for d in [4, 6, 8]:
    results = qae.run(U, good_zero_qubits, d=d, ...)
    quality = results['histogram_peak_height']
    print(f"d={d}: quality={quality:.4f}")
```

---

## Integration with UnitaryLab

Access via UnitaryLab framework:

```python
from unitarylab.engine.algorithms import get_algorithm

# Programmatic access
qae_algo = get_algorithm('amplitude_estimation')
results = qae_algo.run(U, good_zero_qubits=[0], d=6)
```

---

## Related Algorithms

- [Quantum Phase Estimation](../qpe) - Foundation for phase extraction
- [Grover's Algorithm](../../cryptology/grover) - Provides Grover iteration basis
- [Amplitude Amplification](../amplitude_amplification) - Amplitude enhancement technique
- [HHL Algorithm](../../cryptology/hhl) - Application in linear system solving

---

## References

- Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000). "Quantum Amplitude Amplification and Estimation." arXiv preprint quant-ph/0005055.
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
- Suzuki, Y., Uno, S., Raymond, R., et al. (2021). "Amplitude estimation via maximum likelihood on noisy quantum computers." arXiv:2105.01814.
- Giurgescu-Seixas, S., et al. (2022). "Variational quantum amplitude estimation." arXiv:2204.13939.

---

## License

MIT License - Authored by Yuanchun He