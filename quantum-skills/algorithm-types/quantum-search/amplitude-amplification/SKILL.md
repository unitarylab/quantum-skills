---
name: amplitude-amplification
description: Quantum Amplitude Amplification - A universal quantum technique to boost the success probability of target states through Grover iteration. Supports arbitrary state preparation and flexible iteration configuration.
license: MIT
metadata:
    skill-author: Yuanchun He
---

# Quantum Amplitude Amplification

## Algorithm Overview

Amplitude Amplification is a fundamental quantum technique based on Grover's search algorithm, designed to **amplify** the amplitude of target quantum states. When a quantum circuit prepares a target state with low initial probability $p_0$, amplitude amplification leverages iterative phase-reversal and diffusion operations to boost the success probability approaching 1, with iteration complexity merely $O(1/\sqrt{p_0})$—achieving **quadratic speedup** compared to classical methods.

### Key Applications

- **Grover's Search**: Locate specific elements in unstructured databases
- **Quantum Sampling**: Amplify probabilities of desired computational paths
- **Quantum Optimization**: Enhance success rates for optimal solutions
- **Algorithm Acceleration**: Boost success probabilities in hybrid quantum-classical algorithms

---

## Mathematical Foundation

### Problem Formulation

Consider a quantum circuit $U_{\psi_0}$ that prepares a state:

$$U_{\psi_0}|0\rangle^{m+n} = \sqrt{p_0}|0\rangle^{m}|\psi_0\rangle + \sqrt{1-p_0}|\perp\rangle$$

Where:
- $m$ auxiliary qubits encode success/failure indicators
- $n$ data qubits store the target state $|\psi_0\rangle$
- $p_0$ is the initial success probability ($0 < p_0 < 1$)
- $|\perp\rangle$ represents orthogonal background states

### Grover Iteration Mechanism

**Step 1: Phase-Reversal Oracle** - Flip the phase of states with first $m$ qubits all zero:

$$O_f = (I^{\otimes m} - 2|0\rangle^m\langle 0|^m) \otimes I^{\otimes n}$$

**Step 2: Diffusion Operator** - Reflection around the initial state $|s\rangle$:

$$D = U_{\psi_0}(2|0^{m+n}\rangle\langle 0^{m+n}|-I)U_{\psi_0}^{\dagger}$$

**Step 3: Grover Iteration Operator**:

$$G = D \cdot O_f$$

### Amplification Result

After $k$ iterations, the success probability becomes:

$$P_k = \sin^2\big((2k+1)\arcsin(\sqrt{p_0})\big)$$

Choosing $k \approx \frac{\pi}{4\sqrt{p_0}}$ drives $P_k \to 1$

---

## Algorithm Features

| Feature | Description |
|---------|-------------|
| **Speedup Factor** | $O(\sqrt{N})$ - quadratic speedup over classical $O(N)$ |
| **Iteration Complexity** | $k = O(1/\sqrt{p_0})$ |
| **Universality** | Applicable to arbitrary initial states and target conditions |
| **Scalability** | Supports multiple simultaneous target state amplification |
| **Backend Support** | PyTorch and other quantum simulation backends |

---

## Usage Guide

### Prerequisites

```python
from engine.algorithms.fundamental_algorithm.amplitude_amplification import AmplitudeAmplificationAlgorithm
from engine.core import GateSequence, Register, State
import numpy as np
```

### Step 1: Define State Preparation Circuit

Create a `GateSequence` object specifying how to prepare your target state:

```python
# Example: Prepare uniform superposition or custom state
U = GateSequence(n_qubits=3, backend='torch')

# Add quantum gates to prepare the desired state
U.h(0)      # Hadamard on qubit 0
U.h(1)      # Hadamard on qubit 1
U.h(2)      # Hadamard on qubit 2
# U now prepares |+++> (uniform superposition) from |000>
```

### Step 2: Instantiate Algorithm

```python
aa = AmplitudeAmplificationAlgorithm()
```

### Step 3: Configure Algorithm Parameters

Key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `U` | `GateSequence` | State preparation circuit (without ancilla qubits) |
| `good_zero_qubits` | `List[int]` | Indices of qubits defining "good" states (must be \|0\rangle) |
| `p` | `float` | Initial success probability, $0 < p < 1$ |
| `reps` | `int` (optional) | Manually specify iteration count (overrides auto-calculation) |
| `backend` | `str` | Quantum simulation backend, default: 'torch' |
| `algo_dir` | `str` | Result output directory, default: './aa_results' |

### Step 4: Execute Algorithm

```python
# Method 1: Auto-compute iterations from initial probability
results = aa.run(
    U=U,
    good_zero_qubits=[2],           # Qubit 2 must be |0> for success
    p=0.125,                        # Initial probability 1/8
    backend='torch',
    algo_dir='./amplitude_amplification_results'
)

# Method 2: Manually specify iteration count
results = aa.run(
    U=U,
    good_zero_qubits=[1, 2],        # Both qubits 1 and 2 must be |0>
    p=0.25,
    reps=3,                         # Execute 3 iterations
    backend='torch'
)
```

### Step 5: Parse Results

```python
# Results contain key information:
target_probability = results['target_probability']    # Amplified probability
success = results['is_success']                       # Amplification successful?
circuit_path = results.get('circuit_path', None)      # Path to circuit diagram
formatted_data = results.get('formatted_data', {})    # Probability distribution

print(f"Success probability amplified from {results['initial_prob']:.4f} to {target_probability:.4f}")
print(f"Amplification result: {'Successful' if success else 'Failed'}")
```

---

## Practical Examples

### Example 1: Simple Search Problem

Search among 8 items for a single target (initial probability $p=1/8$):

```python
from engine.algorithms.fundamental_algorithm.amplitude_amplification import AmplitudeAmplificationAlgorithm
from engine.core import GateSequence

# Prepare 3-qubit uniform superposition
U = GateSequence(n_qubits=3, backend='torch')
U.h(0)
U.h(1)
U.h(2)

# Define target: all qubits should be |0>
aa = AmplitudeAmplificationAlgorithm()
results = aa.run(
    U=U,
    good_zero_qubits=[0, 1, 2],
    p=1/8,                          # Initial probability
    backend='torch'
)

print(f"Final success probability: {results['target_probability']:.4f}")
```

### Example 2: Fixed Iteration Count

```python
# Use fixed iterations instead of auto-calculation
results = aa.run(
    U=U,
    good_zero_qubits=[0],
    p=0.25,
    reps=2,                         # Execute exactly 2 iterations
    backend='torch'
)
```

### Example 3: Multi-Target Amplification

Define multiple "good state" conditions:

```python
# Target: first two qubits must both be |0> (composite condition)
results = aa.run(
    U=U,
    good_zero_qubits=[0, 1],        # Both conditions required
    p=0.25,
    backend='torch',
    algo_dir='./multi_target_results'
)
```

### Example 4: Custom Phase Gate with Amplitude Amplification

```python
# More complex state preparation
U = GateSequence(n_qubits=3, backend='torch')
U.h(0)
U.rz(0, np.pi/4)                    # Add phase to first qubit
U.cnot(0, 1)                        # Entangle with second qubit
U.h(2)

results = aa.run(
    U=U,
    good_zero_qubits=[2],
    p=0.1875,                       # 3/16 initial probability
    backend='torch'
)
```

---

## Performance Analysis

| Metric | Description |
|--------|-------------|
| **Qubit Support** | Arbitrary number (limited by simulator memory) |
| **Iteration Complexity** | $O(1/\sqrt{p_0})$ |
| **Query Complexity** | $O(1/\sqrt{p_0})$ |
| **Space Complexity** | $O(2^n)$ (state vector size) |
| **Classical Preprocessing** | $O(n)$ (optimal iteration calculation) |

### Complexity Comparison

| Method | Iterations Needed | Success Probability |
|--------|------------------|-------------------|
| Classical Random Sampling | $O(1/p_0)$ | Stochastic |
| Quantum Amplitude Amplification | $O(1/\sqrt{p_0})$ | Deterministic |
| **Speedup** | **$\sqrt{p_0}$ times faster** | — |

---

## Relationship to Grover's Algorithm

- **Grover's Search** is a special case of amplitude amplification ($p_0 = 1/N$)
- **Amplitude Amplification** generalizes Grover to arbitrary initial probabilities
- Both share identical Oracle and Diffusion mechanisms

### When to Use Each

- Use **Grover's Algorithm** for unstructured database search
- Use **Amplitude Amplification** for boosting probabilities in any quantum circuit

---

## Integration with UnitaryLab

This SKILL integrates with UnitaryLab's quantum algorithm framework:

```python
from unitarylab.engine.algorithms import get_algorithm

# Programmatic access
aa_algo = get_algorithm('amplitude_amplification')
results = aa_algo.run(U, good_zero_qubits=[0], p=0.25)
```

---

## Related Algorithms

- [Hadamard Transform](../hadamard_transform) - Foundation of quantum superposition
- [Quantum Phase Estimation](../qpe) - Phase extraction via quantum inference
- [Grover's Search](https://en.wikipedia.org/wiki/Grover's_algorithm) - Unstructured database search

---

## References

- Grover, L. K. (1996). "A Fast Quantum Mechanical Algorithm for Database Search." *Physical Review Letters*, 79(2), 325.
- Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000). "Quantum Amplitude Amplification and Estimation." *Contemporary Mathematics*, 305, 53-74.
- Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.

---

## License

MIT License - Authored by Yuanchun He
