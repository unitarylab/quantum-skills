---
name: quantum-phase-estimation
description: A quantum phase estimation algorithm that can estimate the eigenvalues of a unitary operator with high precision, which is a fundamental component in many quantum algorithms such as Shor's algorithm and quantum simulation.
---

## One Step to Run QPE Example
```bash
python ./scripts/algorithm.py
```

# Quantum Phase Estimation (QPE) Algorithm Skill Guide

## Overview

**Quantum Phase Estimation** (QPE) is one of the **most important algorithms in quantum computing**. It solves the following problem:

Given a unitary operator $U$ and one of its eigenstates $|\psi\rangle$ satisfying:
$$U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$$

where $\phi \in [0, 1)$ is an unknown phase, QPE estimates $\phi$ with **exponential precision** using only $d$ auxiliary qubits.

### Why QPE Matters:

1. **Foundation for Major Quantum Algorithms**:
   - **Shor's Algorithm**: Integer factorization in polynomial time
   - **HHL Algorithm**: Linear system solving
   - **Quantum Chemistry**: Computing molecular properties
   - **Variational Quantum Eigensolver (VQE)**: Finding ground states

2. **Exponential Precision**: With $d$ qubits, achieve precision of $1/2^d$ with query complexity $O(2^d)$

3. **Fundamental Building Block**: Other quantum algorithms are built on top of QPE

### Key Idea:

Convert the phase information (encoded in eigenvalues) into a **quantum superposition**, then use the **Quantum Fourier Transform (QFT)** to extract this information as measurable qubits.

---

## Learning Objectives

After mastering this skill, you will be able to:

1. Understand the mathematical principle of QPE
2. Explain how eigenvalues encode into phases
3. Understand Quantum Fourier Transform (QFT) and its inverse
4. Build controlled-unitary ($cU^{2^k}$) sequences
5. Use the provided `QPEAlgorithm` class effectively
6. Interpret measurement results and phase histograms
7. Apply QPE to estimate eigenvalues
8. Implement QPE from scratch
9. Extend to advanced applications (HHL, Shor's algorithm)

---

## Prerequisites

- **Essential knowledge**:
  - Complex numbers and polar form ($e^{i\theta}$)
  - Eigenvalues and eigenvectors of matrices
  - Controlled unitary gates (CU, CCX, etc.)
  - Quantum Fourier Transform basics
  - Understanding of measurement and probability
- **Mathematical comfort**: Linear algebra, discrete Fourier transforms
- **Recommended**: Study Hadamard Test first (simpler algorithm)

---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms.fundamental_algorithm import QPEAlgorithm
from engine import GateSequence
import numpy as np

# Step 1: Define the unitary U to estimate eigenvalues
# Example: T gate = P(π/4) has eigenvalues e^(iπ/4)
U = GateSequence(1)
U.p(np.pi / 4, 0)  # T gate

# Step 2: Prepare an eigenstate (T|1⟩ = e^(iπ/4)|1⟩)
# For T gate, |1⟩ is an eigenstate with eigenvalue e^(iπ/4)
prepare_eigenstate = GateSequence(1)
prepare_eigenstate.x(0)  # Creates |1⟩ state

# Step 3: Run QPE with d=5 phase qubits
algo = QPEAlgorithm()
result = algo.run(
    U=U,
    d=5,  # 5 bits precision → 1/32 accuracy
    prepare_target=prepare_eigenstate,
    backend='torch'
)

# Step 4: Interpret results
print(result['plot'])
print(f"Estimated phase: {result['estimated_phase']:.6f}")
print(f"Confidence (probability): {result['confidence_probability']:.4f}")

# Theory: T gate phase = π/4/(2π) = 1/8 = 0.125
print(f"Theoretical phase: 0.125000")
```

### Core Parameters Explained

```python
algo.run(
    U: GateSequence,                        # Unitary operator to analyze
    d: int,                                 # Number of phase register qubits
    prepare_target: Optional[GateSequence], # Circuit to prepare eigenstate
    backend: str = 'torch',                 # Simulation backend
    algo_dir: str = './qpe_results'         # Output directory
)
```

**Return dictionary contains:**
- `status`: "success"
- `estimated_phase`: The recovered phase $\hat{\phi}$
- `confidence_probability`: Probability of measuring the best phase
- `full_circuit`: The complete QPE circuit
- `circuit_path`: Path to the circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

**Precision**: With $d$ phase qubits, you get accuracy of order $1/2^d$

---

## Understanding the Core Components

### 1. Circuit Architecture

The complete QPE circuit has three stages:

```python
def build_qpe_circuit(self, U: GateSequence, d: int, 
                     prepare_target: Optional[GateSequence] = None) -> GateSequence:
    """
    Build complete QPE circuit.

    Structure:
    1. Initialize phase register (d qubits) in |0⟩
    2. Initialize target register in eigenstate |ψ⟩
    3. Apply Hadamard to all phase qubits
    4. Apply controlled-U^(2^k) for k=0..d-1
    5. Apply inverse QFT on phase register
    6. Measure phase register → binary representation of φ
    """
    n_target = U.get_num_qubits()
    gs = GateSequence(d + n_target)
    
    phase_qubits = list(range(d))
    target_qubits = list(range(d, d + n_target))
    
    # Stage 1: Prepare eigenstate on target register
    if prepare_target is not None:
        gs.append(prepare_target, target_qubits)
    
    # Stage 2: Superposition on phase register
    for q in phase_qubits:
        gs.h(q)
    
    # Stage 3: Controlled unitary applications (key computation)
    cU = U.control(1)  # Make U into controlled gate
    for k in range(d):
        power = 2 ** k
        for _ in range(power):
            # Apply controlled-U^(2^k) with phase_qubits[k] as control
            gs.append(cU, target_qubits + [phase_qubits[k]])
    
    # Stage 4: Inverse QFT extracts phase information
    iqft = IQFT(d)
    gs.append(iqft, phase_qubits)
    
    return gs
```

**Key insight**: The controlled-$U$ applications create a superposition where the phase qubit state encodes the eigenvalue phase.

### 2. Controlled-U Decomposition

The main computation is applying controlled-$U^{2^k}$:

$$\mathcal{U} = \prod_{k=0}^{d-1} (|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U^{2^k})$$

This is implemented as:

```python
# After Hadamard on phase qubits:
# State = (1/√(2^d)) Σ_j |j⟩ |ψ⟩

# For each phase qubit k:
cU = U.control(1)  # Controlled-U gate
for k in range(d):
    power = 2 ** k  # 2^0, 2^1, 2^2, ..., 2^(d-1)
    for _ in range(power):
        # Apply cU when phase_qubits[k] is |1⟩
        # This applies U^(2^k) to eigenstate
        gs.append(cU, target_qubits + [phase_qubits[k]])

# Result: Phase qubit j now encodes j·φ as relative phase
```

**Why binary powers?** Each phase qubit k controls $U^{2^k}$. When phase qubit state is $|j_{d-1}...j_0\rangle$, the total U applications = $\sum_{k} j_k \cdot 2^k = j$, giving overall phase $e^{2\pi i j \phi}$.

### 3. Quantum Fourier Transform (QFT)

The QFT converts phase information into measurable qubit states:

$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi i jk/N} |k\rangle$$

The Inverse QFT (iQFT) then performs:

```python
def _iqft_circuit(self, n: int) -> GateSequence:
    """
    Inverse Quantum Fourier Transform on n qubits.
    
    Extracts phase information from relative phases into qubit basis states.
    
    Structure:
    - Controlled phase gates (diagonal rotations)
    - Hadamard gates
    - Bit-reversal swaps
    """
    gs = GateSequence(n)
    
    # Bit reversal swaps
    for i in range(n // 2):
        gs.swap(i, n - 1 - i)
    
    # Core iQFT (inverse of QFT)
    for j in range(n):
        for k in range(0, j):
            # Controlled phase with negative angle (inverse)
            angle = -np.pi / (2 ** (j - k))
            gs.mcp(angle, k, j)  # Multi-controlled phase
        gs.h(j)
    
    return gs
```

**Effect**: If the state has phase $2\pi \cdot j \cdot \phi$, iQFT converts it so that measuring the phase qubits gives value close to $\lfloor 2^d \phi \rfloor$ with high probability.

### 4. Phase Extraction from Measurement

Convert measured bitstring to phase estimate:

```python
def extract_phase(phase_qubits: List[int], state_obj: State, d: int) -> float:
    """
    Extract phase from measurement results.
    """
    # Get probability distribution over phase qubits
    phase_probs = state_obj.probabilities_dict(phase_qubits, endian="little")
    
    # Find most probable measurement outcome
    best_bits = max(phase_probs, key=phase_probs.get)
    
    # Convert binary string to decimal
    # best_bits is a d-bit string, e.g., "01011"
    measured_int = int(best_bits, 2)  # Binary to decimal: 0..2^d - 1
    
    # Normalize to phase in [0, 1)
    phi_estimated = measured_int / (2 ** d)
    
    return phi_estimated
```

---

## Hands-On Example: Estimating T-Gate Phase

Let's estimate the eigenphase of the T gate where $T = P(\pi/4) = e^{i\pi/4}$:

```python
from engine.algorithms.fundamental_algorithm import QPEAlgorithm
from engine import GateSequence
import numpy as np

# Create T gate (Phase gate with angle π/4)
# T|1⟩ = e^(iπ/4)|1⟩ = e^(i·2π·(1/8))|1⟩
# So the eigenphase is 1/8 = 0.125
T_gate = GateSequence(1)
T_gate.p(np.pi / 4, 0)

# Prepare eigenstate |1⟩
prep_one = GateSequence(1)
prep_one.x(0)

# Run QPE with different precision levels
algo = QPEAlgorithm()

for d in [3, 4, 5]:
    result = algo.run(
        U=T_gate,
        d=d,
        prepare_target=prep_one
    )
    
    resolution = 1 / (2 ** d)
    print(f"d={d}: Estimated={result['estimated_phase']:.6f}, "
          f"Theoretical=0.125000, "
          f"Error<{resolution:.6f}")
```

**Expected output**:
```
d=3: Estimated=0.125000, Theoretical=0.125000, Error<0.125000
d=4: Estimated=0.125000, Theoretical=0.125000, Error<0.062500
d=5: Estimated=0.125000, Theoretical=0.125000, Error<0.031250
```

Notice how increasing $d$ narrows the uncertainty range!

---

## Implementing Your Own QPE

### Complete Implementation Template

```python
import numpy as np
from typing import Optional, Dict, Any
from engine import GateSequence, State

class MyQuantumPhaseEstimation:
    """User-implemented Quantum Phase Estimation."""
    
    def __init__(self):
        pass
    
    def run(self, U: GateSequence, d: int, 
            prepare_target: Optional[GateSequence] = None) -> Dict[str, Any]:
        """
        Main QPE pipeline.
        
        Args:
            U: Unitary whose eigenvalues we want to estimate
            d: Number of phase register qubits (precision)
            prepare_target: Circuit to prepare eigenstate (defaults to |0⟩)
        
        Returns:
            Dictionary with estimated phase and statistics
        """
        # Step 1: Build QPE circuit
        circuit = self.build_qpe_circuit(U, d, prepare_target)
        
        # Step 2: Execute simulation
        state = circuit.execute()
        
        # Step 3: Extract phase information
        phi = self.extract_phase_from_state(state, d)
        
        return {
            "estimated_phase": phi,
            "circuit": circuit,
            "message": f"Estimated phase: {phi:.6f}"
        }
    
    def build_qpe_circuit(self, U: GateSequence, d: int,
                         prepare_target: Optional[GateSequence]) -> GateSequence:
        """
        Construct the complete QPE circuit.
        
        Pipeline:
        1. Prepare eigenstate on target qubits
        2. Create superposition on phase qubits with Hadamard
        3. Apply controlled-U^(2^k) for exponential power scaling
        4. Apply inverse QFT to phase qubits
        """
        n_target = U.get_num_qubits()
        circuit = GateSequence(d + n_target)
        
        phase = list(range(d))
        target = list(range(d, d + n_target))
        
        # Stage 1: State preparation
        if prepare_target is not None:
            circuit.append(prepare_target, target)
        
        # Stage 2: Superposition via Hadamard
        for q in phase:
            circuit.h(q)
        
        # Stage 3: Controlled unitary applications with exponential scaling
        cU = U.control(1)
        for k in range(d):
            # Number of times to apply controlled-U^(2^k)
            power_of_two = 2 ** k
            
            # Apply 2^k times (faster than applying U separately k times)
            # This is due to binary encoding: we control 2^0, 2^1, 2^2, etc.
            for _ in range(power_of_two):
                circuit.append(cU, target + [phase[k]])
        
        # Stage 4: Inverse QFT on phase register
        iqft = self.build_iqft(d)
        circuit.append(iqft, phase)
        
        return circuit
    
    def build_iqft(self, n: int) -> GateSequence:
        """
        Build inverse Quantum Fourier Transform for n qubits.
        
        The iQFT converts relative phases into measurable basis states.
        """
        circuit = GateSequence(n)
        
        # Bit-reversal swaps (reorder qubits for measurement)
        for i in range(n // 2):
            circuit.swap(i, n - 1 - i)
        
        # Core iQFT (controlled phase gates + Hadamards)
        for j in range(n):
            # Controlled phase gates from earlier qubits
            for k in range(0, j):
                angle = -np.pi / (2 ** (j - k))
                circuit.mcp(angle, k, j)  # Multi-controlled phase
            
            # Hadamard gate
            circuit.h(j)
        
        return circuit
    
    def extract_phase_from_state(self, state, d: int) -> float:
        """
        Extract phase from final quantum state.
        
        Process:
        1. Compute measurement probabilities for phase qubits
        2. Find most probable outcome
        3. Convert to phase in [0, 1)
        """
        state_array = np.asarray(state, dtype=complex).reshape(-1)
        state_obj = State(state_array)
        
        # Get probability distribution on phase qubits (0..d-1)
        phase_qubits = list(range(d))
        probs = state_obj.probabilities_dict(phase_qubits, endian="little")
        
        # Find best (most probable) measurement outcome
        best_bits = max(probs, key=probs.get)
        
        # Convert from binary string to decimal
        # Example: "101" → 5
        measured_int = int(best_bits, 2)
        
        # Normalize to phase in [0, 1)
        phi = measured_int / (2 ** d)
        
        return phi
```

---

## Mathematical Deep Dive

### Initial State Preparation

Start with eigenstate: $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$

Create superposition on phase register:
$$|0^d\rangle |\psi\rangle \xrightarrow{H^{\otimes d} \otimes I} \frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1}|j\rangle|\psi\rangle$$

### Controlled-Unitary Application

Apply $\mathcal{U} = \prod_{k=0}^{d-1} (|0\rangle\langle 0|_k \otimes I + |1\rangle\langle 1|_k \otimes U^{2^k})$:

$$\frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1}|j\rangle|\psi\rangle \xrightarrow{\mathcal{U}} \frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1}e^{2\pi i j\phi}|j\rangle|\psi\rangle$$

**Key calculation**: For phase qubit state $|j\rangle$ where $j = \sum_{k=0}^{d-1} j_k 2^k$:
- Qubit $k$ contributes control signal $j_k$
- If $j_k=1$, apply $U^{2^k}$ to eigenstate
- Total phase shift: $e^{2\pi i \phi \sum_k j_k 2^k} = e^{2\pi i j\phi}$

### Inverse QFT Extraction

Apply iQFT:
$$\frac{1}{\sqrt{2^d}}\sum_{j=0}^{2^d-1}e^{2\pi i j\phi}|j\rangle \xrightarrow{\text{iQFT}} \text{High amplitude on } |k^*\rangle$$

where $k^* = \lfloor 2^d \phi \rfloor$ (the integer part of $2^d \phi$)

### Precision and Probability

Probability of measuring $k$ given true phase $\phi$:

$$P(k) = \frac{1}{2^{2d}}\frac{\sin^2(2^d(\phi - k/2^d))}{\sin^2(\phi - k/2^d)}$$

**Result**: 
- Peak probability $\approx 4/\pi^2 \approx 0.405$ (independent of $d$!)
- Error bound: $|\hat{\phi} - \phi | < 2^{-d}$ with high probability

### Query Complexity

- Each controlled-$U^{2^k}$ evaluation costs one query to $U$
- Need $\sum_{k=0}^{d-1} 2^k = 2^d - 1 \approx 2^d$ evaluations
- Total complexity: $O(2^d) = O(1/\epsilon)$ for $\epsilon$-accuracy
- Classical FFT would require $O(1/\epsilon^2)$ operations

---

## Debugging Tips

1. **Estimated phase looks random?**
   - Verify target qubits have correct order in circuit
   - Check that prepare_target creates a true eigenstate of U
   - If not eigensate, you'll see multiple peaks (not a single peak)

2. **Controlled-U not applying correctly?**
   - Verify control qubit index vs target qubits
   - Ensure control_sequence matches your intent ("0" vs "1")
   - Check that cU gate is properly constructed

3. **iQFT producing wrong output?**
   - Verify bit-reversal swaps are applied
   - Check phase gates have correct angles: $-\pi/2^k$
   - Ensure Hadamard gates are ordered correctly

4. **Precision not improving with larger d?**
   - Remember peak probability plateaus around 40%, doesn't improve with d
   - Measure best outcome, not expected value over distribution
   - Larger d just makes distribution narrower, not taller

5. **Circuit too large/slow?**
   - Doubling $d$ quadruples circuit gates (since we use $2^d$ controlled gates)
   - For $d>15$, expect significant slowdown on classical simulators
   - Consider approximate methods or decomposition for large $d$

---

## Further Learning

- **Related Algorithms**:
  - **Shor's Algorithm**: Uses QPE for order-finding (integer factorization)
  - **HHL Algorithm**: Solves linear systems using QPE
  - **Quantum Amplitude Estimation**: Uses QPE principles
- **Classic References**:
  - "Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer" (Shor, 1994)
  - "A fast quantum mechanical algorithm for database search" (Grover, 1996)

---

## Summary Checklist

Before considering yourself proficient, ensure you can:

- [ ] Explain why QPE is fundamental to quantum computing
- [ ] Understand the eigenvalue-to-phase encoding
- [ ] Describe controlled-U^(2^k) decomposition and why it works
- [ ] Explain QFT and iQFT roles in QPE
- [ ] Draw the complete QPE circuit from scratch
- [ ] Calculate precision: $d$ qubits → $1/2^d$ accuracy
- [ ] Use `QPEAlgorithm` for any unitary/eigenstate pair
- [ ] Interpret phase histogram and extract best estimate
- [ ] Implement QPE from scratch
- [ ] Understand query complexity: $O(2^d)$ or $O(1/\epsilon)$
- [ ] Apply successfully to T-gate, rotation gates, multi-qubit unitaries
- [ ] Debug common implementation issues
- [ ] Appreciate how QPE enables Shor's and HHL algorithms

---

## Real-World Impact: Why QPE Changes Everything

### Shor's Algorithm (Factoring)

1. Reduce factorization to finding period of $f(x) = a^x \mod N$
2. Use QPE to extract the eigenphase
3. Extract period from phase
4. Use classical post-processing to find factors

**Classical**: Super-polynomial time
**Quantum + QPE**: Polynomial time!

### HHL Algorithm (Linear Systems)

Solve $A|x\rangle = |b\rangle$:

1. Diagonalize $A$ using QPE
2. Invert eigenvalues (Phase rotation)
3. Uncompute using inverse QPE
4. Measure to get solution

**Classical Direct Solve**: $O(N^3)$ or $O(N^2)$ iterative
**Quantum + QPE**: $O(\log N)$ queries!

Both algorithms follow the same QPE pattern: **Estimate eigenvalues → Process → Amplitude Amplify → Measure**

---

Master Quantum Phase Estimation to unlock the full power of hybrid quantum-classical computing!
