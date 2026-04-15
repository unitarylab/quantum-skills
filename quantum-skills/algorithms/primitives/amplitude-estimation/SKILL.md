---
name: amplitude-estimation
description: A quantum algorithm for estimating the amplitude of a specific state in a quantum superposition, which can be used for various applications such as Monte Carlo simulations and optimization problems. Provides efficient implementations and educational resources for understanding and utilizing amplitude estimation in quantum algorithm development.
---

## One Step to Run Amplitude Estimation Example
```bash
python ./scripts/algorithm.py
```

# Amplitude Estimation Algorithm Skill Guide

## Overview

**Amplitude Estimation** (also called Quantum Amplitude Estimation or QAE) is a powerful quantum algorithm that estimates the success probability of a quantum state with **quadratic speedup** over classical methods.

**The Problem**: Given a quantum circuit that prepares a state:
$$U|0^{m+n}\rangle = \sqrt{p}|\text{good}\rangle + \sqrt{1-p}|\perp\rangle$$

where $p$ is the unknown **success probability** we want to estimate. Classical Monte Carlo methods require $O(1/\epsilon^2)$ samples to estimate $p$ to accuracy $\epsilon$. Amplitude Estimation achieves this in just $O(1/\epsilon)$ queries!

### How It Works:
1. **Grover Iterations**: Amplify the amplitude difference between good and bad states
2. **Quantum Phase Estimation (QPE)**: Encode the amplitude information into phase information
3. **Phase Measurement**: Extract the phase to recover the success probability

### Key Characteristics:
- **Quadratic Speedup**: $O(1/\epsilon)$ vs classical $O(1/\epsilon^2)$
- **Combines Two Algorithms**: Amplitude Amplification + Quantum Phase Estimation
- **Versatile Applications**: Monte Carlo simulations, option pricing, optimization
- **Phase-Based**: Converts amplitude problems to eigenvalue problems

---

## Learning Objectives

After mastering this skill, you will be able to:

1. Understand the mathematical connection between amplitudes and phases
2. Use the provided `AmplitudeEstimationAlgorithm` class effectively
3. Design oracle operators for custom success predicates
4. Implement and apply Quantum Phase Estimation (QPE)
5. Analyze phase histograms to extract amplitude estimates
6. Implement amplitude estimation from scratch
7. Apply to real-world problems (Monte Carlo, option pricing)

---

## Prerequisites

- **Strong prerequisites**: 
  - Grover's algorithm and amplitude amplification fundamentals
  - Quantum Phase Estimation (QPE) basics
  - Understanding of controlled unitary gates
  - Comfortable with Python, numpy, and phase/angle calculations
- **Recommended reading**: [Amplitude Amplification SKILL](../amplitude-amplification/SKILL.md)

---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms import AmplitudeEstimationAlgorithm
from engine import GateSequence

# Step 1: Create a state preparation circuit
U = GateSequence(2)  # 2-qubit register
U.h(0)              # Hadamard on qubit 0
U.h(1)              # Hadamard on qubit 1
# After this: U|00⟩ = 1/2(|00⟩ + |01⟩ + |10⟩ + |11⟩)

# Step 2: Define which qubits determine "good" state
# In this case, good = |00⟩ (both qubits are 0)
good_zero_qubits = [0, 1]

# Step 3: Run amplitude estimation
algo = AmplitudeEstimationAlgorithm()
result = algo.run(
    U=U,
    good_zero_qubits=good_zero_qubits,
    d=6,  # Phase register precision (6 bits)
    backend='torch'
)

# Step 4: Inspect results
print(result['plot'])  # Beautiful ASCII output
print(f"Estimated p: {result['estimated_amplitude']:.6f}")
print(f"True p: 0.25")  # For uniform superposition over 4 states
print(f"Measured phase: {result['phi']:.6f}")
```

### Core Parameters Explained

```python
algo.run(
    U: GateSequence,           # State preparation unitary
    good_zero_qubits: List[int],  # Qubits that must be |0⟩ to define "good"
    d: int = 6,                # Precision (# of phase register qubits)
    backend: str = 'torch',    # Simulation backend ('torch', 'numpy', etc.)
    algo_dir: str = './qae_results'  # Output directory
)
```

**Return dictionary contains:**
- `status`: "success" (always on valid input)
- `estimated_amplitude`: The recovered success probability $\hat{p}$
- `phi`: The measured phase $\phi$ in range [0, 0.5]
- `histogram`: Raw phase measurement distribution
- `circuit_path`: Location of generated circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

### Understanding the Phase-to-Amplitude Conversion

The algorithm measures a phase and converts it back to probability:

$$\phi_{\text{measured}} \approx \frac{\theta}{\pi}, \quad \text{where} \quad q = \sin^2(\theta)$$

Therefore:
$$\hat{p} = \sin^2(\pi \cdot \phi)$$

The `histogram` shows all measured phases ranked by probability. The **peak corresponds to the most likely estimate**.

---

## Understanding the Core Components

### 1. Grover Operator (Amplitude Amplification)

The Grover operator $Q = S_\psi S_f$ encodes amplitude information as phase:

```python
def _grover_operator_from_zero_oracle(self, U: GateSequence, 
                                      good_zero_qubits: List[int]) -> GateSequence:
    """
    Construct a single Grover iteration.
    
    Key insight: The eigenvalues of Q are e^{±i2θ}, where:
    - θ = arcsin(√p) for success probability p
    - This means the phase encodes the amplitude!
    """
    n_data = U.get_num_qubits()
    data = list(range(n_data))
    ancilla = n_data
    
    gs = GateSequence(n_data + 1, name="GroverIter")
    
    # S_f: Phase flip on "good" states
    self._phase_oracle_all_zeros(gs, zero_qubits=good_zero_qubits, ancilla=ancilla)
    
    # S_ψ: Reflection about prepared state
    self._diffusion_about_prepared_state(gs, U=U, data_qubits=data, ancilla=ancilla)
    
    # Global phase correction (multiply by -1 for QPE compatibility)
    self._prepare_kickback_ancilla_minus(gs, ancilla)
    gs.x(ancilla)
    self._unprepare_kickback_ancilla_minus(gs, ancilla)
    
    return gs
```

**Physical meaning**: Applying $Q$ repeatedly rotates the state in the 2D subspace spanned by $|\text{good}\rangle$ and $|\text{bad}\rangle$. The rotation angle carries amplitude information!

### 2. Quantum Phase Estimation (QPE)

QPE extracts eigenvalue information from $Q$ by storing it in a phase register:

```python
def _qpe_circuit(self, U: GateSequence, d: int, 
                prepare_target: Optional[GateSequence] = None) -> GateSequence:
    """
    Build the complete QPE circuit.
    
    Structure:
    1. Initialize d phase qubits in superposition (H gates)
    2. Apply controlled-U^(2^k) for each phase qubit k
    3. Apply inverse QFT to extract phase information
    
    Returns d + n_target qubit circuit (d phase, n_target data)
    """
    n_target = U.get_num_qubits()
    gs = GateSequence(d + n_target, name=f"QPE_d{d}")
    
    phase = list(range(d))              # Phase register qubits
    target = list(range(d, d + n_target))  # Target (Grover) qubits
    
    # Prepare target state
    if prepare_target is not None:
        gs.append(prepare_target, target)
    
    # Hadamard on all phase qubits
    for q in phase:
        gs.h(q)
    
    # Controlled application of U^(2^k)
    cU = U.control(1)  # Make U a controlled gate
    for k in range(d):
        power = 2 ** k
        for _ in range(power):
            gs.append(cU, target + [phase[k]])
    
    # Inverse QFT to measure phase
    iqft = self._iqft_circuit(d, do_swaps=True)
    gs.append(iqft, phase)
    
    return gs
```

**Complexity insight**: The outer loop runs $2^0 + 2^1 + ... + 2^{d-1} = 2^d - 1$ controlled applications. This is why increasing $d$ (precision) doubles the circuit depth!

### 3. Inverse Quantum Fourier Transform (iQFT)

The iQFT converts from time domain to frequency domain:

```python
def _iqft_circuit(self, n: int, do_swaps: bool = True) -> GateSequence:
    """
    Inverse QFT for n qubits.
    
    Structure:
    - Conditional phase gates (CPhase)
    - Hadamard gates
    - Bit-reversal swaps at the end
    """
    gs = GateSequence(n, name=f"iQFT_{n}")
    
    # Bit reversals help with measurement order
    if do_swaps:
        for i in range(n // 2):
            gs.swap(i, n - 1 - i)
    
    # iQFT core
    for j in range(n):
        for k in range(0, j):
            angle = -np.pi / (2 ** (j - k))
            gs.mcp(angle, k, j)  # Multi-controlled phase gate
        gs.h(j)
    
    return gs
```

### 4. Phase Histogram Extraction

Convert the measured statevector into a probability distribution over phases:

```python
def _phase_histogram(self, statevector: np.ndarray, d: int) -> dict[str, float]:
    """
    Extract the probability distribution of the phase register.
    
    Process:
    1. Compute |⟨i|ψ⟩|² for all computational basis states |i⟩
    2. Extract the phase register bits (lowest d bits)
    3. Sum probabilities for each unique phase value
    4. Return sorted by probability (descending)
    """
    probs = np.abs(statevector) ** 2  # Measurement probabilities
    counts: dict[str, float] = {}
    modulus = 2 ** d
    
    for idx, p in enumerate(probs):
        if p < 1e-12:  # Skip negligible probabilities
            continue
        k = idx % modulus         # Extract phase register (lowest d bits)
        bits = format(k, f'0{d}b')  # Binary representation
        counts[bits] = counts.get(bits, 0.0) + float(p)
    
    # Sort by probability descending
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
```

---

## Hands-On Example: Estimating Custom Success Probability

Let's estimate the success probability for a 1-qubit state with controlled amplitude:

```python
from engine.algorithms.fundamental_algorithm import AmplitudeEstimationAlgorithm
from unitary import GateSequence
import math

# Target success probability
p_target = 0.36

# Construct state: U|0⟩ = √p|0⟩ + √(1-p)|1⟩
# Using RY gate: RY(2θ)|0⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
# We want cos²(θ) = p, so θ = arccos(√p)
theta = math.acos(math.sqrt(p_target))

U_custom = GateSequence(1)
U_custom.ry(2 * theta, 0)

# Run amplitude estimation
algo = AmplitudeEstimationAlgorithm()
result = algo.run(
    U=U_custom,
    good_zero_qubits=[0],  # Good state = |0⟩
    d=7,  # Use 7 bits for better precision
    backend='torch'
)

print("=" * 50)
print(f"Target probability: {p_target:.6f}")
print(f"Estimated probability: {result['estimated_amplitude']:.6f}")
print(f"Measured phase: {result['phi']:.6f}")
print(f"Top 3 phases measured:")
for bits, prob in list(result['histogram'].items())[:3]:
    print(f"  {bits}: {prob:.4f}")
```

**Expected output**: Estimated amplitude should be very close to 0.36 (within ~1/2^7 ≈ 0.008)

---

## Implementing Your Own Amplitude Estimation

### Complete Implementation Template

```python
import math
import numpy as np
from typing import List, Optional, Dict, Any

class MyAmplitudeEstimation:
    """User-implemented quantum amplitude estimation."""
    
    def __init__(self):
        self.log_prefix = "INFO: [My QAE] "
    
    def run(self, U, good_zero_qubits: List[int], d: int = 6) -> Dict[str, Any]:
        """
        Main amplitude estimation pipeline.
        
        Args:
            U: State preparation unitary
            good_zero_qubits: Indices of qubits defining "good" state
            d: Number of phase register qubits (precision)
        
        Returns:
            Dictionary with estimated probability and histogram
        """
        print(f"Starting Amplitude Estimation (d={d} bits precision)...")
        
        # Step 1: Build Grover operator (converts amplitude to phase)
        print("  Building Grover operator...")
        grover_op = self._build_grover(U, good_zero_qubits)
        
        # Step 2: Build QPE circuit using Grover as unitary
        print("  Building QPE circuit...")
        qpe_circuit = self._build_qpe(grover_op, d, U, good_zero_qubits)
        
        # Step 3: Execute simulation
        print("  Running quantum simulation...")
        statevector = qpe_circuit.execute()
        
        # Step 4: Extract and analyze phase histogram
        print("  Analyzing results...")
        histogram = self._extract_phase_histogram(statevector, d)
        best_phase = self._find_best_phase(histogram)
        
        # Step 5: Convert phase to amplitude
        estimated_p = self._phase_to_amplitude(best_phase)
        
        return {
            "estimated_amplitude": estimated_p,
            "best_phase": best_phase,
            "histogram": histogram,
            "message": f"Estimated p ≈ {estimated_p:.6f}"
        }
    
    def _build_grover(self, U, good_zero_qubits: List[int]):
        """
        Build single Grover iteration.
        
        Structure: Q = (2|ψ⟩⟨ψ| - I) * (I - 2|good⟩⟨good|)
        Eigenvalues: e^{±i2θ} where sin(θ) = √p
        """
        # Pseudocode structure:
        # 1. Phase oracle: flip phase when good_zero_qubits are all |0⟩
        # 2. Diffuser: reflection about prepared state
        # 3. Global phase correction for QPE
        pass
    
    def _build_qpe(self, U, d: int, prepare_U, good_zero_qubits: List[int]):
        """
        Build full QPE circuit.
        
        Structure:
        1. State preparation on target register
        2. Hadamard on all phase register qubits
        3. Controlled-U^(2^k) applications
        4. Inverse QFT
        """
        # Pseudocode:
        # circuit = GateSequence(d + n_qubits)
        # Initialize phase qubits with H gates
        # Apply controlled Grover iterations with exponential scaling
        # Apply inverse QFT
        # Return circuit
        pass
    
    def _extract_phase_histogram(self, statevector, d: int) -> Dict[str, float]:
        """
        Extract phase register probability distribution.
        
        Returns: {bitstring: probability} sorted by probability descending
        """
        probs = np.abs(statevector) ** 2
        histogram = {}
        
        for idx, p in enumerate(probs):
            if p < 1e-12:
                continue
            # Extract lowest d bits (phase register)
            phase_idx = idx % (2 ** d)
            bits = format(phase_idx, f'0{d}b')
            histogram[bits] = histogram.get(bits, 0.0) + p
        
        # Sort by probability descending
        return dict(sorted(histogram.items(), 
                          key=lambda x: x[1], 
                          reverse=True))
    
    def _find_best_phase(self, histogram: Dict[str, float]) -> float:
        """Find the most probable phase from histogram."""
        best_bits = next(iter(histogram))
        d = len(best_bits)
        phi_raw = int(best_bits, 2) / (2 ** d)
        # Fold to [0, 0.5] (QPE returns symmetric phases)
        return min(phi_raw, 1.0 - phi_raw)
    
    def _phase_to_amplitude(self, phi: float) -> float:
        """
        Convert phase measurement to success probability.
        
        Relationship: φ ≈ θ/π where sin²(θ) = p
        Therefore: p ≈ sin²(π·φ)
        """
        return float(np.sin(np.pi * phi) ** 2)
```

---

## Mathematical Deep Dive

### State Preparation and Setup

Start with a unitary $U$ that creates:
$$U|0^{m+n}\rangle = \sqrt{p}|\text{good}\rangle + \sqrt{1-p}|\text{bad}\rangle$$

Define angle $\theta$ such that:
$$\sqrt{p} = \sin(\theta), \quad \sqrt{1-p} = \cos(\theta)$$

Therefore: $p = \sin^2(\theta)$ for $\theta \in (0, \pi/2)$

### Grover Operator Eigenvalues

The Grover iteration $Q = S_\psi S_f$ acts on the 2D subspace $\{\text{good}, \text{bad}\}$. In this subspace:

$$Q = \begin{pmatrix} \cos(2\theta) & -\sin(2\theta) \\ \sin(2\theta) & \cos(2\theta) \end{pmatrix}$$

This is a rotation by angle $2\theta$. The eigenvalues are:
$$\lambda_\pm = e^{\pm i 2\theta}$$

**Crucial insight**: The phase information $2\theta$ is encoded in the eigenvalues!

### QPE Extraction

Quantum Phase Estimation with $d$ qubits measures the phase with precision $2\pi / 2^d$:

$$\phi_{\text{measured}} \in \{0, 1, 2, \ldots, 2^d - 1\} / 2^d$$

The measured phase corresponds to:
$$\phi_{\text{measured}} \approx \frac{2\theta}{2\pi} = \frac{\theta}{\pi}$$

(Note: QPE may return $\phi$ or $1 - \phi$ equally; we take $\min(\phi, 1-\phi)$)

### Complete Recovery Formula

$$\phi_{\text{measured}} \approx \frac{\theta}{\pi} \implies \theta \approx \pi \cdot \phi$$

$$\hat{p} = \sin^2(\theta) = \sin^2(\pi \cdot \phi)$$

### Precision Analysis

- **Resolution**: $\Delta\phi = 2^{-d}$
- **Amplitude error**: $\Delta p \lesssim \pi \cdot 2^{-d}$ (worst case)
- **Success probability of peak**: $P_{\text{peak}} \gtrsim 4/\pi^2 \approx 0.405$ (independent of $d$)
- **Query complexity**: $O(2^d)$ Grover iterations per execution
- **Total complexity for $\epsilon$-accuracy**: $O(1/\epsilon)$ queries vs classical $O(1/\epsilon^2)$

---

## Debugging Tips

1. **Phase doesn't match theory?**
   - Check if `good_zero_qubits` matches your intended "good" state
   - Verify state preparation unitary $U$ is correct
   - Try increasing $d$ for better resolution

2. **Histogram has multiple peaks?**
   - This is normal! Peaks at $\phi$ and $1-\phi$ have equal height
   - Take $\min(\phi, 1-\phi)$ to recover the phase (code does this)
   - Multiple peaks suggest inadequate precision (increase $d$)

3. **Estimated amplitude unreasonable?**
   - Verify $\phi \in [0, 0.5]$ after folding
   - Check: $\hat{p} = \sin^2(\pi \cdot \phi)$ is between 0 and 1
   - If $\phi > 0.5$, use $\phi' = 1 - \phi$

4. **Circuit too large?**
   - QPE requires $O(2^d)$ controlled gates
   - For $d=10$, this is 1,024 gates
   - Try smaller $d$ first, increase only as needed

---