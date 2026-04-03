---
name: hadamard-test
description: A quantum algorithm that uses the Hadamard test to estimate the expectation value of a unitary operator with respect to a given quantum state. This algorithm is fundamental in quantum computing and has applications in various quantum algorithms, including quantum phase estimation and variational quantum algorithms.
---

# Hadamard Test Algorithm Skill Guide

## 📚 Overview

**Hadamard Test** is a fundamental quantum algorithm that uses a single **ancilla (auxiliary) qubit** to estimate the complex expectation value of a unitary operator $U$ with respect to a quantum state $|\psi\rangle$:

$$\langle\psi|U|\psi\rangle \in \mathbb{C}$$

### The Key Insight

Instead of measuring the target qubits directly, we measure the ancilla qubit which encodes the unitary's effect as **measurement statistics**. This is far more powerful because it allows us to extract information about $U$ without understanding it classically.

### Core Modes:

1. **Expectation Mode**: Estimate $\text{Re}(\langle\psi|U|\psi\rangle)$ or $\text{Im}(\langle\psi|U|\psi\rangle)$
2. **Swap Test Mode**: Measure state overlap $|\langle\phi|\psi\rangle|^2$
3. **Phase Estimation Mode**: Recover the full eigenphase $\phi$ by combining real and imaginary parts

### Real-World Applications:

- **Variational Quantum Algorithms (VQA)**: Evaluate objective functions in VQE, QAOA
- **Quantum Machine Learning**: Compute kernel matrices, overlap measurements
- **Quantum Chemistry**: Calculate molecular properties and expectation values
- **Phase Detection**: Extract eigenvalues and eigenphases from unitaries

---

## 🎯 Learning Objectives

After mastering this skill, you will be able to:

1. ✅ Understand the mathematical principle of Hadamard test
2. ✅ Build Hadamard test circuits for real and imaginary parts
3. ✅ Use the provided `HadamardTestAlgorithm` class effectively
4. ✅ Implement state overlap (swap test) measurements
5. ✅ Extract eigenphases from unitaries
6. ✅ Handle shot noise and statistical sampling
7. ✅ Apply to real quantum algorithm problems
8. ✅ Implement Hadamard test from scratch

---

## 📋 Prerequisites

- **Essential knowledge**:
  - Hadamard gate and its properties
  - Controlled unitary gates (CU)
  - Basic quantum state preparation
  - Understanding of measurement and expectation values
  - Pauli-Z measurements and probability statistics
- **Mathematical comfort**: Complex numbers, trigonometry, linear algebra

---

## 🚀 Using the Provided Implementation

### Quick Start Example

```python
from algorithm import HadamardTestAlgorithm
from engine.core import GateSequence

# Step 1: Create a unitary U to evaluate
U = GateSequence(1)
U.rz(0.8, 0)  # Rotation Z by 0.8 radians

# Step 2: Prepare the quantum state |ψ⟩
prepare_psi = GateSequence(1)
prepare_psi.h(0)  # |ψ⟩ = |+⟩ = H|0⟩

# Step 3: Run Hadamard test (real part)
algo = HadamardTestAlgorithm()
result = algo.run(
    mode="expectation",
    U=U,
    prepare_psi=prepare_psi,
    imag=False,  # Measure real part
    shots=30000,
    backend='torch'
)

# Step 4: Inspect results
print(result['plot'])
print(f"Estimated Re(<ψ|U|ψ>): {result['estimated_value']:.6f}")

# For imaginary part, run again with imag=True
result_imag = algo.run(
    mode="expectation",
    U=U,
    prepare_psi=prepare_psi,
    imag=True,   # Now measure imaginary part
    shots=30000,
    backend='torch'
)

# Combine to get full complex number
re_part = result['estimated_value']
im_part = result_imag['estimated_value']
full_expectation = re_part + 1j * im_part
print(f"Full expectation value: {full_expectation}")
```

### Core Parameters Explained

```python
algo.run(
    mode: str = "expectation",     # "expectation", "swap_test", or "phase_estimation"
    U: Optional[GateSequence] = None,       # Unitary to evaluate (not needed for swap_test)
    prepare_psi: Optional[GateSequence] = None,  # State preparation |ψ⟩
    prepare_phi: Optional[GateSequence] = None,  # State preparation |φ⟩ (swap_test only)
    imag: bool = False,                    # Measure imaginary (True) or real (False) part
    shots: int = 20000,                    # Number of measurement samples
    backend: str = 'torch',                # Simulation backend
    algo_dir: str = './hadamard_test_results'  # Output directory
)
```

**Return dictionary contains:**
- `status`: "success"
- `estimated_value`: The measured expectation value or overlap
- `circuit_paths`: Paths to generated circuit diagrams
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

---

## 🔧 Understanding the Core Components

### 1. Hadamard Test Circuit (Real Part)

The circuit that estimates $\text{Re}(\langle\psi|U|\psi\rangle)$:

```python
def _build_hadamard_test_circuit(
    self, U: GateSequence, 
    prepare_psi: Optional[GateSequence] = None, 
    imag: bool = False
) -> GateSequence:
    """
    Build the Hadamard test circuit.
    
    Structure:
    1. Initialize ancilla in |0⟩
    2. Apply H to ancilla
    3. If imag=True, apply S† (phase gate)
    4. Prepare state |ψ⟩ on target register
    5. Apply controlled-U (ancilla controls)
    6. Apply H to ancilla
    7. Measure ancilla
    
    Key insight: Controlled-U encodes U's effect onto ancilla's measurement!
    """
    n = U.get_num_qubits()
    qc = GateSequence(1 + n, name="HadamardTest")
    anc = 0           # Ancilla qubit (index 0)
    tgt = list(range(1, 1 + n))  # Target qubits
    
    # Prepare ancilla in |+⟩ state
    qc.h(anc)
    
    # For imaginary part, apply phase gate S†
    if imag:
        qc.sdag(anc)   # Equivalent to adding -i phase
    
    # Prepare target state
    if prepare_psi is not None:
        qc.append(prepare_psi, target=tgt)
    
    # Key step: Controlled-U with ancilla as control
    # control_sequence="1" means U is applied when ancilla=1
    qc.append(U, target=tgt, control=[anc], control_sequence="1")
    
    # Final Hadamard on ancilla to extract phase
    qc.h(anc)
    
    return qc
```

**Physical meaning**:
- The H gates create a superposition in the ancilla
- The controlled-U introduces a **relative phase** between $|0\rangle$ and $|1\rangle$ branches proportional to $\langle\psi|U|\psi\rangle$
- The final H converts this phase into **measurement probabilities**: $P(0) - P(1) = \text{Re}(\langle\psi|U|\psi\rangle)$

### 2. Real vs. Imaginary Parts

To measure the **complex** expectation value, run two circuits:

```python
# Real part circuit (no phase gate)
gs_real = self._build_hadamard_test_circuit(U, prepare_psi, imag=False)

# Imaginary part circuit (with S† phase gate)
gs_imag = self._build_hadamard_test_circuit(U, prepare_psi, imag=True)

# The S† gate adds a -i phase, flipping the role of real/imaginary
# Result: Re version gives Re(<ψ|U|ψ>), Im version gives Im(<ψ|U|ψ>)
```

**Why S†?** The measurement statistics transform as:
- $\langle Z \rangle = p(0) - p(1) = \text{Re}(\langle\psi|U|\psi\rangle)$ (real)
- After S†: $\langle Z \rangle = \text{Im}(\langle\psi|U|\psi\rangle)$ (imaginary)

### 3. Measurement and Sampling

Extract the expectation value from measurement statistics:

```python
def _extract_measurement(self, circ: GateSequence, shots: int):
    """
    Measure ancilla and compute <Z> = p(0) - p(1)
    """
    # Get exact probabilities from statevector
    statevec = self._as_statevector(circ.execute())
    state_obj = State(statevec)
    
    # Extract ancilla qubit (index 0) probability distribution
    anc_probs = state_obj.probabilities_dict([0], endian="little")
    p0_exact = float(anc_probs.get("0", 0.0))
    
    # Simulate shot noise (optional)
    if shots is not None and shots > 0:
        # Binomial sampling: if p0 is true probability, 
        # with 'shots' trials, expect ~shots*p0 outcomes in |0⟩
        c0 = int(self.rng.binomial(int(shots), p0_exact))
        p0 = c0 / float(shots)
    else:
        p0 = p0_exact
    
    p1 = 1.0 - p0
    return p0 - p1  # <Z> expectation value
```

### 4. Swap Test for State Overlap

Measure $|\langle\phi|\psi\rangle|^2$ using Hadamard test principle:

```python
def swap_test(prepare_phi: GateSequence, prepare_psi: GateSequence) -> float:
    """
    Construct SWAP operator that exchanges two registers.
    Apply Hadamard test to this operator to get state overlap.
    """
    n = prepare_phi.get_num_qubits()
    
    # SWAP operator: swaps qubits 0..n-1 with qubits n..2n-1
    U_swap = GateSequence(2 * n, name="SWAP")
    for i in range(n):
        U_swap.swap(i, i + n)
    
    # Prepare joint state |φ⟩⊗|ψ⟩
    prepare_joint = GateSequence(2 * n)
    prepare_joint.append(prepare_phi, target=list(range(n)))
    prepare_joint.append(prepare_psi, target=list(range(n, 2*n)))
    
    # Apply Hadamard test to SWAP operator
    # Result: <Z> = 2|<φ|ψ>|² - 1, so |<φ|ψ>|² = (1 + <Z>)/2
    # But implementation returns clipped (1 + <Z>)/2 ≈ |<φ|ψ>|²
    return run_hadamard_test(U_swap, prepare_joint)
```

### 5. Phase Estimation from Complex Value

Recover eigenphase from real and imaginary components:

```python
def _estimate_phi_from_real_imag(self, cos_est: float, sin_est: float) -> float:
    """
    Given <ψ|U|ψ> ≈ cos_est + i·sin_est (for U = e^(iφ) eigenstate),
    extract the eigenphase φ ∈ [0, 2π).
    """
    # arctan2 computes angle in (-π, π]
    angle = float(np.arctan2(sin_est, cos_est))
    
    # Convert to [0, 1) normalized phase
    phi = float((angle / (2.0 * np.pi)) % 1.0)
    
    return phi
```

---

## 💡 Hands-On Example: Evaluating a Rotation Operator

Let's estimate $\langle+|RZ(0.8)|+\rangle$ where $|+\rangle = H|0\rangle$ and $RZ(\theta) = e^{-i\theta Z/2}$:

**Expected result**: $\langle+|RZ(\theta)|+\rangle = \cos(\theta/2)$

```python
from algorithm import HadamardTestAlgorithm
from engine.core import GateSequence
import numpy as np

# Setup
theta = 0.8
algo = HadamardTestAlgorithm()

# Create unitary: RZ(0.8)
U = GateSequence(1)
U.rz(theta, 0)

# Prepare state: |+⟩ = H|0⟩
prepare_plus = GateSequence(1)
prepare_plus.h(0)

# Run Hadamard test for real part
result_re = algo.run(
    mode="expectation",
    U=U,
    prepare_psi=prepare_plus,
    imag=False,
    shots=30000
)

# Run Hadamard test for imaginary part
result_im = algo.run(
    mode="expectation",
    U=U,
    prepare_psi=prepare_plus,
    imag=True,
    shots=30000
)

# Compare with theory
re_est = result_re['estimated_value']
im_est = result_im['estimated_value']
re_theory = np.cos(theta / 2)
im_theory = 0.0  # Should be zero for |+⟩ and RZ rotation

print(f"Real part - Estimated: {re_est:.6f}, Theory: {re_theory:.6f}")
print(f"Imag part - Estimated: {im_est:.6f}, Theory: {im_theory:.6f}")
print(f"Full expectation: {re_est:.6f} + {im_est:.6f}i")
```

---

## 🎓 Implementing Your Own Hadamard Test

### Complete Implementation Template

```python
import numpy as np
from typing import Optional, Dict, Any
from engine.core import GateSequence, State

class MyHadamardTest:
    """User-implemented Hadamard Test."""
    
    def __init__(self):
        self.rng = np.random.default_rng()
    
    def build_hadamard_test_circuit(
        self, U: GateSequence, 
        prepare_psi: Optional[GateSequence] = None,
        imag: bool = False
    ) -> GateSequence:
        """
        Build the Hadamard test circuit.
        
        Key steps:
        1. Hadamard on ancilla → superposition
        2. Optional S† phase gate (for imaginary part)
        3. Prepare target state |ψ⟩
        4. Controlled-U gate
        5. Final Hadamard on ancilla
        """
        n = U.get_num_qubits()
        circuit = GateSequence(1 + n)
        
        ancilla = 0
        target_qubits = list(range(1, 1 + n))
        
        # Step 1: Initialize superposition
        circuit.h(ancilla)
        
        # Step 2: Phase gate for imaginary part
        if imag:
            circuit.sdag(ancilla)  # S† gate
        
        # Step 3: State preparation
        if prepare_psi is not None:
            circuit.append(prepare_psi, target=target_qubits)
        
        # Step 4: Controlled unitary (key step!)
        # The control qubit is the ancilla
        circuit.append(U, target=target_qubits, control=[ancilla])
        
        # Step 5: Measurement basis transformation
        circuit.h(ancilla)
        
        return circuit
    
    def run_hadamard_test(
        self, U: GateSequence,
        prepare_psi: Optional[GateSequence] = None,
        imag: bool = False,
        shots: int = 20000
    ) -> float:
        """
        Execute Hadamard test and return <Z> = p(0) - p(1)
        """
        # Build circuit
        circuit = self.build_hadamard_test_circuit(U, prepare_psi, imag)
        
        # Execute simulation
        statevector = np.asarray(circuit.execute(), dtype=complex).reshape(-1)
        
        # Extract ancilla probabilities
        state = State(statevector)
        ancilla_probs = state.probabilities_dict([0], endian="little")
        
        p0_ideal = float(ancilla_probs.get("0", 0.0))
        
        # Add shot noise
        if shots > 0:
            counts_0 = self.rng.binomial(shots, p0_ideal)
            p0_measured = counts_0 / shots
        else:
            p0_measured = p0_ideal
        
        p1_measured = 1.0 - p0_measured
        
        return p0_measured - p1_measured
    
    def estimate_expectation(
        self, U: GateSequence,
        prepare_psi: Optional[GateSequence] = None,
        shots: int = 20000
    ) -> Dict[str, Any]:
        """
        Estimate full complex expectation value <ψ|U|ψ>
        """
        # Measure real and imaginary parts
        real_part = self.run_hadamard_test(U, prepare_psi, imag=False, shots=shots)
        imag_part = self.run_hadamard_test(U, prepare_psi, imag=True, shots=shots)
        
        return {
            "real": real_part,
            "imag": imag_part,
            "complex": complex(real_part, imag_part),
            "magnitude": np.sqrt(real_part**2 + imag_part**2)
        }
    
    def swap_test(
        self, prepare_phi: GateSequence,
        prepare_psi: GateSequence,
        shots: int = 20000
    ) -> float:
        """
        Estimate state overlap |<φ|ψ>|²
        """
        n = prepare_phi.get_num_qubits()
        
        # Build SWAP gate on 2n qubits
        swap_gate = GateSequence(2 * n)
        for i in range(n):
            swap_gate.swap(i, i + n)
        
        # Prepare joint state
        joint_prep = GateSequence(2 * n)
        joint_prep.append(prepare_phi, target=list(range(n)))
        joint_prep.append(prepare_psi, target=list(range(n, 2*n)))
        
        # Apply Hadamard test to SWAP
        result = self.run_hadamard_test(swap_gate, joint_prep, shots=shots)
        
        # Convert: <Z> = 2|<φ|ψ>|² - 1, so |<φ|ψ>|² = (1 + <Z>)/2
        overlap_sq = (1.0 + result) / 2.0
        
        return float(np.clip(overlap_sq, 0.0, 1.0))
```

---

## 📊 Mathematical Deep Dive

### Basic Hadamard Test Principle

Start with:
- **Ancilla**: $|0\rangle$
- **Target**: $|\psi\rangle$

Apply the circuit:
1. Hadamard: ancilla $\to$ $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |\psi\rangle$
2. Controlled-$U$: $\to$ $\frac{1}{\sqrt{2}}(|0\rangle \otimes |\psi\rangle + |1\rangle \otimes U|\psi\rangle)$
3. Hadamard on ancilla: $\to$ $\frac{1}{2}\left[|0\rangle \otimes (I + U)|\psi\rangle + |1\rangle \otimes (I - U)|\psi\rangle\right]$

The probability of measuring $0$ is:
$$P(0) = \frac{1}{4}\langle\psi|(I + U^\dagger)(I + U)|\psi\rangle = \frac{1}{2}\left(1 + \text{Re}(\langle\psi|U|\psi\rangle)\right)$$

Therefore:
$$\langle Z \rangle = P(0) - P(1) = 2P(0) - 1 = \text{Re}(\langle\psi|U|\psi\rangle)$$

### Phase Gate for Imaginary Part

Insert $S^\dagger$ after the first Hadamard:

$$S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$$

This introduces a phase factor that effectively extracts the imaginary part:
$$\langle Z \rangle_{\text{with } S^\dagger} = \text{Im}(\langle\psi|U|\psi\rangle)$$

### Complex Expectation Value

Combining both:
$$\langle\psi|U|\psi\rangle = \text{Re}(\cdot) + i \cdot \text{Im}(\cdot)$$

Each component has statistical uncertainty $\sim 1/\sqrt{\text{shots}}$.

### Swap Test for Overlap

For states $|\phi\rangle$ and $|\psi\rangle$, the SWAP operator on the joint state:
$$\text{SWAP}|\phi\rangle \otimes |\psi\rangle = |\psi\rangle \otimes |\phi\rangle$$

Applying Hadamard test to SWAP gives:
$$\langle Z \rangle = 2|\langle\phi|\psi\rangle|^2 - 1$$

Solving: 
$$|\langle\phi|\psi\rangle|^2 = \frac{1 + \langle Z \rangle}{2}$$

---

## ✅ Exercises & Challenges

### Exercise 1: Real and Imaginary Parts (Beginner)

**Goal**: Measure both components of an expectation value.

```python
# TODO: For a unitary U = RX(π/4) and state |ψ⟩ = |+⟩:
# 1. Measure the real part using imag=False
# 2. Measure the imaginary part using imag=True
# 3. Compute the magnitude |<ψ|U|ψ>|
# 4. Compare with analytical results

# Hint: RX rotation: <+|RX(θ)|+> = cos(θ/2)
```

**Verification**: Real and imaginary parts should satisfy the rotation angle.

---

### Exercise 2: State Overlap Measurement (Intermediate)

**Goal**: Use swap test to measure how similar two states are.

```python
# TODO: Create several state pairs and measure their overlaps:
# 1. Same state vs itself (should be 1.0)
# 2. |0⟩ vs |+⟩ (should be 0.5)
# 3. |0⟩ vs |1⟩ (should be 0.0)
# 4. Two random states (varies)

# For each, run swap_test and verify against theory
```

**Challenge**: How does overlap measurement error scale with shots?

---

### Exercise 3: Complex Unitary Analysis (Intermediate)

**Goal**: Analyze a unitary with non-zero imaginary expectation value.

```python
# TODO: Design a unitary U and state |ψ⟩ such that:
# - <ψ|U|ψ> has both real and imaginary parts
# 
# Example: U = RZ(1.3), |ψ⟩ = |0⟩
# <0|RZ(1.3)|0> = cos(1.3/2) - i·sin(1.3/2)
#
# Measure both parts and verify magnitude |<ψ|U|ψ>| = 1

# Hint: |cos(0.65) - i·sin(0.65)|² = cos²(0.65) + sin²(0.65) = 1
```

---

### Exercise 4: Bell State Characterization (Intermediate)

**Goal**: Use Hadamard test on multi-qubit systems.

```python
# TODO: Create a Bell state and apply various unitaries:
# Bell = (|00⟩ + |11⟩)/√2
#
# 1. Apply single-qubit rotations
# 2. Apply two-qubit gates (CX, CZ, etc.)
# 3. Measure expectation values
# 4. Verify entanglement properties
```

**Challenge**: What symmetries can you observe in Bell state expectation values?

---

### Exercise 5: Variational Algorithm Prototyping (Advanced)

**Goal**: Build a simple VQE-like algorithm using Hadamard test.

```python
# TODO: Implement a mini VQE:
# 1. Parameterized ansatz: U(θ) = RZ(θ) on 1-2 qubits
# 2. Define a Hamiltonian (combination of Pauli operators)
# 3. Use Hadamard test to estimate <ψ(θ)|H|ψ(θ)>
# 4. Optimize θ to minimize energy
# 5. Plot energy vs parameter

# Hint: Hadamard test works on individual Pauli operators
# Estimate <ψ|Z|ψ>, <ψ|X|ψ>, etc. separately, then sum
```

---

### Exercise 6: Custom Implementation (Advanced)

**Goal**: Implement Hadamard test from scratch.

```python
# TODO: Build your own HadamardTestAlgorithm with:
# 1. Hadamard test circuit construction
# 2. Real/imaginary measurement
# 3. Swap test implementation
# 4. Shot noise simulation
# 5. Phase estimation from complex values

# Verify: Your results match the provided implementation
```

---

## 🔍 Debugging Tips

1. **Expectation value suspicious?**
   - Verify ancilla qubit is index 0
   - Check that target register qubits are correctly arranged
   - Confirm controlled-U is applied with ancilla as control
   - Verify state preparation happens on target qubits, not ancilla

2. **Swap test returning wrong overlap?**
   - Ensure both states have same number of qubits
   - Check that SWAP gate correctly exchanges registers
   - Verify joint state preparation is correct
   - Apply clipping to [0, 1] range

3. **Shot noise too large?**
   - Increase `shots` parameter (improves as $\sim 1/\sqrt{\text{shots}}$)
   - Expected error ≈ $1/\sqrt{\text{shots}} \approx 1/\sqrt{20000} \approx 0.007$

4. **Phase estimation inaccurate?**
   - Run both real and imaginary measurements
   - Use arctan2(imag, real) to get correct quadrant
   - Normalize angle to [0, 2π)

5. **Circuit too complex?**
   - Hadamard test is inherently ~1 overhead (1 ancilla + n target qubits)
   - Main cost is controlled-U, which depends on U's complexity
   - For variational algorithms, budget shots across multiple parameter evaluations

---

## 📖 Further Learning

- **Mathematical Foundation**: See [README.md](README.md) for detailed derivations
- **Source Implementation**: Study [algorithm.py](algorithm.py) for production code
- **Related Topics**:
  - Variational Quantum Eigensolvers (VQE)
  - Quantum Approximate Optimization Algorithm (QAOA)
  - Quantum Phase Estimation (QPE)
- **Applications**:
  - Quantum Chemistry simulations
  - Machine learning with quantum kernels
  - Optimization problems
- **Classic Paper**: "Simulating physics with computers" (Feynman) and variational algorithm papers

---

## 🎯 Summary Checklist

Before considering yourself proficient, ensure you can:

- [ ] Explain why Hadamard test measures unitary expectation values
- [ ] Draw and understand the Hadamard test circuit
- [ ] Build real part circuit (without phase gate)
- [ ] Build imaginary part circuit (with S† gate)
- [ ] Interpret P(0) - P(1) measurement results
- [ ] Combine real/imaginary to get complex expectation value
- [ ] Understand swap test principle and implementation
- [ ] Apply Hadamard test to multi-qubit systems
- [ ] Handle shot noise correctly
- [ ] Use `HadamardTestAlgorithm` for all three modes
- [ ] Implement from scratch
- [ ] Apply to variational quantum algorithms

---

## 🎓 Real-World Application: Variational Quantum Eigensolver (VQE)

The Hadamard test is the **cornerstone of VQE**, where:

1. **Ansatz**: Parameterized circuit $U(\theta)$ that prepares trial states
2. **Observable**: Measure $\langle\psi(\theta)|H|\psi(\theta)\rangle$ using Hadamard test
3. **Optimization**: Adjust $\theta$ to minimize ground state energy
4. **Repeat**: Use classical optimizer with quantum cost function

Example:
```python
# Pseudo-code VQE loop
for iteration in range(max_iterations):
    # Evaluate cost function at current parameters
    cost = 0
    for pauli_term in hamiltonian:
        expectation = hadamard_test.estimate_expectation(
            U=pauli_term,
            prepare_psi=ansatz(theta),
            shots=shots
        )
        cost += coefficient * expectation.real
    
    # Update parameters using classical optimizer
    theta = optimizer.step(cost)
```

This hybrid classical-quantum approach is how we solve chemistry problems on NISQ devices!

---

*Master Hadamard Test to unlock quantum variational algorithms!* 🚀
