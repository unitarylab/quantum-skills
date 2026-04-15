---
name: amplitude-amplification
description: A quantum algorithm that generalizes Grover's search algorithm, allowing for the amplification of the probability of desired outcomes in a quantum state. It is used to find marked items in an unsorted database with quadratic speedup compared to classical algorithms. This skill provides a comprehensive guide to understanding, implementing (using UnitaryLab's quantum simulator), and utilizing amplitude amplification in quantum computing applications.
---

## One Step to Run Amplitude Amplification Example
```bash
python ./scripts/algorithm.py
```

# Amplitude Amplification Algorithm Skill Guide

## Overview

**Amplitude Amplification** is a powerful quantum algorithm that generalizes Grover's search algorithm. It solves the following problem:

Given a quantum state prepared by some unitary $U$ such that:
$$U|0^{m+n}\rangle = \sqrt{p_0}|0^m\rangle|\psi_0\rangle + \sqrt{1-p_0}|\perp\rangle$$

where $p_0$ is the initial success probability, the goal is to amplify the probability $p_0$ to be close to 1, allowing us to reliably prepare the target state $|\psi_0\rangle$ with only $O(1/\sqrt{p_0})$ repetitions.

### Key Characteristics:
- **Quadratic Speedup**: If initial success probability is $p_0$, we need ~$1/\sqrt{p_0}$ iterations instead of classical $1/p_0$
- **Generalization of Grover**: When $p_0 = 1/N$, reduces to standard Grover's algorithm
- **Wide Applications**: State preparation, optimization, phase estimation enhancement

---

## Learning Objectives

After mastering this skill, you will be able to:

1. Understand the mathematical foundations of amplitude amplification
2. Use the provided `AmplitudeAmplificationAlgorithm` class to amplify probabilities
3. Implement the oracle and diffuser operators
4. Calculate optimal iteration counts
5. Write your own amplitude amplification implementation from scratch

---

## Prerequisites

- Basic understanding of quantum gates (H, X, Z, CX, MCX)
- Familiarity with quantum state preparation
- Understanding of unitary operators and quantum circuits
- Knowledge of the core `GateSequence` and `Register` classes
- Comfort with Python and numpy

---

## Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms import AmplitudeAmplificationAlgorithm
from engine import GateSequence

# Step 1: Create the state preparation unitary U
U_prep = GateSequence(2)  # 2-qubit register
U_prep.h(0)               # Apply Hadamard to each qubit
U_prep.h(1)

# Step 2: Define target condition
# In this case, we want both qubits in |0⟩ state
good_zero_qubits = [0, 1]

# Step 3: Initial success probability
# For uniform superposition over 2 qubits, P(|00⟩) = 1/4 = 0.25
initial_p = 0.25

# Step 4: Run the algorithm
algo = AmplitudeAmplificationAlgorithm()
result = algo.run(
    U=U_prep,
    good_zero_qubits=good_zero_qubits,
    p=initial_p,
    backend='torch'  # or 'numpy'
)

# Step 5: Inspect results
print(result['plot'])  # Beautiful ASCII output
print(f"Initial probability: {initial_p:.4f}")
print(f"Amplified probability: {result['amplified_prob']:.4f}")
print(f"Circuit diagram: {result['circuit_path']}")
```

### Core Parameters Explained

```python
algo.run(
    U: GateSequence,           # Unitary that prepares your state
    good_zero_qubits: List[int],  # Qubit indices that should be |0⟩ in target state
    p: float,                   # Initial success probability (0 < p < 1)
    reps: Optional[int] = None, # Number of iterations (auto-calculated if None)
    backend: str = 'torch',     # Simulation backend
    algo_dir: str = './aa_results'  # Output directory for circuit diagrams
)
```

**Important**: The function returns a dictionary containing:
- `status`: "ok" or "failed"
- `amplified_prob`: Final probability of target state
- `circuit_path`: Path to the generated circuit diagram
- `message`: Detailed execution message
- `plot`: Formatted ASCII result panel

---

## Understanding the Key Components

### 1. Oracle Construction

The oracle marks the target state by applying a phase flip (-1) when all qubits in `good_zero_qubits` are |0⟩:

```python
def _build_oracle(self, gs: GateSequence, zero_qubits: List[int], ancilla: int) -> None:
    """
    Implements: O = I - 2|0⋯0⟩⟨0⋯0|
    This flips the phase of computational basis states matching the condition.
    """
    # Prepare ancilla in |−⟩ state for phase kickback
    self._prepare_kickback_ancilla_minus(gs, ancilla)
    
    # Flip target qubits to use controlled-X
    for q in zero_qubits:
        gs.x(q)
    
    # Multi-controlled X applying phase flip
    if len(zero_qubits) == 0:
        gs.z(ancilla)
    elif len(zero_qubits) == 1:
        gs.cx(zero_qubits[0], ancilla)
    else:
        gs.mcx(zero_qubits, ancilla)  # Multi-controlled X
    
    # Unflip qubits
    for q in zero_qubits:
        gs.x(q)
    
    # Restore ancilla to |0⟩
    self._unprepare_kickback_ancilla_minus(gs, ancilla)
```

### 2. Diffuser Construction

The diffuser performs a reflection about the prepared state:

```python
def _build_diffuser(self, gs: GateSequence, U: GateSequence, 
                   data_qubits: List[int], ancilla: int) -> None:
    """
    Implements: D = U(2|0⋯0⟩⟨0⋯0| - I)U†
    This reflects the state about the prepared state |ψ⟩ = U|0⟩.
    """
    # Undo the preparation
    gs.append(U.dagger(), data_qubits)
    
    # Apply oracle in computational basis
    self._build_oracle(gs, zero_qubits=list(data_qubits), ancilla=ancilla)
    
    # Re-apply the preparation
    gs.append(U, data_qubits)
```

### 3. Optimal Iteration Calculation

The number of iterations needed depends on the initial probability:

```python
def _get_optimal_iterations(self, p: float) -> int:
    """
    Calculate: k ≈ π/(4θ) - 1/2, where θ = arcsin(√p)
    This ensures (2k+1)θ ≈ π/2 for maximum amplitude conversion.
    """
    theta = math.asin(math.sqrt(p))
    r = int(round((math.pi / (4.0 * theta)) - 0.5))
    return max(0, r)
```

---

## Hands-On Example: Amplify Specific State Probability

Let's amplify the probability of finding state |101⟩ in a 3-qubit superposition:

```python
from engine.algorithms.fundamental_algorithm import AmplitudeAmplificationAlgorithm
from engine import GateSequence

# Create a non-uniform superposition: we apply different gates to each qubit
U_prep = GateSequence(3)
U_prep.h(0)  # Create superposition
U_prep.h(1)  # Create superposition  
U_prep.h(2)  # Create superposition
# After this, each computational basis state has probability 1/8 = 0.125

algo = AmplitudeAmplificationAlgorithm()

# We might add some phase manipulation here to prepare a specific state
# For demonstration, let's say we want to amplify |000⟩
result = algo.run(
    U=U_prep,
    good_zero_qubits=[0, 1, 2],  # Target state |000⟩
    p=1/8,  # Initial probability = 0.125
    backend='torch'
)

print("=" * 50)
print(f"Successfully amplified |000⟩ state!")
print(f"Initial prob:  {result['amplified_prob']:.4f}")
print(f"Circuit saved: {result['circuit_path']}")
```

---

## Implementing Your Own Amplitude Amplification

### Full Implementation Template

Here's a skeleton for implementing amplitude amplification from scratch:

```python
import math
from typing import List, Dict, Any

class MyAmplitudeAmplification:
    def __init__(self):
        self.log_prefix = "INFO: "
    
    def run(self, U, good_zero_qubits: List[int], p: float, 
            reps: int = None, backend: str = 'torch') -> Dict[str, Any]:
        """
        Main execution method for amplitude amplification.
        """
        print("Starting Amplitude Amplification...")
        
        # Step 1: Calculate iterations if not provided
        if reps is None:
            reps = self._calc_iterations(p)
            print(f"  Calculated iterations: {reps}")
        
        # Step 2: Initialize quantum circuit
        n = U.get_num_qubits()
        gs = self._init_circuit(U, n, backend)
        
        # Step 3: Build main loop - apply Grover iteration k times
        for iteration in range(reps):
            print(f"  Applying iteration {iteration + 1}/{reps}...")
            
            # Sub-step 3a: Oracle - mark the target state
            self._apply_oracle(gs, good_zero_qubits, n)
            
            # Sub-step 3b: Diffuser - reflect about the prepared state
            self._apply_diffuser(gs, U, n)
        
        # Step 4: Execute quantum simulation
        result_state = gs.execute()
        
        # Step 5: Post-process and verify
        target_prob = self._measure_target_probability(result_state, good_zero_qubits)
        is_success = target_prob > p
        
        print(f"  Result: {target_prob:.4f} (success: {is_success})")
        
        return {
            "status": "ok" if is_success else "failed",
            "amplified_prob": target_prob,
            "message": f"Amplitude amplified from {p:.4f} to {target_prob:.4f}"
        }
    
    def _calc_iterations(self, p: float) -> int:
        """Calculate optimal number of Grover iterations"""
        theta = math.asin(math.sqrt(p))
        return max(0, int(round(math.pi / (4 * theta) - 0.5)))
    
    def _init_circuit(self, U, n_qubits, backend):
        """Initialize GateSequence and prepare initial state"""
        from engine import GateSequence
        gs = GateSequence(n_qubits + 1, backend=backend)  # +1 for ancilla
        gs.append(U, list(range(n_qubits)))
        return gs
    
    def _apply_oracle(self, gs, target_qubits: List[int], n_qubits: int):
        """Apply the oracle: phase flip on target states"""
        ancilla = n_qubits  # Ancilla is the last qubit
        
        # Prepare ancilla in |−⟩
        gs.x(ancilla)
        gs.h(ancilla)
        
        # Apply multi-controlled X
        if len(target_qubits) > 0:
            gs.mcx(target_qubits, ancilla)
        else:
            gs.z(ancilla)
        
        # Restore ancilla
        gs.h(ancilla)
        gs.x(ancilla)
    
    def _apply_diffuser(self, gs, U, n_qubits: int):
        """Apply diffuser: reflection about the prepared state"""
        ancilla = n_qubits
        data_qubits = list(range(n_qubits))
        
        # Undo preparation
        gs.append(U.dagger(), data_qubits)
        
        # Apply oracle in computational basis
        self._apply_oracle(gs, data_qubits, n_qubits)
        
        # Redo preparation
        gs.append(U, data_qubits)
    
    def _measure_target_probability(self, state, target_qubits):
        """Calculate probability of measuring target state"""
        # This depends on your simulator's State interface
        # Pseudocode:
        target_prob = 0.0
        for basis_string, probability in state.items():
            # Check if target qubits match condition
            is_target = all(basis_string[q] == '0' for q in target_qubits)
            if is_target:
                target_prob += probability
        return target_prob
```

---

## Mathematical Deep Dive

### Iteration Count Formula

If the initial success probability is $p_0$, we need approximately:

$$k = \left\lceil \frac{\pi}{4\theta} - \frac{1}{2} \right\rceil \text{ iterations, where } \theta = \arcsin\left(\sqrt{p_0}\right)$$

**Special cases:**
- When $p_0 \ll 1$: $k \approx \frac{\pi}{4\sqrt{p_0}}$ (quadratic speedup)
- When $p_0 = 1/N$ (classical Grover): $k \approx \frac{\pi}{4}\sqrt{N}$

### State Evolution

After $k$ iterations, the state evolves as:

$$\text{Grover}^k|\psi\rangle = \sin((2k+1)\theta)|w\rangle + \cos((2k+1)\theta)|r\rangle$$

where:
- $|w\rangle$ is the superposition of target states
- $|r\rangle$ is the superposition of non-target states
- Optimal when $(2k+1)\theta \approx \pi/2$

---

## Debugging Tips

1. **Probability not increasing?**
   - Verify that `p` accurately represents the initial probability
   - Check that `good_zero_qubits` correctly specifies your target state
   - Confirm the unitary `U` is correct

2. **Mathematical inconsistency?**
   - Manually verify iteration count: $k = \lfloor \frac{\pi}{4\arcsin(\sqrt{p})} - \frac{1}{2} \rfloor$
   - Print intermediate states to debug

3. **Circuit explosion?**
   - Multi-controlled gates grow expensive. For large systems, consider decomposition strategies
   - Try using `mcx` decomposition parameters if available

---

## Further Learning

- **Related Topics**: Grover's Algorithm, Quantum Phase Estimation, Quantum Optimization
- **Papers**: Original Grover paper (1996), Amplitude Amplification generalizations

---

## Summary Checklist

Before considering yourself proficient, ensure you can:

- [ ] Explain what amplitude amplification does and why it's useful
- [ ] Calculate optimal iteration count for a given initial probability
- [ ] Use `AmplitudeAmplificationAlgorithm` to amplify a specific state
- [ ] Understand and explain oracle and diffuser construction
- [ ] Interpret the ASCII output panels
- [ ] Implement amplitude amplification from scratch
- [ ] Debug failing amplification attempts
- [ ] Apply this technique to a novel quantum problem

---
