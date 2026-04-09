---
name: lcu
description: A quantum algorithm for solving linear systems of equations using the Linear Combination of Unitaries (LCU) method, providing exponential speedup over classical methods for certain types of problems. This skill includes efficient implementations and educational resources for understanding and utilizing the LCU algorithm in various applications.
---

# Linear Combination of Unitaries (LCU) Algorithm Skill Guide

## 📚 Overview

**Linear Combination of Unitaries (LCU)** is a fundamental quantum subroutine that implements a **non-unitary matrix** as a linear combination of unitary matrices. Given:

$$M = \sum_{j=0}^{m-1} \alpha_j U_j$$

where $U_j$ are unitary operators and $\alpha_j \geq 0$ are coefficients, LCU probabilistically prepares:

$$|\psi'\rangle \propto M|\psi\rangle$$

### Key Insight

Instead of directly implementing the non-unitary $M$ (which is impossible), LCU:
1. Uses auxiliary qubits to encode which unitary to apply
2. Creates quantum superposition over all unitary choices
3. Applies them simultaneously through quantum parallelism
4. Post-selects on ancilla measurement to extract result

### Why LCU Matters:

1. **Foundation for Quantum Linear Solvers**: HHL and similar algorithms use LCU
2. **Hamiltonian Simulation**: Approximate Hamiltonian evolution
3. **Block-Encoding Framework**: Key component in modern quantum algorithms
4. **Amplitude Amplification Target**: Can boost success probability

### Real Applications:

- Simulating quantum dynamics (Hamiltonian simulation)
- Quantum machine learning (kernel methods)
- Chemistry simulations (molecular property calculation)
- Solving differential equations (PDE discretization)

---

## 🎯 Learning Objectives

After mastering this skill, you will be able to:

1. ✅ Understand why non-unitary matrices require probabilistic implementation
2. ✅ Grasp the LCU three-step procedure (V → SELECT → V†)
3. ✅ Explain auxiliary qubit role in coefficient encoding
4. ✅ Understand the SELECT operator (multi-controlled unitaries)
5. ✅ Use the provided `LCUAlgorithm` class effectively
6. ✅ Calculate success probability and post-selection requirements
7. ✅ Implement LCU from scratch
8. ✅ Apply to Hamiltonian simulation and other problems
9. ✅ Combine with amplitude amplification for improved success rate

---

## 📋 Prerequisites

- **Essential knowledge**:
  - Quantum superposition and entanglement
  - Controlled unitary gates (CU, CCU, MCU)
  - Measurement and post-selection
  - Amplitude encoding and state preparation
  - Basic linear algebra (eigenvalues, norms)
- **Mathematical comfort**: Complex numbers, matrix decomposition
- **Recommended**: Study [Amplitude Amplification](../amplitude-amplification/SKILL.md) for success probability improvement

---

## 🚀 Using the Provided Implementation

### Quick Start Example

```python
from engine.algorithms.linear_algebra import LCUAlgorithm
from engine import GateSequence
import numpy as np

# Step 1: Define the unitaries U_j
# Example: Implement M = 0.6*I + 0.4*X on 1 qubit
# where I is identity, X is Pauli-X

U0 = GateSequence(1)
U0.i(0)  # Identity

U1 = GateSequence(1)
U1.x(0)  # Pauli-X

unitaries = [U0, U1]

# Step 2: Define coefficients α_j
alphas = [0.6, 0.4]  # M = 0.6*I + 0.4*X

# Step 3: Optional: define initial state
initial_state = GateSequence(1)
initial_state.h(0)  # Create |+⟩ state

# Step 4: Run LCU algorithm
algo = LCUAlgorithm()
result = algo.run(
    alphas=alphas,
    unitaries=unitaries,
    n_sys=1,  # 1-qubit system
    initial_state=initial_state,
    backend='torch'
)

# Step 5: Inspect results
print(result['plot'])
print(f"LCU success probability: {result['lcu_success_probability']:.6f}")
print(f"Expected: ~{np.sum(alphas)**2:.6f} (s² where s=sum(alphas))")
```

### Core Parameters Explained

```python
algo.run(
    alphas: List[float],              # Non-negative coefficients [α₀, α₁, ...]
    unitaries: List[GateSequence],    # Unitary circuits [U₀, U₁, ...]
    n_sys: int,                       # Number of system (data) qubits
    initial_state: Optional[GateSequence],  # Initial state preparation
    backend: str = 'torch',           # Simulation backend
    algo_dir: str = './lcu_results'   # Output directory
)
```

**Return dictionary contains:**
- `lcu_success_probability`: Probability of measuring ancilla = |0⟩
- `full_circuit`: Complete LCU quantum circuit
- `circuit_path`: Location of circuit diagram
- `message`: Detailed summary
- `plot`: Formatted ASCII result panel

**Success Probability Formula**:
$$P_{\text{success}} = \frac{\|M|\psi\rangle\|^2}{s^2} \text{ where } s = \sum_{j=0}^{m-1} |\alpha_j|$$

---

## 🔧 Understanding the Core Components

### 1. The Three-Step LCU Pipeline

```python
def run(self, alphas, unitaries, n_sys, initial_state=None):
    """
    Complete LCU pipeline:
    
    Step 1: Prepare coefficient superposition with V operator
    Step 2: Apply controlled unitaries with SELECT operator
    Step 3: Uncompute with V† and post-select on |0⟩
    """
    
    # Extract dimensions
    m = len(alphas)  # Number of unitaries
    n_anc = ceil(log₂(m))  # Auxiliary qubits needed
    s = sum(alphas)  # Normalization constant
    
    # Build three components
    V = build_V(alphas, s, m, n_anc)           # V operator
    SELECT = build_SELECT(unitaries, m, n_anc, n_sys)  # SELECT operator
    V_dagger = V.dagger()                       # V†
    
    # Construct complete circuit
    circuit = GateSequence(n_anc + n_sys)
    
    if initial_state:
        circuit.append(initial_state, system_qubits)
    
    circuit.append(V, ancilla_qubits)           # Apply V
    circuit.append(SELECT, all_qubits)          # Apply SELECT
    circuit.append(V_dagger, ancilla_qubits)    # Apply V†
    
    # Measure ancilla and post-select on |0⟩ state
    return extract_if_ancilla_is_zero(circuit)
```

### 2. The V Operator: Coefficient Encoding

Prepares superposition of coefficients on auxiliary register:

```python
def _build_V(self, alphas, s_norm, m, n_anc, backend='torch'):
    """
    Construct V operator to create superposition:
    
    |0⟩_{anc} → Σⱼ √(αⱼ/s) |j⟩_{anc}
    
    Uses amplitude encoding with binary tree of RY rotations.
    
    Mathematical basis:
    - Normalize: βⱼ = αⱼ/s
    - Create superposition using recursive RY rotations
    - Each qubit encodes one bit position of the index j
    """
    qc = GateSequence(n_anc)
    
    # Normalize coefficients
    normalized_alphas = [a / s_norm for a in alphas]
    
    # Build amplitude encoding tree using RY rotations
    # This is a classical computation to determine angles
    gates = state_preparation_tree(normalized_alphas)
    
    # Apply gates (mostly RY and controlled RY)
    for gate in gates:
        if gate['control'] is None:
            qc.ry(gate['angle'], gate['target'])
        else:
            qc.mcry(gate['angle'], gate['control'], gate['target'])
    
    return qc
```

**Example**: For $m=4$ with $\alpha = [0.6, 0.2, 0.15, 0.05]$:
- Need 2 auxiliary qubits (since $2^2=4$)
- Prepare: $\frac{\sqrt{0.6}}{s}|00\rangle + \frac{\sqrt{0.2}}{s}|01\rangle + \frac{\sqrt{0.15}}{s}|10\rangle + \frac{\sqrt{0.05}}{s}|11\rangle$

### 3. The SELECT Operator: Conditional Unitaries

Applies the unitary corresponding to the auxiliary qubit state:

```python
def _build_select(self, unitaries, m, n_anc, n_sys):
    """
    Construct SELECT operator:
    
    SELECT = Σⱼ |j⟩⟨j|_{anc} ⊗ Uⱼ
    
    This is a multi-controlled multiplexer:
    - If ancilla = |j⟩, apply Uⱼ to system register
    - Uses controlled unitary gates with binary control patterns
    
    For m=4:
    SELECT = |00⟩⟨00| ⊗ U₀ + |01⟩⟨01| ⊗ U₁ + |10⟩⟨10| ⊗ U₂ + |11⟩⟨11| ⊗ U₃
    """
    qc = GateSequence(n_anc + n_sys)
    anc_qubits = list(range(n_anc))
    sys_qubits = list(range(n_anc, n_anc + n_sys))
    
    # For each unitary Uⱼ
    for j, U in enumerate(unitaries):
        # Convert j to binary control pattern
        control_pattern = format(j, f'0{n_anc}b')
        
        # Apply U controlled by ancilla qubits matching pattern
        # Example: if j=2 (binary 10), apply U₂ when anc = |10⟩
        qc.append(U, target=sys_qubits, 
                 control=anc_qubits, 
                 control_sequence=control_pattern)
    
    return qc
```

### 4. Post-Selection: Extracting the Result

After V†, measure ancilla and post-select:

```python
def extract_solution_via_postselection(state, n_anc, n_sys):
    """
    Extract solution by post-selecting on ancilla measurement.
    
    State before measurement:
    |Ψ⟩ = |0⟩_{anc} ⊗ (M|ψ⟩/s) + |0⟩⊥_{anc} ⊗ (orthogonal_part)
    
    After measuring ancilla:
    - If result is |0⟩_{anc}: system register contains M|ψ⟩/s
    - If result is |j⟩_{anc}, j≠0: measurement failed, discard
    
    Success probability: P = ||M|ψ⟩||²/s²
    """
    # Measure ancilla qubits (indices 0 to n_anc-1)
    zero_state = '0' * n_anc
    ancilla_probs = get_probabilities_for_qubits(state, range(n_anc))
    
    success_prob = ancilla_probs.get(zero_state, 0.0)
    
    if success_prob < 1e-10:
        raise ValueError("Post-selection failed: ancilla not measured as |0⟩")
    
    # Extract system state conditioned on ancilla = |0⟩
    system_state = extract_conditional_state(state, ancilla_is_zero=True)
    
    # Normalize
    normalized_state = system_state / np.sqrt(success_prob)
    
    # Scale by s (the sum of coefficients)
    result_state = normalized_state * s
    
    return result_state, success_prob
```

---

## 💡 Hands-On Example: Pauli Combination

Let's implement $M = 0.5Z + 0.3X$ on a single qubit:

```python
from algorithm import LCUAlgorithm
from engine.core import GateSequence
import numpy as np

# Define the unitaries
Z_gate = GateSequence(1)
Z_gate.z(0)  # Pauli-Z

X_gate = GateSequence(1)
X_gate.x(0)  # Pauli-X

# Coefficients
alphas = [0.5, 0.3]  # M = 0.5*Z + 0.3*X

# Initial state: |+⟩ = (|0⟩ + |1⟩)/√2
initial = GateSequence(1)
initial.h(0)

# Run LCU
algo = LCUAlgorithm()
result = algo.run(
    alphas=alphas,
    unitaries=[Z_gate, X_gate],
    n_sys=1,
    initial_state=initial,
    backend='torch'
)

print(result['plot'])

# Theoretical analysis:
# s = 0.5 + 0.3 = 0.8
# ||M|+⟩|| = ||(0.5*Z + 0.3*X)|+⟩||
# Z|+⟩ = |−⟩, X|+⟩ = |+⟩
# M|+⟩ = 0.5|−⟩ + 0.3|+⟩
# ||M|+⟩||² ≈ 0.34, P_success = 0.34/0.64 ≈ 0.53

print(f"Success probability: {result['lcu_success_probability']:.4f}")
print(f"Expected: ~0.53")
```

---

## 🎓 Implementing Your Own LCU

### Complete Implementation Template

```python
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from engine.core import GateSequence, State

class MyLCU:
    """User-implemented Linear Combination of Unitaries."""
    
    def run(self, alphas: List[float], unitaries: List[GateSequence],
            n_sys: int, initial_state: Optional[GateSequence] = None
            ) -> Dict[str, Any]:
        """
        Execute LCU algorithm.
        
        Args:
            alphas: Non-negative coefficients
            unitaries: List of unitary circuits
            n_sys: Number of system qubits
            initial_state: Optional state preparation
        
        Returns: Solution state and success probability
        """
        # STEP 1: Validate and prepare
        m = len(alphas)
        assert m == len(unitaries), "Length mismatch"
        
        n_anc = int(np.ceil(np.log2(m)))
        s = sum(alphas)  # Normalization constant
        
        # STEP 2: Build circuit components
        circuit = GateSequence(n_anc + n_sys)
        anc_qubits = list(range(n_anc))
        sys_qubits = list(range(n_anc, n_anc + n_sys))
        
        # Initialize system state if provided
        if initial_state:
            circuit.append(initial_state, sys_qubits)
        
        # STEP 3: Build and apply V operator
        V = self._prepare_V(alphas, s, m, n_anc)
        circuit.append(V, anc_qubits)
        
        # STEP 4: Build and apply SELECT operator
        SELECT = self._build_SELECT(unitaries, m, n_anc, n_sys)
        circuit.append(SELECT, list(range(n_anc + n_sys)))
        
        # STEP 5: Build and apply V†
        circuit.append(V.dagger(), anc_qubits)
        
        # STEP 6: Execute and post-select
        state = circuit.execute()
        state_array = np.asarray(state, dtype=complex).reshape(-1)
        
        result_state, p_success = self._postselect(
            state_array, n_anc, n_sys, s
        )
        
        return {
            "state": result_state,
            "success_probability": p_success,
            "circuit": circuit,
            "message": f"LCU completed with success prob {p_success:.6f}"
        }
    
    def _prepare_V(self, alphas: List[float], s: float, 
                   m: int, n_anc: int) -> GateSequence:
        """
        Prepare V operator to encode coefficient superposition.
        
        Create state: Σⱼ √(αⱼ/s) |j⟩
        """
        # Normalize amplitudes
        amplitudes = np.array([np.sqrt(a / s) for a in alphas])
        
        # Pad to 2^n_anc
        dim = 2 ** n_anc
        padded = np.zeros(dim, dtype=complex)
        padded[:m] = amplitudes
        
        # Build circuit using amplitude encoding tree
        circuit = GateSequence(n_anc)
        gates = self._amplitude_encoding_tree(padded)
        
        for gate in gates:
            if gate['control'] is None:
                circuit.ry(gate['angle'], gate['target'])
            else:
                circuit.mcry(gate['angle'], gate['control'], gate['target'])
        
        return circuit
    
    def _build_SELECT(self, unitaries: List[GateSequence],
                     m: int, n_anc: int, n_sys: int) -> GateSequence:
        """
        Build SELECT operator to apply controlled unitaries.
        
        For each j, apply Uⱼ when ancilla = |j⟩
        """
        circuit = GateSequence(n_anc + n_sys)
        anc_qubits = list(range(n_anc))
        sys_qubits = list(range(n_anc, n_anc + n_sys))
        
        for j, U in enumerate(unitaries):
            # Control pattern: binary representation of j
            control_pattern = format(j, f'0{n_anc}b')
            
            # Apply U with proper control
            circuit.append(U, target=sys_qubits,
                          control=anc_qubits,
                          control_sequence=control_pattern)
        
        return circuit
    
    def _amplitude_encoding_tree(self, amplitudes: np.ndarray) -> List[Dict]:
        """
        Recursively compute gate sequence for amplitude encoding.
        
        Uses binary tree structure:
        - Each qubit controls a RY rotation at its level
        - Angle determined by amplitude ratios
        """
        # Normalize
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        n = len(amplitudes)
        depth = int(np.log2(n))
        
        gates = []
        
        def build_recursive(level, start, length, controls):
            if level == depth:
                return
            
            mid = length // 2
            left = amplitudes[start:start+mid]
            right = amplitudes[start+mid:start+length]
            
            norm_left = np.linalg.norm(left)
            norm_right = np.linalg.norm(right)
            total = norm_left + norm_right
            
            if total > 0:
                # RY angle: 2 * arctan2(right_norm, left_norm)
                angle = 2 * np.arctan2(norm_right, norm_left)
            else:
                angle = 0
            
            target = level
            
            if len(controls) == 0:
                gates.append({
                    'angle': angle,
                    'target': target,
                    'control': None
                })
            else:
                gates.append({
                    'angle': angle,
                    'target': target,
                    'control': controls.copy()
                })
            
            # Recurse
            build_recursive(level+1, start, mid, controls+[0])
            build_recursive(level+1, start+mid, mid, controls+[1])
        
        build_recursive(0, 0, n, [])
        return gates
    
    def _postselect(self, state: np.ndarray, n_anc: int, 
                   n_sys: int, s: float) -> Tuple[np.ndarray, float]:
        """
        Post-select on ancilla = |0⟩ to extract result.
        """
        # Extract ancilla = |0⟩ slice
        stride = 2 ** (n_anc + n_sys)
        zero_indices = [i for i in range(stride) if (i >> n_sys) == 0]
        
        # Get amplitudes
        success_state = state[zero_indices]
        
        # Success probability
        p_success = float(np.sum(np.abs(success_state) ** 2))
        
        if p_success < 1e-12:
            raise ValueError("Post-selection failed!")
        
        # Normalize
        normalized = success_state / np.sqrt(p_success)
        
        # Scale by s
        final_state = normalized * s
        
        return final_state, p_success
```

---

## 📊 Mathematical Deep Dive

### Coefficient Superposition

The V operator creates:
$$V|0\rangle_{\text{anc}} = \sum_{j=0}^{m-1} \sqrt{\frac{\alpha_j}{s}} |j\rangle_{\text{anc}}$$

where $s = \sum_{j=0}^{m-1} |\alpha_j|$

### SELECT Operation

Given auxiliary state $|j\rangle_{\text{anc}}$, apply $U_j$:
$$\text{SELECT} = \sum_{j=0}^{m-1} |j\rangle\langle j|_{\text{anc}} \otimes U_j$$

Combined with superposition:
$$(\text{SELECT}) \left(\sum_j \sqrt{\frac{\alpha_j}{s}}|j\rangle_{\text{anc}}\right) \otimes |\psi\rangle = \sum_j \sqrt{\frac{\alpha_j}{s}}|j\rangle_{\text{anc}} \otimes U_j|\psi\rangle$$

### V† Mapping Back

$$V^\dagger \left(\sum_j \sqrt{\frac{\alpha_j}{s}}|j\rangle_{\text{anc}}\right) = |0\rangle_{\text{anc}}$$

After applying V† to the full state:
$$\text{Result} = |0\rangle_{\text{anc}} \otimes \frac{1}{s}M|\psi\rangle + |0\rangle_{\text{anc}}^{\perp} \otimes (\text{garbage})$$

### Success Probability

$$P_{\text{success}} = \langle 0|_{\text{anc}} \otimes I_{\text{sys}} \, |\Psi\rangle\langle\Psi| \, |0\rangle_{\text{anc}} \otimes I_{\text{sys}}$$

$$= \frac{\|M|\psi\rangle\|^2}{s^2}$$

**Key insight**: Success probability depends on how well $M$ amplifies the initial state!

---

## 🔍 Debugging Tips

1. **Success probability is zero?**
   - Check if coefficients sum to zero
   - Verify initial state isn't orthogonal to M|ψ⟩
   - Ensure unitaries are correctly implemented

2. **V operator producing wrong phases?**
   - Verify amplitude normalization: Σ|α_j|² = s²
   - Check RY angle calculations in amplitude tree
   - Test V on trivial case (uniform coefficients)

3. **SELECT operator applying wrong unitary?**
   - Verify control patterns match unitary indices
   - Check qubit ordering (ancilla vs system)
   - Test with simple 2-unitary case first

4. **Post-selection failing?**
   - Increase simulation precision (more shots)
   - Check if ancilla qubits properly disentangled
   - Verify V† is exact inverse of V

5. **Circuit growing exponentially?**
   - Amplitude encoding: O(m log m) gates for m terms (good!)
   - SELECT: O(m) for m unitaries
   - Total complexity is polynomial, not exponential

---