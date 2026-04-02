---
name: initialization
description: |
  Quantum state initialization and preparation module for preparing arbitrary quantum states on qubits.
  Automatically constructs optimized quantum circuits that initialize qubits to target normalized state vectors
  using recursive decomposition techniques.
keywords:
  - state preparation
  - quantum initialization
  - state vector
  - quantum circuit synthesis
  - controlled operations
  - amplitude encoding
---

# Initialization Skills Guide

## I. Algorithm Introduction

### 1.1 Basic Concepts

**State Preparation (Quantum State Initialization)** is the process of initializing a quantum system to a specific target state. Given a normalized quantum state vector $|\psi\rangle$, the Initialization module automatically constructs a quantum circuit that evolves the initial ground state $|00...0\rangle$ to the target state through a series of quantum gate operations.

### 1.2 Mathematical Foundation

#### 1.2.1 Quantum State Representation

An n-qubit quantum system state vector is represented as:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

Where:
- $\alpha_i \in \mathbb{C}$ is the amplitude
- $|i\rangle$ is the computational basis state
- Normalization condition: $\sum_{i=0}^{2^n-1} |\alpha_i|^2 = 1$

#### 1.2.2 Phase and Amplitude

Each amplitude can be represented in polar form:

$$\alpha_i = r_i e^{i\phi_i}$$

Where:
- $r_i = |\alpha_i|$ is the amplitude magnitude (non-negative real number)
- $\phi_i = \arg(\alpha_i)$ is the phase

### 1.3 Recursive Decomposition Algorithm

Initialization uses a recursive decomposition method to construct state preparation circuits.

#### 1.3.1 Algorithm Workflow

```
Input: Target state vector |v⟩
    ↓
Check normalization condition
    ↓
Determine if single-qubit scenario
    ├─ Yes → Apply single-qubit gates directly
    └─ No → Vector decomposition
        ↓
    Decompose |v⟩ into two parts
    v1 = v[0:2^(n-1)]
    v2 = v[2^(n-1):]
        ↓
    Calculate probabilities
    p1 = ||v1||²
    p2 = ||v2||²
        ↓
    Apply RY gate to adjust last qubit probability
        ↓
    Recursively prepare v1 and v2
    Using controlled gates
        ↓
    Return complete circuit
```

#### 1.3.2 Single-Qubit Initialization

For a single quantum qubit, the target state is:

$$|\psi\rangle = \alpha_0 |0\rangle + \alpha_1 |1\rangle$$

Initialization steps:

1. **Extract Phase**:
   - $\phi_0 = \arg(\alpha_0)$
   - $\phi_1 = \arg(\alpha_1)$

2. **Apply Global Phase**:
   $$|\psi'\rangle = e^{-i\phi_0} |\psi\rangle$$

3. **Compute Rotation Angle**:
   $$\theta = \arccos(|\alpha_0|)$$

4. **Apply RY Rotation**:
   $$RY(2\theta) |0\rangle \rightarrow |\psi''\rangle$$

5. **Apply Phase Gate**:
   $$P(\phi_1 - \phi_0) |\psi''\rangle \rightarrow |\psi\rangle$$

```python
# Single-qubit initialization process diagram
Initial state: |0⟩
    ↓ [Apply global phase]
Initial state (phase adjusted)
    ↓ [Apply RY(2θ)]
Superposition state
    ↓ [Apply phase gate P(Δφ)]
Target state: α₀|0⟩ + α₁|1⟩
```

#### 1.3.3 Multi-Qubit Recursive Decomposition

For n-qubit scenarios, apply RY rotation on qubit (n-1):

1. **Probability Distribution Calculation**:
   - $p_1 = ||v_1||^2$ (probability of first half being in state 0)
   - $p_2 = ||v_2||^2$ (probability of first half being in state 1)

2. **Rotation Angle Calculation**:
   $$\theta = \arccos(\sqrt{p_1})$$

3. **Controlled State Preparation**:
   - When qubit (n-1) is 0: prepare normalized $v_1/||v_1||$
   - When qubit (n-1) is 1: prepare normalized $v_2/||v_2||$

```
n-qubit decomposition diagram:
┌──────────────────────────────────┐
│  Target state |v⟩ (2ⁿ amplitudes)│
└──────────────────────────────────┘
            ↓ [Split]
    ┌───────────────────┐
    │   v1 (first 2^(n-1))  │  v2 (last 2^(n-1))
    └───────────────────┘
         ↓ [RY rotation]
    Probability redistribution
         ↓ [Controlled preparation]
    ├─ Control=0: Recursively prepare v1
    └─ Control=1: Recursively prepare v2
```

### 1.4 Complexity Analysis

| Metric | Complexity | Description |
|------|---------|------|
| Circuit Depth | $O(n^2)$ | Recursion decomposition depth |
| Gate Count | $O(2^n)$ | Exponential number of gate operations |
| Parameter Participation | $O(2^n)$ | Need $2^n$ amplitudes |
| Time Complexity | $O(2^n)$ | Time to compute circuit |

---

## II. Usage Steps

### 2.1 Basic Usage Workflow

#### Step 1: Import Module
```python
from core.Initialization import state_preparation
from core.GateSequence import GateSequence
import numpy as np
```

#### Step 2: Define Target State Vector
```python
# Target state must be normalized
# Example: Single-qubit ground state
state_0 = np.array([1, 0])
state_1 = np.array([0, 1])

# Example: Single-qubit superposition
superposition = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# Example: Two-qubit Bell state
bell_plus = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
```

#### Step 3: Call state_preparation
```python
# Generate state preparation circuit
circuit = state_preparation(superposition, backend='torch')
```

#### Step 4: Verify and Use Circuit
```python
# Execute circuit to get quantum state
result_state = circuit.execute()

# Draw circuit
circuit.draw(title="State Preparation Circuit")

# Print circuit information
print(f"Circuit name: {circuit.name}")
print(f"Number of qubits: {circuit.get_num_qubits()}")
```

### 2.2 Detailed Usage Examples

#### Example 1: Prepare Single-Qubit Superposition State

```python
import numpy as np
from core.Initialization import state_preparation

# Target state: |+⟩ = (1/√2)(|0⟩ + |1⟩)
target_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# Generate circuit
circuit = state_preparation(target_state)

# Execute
result = circuit.execute()
print(f"Target state: {target_state}")
print(f"Execution result: {result}")

# Draw
circuit.draw(title="Superposition State")
```

#### Example 2: Prepare Two-Qubit Entangled State (Bell State)

```python
import numpy as np
from core.Initialization import state_preparation

# Bell state |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
bell_state = np.array([
    1/np.sqrt(2),  # |00⟩
    0,              # |01⟩
    0,              # |10⟩
    1/np.sqrt(2)   # |11⟩
])

# Generate circuit
circuit = state_preparation(bell_state)

# Execute and verify
result = circuit.execute()

print(f"Bell state: {bell_state}")
print(f"Result: {result}")

# Verify preparation success
print(f"Preparation successful: {np.allclose(result, bell_state)}")
```

#### Example 3: Prepare Complex Three-Qubit State

```python
import numpy as np
from core.Initialization import state_preparation

# Define a GHZ state: |GHZ⟩ = (1/√2)(|000⟩ + |111⟩)
ghz_state = np.zeros(8)
ghz_state[0] = 1/np.sqrt(2)
ghz_state[7] = 1/np.sqrt(2)

# Generate circuit
circuit = state_preparation(ghz_state)

# Execute
result = circuit.execute()

print("GHZ state successfully prepared!")
print(f"Non-zero components: {np.nonzero(result)[0]}")
```

#### Example 4: State Preparation with Phase

```python
import numpy as np
from core.Initialization import state_preparation

# State with phase: |ψ⟩ = (1/√2)|0⟩ + (i/√2)|1⟩
# where i = e^(iπ/2)
target_state = np.array([
    1/np.sqrt(2),           # Real: 1/√2
    1j/np.sqrt(2)           # Imaginary: i/√2
])

# Generate circuit
circuit = state_preparation(target_state)

# Execute
result = circuit.execute()

print(f"State with phase (target): {target_state}")
print(f"Execution result: {result}")
print(f"Phase difference: {np.angle(result[1]) - np.angle(result[0])}")
```

### 2.3 GateSequence Integration

#### Integration Method 1: Direct Initialization in Circuit

```python
from core.GateSequence import GateSequence
from core.Initialization import state_preparation
import numpy as np

# Target initial state
initial_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])

# Create main circuit
main_circuit = GateSequence(3)

# Use initialize method
main_circuit.initialize(initial_state, target=[0, 1])

# Apply more operations after initialization
main_circuit.h(2)
main_circuit.cnot(0, 2)

# Execute
result = main_circuit.execute()
```

#### Integration Method 2: Combine Multiple Preparation Circuits

```python
from core.GateSequence import GateSequence
from core.Initialization import state_preparation
import numpy as np

# Prepare two different states
state1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
state2 = np.array([1/np.sqrt(3), np.sqrt(2/3)])

# Generate corresponding circuits
circuit1 = state_preparation(state1)
circuit2 = state_preparation(state2)

# Create main circuit
main_circuit = GateSequence(4)

# Apply separately on different bits
main_circuit.append(circuit1, [0, 1])
main_circuit.append(circuit2, [2, 3])

# Add interaction between two parts
main_circuit.cnot(1, 2)

# Execute
result = main_circuit.execute()
```

### 2.4 Advanced Scenarios

#### Scenario 1: Parameterized State Preparation

```python
def create_parameterized_state(theta):
    """Create parameterized single-qubit state"""
    # State: |ψ(θ)⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
    state = np.array([np.cos(theta), np.sin(theta)])
    return state_preparation(state)

# Use different parameters
theta_values = [0, np.pi/4, np.pi/2]

for theta in theta_values:
    circuit = create_parameterized_state(theta)
    result = circuit.execute()
    print(f"θ = {theta:.4f}: {result}")
```

#### Scenario 2: Quantum Algorithm Initialization

```python
def prepare_search_state(n):
    """Prepare uniform superposition state for Grover search"""
    # Create uniform superposition: |s⟩ = (1/√2ⁿ) Σᵢ |i⟩
    state = np.ones(2**n) / np.sqrt(2**n)
    return state_preparation(state)

# Prepare for 3-qubit search
circuit = prepare_search_state(3)
result = circuit.execute()

print("Uniform superposition state prepared")
print(f"All amplitudes equal: {np.allclose(np.abs(result), 1/np.sqrt(8))}")
```

#### Scenario 3: Encode Classical Data

```python
def amplitude_encode_data(data):
    """
    Use amplitude encoding to encode classical data
    
    Input data must be normalized to [-1, 1]
    """
    # Normalize data
    normalized = np.array(data, dtype=float)
    normalized = normalized / np.linalg.norm(normalized)
    
    # Create state
    circuit = state_preparation(normalized)
    
    return circuit

# Encode data [1, 2, 3, 4]
data = [1, 2, 3, 4]
circuit = amplitude_encode_data(data)

result = circuit.execute()
print(f"Encoded data: {result}")
```

---

## III. Key API Reference

### 3.1 state_preparation Function

```python
state_preparation(v, backend='torch')
```

**Parameters:**
- `v: array-like` - Target quantum state vector
  - Must be element-wise complex array
  - Must satisfy normalization condition: $\sum_i |v_i|^2 = 1$
  - Length must be integer power of 2 ($2^n$)

- `backend: str` - Backend to use
  - Default: `'torch'`
  - Optional: `'torch'`, other supported backends

**Returns:**
- `GateSequence` - Prepared quantum circuit object

**Exceptions:**
- `ValueError` - If vector is not normalized

### 3.2 Returned GateSequence Object

The returned circuit object supports following operations:

```python
# Execute circuit
state = circuit.execute()

# Get circuit matrix
matrix = circuit.get_matrix()

# Copy circuit
circuit_copy = circuit.copy()

# Draw circuit
circuit.draw(filename=None, title=None)

# Get information
num_qubits = circuit.get_num_qubits()
backend = circuit.get_backend_type()
```

---

## IV. Mathematical Details and Verification

### 4.1 Normalization Condition Check

```python
import numpy as np

def check_normalization(state_vector):
    """Check if vector is normalized"""
    norm = np.linalg.norm(state_vector)
    
    print(f"Vector norm: {norm}")
    print(f"Is normalized (norm=1): {np.isclose(norm, 1)}")
    
    # Check sum of squared amplitudes
    prob_sum = np.sum(np.abs(state_vector)**2)
    print(f"Probability sum: {prob_sum}")
    
    return np.isclose(prob_sum, 1)

# Test
state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
check_normalization(state)
```

### 4.2 Phase Verification

```python
import numpy as np

def analyze_phases(state_vector):
    """Analyze phase information in state vector"""
    amplitudes = np.abs(state_vector)
    phases = np.angle(state_vector)
    
    print("Amplitude and phase analysis:")
    for i, (amp, phase) in enumerate(zip(amplitudes, phases)):
        print(f"  |{i}⟩: amplitude={amp:.4f}, phase={phase:.4f} rad ({np.degrees(phase):.1f}°)")
    
    return amplitudes, phases

# 测试
state = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
analyze_phases(state)
```

### 4.3 保真度计算

```python
def fidelity(state1, state2):
    """
    计算两个量子态之间的保真度
    F = |⟨ψ₁|ψ₂⟩|²
    """
    # 计算内积
    inner_product = np.vdot(state1, state2)
    
    # 计算保真度
    fidelity_value = np.abs(inner_product)**2
    
    return fidelity_value

# 测试
target = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
circuit = state_preparation(target)
result = circuit.execute()
f = fidelity(target, result)
print(f"保真度: {f:.6f}")
```

---

## V. Common Patterns

### Pattern 1: Simple Single-Qubit Initialization

```python
def create_single_qubit_state(theta, phi):
    """
    Create arbitrary single qubit state
    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    """
    state = np.array([
        np.cos(theta/2),
        np.exp(1j * phi) * np.sin(theta/2)
    ])
    return state_preparation(state)
```

### Pattern 2: Multi-Qubit Entangled States

```python
def create_ghz_state(n):
    """Create n-qubit GHZ state"""
    state = np.zeros(2**n)
    state[0] = 1/np.sqrt(2)
    state[-1] = 1/np.sqrt(2)
    return state_preparation(state)

def create_w_state(n):
    """Create n-qubit W state"""
    state = np.zeros(2**n)
    # W state: (1/√n)(|100...0⟩ + |010...0⟩ + ... + |000...1⟩)
    for i in range(n):
        state[2**i] = 1/np.sqrt(n)
    return state_preparation(state)
```

### Pattern 3: Data Encoding

```python
def encode_amplitude(values):
    """
    Amplitude encoding: encode values into quantum amplitudes
    values must sum to 1 in probability (|v|² sum to 1)
    """
    data = np.array(values)
    # Normalize
    data = data / np.linalg.norm(data)
    
    return state_preparation(data)
```

---

## VI. Error Handling Guide

### 6.1 Common Errors

**Error 1: Vector Not Normalized**
```python
# Wrong!
state = np.array([1, 1])  # ||state|| = √2 ≠ 1

try:
    circuit = state_preparation(state)
except ValueError as e:
    print(f"Error: {e}")  # The vector is not unit!

# Correct way
state = state / np.linalg.norm(state)
circuit = state_preparation(state)
```

**Error 2: Vector Length Not a Power of 2**
```python
# Wrong!
state = np.array([1/np.sqrt(3)] * 3)  # 3 ≠ 2^n

# Correct way
state = np.zeros(4)  # 4 = 2²
state[0] = 1/np.sqrt(2)
state[3] = 1/np.sqrt(2)
```

**Error 3: Complex Amplitude Handling**
```python
# Possible error
state = np.array([0.5 + 0.5j, 0.5 - 0.5j])

# Correct way
state = np.array([0.5 + 0.5j, 0.5 - 0.5j])
state = state / np.linalg.norm(state)  # Normalize first
circuit = state_preparation(state)
```

### 6.2 Verification Measures

```python
def safe_state_preparation(state_vector):
    """State preparation with validation"""
    # 1. Convert to numpy array
    state = np.array(state_vector, dtype=complex)
    
    # 2. Check if length is power of 2
    n = len(state)
    if not (n & (n - 1) == 0):
        raise ValueError(f"State length {n} is not a power of 2")
    
    # 3. Check and normalize
    norm = np.linalg.norm(state)
    if np.isclose(norm, 0):
        raise ValueError("State vector is all zeros")
    
    state = state / norm
    
    # 4. Verify again
    if not np.isclose(np.linalg.norm(state), 1):
        raise ValueError("Normalization failed")
    
    # 5. Prepare state
    return state_preparation(state)

# Usage
try:
    circuit = safe_state_preparation([1, 1])
    print("Success!")
except ValueError as e:
    print(f"Preparation failed: {e}")
```

---

## VII. Performance Optimization

### 7.1 Circuit Complexity Management

```python
def estimate_circuit_complexity(n_qubits):
    """Estimate state preparation circuit complexity"""
    
    num_params = 2**n_qubits
    depth = n_qubits**2  # Recursive decomposition depth
    num_gates = 3 * (2**n_qubits - 1)  # Approximate gate count
    
    print(f"n = {n_qubits}:")
    print(f"  Number of parameters: {num_params}")
    print(f"  Circuit depth: {depth}")
    print(f"  Gate count: {num_gates}")
    
    return num_params, depth, num_gates

# Analyze different scales
for n in range(1, 6):
    estimate_circuit_complexity(n)
```

### 7.2 Caching Strategy

```python
# Cache generated circuits
circuit_cache = {}

def get_or_create_circuit(state_key, state_vector):
    """Circuit creation with caching"""
    if state_key not in circuit_cache:
        circuit_cache[state_key] = state_preparation(state_vector)
    
    return circuit_cache[state_key]

# Usage
circuit = get_or_create_circuit("bell", bell_state)
```

---

## VIII. Complete Workflow Example

```python
import numpy as np
from core.Initialization import state_preparation
from core.GateSequence import GateSequence

def complete_workflow():
    """Complete state preparation and handling workflow"""
    
    # 1. Define target state
    print("Step 1: Define target state")
    target_state = np.array([
        1/np.sqrt(8),
        1/np.sqrt(8),
        1/np.sqrt(8),
        1/np.sqrt(8),
        1/np.sqrt(8),
        np.sqrt(3/8)
    ])
    
    # 2. Prepare state
    print("Step 2: Generate state preparation circuit")
    circuit = state_preparation(target_state)
    
    # 3. Execute circuit
    print("Step 3: Execute circuit")
    result_state = circuit.execute()
    
    # 4. Verify results
    print("Step 4: Verify results")
    fidelity = np.abs(np.vdot(target_state, result_state))**2
    print(f"  Fidelity: {fidelity:.6f}")
    
    # 5. Extend circuit
    print("Step 5: Extend circuit")
    extended_circuit = GateSequence(6)
    extended_circuit.append(circuit, [0, 1, 2])
    extended_circuit.h(3)
    extended_circuit.cnot(2, 5)
    
    # 6. Draw circuit
    print("Step 6: Draw circuit")
    circuit.draw(title="State Preparation Circuit")
    
    return circuit, result_state, fidelity

# Execute workflow
circuit, result, f = complete_workflow()
```

---

## IX. Summary Checklist

When using the Initialization module, ensure:

- [ ] Target state vector is normalized ($\sum_i |v_i|^2 = 1$)
- [ ] Vector length is an integer power of 2
- [ ] Using correct data types (complex numpy arrays)
- [ ] Verified generated circuit accuracy (fidelity close to 1)
- [ ] Understood recursive decomposition complexity
- [ ] Consider computational resources for large qubit systems (n > 10)
- [ ] Implement caching mechanism if using same state multiple times
- [ ] Correctly handle state vectors with phases
- [ ] Use correct target qubit indices when integrating to main circuit
- [ ] Consider using specific encoding methods (e.g., amplitude encoding) for complex quantum states

