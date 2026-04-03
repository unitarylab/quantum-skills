---
name: state
description: |
  Quantum state representation and manipulation module using PyTorch backend.
  Provides comprehensive tools for creating, measuring, and analyzing quantum states
  including normalization, inner products, tensor products, and measurement operations.
keywords:
  - quantum state
  - state vector
  - measurement
  - probability distribution
  - state collapse
  - tensor product
  - expectation value
---

# State Skills Guide

## I. Algorithm Introduction

### 1.1 Basic Concepts

**State (Quantum State)** is a core concept in quantum computing, representing a complete description of a quantum system. The State class uses vector notation to express quantum states in the computational basis and performs efficient calculations through PyTorch tensors.

### 1.2 Mathematical Foundation

#### 1.2.1 State Vector Representation

For an $n$ qubit system, the quantum state can be expressed as:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

Where:
- $\alpha_i \in \mathbb{C}$ is the amplitude of the $i$-th basis state
- $|i\rangle$ is the computational basis state $|i_1 i_2 \cdots i_n\rangle$
- Normalization condition: $\sum_{i=0}^{2^n-1} |\alpha_i|^2 = 1$

In the State class, this vector is stored as a one-dimensional PyTorch tensor of length $2^n$.

#### 1.2.2 Mathematical Representation of Quantum Operations

**Measurement in Computational Basis**:
$$P(i) = |\alpha_i|^2$$

**Inner Product**:
$$\langle \psi | \phi \rangle = \sum_{i=0}^{2^n-1} \alpha_i^* \beta_i$$

**Expectation Value**:
$$\langle O \rangle = \langle \psi | O | \psi \rangle$$

**Tensor Product**:
$$|\psi_1\rangle \otimes |\psi_2\rangle = \sum_{i,j} \alpha_i \beta_j |i\rangle \otimes |j\rangle$$

### 1.3 Processing Workflow

```
Create State
    ↓
Initialize Data (number of bits or vector)
    ↓
Auto-normalize
    ↓
Query: probability, expectation value
    or
Operations: inner product, tensor product
    or
Measurement: probability distribution, sampling, state collapse
    ↓
Output results
```

### 1.4 Core Features

**Flexible Initialization**
- Create ground state |0⟩^⊗n with number of qubits
- Initialize arbitrary state with data vector

**Automatic Data Processing**
- Support multiple input formats (Tensor, List, Numpy)
- Automatic dimension check and flattening
- Auto-normalization during creation

**Precise Quantum Operations**
- Support complex amplitudes
- Endian flexibility (big-endian/little-endian)
- Numerical stability guarantee

**Efficient PyTorch Implementation**
- GPU acceleration support
- Vectorized operations
- Memory efficient

### 1.5 Endian Convention

```
Little Endian (little-endian, Qiskit default):
- Bit 0 is rightmost (least significant)
- Tensor representation: [q_{n-1}, q_{n-2}, ..., q_0]
- Binary string "101" means: q_0=1, q_1=0, q_2=1

Big Endian (big-endian):
- Bit 0 is leftmost
- Tensor representation: [q_0, q_1, ..., q_{n-1}]
- Binary string "101" means: q_0=1, q_1=0, q_2=1
```

---

## II. Usage Steps

### 2.1 Basic Usage Workflow

#### Step 1: Import Module
```python
from core.State import State
import torch
import numpy as np
```

#### Step 2: Create State
```python
# Method 1: Create with number of bits (creates |0⟩^⊗n)
state = State(3)  # 3-bit ground state |000⟩

# Method 2: With vector data
state = State([1/np.sqrt(2), 1/np.sqrt(2)])

# Method 3: With PyTorch tensor
data = torch.tensor([1, 1], dtype=torch.complex128)
state = State(data)  # Auto-normalize
```

#### Step 3: Query State Information
```python
# Get basic info
n_qubits = state.num_qubits
dimension = state.dim
prob_dist = state.probabilities()

# Get data
data = state.data
dtype = state.dtype
```

#### Step 4: Perform Quantum Operations
```python
# View probability distribution
probs = state.probabilities_dict([0, 1])

# Calculate expectation value
matrix = torch.eye(4, dtype=torch.complex128)
exp_val = state.expectation_value(matrix)

# Measurement (collapse)
result = state.measure([0, 1])

# Sampling
samples = state.sample_counts(shots=1024)
```

### 2.2 Detailed Usage Examples

#### Example 1: Creating and Checking Basic States

```python
import numpy as np
from core.State import State

# Create |0⟩ state
state_0 = State(1)
print(state_0)
# Output: State: qubits: 1, data: [1+0j, 0+0j]

# Create |1⟩ state
state_1_data = [0, 1]
state_1 = State(state_1_data)
print(state_1)
# Output: State: qubits: 1, data: [0+0j, 1+0j]

# Verify basic information
print(f"Number of bits: {state_0.num_qubits}")    # 1
print(f"Hilbert space dimension: {state_0.dim}") # 2
print(f"Data type: {state_0.dtype}")       # torch.complex128
```

#### Example 2: Creating Superposition States

```python
import numpy as np
from core.State import State

# Create single-qubit superposition |+⟩ = (1/√2)(|0⟩ + |1⟩)
plus_state = State([1/np.sqrt(2), 1/np.sqrt(2)])
print(f"+ state data: {plus_state.data}")

# Create |-⟩ = (1/√2)(|0⟩ - |1⟩)
minus_state = State([1/np.sqrt(2), -1/np.sqrt(2)])
print(f"- state data: {minus_state.data}")

# State with phase |i⟩ = (1/√2)(|0⟩ + i|1⟩)
i_state = State([1/np.sqrt(2), 1j/np.sqrt(2)])
print(f"i state data: {i_state.data}")

# Check probability distributions are 50%-50%
print(f"Probability distribution: {plus_state.probabilities()}")
```

#### Example 3: Multi-qubit Basis States

```python
from core.State import State
import torch
import numpy as np

# Create 3-qubit |000⟩ state
state = State(3)
print(f"Number of bits: {state.num_qubits}")      # 3
print(f"Dimension: {state.dim}")              # 8
print(f"Data: {state.data}")
# Only the first component is 1, others are 0

# Create specific 3-qubit state |101⟩
# Binary 101 = Decimal 5
state_101_data = [0] * 8
state_101_data[5] = 1  # Position 5 corresponds to |101⟩ (little-endian)
state_101 = State(state_101_data)
print(f"|101⟩ state: {state_101.data}")
```

#### Example 4: Probability Distribution Query (Non-Collapse)

```python
from core.State import State
import numpy as np

# Create Bell state |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
bell_plus_data = [
    1/np.sqrt(2), 0, 0, 1/np.sqrt(2)
]
state = State(bell_plus_data)

# Query probability of all bits
full_probs = state.probabilities_dict([0, 1])
print("Complete probability distribution:")
for bits, prob in full_probs.items():
    print(f"  {bits}: {prob:.4f}")

# Query probability of specific bits (partial trace)
partial_0 = state.probabilities_dict([0])
print("\nProbability of qubit 0 only:")
for bits, prob in partial_0.items():
    print(f"  {bits}: {prob:.4f}")
```

#### Example 5: Measurement and State Collapse

```python
from core.State import State
import numpy as np

# Create uniform superposition state
uniform_data = np.ones(4) / 2  # (1/2)(|00⟩ + |01⟩ + |10⟩ + |11⟩)
state = State(uniform_data)

print(f"Initial state probability: {state.probabilities()}")

# Measure qubit 0
result = state.measure([0])
print(f"Measurement result: {result}")
print(f"Probability after measurement: {state.probabilities()}")
# State has collapsed to one of two possibilities

# Measure qubit 1 again (state has changed)
result2 = state.measure([1])
print(f"Second measurement result: {result2}")
print(f"Final state: {state.data}")
# Now state is a definite basis state
```

#### Example 6: Sampling and Statistics

```python
from core.State import State
import numpy as np

# Create non-uniform distribution
# P(|0⟩) = 0.25, P(|1⟩) = 0.75
weighted_state = State([0.5, np.sqrt(0.75)])

# Sample 10000 times
samples = weighted_state.sample_counts(shots=10000)
print("Sampling results:")
for state_str, count in sorted(samples.items()):
    print(f"  {state_str}: {count}/10000 ({count/100:.1f}%)")

# Theoretical values
print("\nTheoretical probability:")
print(f"  |0⟩: 25%")
print(f"  |1⟩: 75%"
```

### 2.3 Integration with Quantum Circuits

#### Integration Method 1: Getting State After Circuit Execution

```python
from engine.core.gate_sequence import GateSequence
from engine.core.State import State
from engine.core.register import Register

# Create circuit
qreg = Register("q", 2)
circuit = GateSequence(qreg)

# Build circuit
circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])

# Execute and get state vector
result_vector = circuit.execute()

# Create State object
state = State(result_vector)
print(f"Circuit result state: {state}")
print(f"Probability distribution: {state.probabilities()}")
```

#### Integration Method 2: Analyzing Circuit Results

```python
from engine.core.gate_sequence import GateSequence
from engine.core.State import State
from engine.core.register import Register

# Create Bell circuit
qreg = Register("q", 2)
circuit = GateSequence(qreg)
circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])

# Execute
result_vector = circuit.execute()
state = State(result_vector)

# Analyze probability distribution
probs = state.calculate_state([0, 1])
print("Complete analysis:")
for bits, info in probs.items():
    print(f"  {bits}: probability={info['prob']:.4f}, int_value={info['int']}")

# Perform measurement sampling
samples = state.sample_counts(shots=1000)
print(f"\n1000 samples: {samples}")
```

### 2.4 Advanced Scenarios

#### Scenario 1: Inner Product Calculation

```python
from engine.core.State import State
import numpy as np

# Create two states
state1 = State([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩
state2 = State([1/np.sqrt(2), -1/np.sqrt(2)])  # |-⟩

# Calculate inner product ⟨+|-⟩
inner = state1.inner_product(state2)
print(f"⟨+|-⟩ = {inner}")
# Theoretical value: 0

# Calculate ⟨+|+⟩ (should be 1)
self_inner = state1.inner_product(state1)
print(f"⟨+|+⟩ = {self_inner}")
```

#### Scenario 2: Expectation Value Calculation

```python
from engine.core.State import State
import torch
import numpy as np

# Create a state
state = State([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩

# Define Pauli-Z matrix
pauli_z = torch.tensor([
    [1, 0],
    [0, -1]
], dtype=torch.complex128)

# Calculate ⟨Z⟩ expectation value
exp_z = state.expectation_value(pauli_z)
print(f"⟨Z⟩ = {exp_z}")
# For |+⟩, ⟨Z⟩ = 0

# Define projection operator P_0 = |0⟩⟨0|
proj_0 = torch.tensor([
    [1, 0],
    [0, 0]
], dtype=torch.complex128)

# Calculate probability of measuring |0⟩
prob_0 = state.expectation_value(proj_0).real
print(f"P(|0⟩) = {prob_0}")
```

#### Scenario 3: Tensor Product

```python
from engine.core.State import State
import numpy as np

# Create two single-qubit states
state_0 = State(1)  # |0⟩
state_plus = State([1/np.sqrt(2), 1/np.sqrt(2)])  # |+⟩

# Calculate tensor product |0⟩ ⊗ |+⟩
combined = state_0.tensor(state_plus)

print(f"Result type: {type(combined)}")
print(f"Number of qubits: {combined.num_qubits}")  # 2
print(f"Data: {combined.data}")
# Should be (1/√2)|00⟩ + (1/√2)|01⟩
```

#### Scenario 4: Continue Operation After Partial Measurement

```python
from engine.core.State import State
import numpy as np

# Create Bell state
bell_data = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
state = State(bell_data)

print("Initial Bell state:")
print(f"  Complete probability: {state.probabilities_dict([0, 1])}")

# Measure qubit 0
result_0 = state.measure([0])
print(f"\nMeasure qubit 0: {result_0}")

# View state after measurement
print(f"  Complete state probability after collapse: {state.probabilities()}")

# Partial state probability
partial = state.calculate_state([1])
print(f"  Distribution of qubit 1: {partial}")

# Continue measuring qubit 1
result_1 = state.measure([1])
print(f"\nMeasure qubit 1: {result_1}")
print(f"  Final state: {state.data}")
```

---

## III. Key API Reference

### 3.1 Constructor

```python
State(data, num_qubits=None)
```

**Parameters:**
- `data: int | array-like` - Initialization data
  - If `int`: Create n-qubit ground state $|0\rangle^{\otimes n}$
  - If array: State vector data (auto-flattened to 1D)
- `num_qubits: int` - Optional, for validation (usually auto-derived)

**Returns:**
- Initialized and normalized State object

**Exceptions:**
- `ValueError` - If dimension is not $2^n$ or normalization fails

### 3.2 Properties Access

```python
# Read-only properties
state.num_qubits    # Number of qubits
state.dim           # Hilbert space dimension
state.data          # State vector (PyTorch tensor)
state.dtype         # Data type (torch.complex128)

# Methods
state.norm()        # Calculate vector norm
```

### 3.3 Core Methods

| Method | Parameter | Return | Description |
|------|------|--------|------|
| `normalize()` | - | self | Normalize state |
| `inner_product(other)` | other: State | complex | Calculate inner product $\langle\psi\|\phi\rangle$ |
| `tensor(other)` | other: State | State | Tensor product $\|\psi\rangle\otimes\|\phi\rangle$ |
| `expectation_value(matrix)` | matrix: Tensor | complex | Expectation value $\langle\psi|O|\psi\rangle$ |
| `probabilities()` | - | Tensor | Probability distribution of all basis states |
| `probabilities_dict(targets)` | targets: list | dict | Probability distribution of specified qubits |
| `measure(targets)` | targets: list | str | Measure and return result string |
| `sample_counts(shots)` | shots: int | dict | Statistics of sampling results for specified times |
| `calculate_state(targets)` | targets: list | dict | Detailed state analysis |

### 3.4 Method Description

#### probabilities_dict

```python
probabilities_dict(target_indices, endian='little', threshold=1e-9) -> dict
```

**Parameters:**
- `target_indices` - Index of qubits to analyze
- `endian` - Endianness ('little' or 'big')
- `threshold` - Probability threshold (values below this are ignored)

**Returns:**
```python
{
    '00': 0.5,
    '11': 0.5
}
```

#### measure

```python
measure(target_indices, endian='little') -> str
```

**Parameters:**
- `target_indices` - Index of qubits to measure
- `endian` - Endianness

**Returns:**
- Binary string of measurement result (e.g., '101')

**Side Effect:**
- Modifies State object (state collapse)

#### calculate_state

```python
calculate_state(target_indices, endian='little', threshold=1e-5) -> dict
```

**Returns:**
```python
{
    '00': {'prob': 0.5, 'int': 0},
    '11': {'prob': 0.5, 'int': 3}
}
```

---

## IV. Mathematical Operations Explained

### 4.1 Mathematical Model of Quantum Measurement

Single measurement process:

1. **Probability Calculation**: $P(i) = |\alpha_i|^2$
2. **Sample Selection**: Sample from $\{0, 1, ..., 2^n-1\}$ according to probability
3. **State Collapse**: $|\psi\rangle \rightarrow \frac{|i\rangle}{\sqrt{P(i)}}$

### 4.2 Partial Bit Measurement and Partial Trace

For probability distribution of a subset of bits, calculate via partial trace:

$$\rho_A = \text{Tr}_B(\rho)$$

The diagonal elements represent the probability of measuring bits in A.

### 4.3 Expectation Value and Projection Operators

Probability of measuring state $|i\rangle$:

$$P(i) = \langle\psi|P_i|\psi\rangle$$

where $P_i = |i\rangle\langle i|$ is the projection operator.

---
## V. Common Patterns

### Pattern 1: Create Standard Basis State

```python
def create_basis_state(n_qubits, target_state_int):
    """Create computational basis state |target_state⟩"""
    data = [0] * (2**n_qubits)
    data[target_state_int] = 1
    return State(data)

# Usage
state_5 = create_basis_state(3, 5)  # |101⟩
```

### Pattern 2: Create Uniform Superposition

```python
def create_uniform_superposition(n_qubits):
    """Create uniform superposition state"""
    amplitude = 1 / np.sqrt(2**n_qubits)
    data = [amplitude] * (2**n_qubits)
    return State(data)

# Usage
uniform = create_uniform_superposition(3)
```

### Pattern 3: Analyze Quantum Circuit Output

```python
def analyze_circuit_output(circuit, num_qubits, shots=1000):
    """Analyze circuit output"""
    result_vector = circuit.execute()
    state = State(result_vector)
    
    # Get statistics
    stats = state.sample_counts(shots=shots)
    
    # Get theoretical probabilities
    theory = state.probabilities_dict(list(range(num_qubits)))
    
    return stats, theory
```

---
## VI. Error Handling Guide

### 6.1 Common Errors

**Error 1: Dimension is not a power of 2**
```python
# Wrong!
try:
    state = State([1, 1, 1])  # 3 ≠ 2^n
except ValueError as e:
    print(f"Error: {e}")  # Input data length 3 is not a power of 2

# Correct
state = State([1, 1, 1, 1]) / 2  # 4 = 2^2
```

**Error 2: Non-normalized vector**
```python
# Acceptable, but will be auto-fixed
state = State([1, 1])  # Not normalized
print(state.data)  # Auto-normalized to [1/√2, 1/√2]
```

**Error 3: Dimension mismatch in operations**
```python
state1 = State(1)   # 1 qubit
state2 = State(2)   # 2 qubits

# Wrong!
try:
    inner = state1.inner_product(state2)
except ValueError as e:
    print(f"Error: {e}")  # Dimension mismatch

# Correct: Use tensor product
combined = state1.tensor(state2)
```

### 6.2 Validation Function

```python
def validate_state(state):
    """Validate State object validity"""
    # Check type
    if not isinstance(state, State):
        raise TypeError("Expected State object")
    
    # Check normalization
    norm = state.norm()
    if not np.isclose(norm, 1.0):
        print(f"Warning: State not properly normalized (norm={norm})")
    
    # Check dimension
    if state.dim != 2**state.num_qubits:
        raise ValueError("Dimension and qubit count mismatch")
    
    return True
```

---
## VII. Performance Optimization

### 7.1 Using GPU Acceleration

```python
import torch
from engine.core.State import State

# If GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create state and move to GPU (requires State class modification)
state_data = torch.tensor([1, 1], dtype=torch.complex128, device=device)
state = State(state_data)
```

### 7.2 Efficiency of Batch Sampling

```python
# Sample large number of qubits
n_qubits = 20
n_shots = 100000

state = State(n_qubits)
samples = state.sample_counts(shots=n_shots)

# More efficient than measuring n_shots times individually
```

### 7.3 Reusing State Objects

```python
# Good practice: Create once, analyze multiple times
state = State([1/np.sqrt(2), 1/np.sqrt(2)])

# Multiple queries (does not modify state)
state.probabilities_dict([0])
state.calculate_state([0])
state.sample_counts(shots=100)

# Avoid repeated creation
# for i in range(100):
#     s = State([1/np.sqrt(2), 1/np.sqrt(2)])  # Bad
```

---
## VIII. Complete Workflow Example

```python
import numpy as np
from engine.core.State import State
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

def complete_quantum_workflow():
    """Complete quantum computing workflow"""
    
    # Step 1: Create quantum circuit
    print("Step 1: Build quantum circuit")
    qreg = Register("q", 2)
    circuit = GateSequence(qreg)
    
    # Step 2: Apply quantum gates to build Bell state
    print("Step 2: Apply quantum gates")
    circuit.h(qreg[0])
    circuit.cnot(qreg[0], qreg[1])
    
    # Step 3: Execute circuit to get state vector
    print("Step 3: Execute circuit")
    result_vector = circuit.execute()
    
    # Step 4: Create State object
    print("Step 4: Create quantum state object")
    state = State(result_vector)
    
    # Step 5: Analyze state
    print("Step 5: Analyze quantum state")
    print(f"  Number of qubits: {state.num_qubits}")
    print(f"  Complete probability distribution: {state.probabilities_dict([0, 1])}")
    
    # Step 6: Verify fidelity
    print("Step 6: Verify results")
    fidelity = np.abs(state.data[0])**2 + np.abs(state.data[3])**2
    print(f"  Bell state fidelity: {fidelity:.4f}")
    
    # Step 7: Sampling
    print("Step 7: Quantum sampling")
    samples = state.sample_counts(shots=1000)
    print(f"  Sampling results: {samples}")
    
    # Step 8: Measurement (modifies state)
    print("Step 8: Execute measurement")
    result = state.measure([0, 1])
    print(f"  Measurement result: {result}")
    print(f"  Final state: {state.data}")

# Execute
complete_quantum_workflow()
```

---
## IX. Summary Checklist

When using the State class, please ensure:

- [ ] State class has been correctly imported
- [ ] Data dimension is $2^n$ or created with qubit count
- [ ] Understand automatic state normalization
- [ ] Know that measurement modifies state (collapse)
- [ ] Understand the role of endian parameter
- [ ] Use correct indexing method to access qubits
- [ ] Distinguish between probability query (non-collapse) and measurement (collapse)
- [ ] Handle complex amplitudes correctly
- [ ] Verify state normalization when necessary
- [ ] Use appropriate numerical threshold for floating-point error handling

