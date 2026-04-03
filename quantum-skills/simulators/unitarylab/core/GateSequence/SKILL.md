---
name: gatesequence
description: |
  Quantum Gate Sequence module for constructing, manipulating, and executing quantum circuits.
  Provides high-level abstractions for quantum gate operations with register management and 
  classical-quantum integration capabilities.
keywords:
  - quantum gates
  - gate sequence
  - quantum circuit
  - register management
  - quantum operations
  - gate composition
  - measurement
---

# GateSequence Skills Guide

## I. Algorithm Introduction

### 1.1 Basic Concepts

**GateSequence (Quantum Gate Sequence)** is the core abstraction of quantum circuits, used for constructing, managing, and executing quantum computing operations. It provides a high-level interface for operating on quantum qubits while managing interactions with classical computing systems.

### 1.2 Core Architecture

#### 1.2.1 Register Management System

GateSequence manages two types of registers:

**Quantum Register**
```
┌─────────────────────────────────────┐
│       Quantum Registers              │
├─────────────────────────────────────┤
│ Register 1: q0, q1, q2 (n_qubits=3) │
│ Register 2: q3, q4 (n_qubits=2)     │
│ ...                                  │
└─────────────────────────────────────┘
        ↓ (Global Index Mapping)
┌─────────────────────────────────────┐
│    Global Qubit Index Space         │
│  q0, q1, q2, q3, q4, ... (total_q) │
└─────────────────────────────────────┘
```

**Classical Register**
- Store quantum measurement results
- Support quantum-classical hybrid algorithms
- Foundation for conditional quantum operations

#### 1.2.2 Index Conversion Mechanism

GateSequence automatically handles local-to-global index conversion:

```python
# Local Index → Global Index Conversion
Register qreg = Register("q", 3)
GateSequence circuit = GateSequence(qreg)

circuit.x(qreg[0])  # Local index 0 → Global index 0
circuit.h(qreg[2])  # Local index 2 → Global index 2
```

#### 1.2.3 Quantum Gate Classification

| Gate Type | Description | Examples |
|--------|--------|------|
| **Single-Qubit Gates** | Operate on single qubit | X, Y, Z, H, S, T, RX, RY, RZ |
| **Rotation Gates** | Parameterized rotation operations | RX(θ), RY(θ), RZ(θ) |
| **Controlled Gates** | One control qubit | CNOT, CZ, CH |
| **Multi-Control Gates** | Multiple control qubits | MCX, MCY, MCZ |
| **Custom Gates** | User-defined unitary operators | Unitary(matrix) |
| **Measurement** | Quantum-to-classical conversion | Measure |

### 1.3 Circuit Processing Workflow

```
Create GateSequence
    ↓
Add Quantum/Classical Registers
    ↓
Build Circuit (Add Quantum Gates)
    ↓
Optional: Circuit Decomposition/Optimization
    ↓
Execute Circuit
    ↓
Get Measurement Results
    ↓
Extract Circuit Matrix or Draw Image
```

### 1.4 Design Characteristics

**Flexible Index System**
- Support register object indexing
- Support integer indexing
- Support slice operations
- Automatic range validation

**Backend Abstraction**
- Independent of specific quantum simulator
- Support multiple backend switching
- Consistent API interface

**Composability**
- Circuits can be copied, decomposed, reversed
- Support cascading circuit composition
- Support control extension

---

## II. Usage Steps

### 2.1 Basic Usage Workflow

#### Step 1: Import Required Modules
```python
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register
from engine.core.classical_register import ClassicalRegister
```

#### Step 2: Create Registers
```python
# Create quantum register
qreg = Register("q", 3)        # 3 quantum qubits

# Create classical register (optional)
creg = ClassicalRegister("c", 3)  # 3 classical bits
```

#### Step 3: Initialize GateSequence
```python
# Method 1: Use register objects
circuit = GateSequence(qreg, creg, name="my_circuit")

# Method 2: Directly specify number of bits
circuit = GateSequence(3)  # Create 3 qubits, automatically named

# Method 3: Mixed approach
circuit = GateSequence(qreg, creg)
```

#### Step 4: Build Circuit
```python
# Add single-qubit gates
circuit.h(qreg[0])      # H gate on bit 0
circuit.x(qreg[1])      # X gate on bit 1
circuit.rz(3.14, qreg[2])  # RZ gate on bit 2

# Add two-qubit gates
circuit.cnot(qreg[0], qreg[1])  # CNOT gate

# Add measurement
circuit.measure(qreg, creg)
```

#### Step 5: Execute and Get Results
```python
# Execute circuit
state = circuit.execute()

# Get measurement results
print(f"Measurement results: {creg.values}")

# Get circuit matrix
matrix = circuit.get_matrix()

# Draw circuit
circuit.draw(filename="my_circuit.png")
```

### 2.2 Detailed Usage Example - Bell State Preparation

```python
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register
from engine.core.classical_register import ClassicalRegister

# Step 1: Create registers
qreg = Register("q", 2)
creg = ClassicalRegister("c", 2)

# Step 2: Initialize circuit
circuit = GateSequence(qreg, creg, name="bell_state")

# Step 3: Build Bell state circuit
# |Φ+⟩ = (1/√2)(|00⟩ + |11⟩)
circuit.h(qreg[0])          # Apply H gate on bit 0
circuit.cnot(qreg[0], qreg[1])  # CNOT gate

# Step 4: Add measurement
circuit.measure(qreg, creg)

# Step 5: Execute
state = circuit.execute()

# Step 6: Check results
print(f"Quantum state: {state}")
print(f"Measurement results: {creg.values}")

# Step 7: Draw circuit
circuit.draw(title="Bell State Circuit")
```

### 2.3 Advanced Usage Scenarios

#### Scenario 1: Multi-Register Circuit

```python
# Create multiple registers
qreg1 = Register("q1", 2)
qreg2 = Register("q2", 3)
creg = ClassicalRegister("c", 5)

# Create circuit with multiple registers
circuit = GateSequence(qreg1, qreg2, creg, name="multi_register")

# Operate on each register independently
circuit.h(qreg1[0])
circuit.x(qreg2[1])
circuit.cnot(qreg1[0], qreg2[0])

# Measurement
circuit.measure(qreg1, creg[0:2])
circuit.measure(qreg2, creg[2:5])
```

#### Scenario 2: Controlled Circuit

```python
circuit = GateSequence(5, name="controlled_circuit")

# Create a basic circuit
base_circuit = GateSequence(3)
base_circuit.h(0)
base_circuit.x(1)

# Add control qubits
controlled = circuit.control(num_ctrl_qubits=2, control_sequence="11")

# Add controlled sub-circuit to main circuit
circuit.append(controlled, [0, 1, 2])
```

#### Scenario 3: Circuit Decomposition and Optimization

```python
circuit = GateSequence(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.rz(1.57, 2)

# Decompose circuit (expand to basic gates)
decomposed = circuit.decompose(times=1)

# Get inverse circuit
inverse = circuit.inverse()

# Get reversed circuit
reversed_circuit = circuit.reverse()

# Repeat circuit n times
repeated = circuit.repeat(times=3)
```

#### Scenario 4: Quantum-Classical Hybrid Algorithm

```python
qreg = Register("q", 3)
creg = ClassicalRegister("c", 3)
circuit = GateSequence(qreg, creg)

# Step 1: Create superposition state
circuit.h(qreg)  # Apply H gate to all qubits

# Step 2: Apply controlled operations (based on implicit results from previous step)
for i in range(3):
    circuit.rz(1.57, qreg[i])

# Step 3: Measurement
circuit.measure(qreg, creg)

# Execute
state = circuit.execute()

# Analyze results
measured_values = creg.values
print(f"Measurement results: {measured_values}")
```

#### Scenario 5: Custom Unitary Operations

```python
import numpy as np

circuit = GateSequence(2)

# Define custom 2-qubit gate (4×4 unitary matrix)
custom_unitary = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
], dtype=complex)

# Apply custom gate
circuit.unitary(custom_unitary, [0, 1])
```

---

## III. Key API Reference

### 3.1 Constructor

```python
GateSequence(*args, **kwargs)
```

**Parameters:**
- `num_qubits: int` - Number of quantum bits (direct specification)
- `registers: Register | ClassicalRegister` - Sequence of register objects
- `name: str` - Circuit name (optional, default 'Sequence')

**Example:**
```python
# Method 1: Number
circuit = GateSequence(3)

# Method 2: Registers
circuit = GateSequence(qreg, creg)

# Method 3: Mixed
circuit = GateSequence(qreg, creg, name="hybrid")
```

### 3.2 Single-Qubit Gates

| Method | Parameter | Description |
|------|------|------|
| `x(qubit)` | qubit | Pauli-X gate |
| `y(qubit)` | qubit | Pauli-Y gate |
| `z(qubit)` | qubit | Pauli-Z gate |
| `h(qubit)` | qubit | Hadamard gate |
| `s(qubit)` | qubit | S gate |
| `t(qubit)` | qubit | T gate |
| `rx(angle, qubit)` | angle, qubit | RX rotation gate |
| `ry(angle, qubit)` | angle, qubit | RY rotation gate |
| `rz(angle, qubit)` | angle, qubit | RZ rotation gate |

### 3.3 Two-Qubit Gates

| Method | Parameter | Description |
|------|------|------|
| `cnot(control, target)` | control, target | Controlled-NOT |
| `cz(control, target)` | control, target | Controlled-Z |
| `ch(control, target)` | control, target | Controlled-H |
| `swap(qubit1, qubit2)` | qubit1, qubit2 | SWAP |

### 3.4 Multi-Control Gates

```python
# Multiple control bits
circuit.mcx(controls, target)      # Multi-Controlled-X
circuit.mcy(controls, target)      # Multi-Controlled-Y
circuit.mcz(controls, target)      # Multi-Controlled-Z
circuit.mcrx(angle, controls, target)  # Multi-Controlled-RX
```

### 3.5 Circuit Operations

```python
# Copy
new_circuit = circuit.copy()

# Decompose
decomposed = circuit.decompose(times=1)

# Inverse
inverse = circuit.inverse()
dagger = circuit.dagger()  # Conjugate transpose

# Reverse
reversed_circuit = circuit.reverse()

# Repeat
repeated = circuit.repeat(times=3)

# Add sub-circuit
circuit.append(sub_circuit, target, control)
circuit.prepend(sub_circuit, target)

# Add control
controlled = circuit.control(num_ctrl_qubits=2)
```

### 3.6 Measurement and Execution

```python
# Measurement
circuit.measure(target_qubits, classical_bits)

# Execute
state = circuit.execute(initial_state=None)

# Get matrix
matrix = circuit.get_matrix(m=0)

# Draw
circuit.draw(filename=None, title=None, style='dark', compact=True)
```

### 3.7 Information Queries

```python
# Get number of qubits
num_qubits = circuit.get_num_qubits()

# Get backend type
backend = circuit.get_backend_type()

# Get circuit name
name = circuit.name

# Get registers
regs = circuit.registers
cl_regs = circuit.classical_registers
```

---

## IV. Indexing Methods Detailed

### 4.1 Supported Index Types

```python
circuit = GateSequence(qreg, creg)

# Single bit
circuit.x(qreg[0])           # Integer index

# Bit range
circuit.h(qreg[0:2])         # Slice

# Multiple bits
circuit.x(qreg[[0, 2, 3]])   # List index

# Tuple index
circuit.y(qreg[(0, 2)])      # Tuple

# All bits
circuit.z(qreg)              # Register object
```

### 4.2 Internal Index Conversion Mechanism

```
Local Index (Local Index)
    ↓
Look up via register object
    ↓
Get global start position
    ↓
Calculate global index (Global Index)
    ↓
Apply quantum gate
```

---

## V. Common Patterns

### Pattern 1: Simple Circuit Construction

```python
def build_simple_circuit(n_qubits):
    circuit = GateSequence(n_qubits)
    
    # Create GHZ state
    circuit.h(0)
    for i in range(1, n_qubits):
        circuit.cnot(0, i)
    
    return circuit
```

### Pattern 2: Circuit Template

```python
def grover_diffusion(circuit, qubits):
    """Grover's diffusion operator"""
    circuit.h(qubits)
    circuit.x(qubits)
    circuit.mcrz(3.14, qubits[:-1], qubits[-1])
    circuit.x(qubits)
    circuit.h(qubits)
```

### Pattern 3: Parameterized Circuit

```python
def parameterized_circuit(angles, qubits):
    circuit = GateSequence(len(qubits))
    
    for i, angle in enumerate(angles):
        circuit.rz(angle, qubits[i])
    
    for i in range(len(qubits)-1):
        circuit.cnot(qubits[i], qubits[i+1])
    
    return circuit
```

---

## VI. Error Handling Guide

### 6.1 Common Errors

**Error 1: Index Out of Range**
```python
qreg = Register("q", 3)
circuit = GateSequence(qreg)

# Wrong!
# circuit.x(qreg[5])  # IndexError

# Correct
circuit.x(qreg[2])
```

**Error 2: Register Not Added**
```python
qreg = Register("q", 3)
circuit = GateSequence(5)  # Create 5 independent qubits

# Wrong!
# circuit.x(qreg[0])  # KeyError: Register not found

# Correct
circuit.x(0)
```

**Error 3: Mismatched Measurement**
```python
qreg = Register("q", 3)
creg = ClassicalRegister("c", 2)
circuit = GateSequence(qreg, creg)

# Wrong!
# circuit.measure(qreg, creg)  # ValueError: Qubit count mismatch

# Correct
circuit.measure(qreg[0:2], creg)
```

### 6.2 Best Practices

```python
# Good approach: Type checking
def safe_apply_gate(circuit, target):
    try:
        if isinstance(target, Register):
            circuit.h(target)
        else:
            raise TypeError("Expected Register or int")
    except (KeyError, IndexError, ValueError) as e:
        print(f"Circuit operation failed: {e}")

# Avoid: Assuming index is valid
# circuit.x(random_index)  # May fail
```

---

## VII. Performance Optimization Recommendations

### 7.1 Batch Operations

```python
# Not recommended: Multiple individual calls
for i in range(10):
    circuit.h(i)

# Recommended: Apply on list
circuit.h([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### 7.2 Circuit Reuse

```python
# Create once, use multiple times
base_circuit = GateSequence(3)
base_circuit.h(0)
base_circuit.cnot(0, 1)

# Copy and extend
circuit1 = base_circuit.copy()
circuit1.x(2)

circuit2 = base_circuit.copy()
circuit2.z(2)
```

### 7.3 Matrix Caching

```python
# First call: Compute and cache
matrix1 = circuit.get_matrix()

# Subsequent calls: If circuit unchanged, use cache
matrix2 = circuit.get_matrix()  # Fast
```

---

## VIII. Integration Example - Complete Quantum Algorithm

```python
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register
from engine.core.classical_register import ClassicalRegister

# Deutsch Algorithm Implementation
def deutsch_algorithm(oracle_type='balanced'):
    """
    Implement Deutsch Algorithm
    
    Parameters:
    oracle_type: 'constant' or 'balanced'
    """
    # Create registers
    qreg = Register("q", 2)
    creg = ClassicalRegister("c", 2)
    
    # Create circuit
    circuit = GateSequence(qreg, creg, name="deutsch")
    
    # Initialize: |01⟩
    circuit.x(qreg[1])
    
    # Apply Hadamard transform
    circuit.h(qreg[0])
    circuit.h(qreg[1])
    
    # Apply Oracle
    if oracle_type == 'constant':
        # Constant oracle: Do nothing or global phase
        # (Implementation details omitted)
        pass
    else:  # balanced
        # Balanced oracle
        circuit.cnot(qreg[0], qreg[1])
    
    # Final Hadamard transform
    circuit.h(qreg[0])
    
    # Measure first qubit
    circuit.measure(qreg[0], creg[0])
    
    return circuit

# Usage
circuit = deutsch_algorithm('balanced')
state = circuit.execute()
print(f"Result: {circuit.classical_registers[0].values}")
```

---

## IX. Summary Checklist

When using GateSequence, please ensure:

- [ ] Quantum and classical registers created correctly
- [ ] GateSequence initialized with all necessary registers
- [ ] Correct indexing method used (integer, register object, slice, etc.)
- [ ] Single-qubit and two-qubit gate parameters are correct
- [ ] Classical register added before measurement
- [ ] Number of measured qubits matches number of classical bits
- [ ] Circuit construction completed before execution
- [ ] Possible errors and exceptions handled
- [ ] Batch operations used for performance-sensitive scenarios
- [ ] Difference between local and global indices understood correctly

