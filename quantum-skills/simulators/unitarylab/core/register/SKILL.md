---
name: register
description: |
  Quantum Register module for managing groups of quantum qubits with flexible indexing.
  Provides high-level abstractions for accessing and manipulating qubits through 
  Python-style indexing patterns (integers, slices, lists, tuples).
keywords:
  - quantum register
  - qubit group
  - quantum indexing
  - qubit management
  - register abstraction
---

# Register Skills Guide

## I. Algorithm Introduction

### 1.1 Basic Concepts

**Register (Quantum Register)** is an abstraction in quantum computing used to organize and manage a group of quantum qubits. It provides a flexible indexing mechanism that allows users to access and manipulate quantum qubits using Python-style indexing, similar to processing array elements.

### 1.2 Design Principles

#### 1.2.1 Register Hierarchy

```
├─ Physical Quantum Hardware
│  └─ Physical Qubits
│
├─ Register Abstraction Layer
│  ├─ Register Object: Logical Grouping
│  │  ├─ Bit 0
│  │  ├─ Bit 1
│  │  ├─ Bit 2
│  │  └─ ...
│  │
│  └─ Index Mapping
│     ├─ Local Index
│     └─ Global Index
│
└─ Application Layer
   ├─ Quantum Circuits (GateSequence)
   ├─ Quantum Algorithms
   └─ Hybrid Algorithms
```

#### 1.2.2 Index System

Register supports multiple indexing methods, converting them to unified internal representation:

| Index Type | Symbol | Return Value | Description |
|---------|------|--------|------|
| Integer | `r[0]` | `[(r, [0])]` | Single bit |
| Slice | `r[0:2]` | `[(r, [0, 1])]` | Consecutive bits |
| List | `r[[0, 2]]` | `[(r, [0, 2])]` | Batch selection |
| Tuple | `r[(0, 2)]` | `[(r, [0, 2])]` | Non-consecutive selection |
| Negative Index | `r[-1]` | `[(r, [n-1])]` | Reverse access |

#### 1.2.3 Return Value Structure

All `__getitem__` operations return a unified format:

```python
[(Register object, [Index list])]
```

Benefits of this design:

1. **Unified Interface**: Regardless of indexing method, return type is consistent
2. **Upward Propagation**: Register information can be passed to GateSequence
3. **Flexibility**: Supports composite operations and chaining
4. **Extensibility**: Easy global index conversion in GateSequence

### 1.3 Uniqueness and Hashing

#### 1.3.1 UUID Identifier

Each Register instance receives a unique UUID:

```python
import uuid

r1 = Register("q", 3)
r2 = Register("q", 3)

print(r1.id)  # e.g.: 'a1b2c3d4e5f6...'
print(r2.id)  # e.g.: 'x9y8z7w6v5u4...'
print(r1.id == r2.id)  # False, even with same name and size
```

#### 1.3.2 Hashing and Dictionary Usage

Register supports use as dictionary keys and set elements:

```python
# Use as dictionary key
mapping = {}
r = Register("q", 3)
mapping[r] = "value"

# Use in sets
registers_set = {r1, r2, r3}

# Use in Python dictionaries
register_info = {
    r: {"type": "quantum", "size": 3}
}
```

### 1.4 Range Validation Mechanism

Register automatically checks index range validity:

```
Index Submission
  ↓
Type Check (int, slice, tuple, list)
  ↓
Range Validation
  ├─ max(indices) < n_qubits
  ├─ min(indices) >= -n_qubits
  └─ Return if all valid, otherwise raise exception
```

**Negative Index Support**:
- Python-style negative indices are correctly handled
- `-1` represents the last bit
- `-n` represents the first bit
- Range checks validate both positive and negative indices

---

## II. Usage Steps

### 2.1 Basic Usage Workflow

#### Step 1: Import Module
```python
from engine.core.register import Register
```

#### Step 2: Create Register
```python
# Create a register with 3 quantum qubits
qreg = Register("q", 3)

# Create multiple different registers
qreg1 = Register("q1", 2)
qreg2 = Register("q2", 4)
```

#### Step 3: Access Bits
```python
# Single bit
bit0 = qreg[0]
bit_last = qreg[-1]

# Bit range
bits_0_1 = qreg[0:2]
bits_all = qreg[:]

# Multiple specific bits
bits_select = qreg[[0, 2]]

# Using tuple
bits_tuple = qreg[(0, 2, 1)]
```

#### Step 4: Query Register Information
```python
# Get number of qubits
num_qubits = len(qreg)
num_qubits = qreg.n_qubits

# Get name
name = qreg.name

# Get unique ID
unique_id = qreg.id

# String representation
print(qreg)  # Register(name='q', n_qubits=3)
```

### 2.2 Detailed Usage Examples

#### Example 1: Basic Register Creation and Access

```python
from engine.core.register import Register

# Create a 5-qubit register
qreg = Register("quantum", 5)

# Information query
print(f"Register name: {qreg.name}")
print(f"Total bits: {len(qreg)}")
print(f"Register ID: {qreg.id}")

# Access single bits
print(qreg[0])  # First bit
print(qreg[4])  # Last bit
print(qreg[-1])  # Penultimate bit
print(qreg[-5])  # Fifth from end (first)

# Access ranges
print(qreg[0:3])  # First three bits
print(qreg[2:])   # All bits starting from second
print(qreg[::2])  # Every other bit
```

#### Example 2: Flexible Indexing Methods

```python
from engine.core.register import Register

qreg = Register("q", 8)

# Method 1: Single index
result1 = qreg[3]
# Returns: [(qreg, [3])]

# Method 2: Slice (Python standard)
result2 = qreg[2:5]
# Returns: [(qreg, [2, 3, 4])]

# Method 3: Stride slice
result3 = qreg[::2]
# Returns: [(qreg, [0, 2, 4, 6])]

# Method 4: List selection
result4 = qreg[[1, 3, 5]]
# Returns: [(qreg, [1, 3, 5])]

# Method 5: Tuple selection
result5 = qreg[(0, 2, 7)]
# Returns: [(qreg, [0, 2, 7])]

# Method 6: Reverse index
result6 = qreg[-1]
# Returns: [(qreg, [7])]

# Method 7: Reverse range
result7 = qreg[-3:]
# Returns: [(qreg, [5, 6, 7])]

print(f"Single: {result1}")
print(f"Slice: {result2}")
print(f"Stride: {result3}")
print(f"List: {result4}")
print(f"Tuple: {result5}")
print(f"Negative: {result6}")
print(f"Negative range: {result7}")
```

#### Example 3: GateSequence Integration

```python
from engine.core.register import Register
from engine.core.gate_sequence import GateSequence

# Create register
qreg = Register("q", 3)

# Create quantum circuit
circuit = GateSequence(qreg, name="demo")

# Use register indices to apply gates
circuit.h(qreg[0])           # H gate on bit 0
circuit.x(qreg[1:3])         # X gate on bits 1, 2
circuit.cnot(qreg[0], qreg[1])  # CNOT: control=0, target=1

# Execute
result = circuit.execute()
print(f"Execution complete")
```

#### Example 4: Multiple Register Management

```python
from engine.core.register import Register
from engine.core.gate_sequence import GateSequence

# Create multiple registers
qreg_data = Register("data", 3)
qreg_ancilla = Register("ancilla", 2)

# Create circuit
circuit = GateSequence(qreg_data, qreg_ancilla)

# Operate on data register
circuit.h(qreg_data)

# Operate on ancilla register
circuit.x(qreg_ancilla[0])

# Cross-register interaction
circuit.cnot(qreg_data[0], qreg_ancilla[1])

# Execute
result = circuit.execute()
print("Multi-register circuit complete")
```

#### Example 5: Register Comparison and Equality

```python
from engine.core.register import Register

# Create registers
qreg1 = Register("q", 3)
qreg2 = Register("q", 3)
qreg3 = qreg1  # Reference same object

# Compare
print(f"qreg1 == qreg2: {qreg1 == qreg2}")  # False, different objects
print(f"qreg1 == qreg3: {qreg1 == qreg3}")  # True, same object
print(f"qreg1.id == qreg2.id: {qreg1.id == qreg2.id}")  # False

# Use in set
regs_set = {qreg1, qreg2, qreg3}
print(f"Set size: {len(regs_set)}")  # 2 (qreg1 and qreg3 are same)
```

### 2.3 Advanced Usage Scenarios

#### Scenario 1: Dynamic Register Creation

```python
def create_registers_for_algorithm(n_data, n_ancilla):
    """Create required registers for specific algorithm"""
    data_reg = Register("data", n_data)
    ancilla_reg = Register("ancilla", n_ancilla)
    return data_reg, ancilla_reg

# Usage
n = 4  # Number of data qubits
m = 2  # Number of ancilla qubits

data_qreg, ancilla_qreg = create_registers_for_algorithm(n, m)

print(f"Data register: {data_qreg}")
print(f"Ancilla register: {ancilla_qreg}")
```

#### Scenario 2: Register Partition Operations

```python
from engine.core.register import Register

qreg = Register("q", 8)

# Partition register into sections
upper_half = qreg[0:4]    # First half
lower_half = qreg[4:]     # Second half

# Even/odd split
odd_qubits = qreg[::2]    # Even positions (0, 2, 4, 6)
even_qubits = qreg[1::2]  # Odd positions (1, 3, 5, 7)

# Specific bits
important_bits = qreg[[0, 3, 7]]  # Key bits

print(f"First half: {upper_half}")
print(f"Second half: {lower_half}")
print(f"Odd bits: {odd_qubits}")
print(f"Even bits: {even_qubits}")
print(f"Key bits: {important_bits}")
```

#### Scenario 3: Register Mapping and Tracking

```python
from engine.core.register import Register

# Create multiple registers
registers = [
    Register("input", 3),
    Register("work", 4),
    Register("output", 2)
]

# Use dictionary for mapping
reg_map = {
    reg.id: {"name": reg.name, "size": len(reg)}
    for reg in registers
}

# Query information
for reg in registers:
    info = reg_map[reg.id]
    print(f"Register {info['name']}: {info['size']} qubits")

# Reverse query
by_name = {
    reg.name: reg
    for reg in registers
}

input_reg = by_name["input"]
print(f"Input register: {input_reg}")
```

#### Scenario 4: Register Indexing Helper Function

```python
from engine.core.register import Register

def get_qubit_indices(register, index_spec):
    """
    Flexible qubit index retrieval function
    
    index_spec can be:
    - int: single index
    - str: 'all', 'even', 'odd', 'first', 'last'
    - slice: slice object
    - list/tuple: index list
    """
    if isinstance(index_spec, str):
        n = len(register)
        if index_spec == 'all':
            return register[:]
        elif index_spec == 'even':
            return register[::2]
        elif index_spec == 'odd':
            return register[1::2]
        elif index_spec == 'first':
            return register[0]
        elif index_spec == 'last':
            return register[-1]
        else:
            raise ValueError(f"Unknown spec: {index_spec}")
    else:
        return register[index_spec]

# Usage
qreg = Register("q", 6)

print(get_qubit_indices(qreg, 'all'))    # All bits
print(get_qubit_indices(qreg, 'even'))   # Even positions
print(get_qubit_indices(qreg, 'odd'))    # Odd positions
print(get_qubit_indices(qreg, 'first'))  # First one
print(get_qubit_indices(qreg, 'last'))   # Last one
print(get_qubit_indices(qreg, [0, 3]))   # Specific positions
```

---

## III. Key API Reference

### 3.1 Constructor

```python
Register(name: str, n_qubits: int)
```

**Parameters:**
- `name: str` - Register identifier (used for debugging and display)
- `n_qubits: int` - Number of quantum bits in the register

**Properties:**
- `name: str` - Register name
- `n_qubits: int` - Number of bits
- `id: str` - Unique UUID identifier

**Example:**
```python
qreg = Register("quantum_register", 4)
```

### 3.2 Indexing Method - `__getitem__`

```python
register[index] -> List[Tuple[Register, List[int]]]
```

**Supported Index Types:**

| Type | Example | Return Value |
|------|------|--------|
| `int` | `r[0]` | `[(r, [0])]` |
| `slice` | `r[0:2]` | `[(r, [0, 1])]` |
| `list` | `r[[0, 2]]` | `[(r, [0, 2])]` |
| `tuple` | `r[(0, 2)]` | `[(r, [0, 2])]` |

**Negative Index Support:**
```python
r = Register("q", 5)
r[-1]      # Last bit → [(r, [4])]
r[-2:]     # Last two → [(r, [3, 4])]
r[:-1]     # All except last → [(r, [0, 1, 2, 3])]
```

**Exceptions:**
- `TypeError` - Unsupported index type
- `IndexError` - Index out of range

### 3.3 Length Method - `__len__`

```python
len(register) -> int
```

Returns the number of qubits in the register.

**Example:**
```python
qreg = Register("q", 5)
print(len(qreg))  # 5
```

### 3.4 Equality Comparison - `__eq__`

```python
register1 == register2 -> bool
```

Two Register objects are equal if and only if they have the same unique ID (i.e., they are the same object).

**Example:**
```python
r1 = Register("q", 3)
r2 = Register("q", 3)
r3 = r1

print(r1 == r2)  # False
print(r1 == r3)  # True (same object)
```

### 3.5 Hash Method - `__hash__`

```python
hash(register) -> int
```

Returns a hash value based on the name, number of bits, and ID, allowing Register to be used as a dictionary key or set element.

**Example:**
```python
qreg = Register("q", 3)
reg_dict = {qreg: "value"}
reg_set = {qreg}
```

### 3.6 String Representation - `__repr__`

```python
repr(register) -> str
```

Returns a readable string representation.

**Example:**
```python
qreg = Register("quantum", 3)
print(qreg)
# Output: Register(name='quantum', n_qubits=3)
```

---

## IV. Index System Explained

### 4.1 Python-Style Indexing

Register fully supports Python list-style indexing:

```python
qreg = Register("q", 10)

# Basic indexing
qreg[0]        # 0-th qubit
qreg[9]        # 9-th qubit
qreg[-1]       # Last qubit (9th)
qreg[-10]      # First qubit

# Slicing
qreg[2:5]      # Qubits 2, 3, 4
qreg[:3]       # First 3 qubits (0, 1, 2)
qreg[7:]       # Starting from 7-th qubit
qreg[:]        # All qubits

# Striding
qreg[::2]      # Every other qubit (0, 2, 4, 6, 8)
qreg[1::2]     # Every other starting from 1 (1, 3, 5, 7, 9)
qreg[::-1]     # Reverse (9, 8, 7, ..., 0)
```

### 4.2 Return Value Uniformity

All indexing operations return the same format, facilitating integration with GateSequence:

```python
qreg = Register("q", 5)

# Although indexing methods differ, return format is the same
print(qreg[0])        # [(qreg, [0])]
print(qreg[0:1])      # [(qreg, [0])]
print(qreg[[0]])      # [(qreg, [0])]
print(qreg[(0,)])     # [(qreg, [0])]
```

### 4.3 Index Range Checking

```python
qreg = Register("q", 3)

# Valid indices: 0, 1, 2, -1, -2, -3
qreg[0]     # ✓
qreg[2]     # ✓
qreg[-1]    # ✓
qreg[-3]    # ✓

# Invalid indices
# qreg[3]    # ✗ IndexError
# qreg[-4]   # ✗ IndexError
```

---

## V. Common Patterns

### Pattern 1: Quantum-Classical Separation

```python
def setup_quantum_classical():
    """Set up quantum and classical registers"""
    from engine.core.register import Register
    from engine.core.classical_register import ClassicalRegister
    
    qreg = Register("q", 3)
    creg = ClassicalRegister("c", 3)
    
    return qreg, creg
```

### Pattern 2: Multi-Part Register Management

```python
def create_modular_registers(n_encoding, n_work, n_output):
    """Create registers for modular algorithm"""
    return {
        'input': Register("input", n_encoding),
        'work': Register("work", n_work),
        'output': Register("output", n_output)
    }

# Usage
regs = create_modular_registers(4, 2, 3)
encoding_reg = regs['input']
work_reg = regs['work']
output_reg = regs['output']
```

### Pattern 3: Register Validation

```python
def validate_register(reg, expected_size=None):
    """Validate register validity"""
    if not isinstance(reg, Register):
        raise TypeError(f"Expected Register, got {type(reg)}")
    
    if expected_size and len(reg) != expected_size:
        raise ValueError(f"Expected size {expected_size}, got {len(reg)}")
    
    return True

# Usage
qreg = Register("q", 3)
validate_register(qreg, expected_size=3)  # ✓
```

### Pattern 4: Dynamic Partitioning

```python
def partition_register(register, partition_size):
    """Partition register into chunks of specified size"""
    n = len(register)
    partitions = []
    
    for i in range(0, n, partition_size):
        end = min(i + partition_size, n)
        partition = register[i:end]
        partitions.append(partition)
    
    return partitions

# Usage
qreg = Register("q", 8)
parts = partition_register(qreg, 2)
# parts = [qreg[0:2], qreg[2:4], qreg[4:6], qreg[6:8]]
```

---

## VI. Error Handling Guide

### 6.1 Common Errors

**Error 1: Index Out of Range**
```python
qreg = Register("q", 3)

# Wrong!
# qreg[3]  # IndexError: Register index [3] out of range

# Correct
qreg[2]  # Last valid index
qreg[-1]  # Or use negative index
```

**Error 2: Invalid Index Type**
```python
qreg = Register("q", 3)

# Wrong!
# qreg[1.5]  # TypeError
# qreg["0"]  # TypeError

# Correct
qreg[1]      # int
qreg[0:2]    # slice
qreg[[0, 1]]  # list
```

**Error 3: Confusing Local and Global Indices**
```python
# Register always uses local indices
qreg = Register("q", 3)
result = qreg[0]  # This is local index 0

# GateSequence converts it to global index
circuit = GateSequence(qreg)
circuit.h(qreg[0])  # GateSequence handles conversion internally
```

### 6.2 Best Practices

```python
def safe_index_register(register, index):
    """Safe register indexing"""
    try:
        # Validate type
        if not isinstance(register, Register):
            raise TypeError(f"Expected Register, got {type(register)}")
        
        # Validate index
        if isinstance(index, int):
            n = len(register)
            if not (-n <= index < n):
                raise IndexError(f"Index {index} out of range [0, {n-1}]")
        
        # Perform indexing
        result = register[index]
        return result
        
    except (TypeError, IndexError) as e:
        print(f"Indexing failed: {e}")
        return None

# Usage
qreg = Register("q", 5)
safe_index_register(qreg, 2)   # ✓
safe_index_register(qreg, 10)  # ✗ With error handling
```

---

## VII. Performance Considerations

### 7.1 Memory Efficiency

```python
import sys

qreg = Register("q", 1000)

# Register object itself occupies small space
size = sys.getsizeof(qreg)
print(f"Register object size: {size} bytes")  # ~60-100 bytes

# Additional space for UUID string
id_size = sys.getsizeof(qreg.id)
print(f"ID string size: {id_size} bytes")
```

### 7.2 Indexing Operation Performance

```python
from engine.core.register import Register
import time

qreg = Register("q", 1000)

# Indexing operation is O(n) (n = number of indices)
start = time.time()
for _ in range(10000):
    result = qreg[500]
end = time.time()

print(f"Single index 10000 times: {end - start:.6f} seconds")

# Range indexing generates index list, also O(n)
start = time.time()
for _ in range(10000):
    result = qreg[0:100]
end = time.time()

print(f"Range index 10000 times: {end - start:.6f} seconds")
```

---

## VIII. Complete Workflow Example

```python
from engine.core.register import Register
from engine.core.gate_sequence import GateSequence
from engine.core.classical_register import ClassicalRegister

def complete_register_workflow():
    """Complete register usage workflow"""
    
    # Step 1: Create registers
    print("Step 1: Create registers")
    qreg = Register("quantum", 4)
    creg = ClassicalRegister("classical", 4)
    
    # Step 2: Check register information
    print(f"  Register: {qreg}")
    print(f"  Number of bits: {len(qreg)}")
    
    # Step 3: Create circuit
    print("Step 2: Create quantum circuit")
    circuit = GateSequence(qreg, creg)
    
    # Step 4: Apply gates using different indexing methods
    print("Step 3: Apply quantum gates")
    circuit.h(qreg[0])          # Single bit
    circuit.x(qreg[1:3])        # Range
    circuit.z(qreg[[3]])        # List
    
    # Step 5: Add interactions
    print("Step 4: Add interactions")
    circuit.cnot(qreg[0], qreg[1])
    circuit.cnot(qreg[2], qreg[3])
    
    # Step 6: Measure
    print("Step 5: Measurement")
    circuit.measure(qreg, creg)
    
    # Step 7: Execute
    print("Step 6: Execute circuit")
    result = circuit.execute()
    
    print(f"  Execution complete")
    print(f"  Measurement results: {creg.values}")
    
    return circuit, result

# Execute
circuit, result = complete_register_workflow()
```

---

## IX. Summary Checklist

When using Register, ensure:

- [ ] Register class has been correctly imported
- [ ] Registers are assigned meaningful names
- [ ] Register size correctly reflects the required number of qubits
- [ ] Understand the return value format `[(register, [indices])]`
- [ ] Use correct indexing methods (int, slice, list, tuple)
- [ ] Note support for negative indices
- [ ] Access indices within range (no out-of-bounds)
- [ ] Use Register as dictionary key or set element when necessary
- [ ] Understand that registers are compared by unique UUID
- [ ] Pass register indices correctly when integrating with GateSequence

