---
name: classicalregister
description: |
  Classical Register module for managing quantum circuit measurement results and classical bit storage.
  Provides flexible indexing and state management for quantum-classical hybrid algorithms.
keywords:
  - classical register
  - measurement results
  - quantum-classical interface
  - bit storage
  - state management
---

# Classical Register Skills Guide

## I. Algorithm Introduction

### 1.1 Basic Concepts

**Classical Register** is a data structure in quantum computing used to store measurement results of quantum qubits. It serves as a bridge between quantum and classical computing, storing the measurement results (0 or 1) of quantum states in classical bits.

### 1.2 Working Principle

#### Initialization Process
```
ClassicalRegister Initialization
  ↓
Allocate space for n_bits classical bits
  ↓
Initialize all bit values to -1 (unmeasured state)
  ↓
Generate unique UUID identifier
  ↓
Complete initialization
```

#### Indexing Mechanism
Classical Register supports multiple indexing methods to enable flexible bit access:

| Index Type | Example | Description |
|---------|------|------|
| Single Index | `creg[0]` | Access bit 0 |
| Slice | `creg[0:3]` | Access bits 0-2 |
| Tuple | `creg[(0, 2)]` | Access bits 0 and 2 |
| List | `creg[[0, 1, 2]]` | Access multiple specified bits |
| Negative Index | `creg[-1]` | Access bit from the end |

#### Range Validation
All index accesses are subject to range validation:
- Valid range: `[-n_qubits, n_qubits)`
- Out-of-range accesses raise `IndexError`
- Ensures access safety and data integrity

### 1.3 Core Features

**Flexible Index Access**
- Supports simultaneous access to single bits, ranges, and multiple bits
- Automatically converts different index formats to unified internal representation

**State Management**
- Initial state: all bit values are -1 (indicating unmeasured)
- After measurement: bit values are 0 or 1
- Supports reading and updating bit states

**Unique Identification**
- Each Classical Register instance has a unique UUID
- Facilitates tracking and managing multiple registers

**Type Safety**
- Strict type checking (int, slice, tuple, list)
- Unsupported index types raise `TypeError`

---

## II. Usage Steps

### 2.1 Basic Usage Workflow

#### Step 1: Import Module
```python
from engine.core.classical_register import ClassicalRegister
```

#### Step 2: Create Classical Register
```python
# Create a classical register to store measurement results of 3 quantum bits
creg = ClassicalRegister("measurement_result", 3)

# Parameter explanation:
# - "measurement_result" is the name of the register
# - 3 is the number of classical bits
```

#### Step 3: Check Register Attributes
```python
# Get register name
name = creg.name              # "measurement_result"

# Get number of bits
num_bits = len(creg)          # 3 or creg.n_qubits

# Get unique identifier
unique_id = creg.id           # UUID hexadecimal string

# Get bit values
values = creg.values          # [-1, -1, -1]

# Print register information
print(creg)
```

#### Step 4: Access Bits
```python
# Method 1: Single bit access
first_bit = creg[0]           # Returns [(creg, [0])]

# Method 2: Range access
first_two = creg[0:2]         # Returns [(creg, [0, 1])]

# Method 3: Multiple bit access
specific_bits = creg[[0, 2]]  # Returns [(creg, [0, 2])]

# Method 4: Tuple indexing
tuple_bits = creg[(1, 2)]     # Returns [(creg, [1, 2])]
```

### 2.2 Complete Workflow Example

#### Scenario: Quantum Circuit Measurement Result Storage

```python
from engine.core.classical_register import ClassicalRegister

# Step 1: Create register
creg = ClassicalRegister("quantum_measurement", 5)
print(f"Created register: {creg.name}, bits: {len(creg)}")

# Step 2: Check initial state
print(f"Initial values: {creg.values}")  # [-1, -1, -1, -1, -1]
print(f"Register ID: {creg.id}")

# Step 3: Simulate storing measurement results
# (In actual applications, these values are obtained from quantum measurement)
creg.values[0] = 1  # First measurement result is 1
creg.values[1] = 0  # Second measurement result is 0
creg.values[2] = 1  # Third measurement result is 1

print(f"After measurement: {creg.values}")  # [1, 0, 1, -1, -1]

# Step 4: Access specific bits
measured_bits = creg[[0, 1, 2]]
print(f"Measured bits: {measured_bits}")

# Step 5: Access range
range_bits = creg[0:3]
print(f"First three bits: {range_bits}")

# Step 6: Count measured bits
measured_count = sum(1 for v in creg.values if v != -1)
print(f"Number of measured bits: {measured_count}")
```

### 2.3 Advanced Usage Scenarios

#### Scenario 1: Managing Multiple Algorithm Results

```python
# Create different classical registers for different algorithms
creg_algorithm1 = ClassicalRegister("algo1_result", 4)
creg_algorithm2 = ClassicalRegister("algo2_result", 4)
creg_control = ClassicalRegister("control_bits", 2)

# Distinguish different registers by ID
registers = {
    creg_algorithm1.id: creg_algorithm1,
    creg_algorithm2.id: creg_algorithm2,
    creg_control.id: creg_control,
}

print(f"Managing {len(registers)} registers")
```

#### Scenario 2: Conditional Quantum Operations

```python
# Conditional operations based on measurement results
creg = ClassicalRegister("conditional", 3)

# Simulate measurement
creg.values = [1, 0, 1]

# Get measurement result at specific position
first_result = creg[0]

# Execute different operations based on result
if creg.values[0] == 1:
    print("Execute conditional operation A (because first bit is 1)")
else:
    print("Execute conditional operation B")
```

#### Scenario 3: Batch Index Operations

```python
creg = ClassicalRegister("batch_operation", 8)
creg.values = [0, 1, 0, 1, 1, 0, 1, 0]

# Access using multiple index methods
even_indices = creg[0:8:2]      # Even indices
odd_indices = creg[1:8:2]       # Odd indices
first_half = creg[0:4]          # First half
specific = creg[[0, 3, 6]]      # Specific positions

print(f"Even indices result: {even_indices}")
print(f"Odd indices result: {odd_indices}")
```

---

## III. Key API Reference

### 3.1 Constructor
```python
ClassicalRegister(name: str, n_bits: int)
```
- **name**: Name identifier of classical register
- **n_bits**: Number of classical bits to allocate
- **Returns**: ClassicalRegister instance

### 3.2 Indexing Method - `__getitem__(index)`
```python
creg[index] -> List[Tuple[ClassicalRegister, List[int]]]
```
- **index**: int | slice | tuple | list
- **Returns**: List of tuples containing register reference and index list
- **Exceptions**: 
  - `TypeError`: Unsupported index type
  - `IndexError`: Index out of range

### 3.3 Length Method - `__len__()`
```python
len(creg) -> int
```
- **Returns**: Number of classical bits

### 3.4 String Representation - `__repr__()`
```python
repr(creg) -> str
```
- **Returns**: String representation of register (includes name, n_bits and values)

---

## IV. Error Handling Guide

### 4.1 Index Range Error

```python
creg = ClassicalRegister("test", 3)

# Error: index out of range
try:
    result = creg[5]  # Only 0-2 are valid
except IndexError as e:
    print(f"Error: {e}")
    # Output: Index [5] out of range for 'test'
```

### 4.2 Type Error

```python
creg = ClassicalRegister("test", 3)

# Error: unsupported index type
try:
    result = creg[1.5]  # float not supported
except TypeError as e:
    print(f"Error: {e}")
    # Output: Indices must be int, slice, or list, not float
```

### 4.3 Best Practices

```python
def safe_access(creg: ClassicalRegister, index):
    """Safely access register bits"""
    try:
        if not isinstance(index, (int, slice, tuple, list)):
            raise TypeError(f"Unsupported index type: {type(index)}")
        
        result = creg[index]
        return result
    except (IndexError, TypeError) as e:
        print(f"Access failed: {e}")
        return None
```

---

## V. Integration Guide

### 5.1 Integration with Quantum Circuits

```python
from engine.core.classical_register import ClassicalRegister
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

# Create quantum and classical registers
qreg = Register("qreg", 3)
creg = ClassicalRegister("creg", 3)

# Use in quantum circuits
# Execute quantum gate operations
# ...
# creg.values = quantum measurement results
```

### 5.2 Integration with Quantum Algorithms

```python
class QuantumAlgorithm:
    def __init__(self, name: str, n_qubits: int):
        self.name = name
        self.n_qubits = n_qubits
        self.creg = ClassicalRegister(f"{name}_result", n_qubits)
        self.measurements = []
    
    def execute(self):
        """Execute algorithm and store measurement results"""
        # Execute quantum operations
        # ...
        # Store measurement results
        self.measurements.append(self.creg.values.copy())
        return self.creg
```

---

## VI. Performance Considerations

### 6.1 Memory Efficiency
- Pre-allocate all bit space during initialization: `self.values = [-1] * n_bits`
- Avoid dynamic expansion, fixed-size registers

### 6.2 Access Speed
- Simple bit access: O(1)
- Range access: O(n), where n is the number of bits accessed
- Validation operations: O(k), where k is the number of indices

### 6.3 Best Practices
```python
# Good: Batch access to multiple bits at once
bits = creg[[0, 1, 2, 3]]  # One operation

# Avoid: Repeated single accesses
# for i in range(4):
#     bit = creg[i]  # Multiple operations, less efficient
```

---

## VII. Common Patterns

### Pattern 1: Register Initialization and Verification
```python
def create_and_verify_register(name: str, n_bits: int):
    creg = ClassicalRegister(name, n_bits)
    assert len(creg) == n_bits
    assert creg.name == name
    assert all(v == -1 for v in creg.values)
    return creg
```

### Pattern 2: Measurement Result Aggregation
```python
def aggregate_measurements(registers: List[ClassicalRegister]):
    all_results = {}
    for creg in registers:
        all_results[creg.id] = creg.values
    return all_results
```

### Pattern 3: Bit State Checking
```python
def check_register_state(creg: ClassicalRegister):
    total_bits = len(creg)
    measured_bits = sum(1 for v in creg.values if v != -1)
    unmeasured_bits = sum(1 for v in creg.values if v == -1)
    
    return {
        "total": total_bits,
        "measured": measured_bits,
        "unmeasured": unmeasured_bits,
        "completion": measured_bits / total_bits
    }
```

---

## VIII. Troubleshooting

### Problem 1: IndexError - Index Out of Range
**Symptoms**: `IndexError: Index [n] out of range for 'register_name'`
**Cause**: Accessing an index that exceeds the number of bits in the register
**Solution**:
```python
creg = ClassicalRegister("creg", 3)  # Only 0, 1, 2 are valid
# Error
# result = creg[3]  # Wrong!

# Correct
if index < len(creg):
    result = creg[index]
```

### Problem 2: TypeError - Unsupported Index Type
**Symptoms**: `TypeError: Indices must be int, slice, or list, not <type>`
**Cause**: Using an unsupported index type
**Solution**:
```python
# Supported types
creg[0]              # int
creg[0:2]            # slice
creg[[0, 1]]         # list
creg[(0, 1)]         # tuple

# Not supported
# creg[0.5]  # float
# creg["0"]  # string
```

---

## IX. Checklist Summary

When using Classical Register, ensure:

- [ ] Correctly imported `ClassicalRegister` class
- [ ] Specified valid name and number of bits for register
- [ ] Used supported index types (int, slice, list, tuple)
- [ ] Validated index range before accessing
- [ ] Properly handled initial state (-1 means unmeasured)
- [ ] Used batch access instead of single access when necessary
- [ ] Used different names and unique IDs for registers with different purposes
- [ ] Implemented error handling mechanisms

