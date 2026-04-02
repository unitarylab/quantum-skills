---
name: unitarylab
description: |
  Comprehensive quantum computing simulator framework integrating quantum state representation, 
  register management, gate operations, state initialization, and circuit visualization.
  Provides complete toolkit for building and simulating quantum circuits with PyTorch backend support.
version: 2.0
requirements:
  - PyTorch for tensor operations and GPU acceleration
  - Matplotlib for circuit visualization
  - Numpy for numerical operations
  - Python 3.7+ for type hints and modern syntax
tags:
  - quantum simulator
  - quantum circuits
  - gate sequences
  - state preparation
  - quantum visualization
  - register management
---

# UnitaryLab 量子计算框架完全指南

## I. System Overview

### 1.1 System Architecture Overview

UnitaryLab is a comprehensive quantum computing simulation framework composed of six interconnected core modules. The table below summarizes the relationships and responsibilities of each component:

| Module | Purpose | Key Class | Role in Framework |
|--------|---------|-----------|------------------|
| **GateSequence** | Build and execute quantum circuits | `GateSequence` | Central orchestrator for circuit operations |
| **Register** | Manage quantum qubits | `Register` | Abstract qubit grouping and indexing |
| **ClassicalRegister** | Store measurement results | `ClassicalRegister` | Quantum-classical interface bridge |
| **State** | Represent and manipulate quantum states | `State` | State vector analysis and operations |
| **Initialization** | Prepare arbitrary quantum states | `StatePreparation` | Automated circuit synthesis for state preparation |
| **CircuitDrawer** | Visualize quantum circuits | `CircuitDrawer` | Visual representation and export |

**Data Flow and Dependencies:**

```
User Algorithm
    ↓
Register ←──────── GateSequence ────────→ ClassicalRegister
    ↓                   ↓                        ↓
   [Qubit          [Execute Circuit]      [Store Results]
    Setup]              ↓
                   State / Initialization
                        ↓
                   CircuitDrawer
                        ↓
                 [Visualization Output]
```

Each module provides specific functionality while maintaining a clean separation of concerns. The GateSequence acts as the central hub, coordinating interactions between quantum and classical components, while CircuitDrawer provides visualization capabilities orthogonal to the computational core.

### 1.3 Typical Workflow

```
Step 1: Design quantum algorithm
        ↓
Step 2: Create Register(s) for qubits
        ↓
Step 3: Initialize GateSequence
        ↓
Step 4: Build circuit or prepare state
   
   Option A: Manual Circuit        Option B: Automatic Initialization
   ├─ circuit.h(qreg[0])    or    ├─ target_state = [1/√2, 1/√2]
   ├─ circuit.cx(qreg[0], qreg[1]) └─ circuit.init(target_state, qreg)
   └─ ... more gates
        ↓
Step 5: Execute on simulator/hardware
        ↓
Step 6: Store results in ClassicalRegister
        ↓
Step 7: Analyze and visualize
        ↓
Step 8: Iterate and optimize
```

---

## II. Core Modules in Detail

### 2.1 Register (Quantum Register)

#### Overview
- **Purpose**: Organize and manage groups of quantum qubits
- **Features**: Flexible Python-style indexing, range validation, UUID-based unique identification

#### Core Concepts

**What is a Register?**
A quantum register is a logical grouping of quantum qubits. Similar to variable names in classical computing, registers provide meaningful names and indexing mechanisms for quantum qubits.

**Indexing System**:

```python
from core.register import Register

qreg = Register("q", 3)  # Create 3-qubit register

# Various indexing methods
qreg[0]        # Qubit 0                → [(qreg, [0])]
qreg[0:2]      # Qubits 0-1            → [(qreg, [0, 1])]
qreg[[0, 2]]   # Qubits 0 and 2        → [(qreg, [0, 2])]
qreg[(0, 1)]   # Qubits 0 and 1        → [(qreg, [0, 1])]
qreg[-1]       # Last qubit             → [(qreg, [2])]
```

#### Key Features
- **Flexible Indexing**: Support for integers, slices, lists, tuples, and negative indices
- **Automatic Range Checking**: Out-of-range access raises exceptions automatically
- **UUID Uniqueness**: Each Register instance has a unique identifier
- **Collection-Friendly**: Can be used as dictionary keys and set elements

#### Usage Example
```python
# Create multiple registers
q1 = Register("q1", 2)
q2 = Register("q2", 3)

# Use in circuits
from core.GateSequence import GateSequence
circuit = GateSequence(q1, q2)
circuit.h(q1[0])
circuit.cx(q1[0], q2[1])
```

---

### 2.2 ClassicalRegister (Classical Register)

#### Overview
- **Purpose**: Store quantum measurement results and enable quantum-classical interactions
- **Features**: Indexing similar to Register, supports state tracking

#### Core Workflow

```
Initialization
  ↓ (All bits initialized to -1, representing unmeasured state)
Measure quantum qubits
  ↓ (Obtain 0 or 1 results)
Update classical bit values
  ↓
Use these values in classical algorithm
  ↓ (Conditional branching, computation, etc.)
Return to quantum operations
  ↓ (Based on classical results)
Complete hybrid algorithm
```

#### Indexing Methods

```python
from core.Classicalregister import ClassicalRegister

creg = ClassicalRegister("result", 3)

# Same indexing methods as Register
creg[0]
creg[0:2]
creg[[0, 1]]
```

#### Usage Example
```python
from core.GateSequence import GateSequence
from core.Classicalregister import ClassicalRegister
from core.register import Register

qreg = Register("q", 2)
creg = ClassicalRegister("c", 2)

circuit = GateSequence(qreg, creg)
circuit.h(qreg[0])
circuit.measure(qreg, creg)  # Measure and store results
```

---

### 2.3 GateSequence (Quantum Gate Sequence)

#### Overview
- **Purpose**: Build, manage, and execute quantum circuits
- **Features**: High-level API, automatic index translation, multiple gate types

#### Architecture Design

**Register Management**:
```
GateSequence
├─ Quantum registers list
│  ├─ Register "q": [q0, q1, q2]
│  └─ Register "anc": [q3, q4]
├─ Classical registers list
│  └─ ClassicalRegister "c": [c0, c1]
└─ Global index mapping
   [0→(q,0), 1→(q,1), 2→(q,2), 3→(anc,0), 4→(anc,1)]
```

**Gate Types**:

| Gate Category | Description | Examples |
|--------|------|------|
| Single-qubit gates | Operations on single qubit | H, X, Y, Z, S, T |
| Parameterized gates | Rotation-type operations | RX(θ), RY(θ), RZ(θ) |
| Two-qubit gates | Controlled or swap operations | CNOT, CZ, SWAP |
| Multi-control gates | Multiple control qubits | MCX, MCY, MCZ |
| Measurement | Quantum to classical conversion | MEASURE |
| Custom gates | User-defined | UNITARY |

#### Basic Workflow

```python
from core.GateSequence import GateSequence
from core.register import Register

# Step 1: Create register
qreg = Register("q", 3)

# Step 2: Create circuit
circuit = GateSequence(qreg)

# Step 3: Build circuit
circuit.h(qreg[0])                          # Hadamard gate
circuit.cx(qreg[0], qreg[1])                # CNOT gate
circuit.ry(qreg[2], 1.57)                   # RY(π/2) gate
circuit.measure(qreg[0])                    # Measurement

# Step 4: Execute and get results
result = circuit.execute(backend='torch')
print(result)
```

---

### 2.4 State (Quantum State)

#### Overview
- **Purpose**: Represent and manipulate quantum states
- **Features**: Automatic normalization, complex amplitude support, PyTorch backend

#### Mathematical Foundation

**State Vector Representation**:
$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

其中 $\alpha_i$ 是复数振幅，满足归一化条件 $\sum_i |\alpha_i|^2 = 1$

**Operation Types**:

| Operation | Description | Formula |
|------|------|------|
| Measurement | Get probability distribution | $P(i) = \|\alpha_i\|^2$ |
| Inner product | Overlap of two states | $\langle \psi \| \phi \rangle$ |
| Tensor product | Combine two systems | $\|\psi_1\rangle \otimes \|\psi_2\rangle$ |
| Expectation value | Observable expectation | $\langle O \rangle$ |

#### Creation and Usage

```python
from core.State import State
import numpy as np

# Create ground state
state = State(3)  # |000⟩

# Create from vector
data = [1/np.sqrt(2), 1/np.sqrt(2)]
state = State(data)  # (|0⟩ + |1⟩)/√2, auto-normalized

# Query and operate
probs = state.probabilities()              # Probability distribution
data = state.data                          # Get amplitude data
inner = state.inner_product(other_state)   # Inner product calculation
```

---

### 2.5 Initialization (Quantum State Preparation)

#### Overview
- **Purpose**: Automatically build quantum circuits to prepare target states
- **Features**: Recursive decomposition, parameter optimization, automation

#### Algorithm Principles

**Single-Qubit Initialization**:
```
Target state: α₀|0⟩ + α₁|1⟩
  ↓ [Extract phase]
Apply global phase compensation
  ↓ [Compute rotation angle]
θ = arccos(|α₀|)
  ↓ [Apply RY(2θ)]
Get correct amplitude superposition
  ↓ [Apply phase gate]
Add relative phase
  ↓
Final state accurate
```

**Multi-Qubit Recursive Decomposition**:
```
n-qubit target state |v⟩
  ↓ [Split]
First 2^(n-1) amplitudes v1 | Last 2^(n-1) amplitudes v2
  ↓ [Compute probabilities]
p1 = ||v1||²  (probability of qubit(n-1)=0)
p2 = ||v2||²  (probability of qubit(n-1)=1)
  ↓ [RY rotation]
Apply RY(θ) on qubit(n-1), where cos(θ/2) = √p1
  ↓ [Controlled recursion]
When qubit(n-1)=0: Recursively prepare v1/||v1||
When qubit(n-1)=1: Recursively prepare v2/||v2||
  ↓
Preparation complete
```

#### Usage Example

```python
from core.GateSequence import GateSequence
from core.Initialization import StatePreparation
from core.register import Register
import numpy as np

# Target state: (|00⟩ + |11⟩)/√2 (Bell state)
target_state = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]

qreg = Register("q", 2)
circuit = GateSequence(qreg)

# Use initialization module
StatePreparation.prepare(circuit, qreg, target_state)

# Circuit now contains all gates needed to prepare this state
result = circuit.execute()
```

---

### 2.6 CircuitDrawer (Circuit Visualization)

#### Overview
- **Purpose**: Convert quantum circuits to visual images
- **Features**: Flexible layout, custom styling, multiple output formats

#### Drawing Process

```
Quantum circuit GateSequence
    ↓
Extract and sort gate list
    ↓
Gate layer folding (optimize space utilization)
    ├─ Standard folding: arrange in time order
    └─ Compact folding: more efficient space utilization
    ↓
Compute coordinate information
    ├─ (x, y) position for each gate
    ├─ Font size, line width
    └─ Colors and styling
    ↓
Draw layer-by-layer with Matplotlib
    ├─ Background and grid
    ├─ Qubit lines and register labels
    ├─ Quantum gates
    ├─ Control points and connection lines
    └─ Parameter labels
    ↓
Export or display
    ├─ PNG/SVG files
    ├─ Screen display
    └─ PDF (depending on backend)
```

#### Coordinate System

```
y-axis (downward is positive):
  0 ← Top (register labels)
  ↓
  -1, -2, -3 ← Qubit lines
  ↓
  -(n+1) ← Second layer start

x-axis (rightward is positive):
  0 ← x offset start
  →
  gate_x ← x coordinate of first gate
  →
  ...
```

#### Style System

```python
style = {
    'backgroundColor': 'white',
    'linecolor': 'black',
    'gatetextcolor': 'black',
    'fontsize': 12,
    'lwidth1': 1.0,           # Thin lines
    'lwidth2': 2.0,           # Medium lines
    'width_per_layer': 2.0,   # Width per layer
    'margin': 0.5,            # Margin
    'displaytext': {          # Gate name mapping
        'h': 'H',
        'x': 'X',
        # ...
    }
}
```

#### Usage Example

```python
from drawer.circuit_drawer import CircuitDrawer
from core.GateSequence import GateSequence
from core.register import Register

qreg = Register("q", 3)
circuit = GateSequence(qreg)
circuit.h(qreg[0])
circuit.cx(qreg[0], qreg[1])

# Draw circuit
drawer = CircuitDrawer(circuit)
drawer.draw('my_circuit.png', style='default')
drawer.show()  # Display on screen
```

---

## III. Integrated Workflows

### 3.1 Complete Quantum Algorithm Example: Bell State Preparation and Measurement

```python
from core.register import Register
from core.Classicalregister import ClassicalRegister
from core.GateSequence import GateSequence
from drawer.circuit_drawer import CircuitDrawer

# Step 1: Create registers
qreg = Register("q", 2)
creg = ClassicalRegister("result", 2)

# Step 2: Create circuit
circuit = GateSequence(qreg, creg)

# Step 3: Build Bell state circuit
circuit.h(qreg[0])                    # (|0⟩ + |1⟩)/√2 on q0
circuit.cx(qreg[0], qreg[1])          # CNOT: (|00⟩ + |11⟩)/√2

# Step 4: Measure
circuit.measure(qreg, creg)

# Step 5: Execute
result = circuit.execute(shots=1000)
print(f"Measurement distribution: {result.counts}")

# Step 6: Visualize
drawer = CircuitDrawer(circuit)
drawer.draw('bell_circuit.png')
drawer.show()
```

### 3.2 Complex State Preparation Using Initialization

```python
from core.register import Register
from core.GateSequence import GateSequence
from core.Initialization import StatePreparation
import numpy as np

# Target: Quantum Fourier Transform eigenstate
# |ψ⟩ = (|0⟩ + e^(2πi/8)|1⟩ + e^(4πi/8)|2⟩ + ...)/√8
target = np.array([
    1/np.sqrt(8) * np.exp(2j*np.pi*k/8) 
    for k in range(8)
])

qreg = Register("q", 3)
circuit = GateSequence(qreg)

# Automatically prepare state
StatePreparation.prepare(circuit, qreg, target)

# Circuit now contains all necessary gates
```

### 3.3 Quantum State Analysis

```python
from core.State import State
import numpy as np

# Create Bell state (|00⟩ + |11⟩)/√2
bell_state = State([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

# Analysis
probs = bell_state.probabilities()
print("Measurement probabilities:", probs)          # [0.5, 0, 0, 0.5]

# Inner product with other state
other = State([1, 0, 0, 0])      # |00⟩
overlap = bell_state.inner_product(other)
print("|⟨00|ψ⟩|²:", abs(overlap)**2)  # 0.5
```

---

## IV. Learning Path

### Stage 1: Fundamentals (1-2 weeks)
1. Understand Register and ClassicalRegister concepts
2. Learn basic GateSequence usage
3. Implement simple quantum circuits (Bell state, GHZ state)
4. Use CircuitDrawer to visualize circuits

**Key Projects**:
- Build 2-3 qubit quantum circuits
- Execute and analyze measurement results
- Generate circuit diagrams

### Stage 2: Intermediate (2-3 weeks)
1. Deep dive into State class mathematics
2. Learn Initialization module recursive decomposition
3. Implement parameterized quantum circuits
4. Handle complex multi-qubit states

**Key Projects**:
- Build variational circuits with parameterized gates
- Prepare specific quantum states (such as eigenstates)
- Design hybrid quantum-classical algorithms

### Stage 3: Advanced (3-4 weeks)
1. Complete quantum algorithm implementations (Grover, QFT, VQE, etc.)
2. Circuit optimization and compilation
3. Error analysis and mitigation
4. Large-scale simulation

**Key Projects**:
- Implement complete quantum algorithms
- Performance optimization and analysis
- Interface with actual quantum hardware

---

## V. Quick Reference

### Common Tasks Quick Guide

**Task 1: Create Basic Quantum Circuit**
```python
qreg = Register("q", 3)
circuit = GateSequence(qreg)
circuit.h(qreg)         # Apply H gate on all qubits
circuit.cx(qreg[0], qreg[1])  # CNOT
```

**Task 2: Prepare Specific Quantum State**
```python
target_state = [1/np.sqrt(2), 0, 1/np.sqrt(2), 0]
StatePreparation.prepare(circuit, qreg, target_state)
```

**Task 3: Execute and Measure**
```python
creg = ClassicalRegister("c", 3)
circuit = GateSequence(qreg, creg)
circuit.measure(qreg, creg)
result = circuit.execute(shots=1000)
```

**Task 4: Visualize Circuit**
```python
drawer = CircuitDrawer(circuit)
drawer.draw('output.png')
```

---

## VI. Architecture Dependencies

```
CircuitDrawer
  └─ GateSequence
      ├─ Register (manages qubits)
      ├─ ClassicalRegister (stores measurement results)
      ├─ Initialization (optional state preparation)
      └─ State (for analysis)

State
  └─ PyTorch (computational backend)

GateSequence
  └─ All other modules
```

---

## VII. Version History

- **v2.0**: Complete integration of all sub-modules, added comprehensive documentation and workflows
- **v1.0**: Initial version with basic gate sequences and register functionality
