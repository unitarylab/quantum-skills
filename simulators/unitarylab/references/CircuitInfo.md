# CircuitInfo User Guide

## Introduction

`CircuitInfo` is a powerful helper class for analyzing quantum gate circuits. It takes a gate sequence object `qc`, extracts the gate data and the number of qubits, and provides comprehensive analytical capabilities:

- **Circuit Overview**: Basic statistics including qubit count, gate count, and circuit depth
- **Gate Statistics**: Detailed analysis of gate types and counts (single-qubit, two-qubit, multi-qubit, parameterized)
- **Structural Analysis**: Layered gate scheduling, instruction lists, and coupling maps
- **Qubit Analysis**: Qubit usage patterns and operation histories
- **Parameter Tracking**: Collection and analysis of parameterized gate parameters
- **Visualization**: Pretty-print methods for easy viewing of circuit information

This tool is essential for quickly obtaining circuit structure information when building, debugging, or optimizing quantum circuits.

## Quick Start

### 1. Import and Create the Circuit

```python
from unitarylab import Circuit
import numpy as np

qc = Circuit(4)

qc.x(0)
qc.cx(0, 1)
qc.h(0)
qc.rz(np.pi / 4, 1)
qc.mcx([0,1],3)

qc.analyze()

```

**Output example:**

```python


====================================================
                  Circuit Overview
====================================================
Qubits              : 4
Total Gates         : 5
Circuit Depth       : 4
Single-Qubit Gates  : 3
Two-Qubit Gates     : 1
Multi-Qubit Gates   : 1
Parameterized Gates : 1
Parameterized       : True

Gate Counts
  - x               : 1
  - cx              : 1
  - h               : 1
  - rz              : 1
  - mcx             : 1

Qubit Usage
  - q0              : 4
  - q1              : 3
  - q2              : 0
  - q3              : 1

Coupling Map
  - q0 <-> q1
  - q0 <-> q3
  - q1 <-> q3
====================================================

====================================================
                  Instructions
====================================================
[0] x
    raw_name         : x
    target           : [0]
    control          : []
    params           : {}
    control_sequence : []
----------------------------------------------------
[1] cx
    raw_name         : cx
    target           : [1]
    control          : [0]
    params           : {}
    control_sequence : [1]
----------------------------------------------------
[2] h
    raw_name         : h
    target           : [0]
    control          : []
    params           : {}
    control_sequence : []
----------------------------------------------------
[3] rz
    raw_name         : rz
    target           : [1]
    control          : []
    params           : {'value': 0.7853981633974483}
    control_sequence : []
----------------------------------------------------
[4] mcx
    raw_name         : mcx
    target           : [3]
    control          : [0, 1]
    params           : {}
    control_sequence : [1, 1]
----------------------------------------------------
====================================================

====================================================
                     Layers
====================================================
Layer 0
  - x         target=[0]  control=[]  params={}
  - rz        target=[1]  control=[]  params={'value': 0.7853981633974483}
----------------------------------------------------
Layer 1
  - cx        target=[1]  control=[0]  params={}
----------------------------------------------------
Layer 2
  - h         target=[0]  control=[]  params={}
----------------------------------------------------
Layer 3
  - mcx       target=[3]  control=[0, 1]  params={}
----------------------------------------------------
====================================================

```

### 2. Display Circuit Overview

```python
info = qc.analyze(show=False)


info.show("overview") 
# or
info.show("summary")  
```

**Output example:**

```python

====================================================
                  Circuit Overview
====================================================
Qubits              : 4
Total Gates         : 5
Circuit Depth       : 4
Single-Qubit Gates  : 3
Two-Qubit Gates     : 1
Multi-Qubit Gates   : 1
Parameterized Gates : 1
Parameterized       : True

Gate Counts
  - x               : 1
  - cx              : 1
  - h               : 1
  - rz              : 1
  - mcx             : 1

Qubit Usage
  - q0              : 4
  - q1              : 3
  - q2              : 0
  - q3              : 1

Coupling Map
  - q0 <-> q1
  - q0 <-> q3
  - q1 <-> q3
====================================================
```

### 3. View Instruction List

```python
info.show("instructions")
```

**Output example:**

```python

====================================================
                  Instructions
====================================================
[0] x
    raw_name         : x
    target           : [0]
    control          : []
    params           : {}
    control_sequence : []
----------------------------------------------------
[1] cx
    raw_name         : cx
    target           : [1]
    control          : [0]
    params           : {}
    control_sequence : [1]
----------------------------------------------------
[2] h
    raw_name         : h
    target           : [0]
    control          : []
    params           : {}
    control_sequence : []
----------------------------------------------------
[3] rz
    raw_name         : rz
    target           : [1]
    control          : []
    params           : {'value': 0.7853981633974483}
    control_sequence : []
----------------------------------------------------
[4] mcx
    raw_name         : mcx
    target           : [3]
    control          : [0, 1]
    params           : {}
    control_sequence : [1, 1]
----------------------------------------------------
====================================================
```

### 4. View Layered Structure

```python
info.show("layers")
```

**Output example:**

```python

====================================================
                     Layers
====================================================
Layer 0
  - x         target=[0]  control=[]  params={}
  - rz        target=[1]  control=[]  params={'value': 0.7853981633974483}
----------------------------------------------------
Layer 1
  - cx        target=[1]  control=[0]  params={}
----------------------------------------------------
Layer 2
  - h         target=[0]  control=[]  params={}
----------------------------------------------------
Layer 3
  - mcx       target=[3]  control=[0, 1]  params={}
----------------------------------------------------
====================================================
```

### 5. Query Qubit Usage

```python
info.show("qubit_usage")
```

**Output example:**

```python

====================================================
                  Qubit Usage
====================================================
  - q0              : 4
  - q1              : 3
  - q2              : 0
  - q3              : 1
====================================================

```

### 6. Get Coupling Map

```python
info.show("coupling_map")

```

**Output example:**

```

====================================================
                 Coupling Map
====================================================
  - q0 <-> q1
  - q0 <-> q3
  - q1 <-> q3
====================================================
```

### 7. View Parameterized Gates

```python
info.show("parameters")
```

**Output example:**

```
====================================================
                 Gate Parameters
====================================================
rz
  [0] {'value': 0.7853981633974483}
----------------------------------------------------
====================================================
```

### 8. Get Qubit Operation History

```python
info.show("qubit_history", qubit=0)
```

**Output example:**

```

====================================================
                 Qubit History: q0                 
====================================================
[0] x
    raw_name : x
    role     : target
    target   : [0]
    control  : []
    params   : {}
----------------------------------------------------
[1] cx
    raw_name : cx
    role     : control
    target   : [1]
    control  : [0]
    params   : {}
----------------------------------------------------
[2] h
    raw_name : h
    role     : target
    target   : [0]
    control  : []
    params   : {}
----------------------------------------------------
[4] mcx
    raw_name : mcx
    role     : control
    target   : [3]
    control  : [0, 1]
    params   : {}
----------------------------------------------------
====================================================
```

