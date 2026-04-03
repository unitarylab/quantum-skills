---
name: circuitdrawer
description: |
  Quantum circuit visualization and drawing module using Matplotlib.
  Provides comprehensive tools for rendering quantum circuits with flexible layout options,
  gate styling, and support for various quantum gate types including single-qubit, multi-qubit,
  and controlled gates with customizable appearance and formatting.
keywords:
  - circuit drawer
  - quantum visualization
  - circuit rendering
  - gate drawing
  - circuit layout
  - matplotlib
  - quantum circuit diagram
---

# Circuit Drawer User Guide

## I. Algorithm Introduction

### 1.1 Core Concepts

**Circuit Drawer** is a tool that converts abstract representations of quantum circuits into visual images. It is responsible for:
- Converting quantum gate sequences into understandable circuit diagrams
- Managing circuit layout and space utilization
- Applying styling and aesthetic design
- Supporting interactive or static image export

### 1.2 Core Architecture

#### 1.2.1 Drawing Process

```
Quantum Circuit (GateSequence)
    ↓
Extract gate list
    ↓
Gate layer folding (Layering)
    ├─ Standard folding: get_layered_gates
    └─ Compact folding: compact_get_layered_gates
    ↓
Compute coordinate information
    ├─ Gate position (x, y)
    ├─ Font size
    ├─ Line width
    └─ Color configuration
    ↓
Draw layer-by-layer
    ├─ Draw register labels and lines
    ├─ Draw various quantum gates
    ├─ Draw control points and connection lines
    └─ Draw other elements (labels, parameters)
    ↓
Final image (PNG/SVG/Display)
```

#### 1.2.2 Coordinate System

```
y-axis direction (downward is positive):
  ↓ 0 ← Top
  ↓
  ↓ -1, -2, -3, ... ← Each qubit downward
  ↓
  ↓ -(n+1) ← Second layer start after folding
  ↓

x-axis direction (rightward is positive):
  ←─── 0 (x_offset)
       │
       gate1 ← [x, width]
       │
  gate2 ← [x+width, width]
```

#### 1.2.3 Gate Folding Mechanism

**Standard Folding (Non-compact)**
```
Gate1  Gate2  Gate3
|─────|─────|─────|
Layer 0: Place in time order, fold when exceeding width limit

Gate4  Gate5
|─────|─────|
Layer 1: Continue placing subsequent gates
```

**Compact Folding**
```
Key optimizations:
1. Each qubit independently maintains "cursor position"
2. Gates without dependencies can be placed in parallel
3. Measurement gates automatically align all qubits

Result: More efficient space utilization
```

### 1.3 Style System

CircuitDrawer uses a layered style system:

```python
Style Configuration (Style Dictionary)
├─ Color
│  ├─ Background color (backgroundColor)
│  ├─ Line color (linecolor)
│  ├─ Gate text color (gatetextcolor)
│  └─ Display color (displaycolor)
├─ Size
│  ├─ Font size (fontsize)
│  ├─ Line width (lwidth1/2/3/4)
│  ├─ Width per layer (width_per_layer)
│  └─ Margin (margin)
├─ Display Text
│  └─ Gate name mapping (displaytext)
└─ Layout Parameters
   ├─ x/y offset (x_offset, y_offset)
   └─ Scale factor (scale_char)
```

### 1.4 Gate Type Classification

| Gate Type | Draw Method | Characteristics | Examples |
|--------|---------|------|------|
| **Single-qubit gates** | `_gate()` | Operations on single qubit | H, X, Y, Z, S, T, RX, RY, RZ |
| **Multi-qubit gates** | `_multiqubit_gate()` | Operations on multiple qubits | SWAP, CX, CZ, CH |
| **Controlled gates** | `_control_gate()` | Quantum gates with control points | CNOT, CCX, CRX |
| **Swap gates** | `_swap()` | Swap two qubits | SWAP |

---

## II. Usage Steps

### 2.1 Basic Usage Workflow

#### Step 1: Import Modules
```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register
```

#### Step 2: Create Quantum Circuit
```python
# Create register and circuit
qreg = Register("q", 3)
circuit = GateSequence(qreg)

# Build circuit
circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])
circuit.cnot(qreg[1], qreg[2])
```

#### Step 3: Create CircuitDrawer
```python
# Create drawer
drawer = CircuitDrawer(
    scale=1.0,           # Scale factor
    width_per_layer=30,  # Max width per layer
    style='dark'         # Style: 'dark' or 'light'
)
```

#### Step 4: Draw Circuit
```python
# Draw to screen
fig = drawer.draw(circuit, title="My Quantum Circuit")

# Or save to file
drawer.draw(circuit, filename="circuit.png", title="Bell State")
```

### 2.2 Detailed Usage Examples

#### Example 1: Simple Circuit Drawing

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

# Create 2-qubit circuit
qreg = Register("q", 2)
circuit = GateSequence(qreg, name="simple_circuit")

# Add gates
circuit.h(qreg[0])
circuit.h(qreg[1])
circuit.cnot(qreg[0], qreg[1])

# Draw
drawer = CircuitDrawer()
figure = drawer.draw(circuit, title="Simple Circuit", filename="simple.png")
```

#### Example 2: Bell State Circuit

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

# Create Bell state circuit
qreg = Register("q", 2)
circuit = GateSequence(qreg, name="Bell State")

circuit.h(qreg[0])
circuit.cnot(qreg[0], qreg[1])

# Draw with compact layout
drawer = CircuitDrawer(width_per_layer=50, style='light')
drawer.draw(circuit, title="Bell State Circuit", compact=True)
```

#### Example 3: Complex Multi-qubit Circuit

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

# Create Grover-style circuit
qreg = Register("q", 4)
circuit = GateSequence(qreg, name="Grover Circuit")

# Initialization layer
for i in range(4):
    circuit.h(qreg[i])

# Oracle
circuit.mcx([qreg[0], qreg[1], qreg[2]], qreg[3])

# Diffusion
for i in range(4):
    circuit.h(qreg[i])
    circuit.x(qreg[i])

circuit.h(qreg[3])
circuit.mcx([qreg[0], qreg[1], qreg[2]], qreg[3])
circuit.h(qreg[3])

for i in range(4):
    circuit.x(qreg[i])
    circuit.h(qreg[i])

# Draw
drawer = CircuitDrawer(width_per_layer=40)
drawer.draw(circuit, title="Grover Algorithm", filename="grover.png")
```

#### Example 4: Parameterized Circuit

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register
import numpy as np

# Create parameterized circuit
qreg = Register("q", 2)
circuit = GateSequence(qreg)

# Apply parameterized gates
angles = [np.pi/4, np.pi/3]
circuit.rx(angles[0], qreg[0])
circuit.rz(angles[1], qreg[1])
circuit.cnot(qreg[0], qreg[1])

# Draw (parameters will be auto-formatted)
drawer = CircuitDrawer(style='dark')
drawer.draw(circuit, title="Parameterized Circuit")
```

#### Example 5: Different Styles

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence


# Create circuit
circuit = GateSequence(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.x(2)

# Dark style
drawer_dark = CircuitDrawer(style='dark')
drawer_dark.draw(circuit, title="Dark Style", filename="circuit_dark.png")

# Light style
drawer_light = CircuitDrawer(style='light')
drawer_light.draw(circuit, title="Light Style", filename="circuit_light.png")

# Custom style
custom_style = {
    'backgroundcolor': '#1a1a1a',
    'fontsize': 12,
    'gatefacecolor': '#00ff00'
}
drawer_custom = CircuitDrawer(style=custom_style)
drawer_custom.draw(circuit, title="Custom Style")
```

### 2.3 Advanced Scenarios

#### Scenario 1: Large Circuit Folding

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence

# Create large circuit
circuit = GateSequence(20)
for i in range(100):
    circuit.h(i % 20)
    circuit.cnot(i % 20, (i + 1) % 20)

# Use smaller width_per_layer to force folding
drawer = CircuitDrawer(width_per_layer=20, scale=0.8)
drawer.draw(circuit, filename="large_circuit.png", compact=True)
```

#### Scenario 2: Batch Circuit Saving

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence

def save_circuit_family(n_qubits_list, output_dir='./circuits'):
    """Save a series of circuits with different sizes"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for n in n_qubits_list:
        circuit = GateSequence(n)
        for i in range(n):
            circuit.h(i)
        
        drawer = CircuitDrawer()
        filename = os.path.join(output_dir, f"circuit_{n}qubits.png")
        drawer.draw(circuit, filename=filename, title=f"{n}-Qubit Circuit")
        print(f"Saved: {filename}")

# Usage
save_circuit_family([2, 3, 4, 5, 6])
```

#### Scenario 3: Custom Labels and Titles

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

# Create registers with meaningful names
qreg_data = Register("data", 2)
qreg_work = Register("work", 1)

circuit = GateSequence(qreg_data, qreg_work)
circuit.h(qreg_data)

# Draw with meaningful title
drawer = CircuitDrawer()
drawer.draw(circuit, title="Quantum Fourier Transform Circuit", filename="qft.png")
```

---

## III. Key API Reference

### 3.1 Constructor

```python
CircuitDrawer(scale=1.0, ax=None, width_per_layer=30, style='dark')
```

**Parameters:**
- `scale: float` - Overall scale factor (default 1.0)
- `ax: matplotlib.axes.Axes` - Optional matplotlib axis object
- `width_per_layer: float` - Maximum width of gates per layer (default 30)
- `style: str | dict` - Style configuration ('dark', 'light', JSON file path, or dictionary)

### 3.2 Main Methods

| Method | Parameters | Return | Description |
|------|------|--------|------|
| `draw()` | circuit, filename, title, compact | Figure | Draw and return/save circuit |
| `get_layered_gates()` | gate_list | - | Standard gate layer folding |
| `compact_get_layered_gates()` | gate_list | - | Compact gate layer folding |
| `get_text_width()` | text | float | Compute text width |

### 3.3 draw() Method Details

```python
draw(gate_sequence: GateSequence, filename=None, title=False, compact=True) -> Figure
```

**Parameters:**
- `gate_sequence` - GateSequence object to draw
- `filename` - Output filename (e.g., 'circuit.png'), if not specified, display on screen
- `title` - Circuit title (default: use circuit name)
- `compact` - Whether to use compact layout (default True)

**Returns:**
- Matplotlib Figure object

**Examples:**
```python
# Display to screen
fig = drawer.draw(circuit, title="My Circuit")

# Save to file
drawer.draw(circuit, filename="output.png", title="Saved Circuit")
```

---

## IV. Style Configuration Details

### 4.1 Predefined Styles

```python
# Dark theme
drawer = CircuitDrawer(style='dark')

# Light theme
drawer = CircuitDrawer(style='light')

# Load from JSON file
drawer = CircuitDrawer(style='my_style.json')

# JSON string format
json_string = '{"backgroundcolor": "#ffffff", "fontsize": 14}'
drawer = CircuitDrawer(style=json_string)
```

### 4.2 Custom Style Parameters

```python
custom_style = {
    # Color configuration
    'backgroundcolor': '#1a1a1a',      # Background color
    'linecolor': '#666666',            # Line color
    'gatetextcolor': 'white',          # Gate text color
    'textcolor': '#999999',            # Regular text color
    'subtextcolor': '#cccccc',         # Small text color
    
    # Size configuration
    'fontsize': 11,                    # Main font size
    'subfontsize': 8,                  # Small font size
    'gatewidth': 0.9,                  # Minimum gate width
    'scale_char': 0.15,                # Character scale factor
    
    # Line width configuration
    'lwidth1': 0.5,                    # Thin lines
    'lwidth15': 1.0,                   # Medium-thin lines
    'lwidth2': 1.5,                    # Medium lines
    'lwidth3': 2.0,                    # Thick lines
    'lwidth4': 2.5,                    # Thickest lines
    
    # Layout configuration
    'x_offset': 1.0,                   # x-direction offset
    'y_offset': 0.5,                   # y-direction offset
    'margin': [1.5, 1.5, 1.5, 1.5],   # [left, right, bottom, top]
    'figwidth': 10.0,                  # Figure width
    'dpi': 150,                        # Resolution
    
    # Gate text mapping
    'displaytext': {
        'h': 'H',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        # ... more mappings
    },
    
    # Display color mapping
    'displaycolor': {
        'target': ['#1F52F0', 'white'],
        'control': ['#666666', 'white'],
    }
}

drawer = CircuitDrawer(style=custom_style)
```

---

## V. Layout Algorithm Details

### 5.1 Standard Folding Algorithm

```
Process:
1. Iterate through all gates
2. For each gate, compute width
3. If (current_x + gate_width) > limit:
   - Move to next layer
   - Reset x coordinate
4. Place gate
5. Update x coordinate
```

**Complexity:**
- Time: O(n × m) (n gates, m layers)
- Space: O(n)

### 5.2 Compact Folding Algorithm

```
Process:
1. Maintain a "cursor" (layer, x) for each qubit
2. For each gate:
   a. Determine range of qubits involved
   b. Find the latest layer among these qubits
   c. Compute the farthest x position in that layer
   d. If measurement gate, align cursors of all qubits
   e. Update cursors of involved qubits
```

**Advantages:**
- More compact layout
- Measurement gates automatically aligned
- Gates without dependencies can be placed in parallel

### 5.3 Coordinate Calculation

```python
# Single-qubit gate position
gate['x'] = current_x              # Top-left
gate['y'] = y_base - qmax - y_offset

# Multi-qubit gate position
gate['x'] = current_x
gate['y'] = y_base - qmax - y_offset
height = qmax - qmin + 1

# y-coordinate adjustment after folding
y_base = -(layer * (n_qubits + 1))
```

---

## VI. Common Patterns

### Pattern 1: Quick Draw

```python
def quick_draw(circuit, title="Circuit"):
    """Quick circuit drawing"""
    from engine.drawer.circuit_drawer import CircuitDrawer
    
    drawer = CircuitDrawer()
    return drawer.draw(circuit, title=title)
```

### Pattern 2: Batch Export

```python
def export_circuits(circuits_dict, output_dir='./outputs'):
    """Export multiple circuits"""
    import os
    from engine.drawer.circuit_drawer import CircuitDrawer
    
    os.makedirs(output_dir, exist_ok=True)
    drawer = CircuitDrawer()
    
    for name, circuit in circuits_dict.items():
        filename = os.path.join(output_dir, f'{name}.png')
        drawer.draw(circuit, filename=filename, title=name)
```

### Pattern 3: Style Presets

```python
def get_publication_style():
    """High-quality publication style"""
    return {
        'backgroundcolor': 'white',
        'fontsize': 12,
        'figwidth': 12.0,
        'dpi': 300,
        'lwidth2': 1.2,
    }

def get_presentation_style():
    """Presentation slide style"""
    return {
        'backgroundcolor': '#2B2B2B',
        'fontsize': 14,
        'figwidth': 10.0,
        'dpi': 150,
    }
```

---

## VII. Performance Optimization

### 7.1 Large Circuit Optimization

```python
# Use compact layout for large circuits
drawer = CircuitDrawer(
    width_per_layer=25,    # Smaller width forces more folding
    style='light'          # Light may be faster
)
drawer.draw(circuit, compact=True)

# Reduce DPI for faster rendering
custom_style = {'dpi': 72}  # Screen preview
drawer = CircuitDrawer(style=custom_style)
```

### 7.2 Text Width Caching

```python
# Text width caching is already implemented internally
# Uses pylatexenc library for efficient computation
```

---

## VIII. Error Handling Guide

### 8.1 Common Errors

**Error 1: Missing circuit object**
```python
# Wrong!
drawer = CircuitDrawer()
drawer.draw(None)  # TypeError

# Correct
from engine.core.gate_sequence import GateSequence
circuit = GateSequence(3)
drawer.draw(circuit)
```

**Error 2: Invalid style string**
```python
# Wrong!
drawer = CircuitDrawer(style='invalid_style')

# Correct
drawer = CircuitDrawer(style='dark')  # or 'light'
```

**Error 3: File overwrite**
```python
# Check if file exists before saving
import os
filename = "circuit.png"

if os.path.exists(filename):
    print(f"File {filename} already exists")
else:
    drawer.draw(circuit, filename=filename)
```

---

## IX. Complete Workflow Example

```python
from engine.drawer.circuit_drawer import CircuitDrawer
from engine.core.gate_sequence import GateSequence
from engine.core.register import Register

def complete_visualization_workflow():
    """Complete quantum circuit visualization workflow"""
    
    # Step 1: Create quantum circuit
    print("Step 1: Create quantum circuit")
    qreg = Register("q", 4)
    circuit = GateSequence(qreg, name="Complex Circuit")
    
    # Step 2: Build circuit
    print("Step 2: Add quantum gates")
    for i in range(4):
        circuit.h(qreg[i])
    
    for i in range(3):
        circuit.cnot(qreg[i], qreg[i+1])
    
    circuit.rx(1.57, qreg[0])
    circuit.rz(3.14, qreg[3])
    
    # Step 3: Create drawer
    print("Step 3: Configure drawing parameters")
    drawer = CircuitDrawer(
        style='dark',
        width_per_layer=40,
        scale=1.2
    )
    
    # Step 4: Draw to screen
    print("Step 4: Preview circuit")
    fig = drawer.draw(
        circuit,
        title="Complex Quantum Circuit",
        compact=True
    )
    
    # Step 5: Save to file
    print("Step 5: Save circuit diagram")
    drawer.draw(
        circuit,
        filename="complex_circuit.png",
        title="Complex Quantum Circuit"
    )
    
    print("Done! Circuit diagram saved")

# Execute
complete_visualization_workflow()
```

---

## X. Checklist Summary

When using CircuitDrawer, ensure:

- [ ] Correctly imported CircuitDrawer and GateSequence
- [ ] Created a valid quantum circuit
- [ ] Selected appropriate style (dark/light or custom)
- [ ] Set appropriate width_per_layer value
- [ ] Chose correct folding method if specific layout needed
- [ ] Specified meaningful circuit title
- [ ] Specified correct filename and path if saving
- [ ] Checked for file overwrite issues
- [ ] Used compact layout for large-scale circuits
- [ ] Customized style when necessary for publication or presentation

