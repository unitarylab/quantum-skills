# Quantum Circuit Building

## Creating a Quantum Circuit

Create circuits using the `Circuit` class. Three constructor forms are supported:

```python
from unitarylab import Circuit, Register, ClassicalRegister
import numpy as np

# Form 1: integer — n-qubit circuit, initial state |0...0⟩
qc = Circuit(4)

# Form 2: Register and ClassicalRegister objects
qr = Register('qr', 2)
cr = ClassicalRegister('cr', 2)
qc = Circuit(qr, cr)

# Form 3: state vector — initial state set to the given vector (must be length 2^n)
sv = np.array([0, 1, 0, 0], dtype=complex)
qc = Circuit(sv)
```
## Single-Qubit Gates

### Pauli Gates
```python

qc.x(0) # Pauli-X gate, applied to qubit 0

qc.y(1) # Pauli-Y gate, applied to qubit 1

qc.z(0) # Pauli-Z gate, applied to qubit 0
```

### Hadamard Gate

```python

qc.h(1) # Hadamard gate, applied to qubit 1
```
### Phase Gate
```python
qc.s(0) # S gate, applied to qubit 0

qc.sdag(1) # S† gate (conjugate transpose of S), applied to qubit 1
qc.t(0) # T gate, applied to qubit 0
qc.tdag(1) # T† gate (conjugate transpose of T), applied to qubit 1
qc.gp(np.pi / 4) # Global phase gate, sets global phase to π/4
```
### Square root gate
```python
qc.sqrtx(0) # √X gate, applied to qubit 0
qc.sqrtxdag(1) # √X† gate, applied to qubit 1
qc.sqrty(0) # √Y gate, applied to qubit 0
qc.sqrtydag(1) # √Y† gate, applied to qubit 1
```
### Identity gate
```python
qc.i(0) # I gate (identity gate), applied to qubit 0
```


## Parameterized Single-Qubit Gates
Parameterized single-qubit gates apply rotations or phase changes using one or more parameters.
### Rotation Gates
```python
import numpy as np

qc.rx(np.pi / 4, 0) # RX rotation gate, rotates π/4 around the X-axis, acting on qubit 0

qc.ry(np.pi / 3, 1) # RY rotation gate, rotates π/3 around the Y-axis, acting on qubit 1

qc.rz(np.pi / 2, 0) # RZ rotation gate, rotates π/2 around the Z-axis, acting on qubit 0
```

### U-Series Universal Single-Qubit Gates
```python
qc.u1(np.pi / 5, 1) # U1 gate, parameter π/5, operates on qubit 1

qc.u2(np.pi / 7, np.pi / 8, 0) # U2 gate, parameter (π/7, π/8), operates on qubit 0

qc.u3(np.pi / 9, np.pi / 10, np.pi / 11, 1) # U3 gate, parameter (π/9, π/10, π/11), operates on qubit 1
```
### Phase Rotation Gate
```python
qc.p(np.pi / 6, 0) # P phase gate, phase parameter π/6, operates on qubit 0
```


## Two-Qubit and Controlled Gates



### Two-bit swap gate
```python
qc.swap(0, 1) # SWAP gate, swaps the states of qubit 0 and qubit 1
```

### Basic controlled gate
```python
qc.cnot(0, 2) # CNOT gate, control bit 0, target bit 2

qc.cx(2, 1) # CX gate, control bit 2, target bit 1 (essentially the same as CNOT)

qc.cz(1, 2) # CZ gate, control bit 1, target bit 2

qc.cy(0, 2) # CY gate, control bit 0, target bit 2

qc.ch(0, 1) # CH gate, control bit 0, target bit 1

qc.cs(1, 0) # CS gate, control bit 1. Target bit 0
```

### Parameterized Controlled Gate
```python
qc.cp(np.pi / 4, 0, 1) # CP gate, control bit 0, target bit 1, phase parameter π/4

qc.crx(np.pi / 5, 1, 2) # CRX gate, control bit 1, target bit 2, rotates π/5 around the X-axis

qc.cry(np.pi / 6, 0, 2) # CRY gate, control bit 0, target bit 2, rotates π/6 around the Y-axis

qc.crz(np.pi / 7, 2, 1) # CRZ gate, control bit 2, target bit 1, rotates π/7 around the Z-axis

```

## Multi-Control Gate Gates

```python

qc = Circuit(4, name="multi_control_test") # Create a 4-qubit circuit for multi-control gate testing
```

### Multi-control fixed gate
```python
qc.mcx([0, 1], 2) # Multi-control X gate, control bit [0, 1], target bit 2

qc.mcy([0, 2], 3) # Multi-control Y gate, control bit [0, 2], target bit 3

qc.mcz([1, 2], 0) # Multi-control Z gate, control bit [1, 2], target bit 0

qc.mch([0, 1], 3) # Multi-control H gate, control bit [0, 1], target bit 3
```

### Multiple control parameterized gates
```python
qc.mcrx(np.pi / 4, [0, 1], 2) # Multiple control RX gate, control bit [0, 1], target bit 2, rotates π/4 around the X-axis

qc.mcry(np.pi / 5, [1, 2], 3) # Multiple control RY gate, control bit [1, 2], target bit 3, rotates π/5 around the Y-axis

qc.mcrz(np.pi / 6, [0, 2], 1) # Multiple control RZ gate, control bit [0, 2], target bit 1, rotates π/6 around the Z-axis

qc.mcp(np.pi / 7, [0, 1, 2], 3) # Multiple control phase gate, control bit [0, 1, 2], target bit 3, phase parameter is π/7
```

## Custom Unitary Gate

The `unitary()` method applies a custom unitary matrix to one or more target qubits.  
It can also be used together with control qubits and a specified `control_sequence`.

Applying a Custom Unitary Matrix
```python
qc = Circuit(n + 1)  # Create an (n+1)-qubit circuit for custom unitary testing

U = np.random.randn(2**n, 2**n) + 1j * np.random.randn(2**n, 2**n)
U = (U + U.conj().T) / 2
U = expm(1j * U)  # Construct a random unitary matrix U of dimension 2**n
```
### Example Test

```python
def test_gs_unitary_apply(n):
    # Generate a random unitary matrix U of dimension 2**n
    U = np.random.randn(2**n, 2**n) + 1j * np.random.randn(2**n, 2**n)
    U = (U + U.conj().T) / 2
    U = expm(1j * U)

    # Create an (n+1)-qubit circuit
    qc = Circuit(n + 1)

    # Apply U to the first n qubits, using qubit n as a control qubit with control state '0'
    qc.unitary(U, range(n), n, '0')

    for j in range(2 ** (n + 1)):
        initial_state = np.zeros(2 ** (n + 1), dtype=complex)
        initial_state[j] = 1.0

        # Execute the circuit
        actual = qc.execute(initial_state.copy())

        # Construct the expected result
        expected = initial_state.copy()

        # U is applied only when the control qubit is in state |0⟩
        if j < 2**n:
            expected[:2**n] = U @ expected[:2**n]

        # Check whether the actual result matches the expected result
        assert np.allclose(actual, expected), f"qc.unitary result does not match expectation, basis index = {j}"

    print(f"test_gs_unitary_apply({n}) passed")

```
### Example run
```python
test_gs_unitary_apply(1)
```



## Measurement

The `measure()` method is used to measure one or more qubits in the circuit.

```python
qc = Circuit(2, name="measure_test")  # Create a 2-qubit circuit for measurement testing

qc.h(0)  # Apply a Hadamard gate to qubit 0 before measurement

qc.measure(0)  # Measure qubit 0

qc.measure([0, 1])  # Measure qubits 0 and 1

data = qc.data()
for i, g in enumerate(data):
    print(i, g)
```


## Circuit Transformations

The `Circuit` class provides several transformation methods for modifying or deriving new circuits from an existing circuit, such as `copy()`, `dagger()`, `reverse()`, `inverse()`, `repeat()`, and `decompose()`.

```python

# Building the Original Circuit
qc = Circuit(2, name="transform_test")  # Create a 2-qubit circuit for circuit transformation testing

qc.h(0)  # Apply a Hadamard gate to qubit 0

qc.cx(0, 1)  # Apply a CX gate with control qubit 0 and target qubit 1

qc.rz(np.pi / 3, 1)  # Apply an RZ gate to qubit 1 with rotation angle π/3

# Circuit Transformation Examples
qs_copy = qc.copy()  # Create a copy of the circuit

qs_dagger = qc.dagger()  # Create the dagger (Hermitian conjugate) of the circuit

qs_reverse = qc.reverse()  # Create a circuit with the gate order reversed

qs_inverse = qc.inverse()  # Create the inverse of the circuit

qs_repeat = qc.repeat(2)  # Repeat the circuit two times

qs_decompose = qc.decompose(1)  # Decompose the circuit to the specified level
```


## Controlled Subcircuit

The `control()` method is used to convert an existing circuit into a controlled circuit by adding one or more control qubits.

```python
qc = Circuit(1, name="sub")  # Create a 1-qubit subcircuit

# Building the Original Subcircuit
qc.h(0)  # Apply a Hadamard gate to qubit 0

qc.t(0)  # Apply a T gate to qubit 0

# Creating a Controlled Subcircuit
ctrl_qc = qc.control(2, control_sequence="11")  # Add 2 control qubits and require the control state to be |11⟩

# Viewing Circuit Information

print("Number of qubits in the original subcircuit:", qc.get_num_qubits())` # Print the number of qubits in the original subcircuit.

print("Number of qubits after control:", ctrl_qc.get_num_qubits())` # Print the number of qubits in the controlled circuit.

print("Name of the controlled circuit:", ctrl_qc.name)` # Print the name of the controlled circuit.

```

## Append and Prepend Subcircuits

The `append()` and `prepend()` methods are used to attach an existing subcircuit to the end or the beginning of another circuit.

```python
qc = Circuit(3, name="main")  # Create a 3-qubit main circuit
sub_qc = Circuit(1, name="sub")  # Create a 1-qubit subcircuit

# Building the Subcircuit
sub_qc.h(0)  # Apply a Hadamard gate to qubit 0 of the subcircuit

sub_qc.t(0)  # Apply a T gate to qubit 0 of the subcircuit

# Building the Main Circuit
qc.x(0)  # Apply an X gate to qubit 0 of the main circuit

# Appending and Prepending the Subcircuit

qc.append(sub_qc, target=[1])  # Append the subcircuit to the end of the main circuit, mapped to qubit 1

qc.prepend(sub_qc, target=[2])  # Prepend the subcircuit to the beginning of the main circuit, mapped to qubit 2

# Viewing Internal Gate Data

for g in qc.data():
    print(g)  # Print the gate data of the combined circuit

```

## Initialize, Execute, and Get Matrix

The `initialize()`, `execute()`, and `get_matrix()` methods are used to set the initial quantum state, run the circuit, and retrieve the unitary matrix representation of the circuit.

```python
qc = Circuit(1, name="exec_test")  # Create a 1-qubit circuit for execution testing

# Initializing the Quantum State
v = np.array([0, 1], dtype=complex)  # Define the initial state vector |1⟩

qc.initialize(v, [0])  # Initialize qubit 0 to the state |1⟩

# Applying Gates
qc.h(0)  # Apply a Hadamard gate to qubit 0

# Executing the Circuit
# execute() returns an ExecutionResult object, not a raw array
result = qc.execute()           # execute from |0...0⟩
result = qc.execute(initial_state=np.array([0, 1], dtype=complex))  # from a given state

state = result.state            # final statevector (1-D complex ndarray)
probs = result.probabilities    # dict {bitstring: probability} over all basis states
classical = result.classical_results_map  # {classical_bit_index: measured_value}

# Getting the Circuit Matrix
matrix = qc.get_matrix()  # Get the matrix representation of the circuit
```

## Drawing a Quantum Circuit

The `draw()` method is used to visualize the structure of a quantum circuit.

```python
qc = Circuit(2, name="draw_test")  # Create a 2-qubit circuit for drawing

# Building a Simple Circuit

qc.h(0)  # Apply a Hadamard gate to qubit 0

qc.cx(0, 1)  # Apply a CX gate with control qubit 0 and target qubit 1

qc.rz(np.pi / 4, 1)  # Apply an RZ gate to qubit 1 with rotation angle π/4

qc.draw() # Draw the circuit

```

## Exception Handling

The `Circuit` class should raise exceptions when invalid inputs are provided, such as invalid qubit indices, non-unitary matrices, or mismatched control sequences.

### Exception Examples

```python
# SWAP on the same qubit should raise an error
qs = Circuit(2)
qs.swap(0, 0)  # Invalid SWAP operation: the two target qubits must be different


# Non-unitary matrix should raise an error
qs = Circuit(1)
bad_u = np.array([[1, 1],
                  [0, 1]], dtype=complex)
qs.unitary(bad_u, [0])  # Invalid unitary operation: the input matrix is not unitary

# control_sequence length mismatch should raise an error
qs = Circuit(3)
qs.mcx([0, 1], 2, control_sequence="1")  # Invalid control sequence: length does not match the number of control qubits
```
