# PennyLane

Follow [SKILL.md](../SKILL.md) for task routing, framework selection, and execution permission.

## Scope and Version

This reference covers devices, QNodes, gates, analytic states/probabilities, sampling, expectations, and differentiable parameters. Device/plugin capabilities determine which operations, measurements, and differentiation methods are available.

[PennyLane 0.45.1](https://pypi.org/project/pennylane/0.45.1/) requires Python `>=3.11`. Use the official [QNode documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html), [measurement API](https://docs.pennylane.ai/en/stable/introduction/measurements.html), and [interfaces guide](https://docs.pennylane.ai/en/stable/introduction/interfaces.html).

## Environment

For an authorized environment setup:

```bash
python -m pip install "pennylane==0.45.1"
```

Add Torch, JAX, or a device plugin only for the selected interface or backend. Do not install several machine-learning frameworks for a basic simulator example. An interface's trainable arrays and supported versions are part of that interface's contract.

## Bell State and Probabilities

```python
import pennylane as qml

device = qml.device("default.qubit", wires=2, shots=None)

def prepare_bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

@qml.qnode(device)
def bell_state():
    prepare_bell()
    return qml.state()

@qml.qnode(device)
def bell_probabilities():
    prepare_bell()
    return qml.probs(wires=[0, 1])

print(bell_state())
print(bell_probabilities())
```

The ideal amplitudes are `[1/sqrt(2), 0, 0, 1/sqrt(2)]` and probabilities are `[0.5, 0, 0, 0.5]`. `shots=None` selects analytic calculations for this device; these probabilities are not finite-shot counts. State measurements require a compatible analytic simulator.

## Trainable Parameters and Gradients

```python
from pennylane import numpy as np

@qml.qnode(device, interface="autograd", diff_method="parameter-shift")
def expectation(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(wires=0))

theta = np.array(0.5, requires_grad=True)
print(expectation(theta))
print(qml.grad(expectation)(theta))
```

This snippet continues the imports and device above. Its theoretical expectation is `cos(theta)` and derivative is `-sin(theta)`. Use PennyLane NumPy with explicit trainability for this Autograd example; ordinary NumPy arrays do not establish trainability. For Torch or JAX, follow that interface's array and differentiation contract. A device's ability to execute a circuit does not imply support for every gradient method.

## Measurement and Circuit Interfaces

| Need | Interface and condition |
|---|---|
| Define a device | `qml.device(name, wires=..., ...)` |
| Define an executable quantum function | `@qml.qnode(device, interface=..., diff_method=...)` |
| Basic gates | `qml.Hadamard`, `qml.CNOT`, `qml.RX`, `qml.RY`, `qml.RZ` with explicit wires |
| Statevector | `qml.state()` on a compatible analytic device |
| Probabilities | `qml.probs(wires=[...])`; exact in this analytic example, estimated with finite shots |
| Observable expectation | `qml.expval(observable)` |
| Samples/counts | `qml.sample(...)`, `qml.counts(...)` with finite shots |
| Inspect the circuit | `qml.draw(qnode)(...)`, or the documented drawing/specification tools |
| Select finite shots | `qml.set_shots(qnode, shots=1000)` returns a shot-configured QNode |

PennyLane basis order follows the device wire order; for probabilities, the supplied `wires` sequence specifies the subsystem order, with its first wire as the most significant bit. Do not assume Qiskit's or UnitaryLab's displayed-string convention. Use an asymmetric basis state when checking a conversion, because the Bell state alone does not distinguish opposite bit orders.

## Capabilities and Limits

`default.qubit` supports the qubit workflow above. Mixed-state noise, continuous-variable operations, acceleration, compilation, and hardware access require a compatible device or additional package; they are not universal QNode capabilities. Check the selected device's measurements, shots, operations, and differentiation support before generating framework-specific code.

A valid gradient is evidence about a specific objective and parameter point, not a convergence guarantee or quantum advantage. Dense simulation still has exponential state-size costs. Finite-shot estimates have sampling uncertainty even when the theoretical probabilities are known.
