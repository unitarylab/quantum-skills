# Qiskit

Follow [SKILL.md](../SKILL.md) for task routing, framework selection, and execution permission.

## Scope and Version

This reference covers circuit construction, statevector inspection, measurement sampling, parameter binding, and expectation values. Qiskit is an SDK; its core statevector tools, the separate Aer simulator package, and hardware-provider runtimes are distinct components.

[Qiskit 2.6.1](https://pypi.org/project/qiskit/2.6.1/) requires Python `>=3.10`. The examples follow the official [circuit API](https://quantum.cloud.ibm.com/docs/api/qiskit/circuit), [quantum-information API](https://quantum.cloud.ibm.com/docs/api/qiskit/quantum_info), and [V2 primitives](https://quantum.cloud.ibm.com/docs/api/qiskit/primitives).

## Environment

For an authorized environment setup:

```bash
python -m pip install "qiskit==2.6.1"
```

Install `qiskit-aer` only for Aer-specific simulation or noise features, `qiskit-ibm-runtime` for a requested IBM provider workflow, and `qiskit-algorithms` for the APIs documented by the gradient/eigensolver references. Those packages have independent versions; installation of core Qiskit does not establish their compatibility. Do not use the old `qiskit-ibmq` installation instructions.

## Circuit and Exact State

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

state = Statevector.from_instruction(circuit)
print(state.data)
print(state.probabilities_dict())
```

The ideal Bell-state amplitudes are `[1/sqrt(2), 0, 0, 1/sqrt(2)]`; the only nonzero probabilities are `00: 0.5` and `11: 0.5`, up to floating-point error. Use a circuit without measurements for this unitary statevector workflow.

## Finite-Shot Sampling

```python
from qiskit.primitives import StatevectorSampler

measured = circuit.copy()
measured.measure_all()
sampler = StatevectorSampler(seed=7)
result = sampler.run([measured], shots=1000).result()
counts = result[0].data.meas.get_counts()
print(counts)
```

This snippet continues the circuit example. `measure_all()` creates a classical register named `meas`; use the actual register name for a different circuit. Counts sum to 1000 and fluctuate around the ideal probabilities. There is no guarantee of exactly 500 occurrences of each Bell outcome. V2 primitive results are indexed per input PUB; do not read the removed concrete V1 `Sampler`/`Estimator` result layout from them.

## Parameter Binding and Expectation Values

```python
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

theta = Parameter("theta")
parametric = QuantumCircuit(1)
parametric.ry(theta, 0)
bound = parametric.assign_parameters({theta: 0.5})

observable = SparsePauliOp.from_list([("Z", 1.0)])
estimate = StatevectorEstimator().run([(bound, observable)]).result()
print(estimate[0].data.evs)
```

This snippet also uses `QuantumCircuit` from the first example. The ideal expectation is `cos(0.5)`. `assign_parameters()` returns a new circuit by default; `inplace=True` mutates the original. A parameterized V2 PUB may instead supply parameter values alongside the circuit and observable; confirm parameter order when passing positional values.

## Result and Ordering Contracts

| Need | Interface |
|---|---|
| Gate construction | `QuantumCircuit.h`, `.cx`, `.rx`, `.ry`, `.rz` |
| Bind parameters | `circuit.assign_parameters(...)` |
| Exact statevector | `Statevector.from_instruction(circuit).data` |
| Exact probabilities | `state.probabilities_dict()` |
| Exact observable expectation | `state.expectation_value(observable)` |
| V2 expectation estimates | `StatevectorEstimator`; each PUB result has `.data.evs` |
| V2 measurement counts | `StatevectorSampler`; each classical register provides `.get_counts()` |
| Backend compilation | `qiskit.transpile` or the documented pass-manager workflow |

Displayed computational-basis strings place qubit 0 on the right, and statevector index `i` is the basis state's binary integer value. Qiskit Pauli labels also place the qubit-0 operator on the right: `IZ` acts as Z on qubit 0. This differs from UnitaryLab's default Pauli-string character-to-qubit order. Explicit classical register mappings determine count keys.

## Capabilities and Limits

Aer supplies additional simulation methods and noise models; supported methods, devices, and noise features depend on its release and backend configuration. Hardware execution requires a separate provider interface and a circuit compatible with the selected target. Do not infer an available processor size, GPU speedup, noise capability, or hardware access from Qiskit's installation alone.

Dense statevectors require memory exponential in the number of qubits. Exact simulation, finite-shot sampling, and real-device results provide different evidence and should be reported accordingly. See the [official bit-ordering guide](https://quantum.cloud.ibm.com/docs/guides/bit-ordering) and [Aer documentation](https://qiskit.github.io/qiskit-aer/) for task-specific details.
