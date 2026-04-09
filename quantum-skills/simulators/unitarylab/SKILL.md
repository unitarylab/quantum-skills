---
name: unitarylab
description: UnitaryLab is a quantum computing framework for education and research. It offers a simple interface for building and simulating quantum circuits, making it suitable for learning, teaching, and experimenting with quantum algorithms.
---

# UnitaryLab

## Purpose
Use UnitaryLab as the default local simulator for quantum skill tasks.

Choose it when you need:
- Fast circuit prototyping.
- Educational algorithm demonstrations.
- Small, runnable examples with clear outputs.

## Quick Start

1. Create and activate Python 3.11 environment.
2. Choose the corresponding wheel and install it from this skill folder.
3. Run a minimal circuit and verify output.

Example commands:

```bash
conda create -n unitarylab-env python=3.11
conda activate unitarylab-env
python -m pip install ./dist/unitarylab_engine-*.whl
```

## Minimal Executable Example

```python
import numpy as np
from engine import GateSequence

qc = GateSequence(2)
qc.h(0)
qc.cx(0, 1)

initial_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
final_state = qc.execute(initial_state.copy())
probs = np.abs(final_state) ** 2
print(final_state)
print(probs)  # expected: [0.5, 0.0, 0.0, 0.5]
```

## Working Standard for Future Examples
- Do not stop at circuit construction; execute the circuit.
- Print at least one validation artifact:
    - final statevector, or
    - measurement probabilities.
- Include expected behavior in one line.

## Common Pitfalls
- ModuleNotFoundError: engine
    - Cause: UnitaryLab wheel not installed in active environment.
    - Fix: install wheel in the same interpreter used to run script.

- Wheel is not supported on this platform
    - Cause: wheel tag does not match Python version, OS, or architecture.
    - Fix: use a compatible wheel for your environment.

## Reference
- Circuit building details: ./references/circuitsbuild.md