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

---

## Environment Installation — When to Act

> **Rule: Only install the environment when the user is about to run or execute code.**
> If the task is explanation, circuit design, concept review, or code writing without execution, skip all installation steps and proceed directly to the relevant section.

### Decision: Do I need to install?

```
Is the user asking to RUN or EXECUTE a script?
│
├─ YES → Check environment (see "Installation Steps" below)
│
└─ NO  → Skip installation entirely.
         Introduce UnitaryLab concepts, write code, explain circuits.
         No conda, no pip, no wheel needed.
```

### Conditions that require installation

Install **only** when one of these is true:
1. The user says "run", "execute", or "try" the code.
2. You are about to call a terminal command that imports `engine`.
3. The user reports a `ModuleNotFoundError: engine` error.
4. A script exists and the user wants to verify its output.

### Conditions that do NOT require installation

Skip installation when:
- Explaining what UnitaryLab is or how it works.
- Writing or reviewing circuit code without running it.
- Answering questions about API methods or gate behavior.
- Designing an algorithm at a conceptual level.
- Showing code examples inline (without executing them).

---

## Installation Steps (run only when needed)

### Available Wheel
The pre-built wheel is located in this skill folder:

```
./dist/unitarylab_engine-*.whl
```

**Compatibility requirements:**
| Requirement | Value |
|-------------|-------|
| Python | 3.11 exactly |
| OS | Windows |
| Architecture | x86-64 (AMD64) |

If the environment does not match these constraints, the wheel will fail to install. Do not attempt to install on Linux, macOS, or Python 3.10/3.12.

### Step-by-Step Installation

```bash
# Step 1 — Create a Python 3.11 conda environment (skip if already exists)
conda create -n unitarylab-env python=3.11

# Step 2 — Activate it
conda activate unitarylab-env

# Step 3 — Install UnitaryLab from the local wheel
#          Run this from the directory containing the dist/ folder,
#          i.e. from: .agents/skills/quantum-skills/simulators/unitarylab/
python -m pip install ./dist/unitarylab_engine-*.whl

# Step 4 — Verify installation
python -c "from engine import GateSequence; print('UnitaryLab OK')"
```

### Verify Before Running Any Script

Before executing user code, confirm:
```bash
python -c "from engine import GateSequence; print('OK')"
```
If this prints `OK`, proceed. If it raises `ModuleNotFoundError`, go back to Step 3.

---

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

---

## Working Standard for Future Examples
- Do not stop at circuit construction; execute the circuit.
- Print at least one validation artifact:
    - final statevector, or
    - measurement probabilities.
- Include expected behavior in one line.

---

## Common Pitfalls

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: engine` | Wheel not installed in active environment. | Run `pip install ./dist/unitarylab_engine-*.whl` in the active env. |
| `Wheel is not supported on this platform` | Python version, OS, or architecture mismatch. | Use Python 3.11 on Windows x64, or switch to Qiskit/PennyLane on other platforms. |
| Wrong conda environment active | `conda activate` not run. | Run `conda activate unitarylab-env` before executing. |
| `engine` imports but results are wrong | Initial state not copied before passing to `execute()`. | Always pass `initial_state.copy()` to preserve the original. |

---

## API Read-and-Recall Workflow
Before writing code, read the local API docs in this order:

### Reference
- Simulator API reference: `./references/api-reference.md`
- Circuit building details: `./references/circuitsbuild.md`


When to re-open the API docs:
- You are unsure about method names or argument order.
- You are unsure about expected input state format.
- You need multi-qubit gates, controlled gates, or less common operations.
- A script runs but outputs look physically wrong (normalization or basis-order issues).
