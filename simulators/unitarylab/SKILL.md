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

## Environment Installation

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
2. You are about to call a terminal command that imports `unitarylab`.
3. The user reports a `ModuleNotFoundError: unitarylab` error.
4. A script exists and the user wants to verify its output.

### Conditions that do NOT require installation

Skip installation when:
- Explaining what UnitaryLab is or how it works.
- Writing or reviewing circuit code without running it.
- Answering questions about API methods or gate behavior.
- Designing an algorithm at a conceptual level.
- Showing code examples inline (without executing them).

---

### Step-by-Step Installation

### Install `uv` First (before any `uv ...` command)

Use a Python-based install first (works across Windows, macOS, and Linux):

```bash
python -m pip install -U uv
```

If `python` is not on PATH, use your interpreter explicitly:

```bash
python3 -m pip install -U uv
```

If pip-based install is unavailable in your environment, use OS-specific installers:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify:

```bash
uv --version
```

#### Using `uv` (Recommended)

download via pip
```bash
# Ensure `uv` is installed first (see section above)
uv venv --python 3.11
uv pip install unitarylab
uv run python -c "import unitarylab; print('UnitaryLab OK')"
```

#### Using `conda`
```bash
# Step 1 — Create a Python 3.11 conda environment (skip if already exists)
conda create -n unitarylab-env python=3.11

# Step 2 — Activate it
conda activate unitarylab-env

# Step 3 — Install UnitaryLab from the pip matching your OS
python -m pip install unitarylab

# Step 4 — Verify installation
python -c "import unitarylab; print('UnitaryLab OK')"
```

**Note**

- Package name: `unitarylab`
- Import name: `unitarylab`
- Wheels are platform-specific; using the wrong wheel will result in a "not supported on this platform" error.

### Verify Before Running Any Script

Before executing user code, confirm:
```bash
python -c "from unitarylab import Circuit; print('OK')"
```
If this prints `OK`, proceed. If it raises `ModuleNotFoundError`, go back to Step 3.

---

## Minimal Executable Example

```python
import numpy as np
from unitarylab import Circuit

qc = Circuit(2)
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
| `ModuleNotFoundError: unitarylab` | Wheel not installed in active environment. | Run `pip install unitarylab` in the active env. |
| `Wheel is not supported on this platform` | Wrong wheel for your OS/architecture, or wrong Python version. | Use Python 3.11 and pick the correct wheel: `win_amd64` (Windows), `macosx_11_0_arm64` (macOS), or `linux_x86_64` (Linux). |
| Wrong conda environment active | `conda activate` not run. | Run `conda activate unitarylab-env` before executing. |
| `unitarylab` imports but results are wrong | Initial state not copied before passing to `execute()`. | Always pass `initial_state.copy()` to preserve the original. |

---

## Library API Quick Reference

> **Agent instruction:** Before writing any code that calls a high-level quantum algorithm, scan this table first to find the right symbol. Then open `./references/api-reference.md` → `unitarylab.library` for the full parameter list and return types.

All high-level quantum algorithms are in `unitarylab.library`:

```python
from unitarylab.library import <symbol>
```

| Symbol | Key signature | What it does |
|--------|--------------|--------------|
| `QFT` | `QFT(n)` | Returns an `n`-qubit QFT `Circuit` |
| `IQFT` | `IQFT(n)` | Returns an `n`-qubit inverse QFT `Circuit` (dagger of QFT) |
| `QPE` | `QPE(U, d, prepare_target=None, return_circuit=False)` | Quantum Phase Estimation — estimates the phase of unitary `U` using `d` ancilla qubits; returns `(circuit, phi_est, probability)` |
| `LCU` | `LCU(decompositions)` | Linear Combination of Unitaries — implements `A = Σ αⱼ Uⱼ` from a list of `(Circuit, coeff)` pairs |
| `QSP` | `QSP(U, n, m, coef, parity, ...)` | Quantum Signal Processing — builds a circuit that applies a polynomial transformation to a block-encoded unitary |
| `QSP_hamiltonian_simulation` | `QSP_hamiltonian_simulation(U_H, n, alpha, m, t, epsilon, beta, flag)` | Hamiltonian simulation via QSP; approximates `exp(−iHt)` from a block-encoding `U_H` of `H` |
| `QSVT` | `QSVT(H, function, target_error=1e-6, block_encoding_method='nagy')` | Quantum Singular Value Transformation — applies a scalar function `f(H)` to Hermitian matrix `H`; returns `QSVTResult` |
| `block_encode` | `block_encode(matrix, method='fable', eps=1e-3, verbose=False)` | Block-encodes a matrix; returns `BlockEncodingResult` (`.circuit`, `.alpha`, `.get_encoded_matrix()`, …) |
| `hamiltonian_simulation` | `hamiltonian_simulation(H, t, method='trotter', target_error=1e-6, **kwargs)` | Simulate `exp(−iHt)` for Hermitian `H`; available methods: `'trotter'`, `'qdrift'`, `'taylor'`, `'qsp'`/`'qsvt'`, `'cartan-lax'`, `'cartan-optimization'` |
| `solve` | `solve(A, b, method='hhl', **kwargs)` | Quantum linear-system solver for `Ax = b`; methods: `'hhl'` (default), `'qsvt'`, `'schro'`, `'schro_trotter'`, `'schro_classical'` |

**Choose a function:**
- Need to encode a matrix into a circuit? → `block_encode`
- Need to evolve a Hamiltonian in time? → `hamiltonian_simulation`
- Need to solve a linear system? → `solve`
- Need phase estimation on a known unitary? → `QPE`
- Need a polynomial function of a matrix? → `QSVT`
- Need to decompose an operator as a sum of unitaries? → `LCU`
- Need QFT as a sub-circuit? → `QFT` / `IQFT`

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
