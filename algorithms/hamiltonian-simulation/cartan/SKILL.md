---
name: cartan
description: "Simulate the time evolution of a quantum system using Cartan decomposition."
---

# Cartan Decomposition Hamiltonian Simulation Skill Guide

## Algorithm Overview

**Category:** Hamiltonian Simulation — Structural Decomposition

Cartan decomposition computes the time-evolution operator

$$
U(t) = e^{-iHt}
$$

by splitting the Lie algebra $\mathfrak{g}$ associated with $H$ into a symmetric subalgebra $\mathfrak{k}$ and an antisymmetric complement $\mathfrak{m}$, then iterating a Lax flow to build an approximate quantum circuit in the form $K \cdot e^{-i\eta} \cdot K^\dagger$. This implementation uses the `cartan-lax` method via `unitarylab.library.hamiltonian.hamiltonian_simulation`.

---

## Mathematical Principles

The Lie algebra $\mathfrak{g}$ is decomposed as a direct sum:

$$
\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}
$$

with the canonical closure relations:

$$
[\mathfrak{k},\, \mathfrak{k}] \subseteq \mathfrak{k}, \qquad
[\mathfrak{k},\, \mathfrak{m}] \subseteq \mathfrak{m}, \qquad
[\mathfrak{m},\, \mathfrak{m}] \subseteq \mathfrak{k}
$$

The target unitary is factored as:

$$
U(t) = K \cdot e^{-i\eta} \cdot K^\dagger, \qquad K \in e^{\mathfrak{k}},\quad \eta \in \mathfrak{m}
$$

The `cartan-lax` flow iteratively updates $K$ via gradient-descent-like steps until the off-$\mathfrak{h}$ component norm falls below the target error $\epsilon$. The exact reference matrix

$$
U_{\text{exact}} = e^{-iH\,t_{\text{evol}}}
$$

is computed via `scipy.linalg.expm` for error benchmarking.

---

## Core Parameters

### `CartanDecompositionAlgorithm.__init__()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text_mode` | `str` | `"plain"` | Output text format mode. |
| `algo_dir` | `str\|None` | `None` | Directory for saving result files. Auto-generated if `None`. |

### `CartanDecompositionAlgorithm.run()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `H` | `np.ndarray \| list` | required | Hermitian Hamiltonian matrix (2D array). Must be real symmetric in the current implementation. |
| `t` | `float` | required | Total evolution time. Also used as default for `evol_time`. |
| `error` | `float` | required | Stopping tolerance for the off-$\mathfrak{h}$ component norm. Range: `[1e-10, 1e-2]`. |
| `evol_time` | `float` | `t` | Override for the evolution time passed to the simulator. |
| `lr` | `float` | `1e-3` | Base integration step size for the Lax flow. Range: `[1e-5, 1.0]`. |
| `max_steps` | `int` | `100000` | Hard cap on the number of Lax update steps. Range: `[1000, 1000000]`. |
| `reps` | `int` | `5000` | Baseline iteration budget before adaptive scaling. Range: `[100, 10000]`. |

---

## Inputs and Outputs

### Inputs

| Name | Type | Constraints |
|---|---|---|
| `H` | `np.ndarray` | Hermitian (real symmetric) matrix; shape `(2^n, 2^n)` |
| `t` | `float` | Positive real number |
| `error` | `float` | Positive tolerance; `1e-10 ≤ error ≤ 1e-2` |

### Outputs

The `run()` method returns a dictionary built by `_build_return_dict(success, circuit_path, filepath, circuit)`. The `self.output` fields are merged into the result via `result.update(self.output)`, so all keys below are accessible directly on the returned dict:

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` otherwise |
| `circuit_path` | `str` | Local path to the saved circuit diagram (SVG) |
| `plot` | `list` | List of saved result files, each as `{"format": str, "filename": str}` (format is the 3-char file extension) |
| `circuit` | `object` | Raw circuit object from `runable.circuit` |
| `Evolution result` | `np.ndarray` | Approximate unitary from the Cartan-Lax flow (`runable.evolution_result`) |
| `Final total error` | `float` | Achieved approximation error (`runable.total_error`) |
| `Computation time (s)` | `float` | Wall-clock runtime in seconds |
| `Exact evolution` | `np.ndarray` | Exact matrix $U_{\text{exact}} = e^{-iHt_{\text{evol}}}$ computed via `scipy.linalg.expm` |

---

## Implementation Notes

### Execution Flow

1. **Input recording** — Store `H`, `t`, and `error` via `self.update_input(...)`.
2. **Parameter expansion** — Read `evol_time`, `lr`, `max_steps`, `reps` from `**kwargs`; fall back to defaults when absent.
3. **Cartan-Lax simulation** — Call `hamiltonian_simulation(H, evol_time, method='cartan-lax', target_error=error, lr=lr, max_steps=max_steps, reps=reps)`.
4. **Exact reference** — Compute $U_{\text{exact}} = \text{expm}(-i H\, t_{\text{evol}})$ via `scipy.linalg.expm`.
5. **Result export** — Save the circuit diagram and text report; return the unified result dictionary via `self._build_return_dict(...)`.

### Engineering Constraints

- The current implementation handles Hamiltonians through the **matrix path** only; Pauli-string input is not yet supported.
- Computational cost scales with `lr`, `max_steps`, and `reps`. Tight `error` thresholds combined with small `lr` will increase runtime significantly.
- `max_steps` acts as a hard termination guard; convergence is not guaranteed if the budget is exhausted before `error` is met.
- Classical overhead from `expm` grows cubically with Hilbert-space dimension; this implementation is suited for small-to-medium systems.

---

## Sample Code

```python
import numpy as np
from unitarylab_algorithms import CartanDecompositionAlgorithm

# Define a 2x2 real symmetric Hamiltonian
H = np.array([[2.0, 1.0],
              [1.0, 2.0]])

algo = CartanDecompositionAlgorithm()

result = algo.run(
    H=H,
    t=1.0,
    error=1e-3,
    lr=1e-3,
    max_steps=100000,
    reps=5000,
)

print("status      :", result['status'])
print("circuit_path:", result['circuit_path'])
print("plot        :", result['plot'])          # e.g. [{'format': 'txt', 'filename': '/path/to/result.txt'}]
```

### Accessing Detailed Output

All `self.output` fields are merged into the returned dict, so they can be accessed either from `result` directly or from `algo.output`:

```python
# Approximate evolution unitary
U_approx = result["Evolution result"]        # same as algo.output["Evolution result"]

# Exact reference unitary
U_exact  = result["Exact evolution"]

# Achieved error
error_val = result["Final total error"]

# Saved files
for f in result["plot"]:
    print(f"Saved {f['format']} file: {f['filename']}")

print(f"Total error: {error_val:.2e}")
```