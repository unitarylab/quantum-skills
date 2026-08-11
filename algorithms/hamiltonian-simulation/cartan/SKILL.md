---
name: cartan
description: "Simulate the time evolution of a quantum system using Cartan decomposition. Skill-first for covered code generation, runnable examples, execution, debugging, validation, and fixed workflows."
---

# Cartan Decomposition Hamiltonian Simulation Skill Guide

## Overview

**Category:** Hamiltonian Simulation — Structural Decomposition

Cartan decomposition computes the time-evolution operator

$$
U(t) = e^{-iHt}
$$

by splitting the Lie algebra $\mathfrak{g}$ associated with $H$ into a symmetric subalgebra $\mathfrak{k}$ and an antisymmetric complement $\mathfrak{m}$, then iterating a Lax flow to build an approximate quantum circuit in the form $K \cdot e^{-i\eta} \cdot K^\dagger$. This implementation uses the `cartan-lax` method via `unitarylab.library.hamiltonian.hamiltonian_simulation`.

---

## How to Use This Skill

Use this skill when the user asks to explain, run, debug, modify, or reimplement Cartan Decomposition Hamiltonian Simulation.

Cartan decomposition splits the Lie algebra of the Hamiltonian into symmetric ($\mathfrak{k}$) and antisymmetric ($\mathfrak{m}$) subalgebras, then uses a Lax flow to iteratively build the time-evolution operator $U(t) = K e^{-i\eta} K^\dagger$.

Use this skill when:
- The Hamiltonian is a real symmetric matrix suitable for Cartan-Lax decomposition
- High-precision Hamiltonian simulation is needed with controllable error tolerance
- Lie-algebraic structure of the Hamiltonian is important

When using this skill:
- **Explanation:** Explain the algorithm, assumptions, mathematical model, and limitations. Do not generate code unless the user requests it.
- **Run or reuse:** Generate standalone task code first. Do not import from or depend on this skill's `scripts/` directory at runtime.
- **Debugging:** Run the smallest documented example first. Compare the observed result with the documented inputs, outputs, status fields, and numerical tolerances before changing code.
- **Modification or reimplementation:** Follow the implementation architecture and theory-to-code mapping. Preserve the documented parameter schema, execution flow, and return contract.
- **Reference scripts:** Treat `scripts/algorithm.py` and any `*_implementation.py` files as reference-only material for troubleshooting, API comparison, and validation.
- **Validation:** When practical, validate with a small deterministic example and report backend, dependency, and scale limitations.

## Reference Implementation Example

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

## Core Parameters Explained

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

## Return Fields

The `run()` method returns a dictionary built by `_build_return_dict(success, circuit_path, filepath, circuit)`. The `self.output` fields are merged into the result via `result.update(self.output)`.

| Key | Type | Description |
|---|---|---|
| `status` | `str` | `'ok'` on success, `'failed'` otherwise |
| `circuit_path` | `str` | Local path to the saved circuit diagram (SVG) |
| `plot` | `list` | List of saved result files, each as `{"format": str, "filename": str}` |
| `circuit` | `object` | Raw circuit object from `runable.circuit` |
| `Evolution result` | `np.ndarray` | Approximate unitary from the Cartan-Lax flow |
| `Final total error` | `float` | Achieved approximation error |
| `Computation time (s)` | `float` | Wall-clock runtime in seconds |
| `Exact evolution` | `np.ndarray` | Exact matrix $U_{\text{exact}} = e^{-iHt_{\text{evol}}}$ via `scipy.linalg.expm` |

## Implementation Architecture

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

## Mathematical Deep Dive

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

## Prerequisites

- Lie algebra basics: decomposition $\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}$ with canonical closure relations
- Hamiltonian simulation fundamentals: $U(t) = e^{-iHt}$
- Python: `numpy`, `scipy.linalg.expm`
- Understanding of gradient-descent-like iterative flows on Lie algebras

## Understanding the Key Quantum Components

1. **Lie algebra decomposition**: The Hamiltonian's Lie algebra $\mathfrak{g}$ is split into a symmetric subalgebra $\mathfrak{k}$ (closed under commutator, $[\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k}$) and an antisymmetric complement $\mathfrak{m}$ (where $[\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}$). This structural split is the foundation of the Cartan factorization.
2. **Lax flow iteration**: `cartan-lax` iteratively updates $K \in e^{\mathfrak{k}}$ via gradient-descent-like steps, driving the off-$\mathfrak{h}$ component norm below the target error $\epsilon$. The `lr` parameter controls step size; `max_steps` acts as a hard termination guard.
3. **Target factorization**: $U(t) = K \cdot e^{-i\eta} \cdot K^\dagger$ with $K \in e^{\mathfrak{k}}$ and $\eta \in \mathfrak{m}$. The circuit construction follows from this factored form.
4. **Error control**: `error` sets the stopping tolerance for the off-$\mathfrak{h}$ norm. Tight tolerances combined with small `lr` increase iteration count significantly.
5. **Matrix-only path**: The current implementation operates on dense matrices — Pauli-string input is not yet supported. The classical `expm` overhead grows cubically with Hilbert-space dimension.

## Theory-to-Code Mapping

| Theory Concept | Code Object or Location |
|---|---|
| Hamiltonian $H$ | `H` — 2D `np.ndarray`, real symmetric |
| Evolution time $t$ | `t` parameter; also `evol_time` override |
| Target error $\epsilon$ | `error` parameter; range `[1e-10, 1e-2]` |
| Lax step size | `lr` parameter; range `[1e-5, 1.0]` |
| Maximum iterations | `max_steps`; hard cap `[1000, 1000000]` |
| Iteration budget | `reps`; baseline before adaptive scaling `[100, 10000]` |
| Cartan-Lax engine | `hamiltonian_simulation(H, evol_time, method='cartan-lax', ...)` from `unitarylab.library` |
| Exact reference $e^{-iHt}$ | `scipy.linalg.expm(-1j * H * t_evol)` |
| Approximate unitary $U_{\text{approx}}$ | `result["Evolution result"]` |
| Achieved error | `result["Final total error"]` |

## Hands-On Example

Sweep error tolerance to observe accuracy-vs-runtime trade-offs:

```python
import numpy as np
from unitarylab_algorithms import CartanDecompositionAlgorithm

H = np.array([[2.0, 1.0],
              [1.0, 2.0]])

for error in [1e-2, 1e-3, 1e-5]:
    algo = CartanDecompositionAlgorithm()
    result = algo.run(H=H, t=1.0, error=error, lr=1e-3, max_steps=100000)
    print(f"error_tol={error:.0e}, achieved={result['Final total error']:.2e}, "
          f"runtime={result['Computation time (s)']:.3f}s, status={result['status']}")
```

Expected behavior: tighter `error` produces lower `Final total error` but increases `Computation time`.

## Minimal Manual Implementation

```python
import numpy as np
from scipy.linalg import expm

def cartan_simulation_skeleton(H, t, lr=1e-3, max_steps=100000, error=1e-3):
    """Simplified Cartan-Lax flow — mirrors the structural decomposition.

    In practice, the full implementation uses `hamiltonian_simulation()`
    from `unitarylab.library` with `method='cartan-lax'`. This skeleton
    illustrates the conceptual factorization U = K @ expm(-1j*eta) @ K†.
    """
    n = H.shape[0]
    # 1. Decompose Lie algebra into k and m components
    #    (simplified: use anti-symmetric / symmetric split)
    k_component = (H - H.T) / 2       # anti-symmetric ≈ k
    m_component = (H + H.T) / 2       # symmetric ≈ m

    # 2. Iterative Lax flow to converge K
    K = np.eye(n, dtype=complex)
    for step in range(max_steps):
        # Compute off-m component residual
        residual = K @ m_component @ K.conj().T
        off_m_norm = np.linalg.norm(residual - np.diag(np.diag(residual)))
        if off_m_norm < error:
            break
        # Gradient step on k
        dK = -lr * (k_component @ K)
        K = K + dK
        # Re-orthogonalize (project back to unitary)
        U_k, _, Vh = np.linalg.svd(K)
        K = U_k @ Vh

    # 3. Diagonalize residual in m
    eta = np.diag(np.diag(K.conj().T @ H @ K)) * t

    # 4. Assemble: U ≈ K @ expm(-1j * eta) @ K†
    U_approx = K @ expm(-1j * eta) @ K.conj().T
    U_exact = expm(-1j * H * t)
    return U_approx, np.linalg.norm(U_approx - U_exact, 'fro')
```

Note: This skeleton captures the conceptual flow. The actual implementation invokes `hamiltonian_simulation()` from the UnitaryLab library with full `cartan-lax` support.

## Debugging Tips

1. **Convergence failure**: If `max_steps` is exhausted before `error` is met, increase `max_steps` or relax `error`. Small `lr` with tight `error` is the most common cause of slow convergence.
2. **lr too large**: The Lax flow may oscillate or diverge. If the final error is worse than expected, halve `lr` and re-run. Start with `1e-3` and adjust.
3. **Non-symmetric H**: The current implementation expects real symmetric Hamiltonians. If `H` has significant imaginary components, the `k`/`m` split may produce unexpected results — verify symmetry with `np.allclose(H, H.T)`.
4. **Dimension scaling**: `expm` cost scales as $O((2^n)^3)$. For $n > 4$ qubits (dimension 16), the classical reference computation dominates runtime. Use for small-to-medium systems.
5. **reps vs max_steps**: `reps` sets the baseline iteration budget before adaptive scaling; `max_steps` is the absolute hard cap. If the flow converges near `reps`, increasing it can reduce total iterations by enabling better adaptive scaling.
